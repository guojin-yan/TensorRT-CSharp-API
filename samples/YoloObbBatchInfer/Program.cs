using JYPPX.TensorRtSharp.Cuda;
using JYPPX.TensorRtSharp.Nvinfer;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace YoloObbBatchInfer
{
    internal class Program
    {
        // ================= 配置参数 =================
        // 模型输入尺寸 (宽=高)
        private const int InputSize = 1024;


        // 建议根据实际模型动态获取或使用 Netron 查看
        private const int OutputSize = 21504;

        // 模型类别数 (根据您的具体数据集修改，此处假设为15类)
        private const int CategoryNum = 15;

        // 置信度阈值
        private const float ConfThreshold = 0.25f;

        // NMS IOU 阈值
        private const float NmsThreshold = 0.3f;

        private const int MaxBatchSize = 24;

        static void Main(string[] args)
        {
            //  ============= 配置 TensorRT 日志回调 =============
            // 定义一个委托，用于处理 TensorRT 内部产生的日志消息。
            // 这允许我们将 C++ 层面的日志输出到 C# 的控制台。
            LogCallbackFunction _callbackDelegate = (message) =>
            {
                Console.WriteLine(message);
            };

            // 将自定义的回调函数注册给 TensorRT 的全局 Logger 实例。
            Logger.Instance.SetCallback(_callbackDelegate);

            // 设置日志的严重性级别阈值。
            // LoggerSeverity.kINFO: 打印信息、警告和错误。
            // 开发调试阶段通常设为 kINFO 或 kVERBOSE；生产环境可设为 kWARNING 或 kERROR 以减少输出。
            Logger.Instance.SetThreshold(LoggerSeverity.kINFO);

            string enginePath = "yolov8s-obb_b.engine";
            string[] imagePaths = { 
                "P0006.png" , "P0016.png", "P0456.png", "P0813.png"};


            // ================= 1. 加载 TensorRT Engine =================
            // 使用 using 语句确保文件流正确关闭
            byte[] engineData;
            using (FileStream fs = new FileStream(enginePath, FileMode.Open, FileAccess.Read))
            using (BinaryReader br = new BinaryReader(fs))
            {
                engineData = br.ReadBytes((int)fs.Length);
            }

            // 反序列化 Engine
            // Runtime 必须在 Engine 生命周期内保持存活，通常建议设为全局或静态，或者确保它最后释放
            Runtime runtime = new Runtime();
            runtime.setMaxThreads(10);
            // 创建 CudaEngine (此处使用 using 确保推理完成后引擎被销毁)
            using (CudaEngine cudaEngine = runtime.deserializeCudaEngineByBlob(engineData, (ulong)engineData.Length))
            {
                // ================= 2. 初始化推理上下文与显存 =================
                // 创建执行上下文
                using (JYPPX.TensorRtSharp.Nvinfer.ExecutionContext executionContext = cudaEngine.createExecutionContext(TrtExecutionContextAllocationStrategy.kSTATIC))
                using (CudaStream cudaStream = new CudaStream()) // 创建 CUDA 流用于异步执行
                {
                    // 获取输入维度信息 (用于校验)
                    Dims inputDims = executionContext.getTensorShape("images");
                    Logger.Instance.INFO($"Input Shape: {inputDims.d[0]}x{inputDims.d[1]}x{inputDims.d[2]}x{inputDims.d[3]}");

                    // 计算所需显存大小
                    // 输入: Batch=1, Channel=3, Height=1024, Width=1024
                    ulong inputSizeInBytes = MaxBatchSize * 3 * InputSize * InputSize;
                    // 输出: Batch=1, Channels=CategoryNum+4(box)+1(angle), Num=8400
                    int outputChannels = CategoryNum + 5; // 4坐标 + 1角度 + N类别
                    ulong outputSizeInBytes = (ulong)(MaxBatchSize * outputChannels * OutputSize);

                    Stopwatch sw = new Stopwatch();
                    // 分配 GPU 显存
                    using (Cuda1DMemory<float> inputGpuMemory = new Cuda1DMemory<float>(inputSizeInBytes))
                    using (Cuda1DMemory<float> outputGpuMemory = new Cuda1DMemory<float>(outputSizeInBytes))
                    {
                        // 绑定显存地址到 TensorRT 上下文
                        executionContext.setInputTensorAddress("images", inputGpuMemory.get());
                        executionContext.setOutputTensorAddress("output0", outputGpuMemory.get());

                        // 关键一步，修改本次推理的形状
                        executionContext.setinputShape("images", new Dims(imagePaths.Count(), 3, 1024, 1024));
                        // 预热推理 (可选，但推荐，尤其是首次推理时)
                        executionContext.executeV3(cudaStream);
                        cudaStream.Synchronize();

                        // ================= 3. 图像预处理 =================
                        List<Mat> images = new List<Mat>();
                        foreach (var path in imagePaths) 
                        {
                            Mat img = Cv2.ImRead(path);
                            if (img.Empty())
                            {
                                Logger.Instance.INFO("Image not found!");
                                return;
                            }
                            images.Add(img);
                        }

                        (float[] inputData1, float[] scales1, int[] xOffsets1, int[] yOffsets1) = PreProcessBatch(images);
                        sw.Start();
                        (float[] inputData, float[] scales, int[] xOffsets, int[] yOffsets) = PreProcessBatch(images);
                        sw.Stop();
                        Logger.Instance.INFO($"Pre-processing time: {sw.ElapsedMilliseconds} ms");
                        // ================= 4. 推理 =================

                        // 准备主机内存接收结果
                        float[] outputData1 = new float[imagePaths.Count() * outputChannels * OutputSize];
                        // 将数据从主机 拷贝到设备
                        inputGpuMemory.copyFromHostAsync(inputData, cudaStream);

                        // 执行推理 (enqueueV3 是异步的)
                        executionContext.executeV3(cudaStream);
                        // 等待推理完成
                        cudaStream.Synchronize();
                        // 将结果从设备 拷贝回主机
                        // 这里的拷贝是同步的，会等待 GPU 计算完成
                        outputGpuMemory.copyToHostAsync(outputData1, cudaStream);

                        sw.Restart();
                        // 准备主机内存接收结果
                        float[] outputData = new float[imagePaths.Count() * outputChannels * OutputSize];
                        // 将数据从主机 拷贝到设备
                        inputGpuMemory.copyFromHostAsync(inputData, cudaStream);

                        // 执行推理 (enqueueV3 是异步的)
                        executionContext.executeV3(cudaStream);
                        // 等待推理完成
                        cudaStream.Synchronize();
                        // 将结果从设备 拷贝回主机
                        // 这里的拷贝是同步的，会等待 GPU 计算完成
                        outputGpuMemory.copyToHostAsync(outputData, cudaStream);

                        sw.Stop();
                        Logger.Instance.INFO($"Inference time: {sw.ElapsedMilliseconds} ms");
                        // ================= 5. 后处理 =================
                        List<List<ObbData>> results1 = PostProcessBatch(outputData, scales, xOffsets, yOffsets);
                        sw.Restart();
                        List<List<ObbData>> results = PostProcessBatch(outputData, scales, xOffsets, yOffsets);
                        sw.Stop();
                        Logger.Instance.INFO($"Post-processing time: {sw.ElapsedMilliseconds} ms");

                        // ================= 6. 结果可视化 =================
                        List<Mat> resultMats = new List<Mat>();
                        for(int i = 0; i < results.Count; ++i)
                        {
                            resultMats.Add(DrawObbResult(results[i], images[i]));
                        }
                        Mat putResultImgs = StitchHorizontalWithPadding(resultMats);
                        Cv2.ImWrite("YOLO11-OBB Result.png", putResultImgs);
                        Cv2.ImShow("YOLO11-OBB Result", putResultImgs);
                        Cv2.WaitKey(0);
                    }
                }
            }
        }

        /// <summary>
        /// 图像预处理：Letterbox 缩放、归一化、HWC 转 CHW
        /// </summary>
        private static (float[], float[] ,  int[] , int[] ) PreProcessBatch(List<Mat> imgs)
        {
            int dataLen = 3 * InputSize * InputSize;
            float[] chwData = new float[imgs.Count * dataLen];
            float[] scales = new float[imgs.Count];
            int[] xOffsets = new int[imgs.Count];
            int[]  yOffsets = new int[imgs.Count];
            Parallel.For(0, imgs.Count, i =>
            {
                Mat img = imgs[i];
                // 转换颜色空间 BGR -> RGB
                Mat rgbImg = new Mat();
                Cv2.CvtColor(img, rgbImg, ColorConversionCodes.BGR2RGB);

                // 计算 Letterbox 缩放比例
                int maxDim = Math.Max(rgbImg.Width, rgbImg.Height);
                scales[i] = (float)maxDim / InputSize;

                // 计算缩放后的尺寸
                int newWidth = (int)(rgbImg.Width / scales[i]);
                int newHeight = (int)(rgbImg.Height / scales[i]);

                // Resize 图像
                Mat resizedImg = new Mat();
                Cv2.Resize(rgbImg, resizedImg, new Size(newWidth, newHeight));

                // 创建黑色背景 Canvas (InputSize x InputSize)
                Mat paddedImg = Mat.Zeros(InputSize, InputSize, MatType.CV_8UC3);

                // 计算粘贴位置 (居中)
                xOffsets[i] = (InputSize - newWidth) / 2;
                yOffsets[i] = (InputSize - newHeight) / 2;

                // 将图像拷贝到 Canvas 中央
                Rect roi = new Rect(xOffsets[i], yOffsets[i], newWidth, newHeight);
                resizedImg.CopyTo(new Mat(paddedImg, roi));

                // 归一化 (0-255 -> 0-1) 并转为 float 类型
                Mat floatImg = new Mat();
                paddedImg.ConvertTo(floatImg, MatType.CV_32FC3, 1.0 / 255.0);

                // HWC 转 CHW 并展平为一维数组
                Mat[] channels = Cv2.Split(floatImg);


                // 拷贝数据：R通道 -> C通道 -> B通道 (OpenCV Split 出来顺序是 B, G, R，对应索引 0, 1, 2)
                int channelSize = InputSize * InputSize;
                // 将 R, G, B 依次拷入数组
                Marshal.Copy(channels[0].Data, chwData, dataLen * i, channelSize); // R
                Marshal.Copy(channels[1].Data, chwData, dataLen * i + channelSize, channelSize); // G
                Marshal.Copy(channels[2].Data, chwData, dataLen * i + channelSize * 2, channelSize); // B

                // 释放临时 Mat
                rgbImg.Dispose();
                resizedImg.Dispose();
                paddedImg.Dispose();
                floatImg.Dispose();
                foreach (var c in channels) c.Dispose();
            });



            return (chwData, scales, xOffsets, yOffsets);
        }

        /// <summary>
        /// 后处理：解析 TensorRT 输出、NMS 过滤
        /// </summary>
        private static List<List<ObbData>> PostProcessBatch(float[] result, float[] scales, int[] xOffsets, int[] yOffsets)
        {
            List<ObbData>[] obbDatas = new List<ObbData>[scales.Length];

            Parallel.For(0, scales.Length, b =>
            {
                List<RotatedRect> boxes = new List<RotatedRect>();
                List<float> confidences = new List<float>();
                List<int> classIds = new List<int>();

                // 遍历所有预测框 (OutputSize)
                // 数据布局: [4(box) + 15(classes) + 1(angle)] * OutputSize
                // 展平数组中，同一属性的数据是连续存储的，例如所有 cx 在一起，所有 cy 在在一起...
                int stride = OutputSize; // 步长，不同属性在数组中的偏移量

                int resultDataOffset = OutputSize * (CategoryNum + 5) * b;

                for (int i = 0; i < OutputSize; i++)
                {
                    // 查找最大类别概率及其索引
                    float maxConf = 0;
                    int maxClassId = -1;

                    // 遍历类别 
                    for (int c = 0; c < CategoryNum; c++)
                    {
                        // 数组索引：(坐标/角度偏移量 + 类别偏移) * 框索引
                        // 注意：原始代码中 result[outputSize * j + i] 这种访问方式基于 Transposed 数据布局
                        float conf = result[(4 + c) * stride + i + resultDataOffset];
                        if (conf > maxConf)
                        {
                            maxConf = conf;
                            maxClassId = c;
                        }
                    }

                    // 置信度过滤
                    if (maxConf > ConfThreshold)
                    {
                        // 提取坐标 (cx, cy, w, h)
                        float cx = result[0 * stride + i + resultDataOffset];
                        float cy = result[1 * stride + i + resultDataOffset];
                        float w = result[2 * stride + i + resultDataOffset];
                        float h = result[3 * stride + i + resultDataOffset];

                        // 提取角度 (通常在第 5 个位置，即类别之前)
                        float angleRad = result[(CategoryNum + 4) * stride + i + resultDataOffset];

                        // 还原坐标到原图尺寸
                        float rx = (cx - xOffsets[b]) * scales[b];
                        float ry = (cy - yOffsets[b]) * scales[b];
                        float rw = w * scales[b];
                        float rh = h * scales[b];

                        // 将弧度转换为角度
                        // Normalize angle to [-π/2, π/2] range
                        // 将角度归一化到[-π/2, π/2]范围
                        if (angleRad >= Math.PI && angleRad <= 0.75 * Math.PI)
                        {
                            angleRad -= (float)Math.PI;
                        }
                        float angleDeg = angleRad * (float)(180f / Math.PI);  // Convert to degrees/转换为角度制

                        boxes.Add(new RotatedRect(new Point2f(rx, ry), new Size2f(rw, rh), angleDeg));
                        confidences.Add(maxConf);
                        classIds.Add(maxClassId);
                    }
                }

                // 执行 NMS (旋转框 NMS)
                // OpenCV 的 NMSBoxes 支持 RotatedRect
                int[] indices;
                CvDnn.NMSBoxes(boxes, confidences, ConfThreshold, NmsThreshold, out indices);

                List<ObbData> finalResults = new List<ObbData>();
                foreach (int idx in indices)
                {
                    finalResults.Add(new ObbData
                    {
                        index = classIds[idx],
                        score = confidences[idx],
                        box = boxes[idx]
                    });
                }
                obbDatas[b] = finalResults;
            });

           

            return obbDatas.Select(x => x?.ToList() ?? new List<ObbData>()).ToList();
        }

        /// <summary>
        /// 绘制旋转检测结果
        /// </summary>
        public static Mat DrawObbResult(List<ObbData> results, Mat image)
        {
            // 克隆图像以免修改原图
            Mat mat = image.Clone();

            foreach (var item in results)
            {
                // 获取旋转矩形的四个顶点
                Point2f[] points = item.box.Points();

                // 绘制多边形框
                for (int j = 0; j < 4; j++)
                {
                    Cv2.Line(mat, (Point)points[j], (Point)points[(j + 1) % 4],
                            new Scalar(0, 255, 0), 2);
                }

                // 绘制标签 (类别 - 置信度)
                string label = $"{item.index} - {item.score:F2}";
                Point2f textPos = points[0]; // 左上角

                Cv2.PutText(mat, label, (Point)textPos, HersheyFonts.HersheySimplex, 0.8,
                            new Scalar(255, 0, 0), 2);
            }

            return mat;
        }

        public class ObbData
        {
            public int index;
            public float score;
            public RotatedRect box;
        }


        /// <summary>
        /// 智能水平拼接：自动处理高度不一致的图片
        /// </summary>
        /// <param name="images">图片列表</param>
        /// <param name="backgroundColor">填充背景颜色，默认为黑色</param>
        /// <returns>拼接后的 Mat</returns>
        public static Mat StitchHorizontalWithPadding(List<Mat> images, Scalar? backgroundColor = null)
        {
            if (images == null || images.Count == 0)
                return new Mat();
            // 1. 找到所有图片中的最大高度
            int maxHeight = images.Max(img => img.Rows);
            // 计算总宽度
            int totalWidth = images.Sum(img => img.Cols);
            // 2. 准备结果画布
            Mat result = new Mat(maxHeight, totalWidth, images[0].Type(), backgroundColor ?? Scalar.Black);
            // 3. 将每一张图片复制到画布的对应位置
            int currentX = 0; // 当前 X 轴偏移量
            foreach (var img in images)
            {
                if (img.Empty()) continue;
                // 计算当前图片需要垂直偏移多少（底部对齐逻辑）
                // 如果想顶部对齐，yOffset = 0
                // 如果想居中，yOffset = (maxHeight - img.Rows) / 2
                int yOffset = maxHeight - img.Rows;
                // 定义 ROI (感兴趣区域)
                Rect roi = new Rect(currentX, yOffset, img.Cols, img.Rows);

                // 将原图片拷贝到结果图的 ROI 区域
                img.CopyTo(new Mat(result, roi));
                // 移动 X 轴指针
                currentX += img.Cols;
            }
            return result;
        }
    }
}

