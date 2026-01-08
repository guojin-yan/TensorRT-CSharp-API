using JYPPX.TensorRtSharp.Cuda;
using JYPPX.TensorRtSharp.Nvinfer;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace YoloObbInfer
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

            string enginePath = "yolo11s-obb.engine";
            string imagePath = "plane.png";


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
                    ulong inputSizeInBytes = 1 * 3 * InputSize * InputSize;
                    // 输出: Batch=1, Channels=CategoryNum+4(box)+1(angle), Num=8400
                    int outputChannels = CategoryNum + 5; // 4坐标 + 1角度 + N类别
                    ulong outputSizeInBytes = (ulong)(1 * outputChannels * OutputSize);

                    Stopwatch sw = new Stopwatch();
                    // 分配 GPU 显存
                    using (Cuda1DMemory<float> inputGpuMemory = new Cuda1DMemory<float>(inputSizeInBytes))
                    using (Cuda1DMemory<float> outputGpuMemory = new Cuda1DMemory<float>(outputSizeInBytes))
                    {
                        // 绑定显存地址到 TensorRT 上下文
                        executionContext.setInputTensorAddress("images", inputGpuMemory.get());
                        executionContext.setOutputTensorAddress("output0", outputGpuMemory.get());
                        // 预热推理 (可选，但推荐，尤其是首次推理时)
                        executionContext.executeV3(cudaStream);

                        // ================= 3. 图像预处理 =================
                        Mat img = Cv2.ImRead(imagePath);
                        if (img.Empty())
                        {
                            Logger.Instance.INFO("Image not found!");
                            return;
                        }

                        sw.Start();
                        float[] inputData = PreProcess(img, out float scale, out int xOffset, out int yOffset);
                        sw.Stop();
                        Logger.Instance.INFO($"Pre-processing time: {sw.ElapsedMilliseconds} ms");
                        // ================= 4. 推理 =================
                        sw.Restart();
                        // 将数据从主机 拷贝到设备
                        inputGpuMemory.copyFromHost(inputData);

                        // 执行推理 (enqueueV3 是异步的)
                        executionContext.executeV3(cudaStream);
                        // 等待推理完成
                        cudaStream.Synchronize(); 

                        // 准备主机内存接收结果
                        float[] outputData = new float[outputChannels * OutputSize];

                        // 将结果从设备 拷贝回主机
                        // 这里的拷贝是同步的，会等待 GPU 计算完成
                        outputGpuMemory.copyToHost(outputData);
                        sw.Stop();
                        Logger.Instance.INFO($"Inference time: {sw.ElapsedMilliseconds} ms");
                        // ================= 5. 后处理 =================

                        sw.Restart();
                        List<ObbData> results = PostProcess(outputData, scale, xOffset, yOffset);
                        sw.Stop();
                        Logger.Instance.INFO($"Post-processing time: {sw.ElapsedMilliseconds} ms");

                        // ================= 6. 结果可视化 =================
                        Mat resultImg = DrawObbResult(results, img);
                        Cv2.ImShow("YOLO11-OBB Result", resultImg);
                        Cv2.WaitKey(0);
                    }
                }
            }
        }

        /// <summary>
        /// 图像预处理：Letterbox 缩放、归一化、HWC 转 CHW
        /// </summary>
        private static float[] PreProcess(Mat img, out float scale, out int xOffset, out int yOffset)
        {
            // 转换颜色空间 BGR -> RGB
            Mat rgbImg = new Mat();
            Cv2.CvtColor(img, rgbImg, ColorConversionCodes.BGR2RGB);

            // 计算 Letterbox 缩放比例
            int maxDim = Math.Max(rgbImg.Width, rgbImg.Height);
            scale = (float)maxDim / InputSize ;

            // 计算缩放后的尺寸
            int newWidth = (int)(rgbImg.Width / scale);
            int newHeight = (int)(rgbImg.Height / scale);

            // Resize 图像
            Mat resizedImg = new Mat();
            Cv2.Resize(rgbImg, resizedImg, new Size(newWidth, newHeight));

            // 创建黑色背景 Canvas (InputSize x InputSize)
            Mat paddedImg = Mat.Zeros(InputSize, InputSize, MatType.CV_8UC3);

            // 计算粘贴位置 (居中)
            xOffset = (InputSize - newWidth) / 2;
            yOffset = (InputSize - newHeight) / 2;

            // 将图像拷贝到 Canvas 中央
            Rect roi = new Rect(xOffset, yOffset, newWidth, newHeight);
            resizedImg.CopyTo(new Mat(paddedImg, roi));

            // 归一化 (0-255 -> 0-1) 并转为 float 类型
            Mat floatImg = new Mat();
            paddedImg.ConvertTo(floatImg, MatType.CV_32FC3, 1.0 / 255.0);

            // HWC 转 CHW 并展平为一维数组
            Mat[] channels = Cv2.Split(floatImg);
            float[] chwData = new float[3 * InputSize * InputSize];

            // 拷贝数据：R通道 -> C通道 -> B通道 (OpenCV Split 出来顺序是 B, G, R，对应索引 0, 1, 2)
            int channelSize = InputSize * InputSize;
            // 将 R, G, B 依次拷入数组
            Marshal.Copy(channels[0].Data, chwData, 0, channelSize); // R
            Marshal.Copy(channels[1].Data, chwData, channelSize, channelSize); // G
            Marshal.Copy(channels[2].Data, chwData, channelSize * 2, channelSize); // B

            // 释放临时 Mat
            rgbImg.Dispose();
            resizedImg.Dispose();
            paddedImg.Dispose();
            floatImg.Dispose();
            foreach (var c in channels) c.Dispose();

            return chwData;
        }

        /// <summary>
        /// 后处理：解析 TensorRT 输出、NMS 过滤
        /// </summary>
        private static List<ObbData> PostProcess(float[] result, float scale, int xOffset, int yOffset)
        {
            List<RotatedRect> boxes = new List<RotatedRect>();
            List<float> confidences = new List<float>();
            List<int> classIds = new List<int>();

            // 遍历所有预测框 (OutputSize)
            // 数据布局: [4(box) + 15(classes) + 1(angle)] * OutputSize
            // 展平数组中，同一属性的数据是连续存储的，例如所有 cx 在一起，所有 cy 在在一起...
            int stride = OutputSize; // 步长，不同属性在数组中的偏移量

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
                    float conf = result[(4 + c) * stride + i];
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
                    float cx = result[0 * stride + i];
                    float cy = result[1 * stride + i];
                    float w = result[2 * stride + i];
                    float h = result[3 * stride + i];

                    // 提取角度 (通常在第 5 个位置，即类别之前)
                    float angleRad = result[(CategoryNum + 4) * stride + i];

                    // 还原坐标到原图尺寸
                    float rx = (cx - xOffset) * scale;
                    float ry = (cy - yOffset) * scale;
                    float rw = w * scale;
                    float rh = h * scale;

                    // 将弧度转换为角度
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

            return finalResults;
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
    }
}

