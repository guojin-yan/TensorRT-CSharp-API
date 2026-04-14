using JYPPX.TensorRtSharp.Cuda;
using JYPPX.TensorRtSharp.Nvinfer;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace YoloDetInferVideo
{
    internal class Program
    {
        // ================= 配置参数 =================
        // 模型输入尺寸 (宽=高)
        private const int InputSize = 640;


        // 建议根据实际模型动态获取或使用 Netron 查看
        private const int OutputSize = 8400;

        // 模型类别数 (根据您的具体数据集修改，此处假设为15类)
        private const int CategoryNum = 80;

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

            string enginePath = "yolov8s.engine";
            string imagePath = @"E:\Data\NY.mp4";


            // ================= 1. 加载 TensorRT Engine =================
            // 使用 using 语句确保文件流正确关闭
            byte[] engineData;
            using (FileStream fs = new FileStream(enginePath, FileMode.Open, FileAccess.Read))
            using (BinaryReader br = new BinaryReader(fs))
            {
                engineData = br.ReadBytes((int)fs.Length);
            }


            ImagePreprocessor preprocessor = new ImagePreprocessor(InputSize);
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
                    // 输入: Batch=1, Channel=3, Height=640, Width=640
                    ulong inputSizeInBytes = 1 * 3 * InputSize * InputSize;
                    // 输出: Batch=1, Channels=CategoryNum+4(box)+1(angle), Num=8400
                    int outputChannels = CategoryNum + 4; // 4坐标 + N类别
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
                        cudaStream.Synchronize();
                        // ================= 3. 图像预处理 =================
                        // 创建一个命名窗口，使用 WindowMode.Normal 允许调整窗口大小
                        // 如果使用 WindowMode.AutoSize，窗口大小会随每一帧图像尺寸变化而闪烁
                        using (var window = new Window("Video Player"))
                        using (var capture = new VideoCapture(imagePath))
                        {
                            // ==========================================
                            // 2. 检查视频是否成功打开
                            // ==========================================
                            if (!capture.IsOpened())
                            {
                                Console.WriteLine("无法打开视频文件，请检查路径是否正确！");
                                return;
                            }
                            // 打印视频基本信息
                            Console.WriteLine($"视频帧率: {capture.Fps}");
                            Console.WriteLine($"总帧数: {capture.FrameCount}");
                            Console.WriteLine($"分辨率: {capture.FrameWidth} x {capture.FrameHeight}");
                            // ==========================================
                            // 3. 视频循环读取与播放
                            // ==========================================

                            Mat frame = new Mat(); // 用于存储每一帧的图像
                            int frameCount = 0;
                            // 计算帧间隔 (毫秒)
                            // Capture.Fps 有时可能返回0或错误值，通常设置为 30fps 左右 (33ms)
                            double delay = 1; //capture.Fps > 0 ? 1000.0 / capture.Fps : 33;
                            while (true)
                            {
                                Stopwatch swSum = new Stopwatch();
                                swSum.Start();
                                // Read() 方法会自动读取下一帧
                                // 如果读取失败或视频结束，返回 false
                                if (!capture.Read(frame))
                                {
                                    Console.WriteLine("视频播放结束或读取下一帧失败。");
                                    break;
                                }
                                frameCount++;
                                string mes1 = $"Frame: {frameCount}";
                                sw.Start();
                                float[] inputData = preprocessor.PreProcess(frame, out float scale, out int xOffset, out int yOffset);
                                sw.Stop();
                                Logger.Instance.INFO($"Pre-processing time: {sw.ElapsedMilliseconds} ms");
                                string mes2 = $"Pre-processing time: {sw.ElapsedMilliseconds} ms";
                                // ================= 4. 推理 =================
                                // 准备主机内存接收结果
                                float[] outputData = new float[outputChannels * OutputSize];

                                sw.Restart();
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

                                string mes3 = $"Inference time: {sw.ElapsedMilliseconds} ms";
                                // ================= 5. 后处理 =================

                                sw.Restart();
                                List<DetData> results = PostProcess(outputData, scale, xOffset, yOffset);
                                sw.Stop();
                                Logger.Instance.INFO($"Post-processing time: {sw.ElapsedMilliseconds} ms");
                                string mes4 = $"Post-processing time: {sw.ElapsedMilliseconds} ms";


                                // ================= 6. 结果可视化 =================
                                Mat resultImg = DrawDetResult(results, frame);



                                swSum.Stop();
                                Logger.Instance.INFO($"Total time per frame: {swSum.ElapsedMilliseconds} ms");
                                string mes5 = $"Total time per frame: {swSum.ElapsedMilliseconds} ms";
                                string mes6 = $"FPS: {1000.0 / swSum.ElapsedMilliseconds:F2}";
                                // 可选：在画面上显示帧数

                                // 使用 Environment.NewLine 或 \n 将所有消息拼接
                                string fullMessage = $"{mes1}{Environment.NewLine}{mes2}{Environment.NewLine}{mes3}{Environment.NewLine}{mes4}{Environment.NewLine}{mes5}{Environment.NewLine}{mes6}";

                                DrawTextLines(resultImg, new[] { mes1, mes2, mes3, mes4, mes5, mes6 },
                                    10, 30, 1.0, new Scalar(0, 255, 0), 2);

                                // ==========================================
                                // 4. 显示图像
                                // ==========================================
                                window.ShowImage(resultImg);
                                // ==========================================
                                // 5. 控制播放速度与退出检测
                                // ==========================================

                                // WaitKey(int delay): 等待按键，单位毫秒。
                                // 这里的 delay 决定了视频的播放速度。
                                // 如果 delay = 1，表示尽可能快地播放 (CPU全速运行)。
                                // 如果 delay = 33，大约对应 30fps。
                                int key = Cv2.WaitKey((int)delay);
                                // 检测按键
                                // ESC 键 (ASCII码 27) 或 'q' 键退出
                                if (key == 27 || key == 113)
                                {
                                    Console.WriteLine("用户请求退出播放。");
                                    break;
                                }
                            }
                        }

              
                        Mat img = Cv2.ImRead(imagePath);
                        if (img.Empty())
                        {
                            Logger.Instance.INFO("Image not found!");
                            return;
                        }

                      
                    }
                }
            }
        }

        /// <summary>
        /// 后处理：解析 TensorRT 输出、NMS 过滤
        /// </summary>
        private static List<DetData> PostProcess(float[] result, float scale, int xOffset, int yOffset)
        {
            List<Rect> boxes = new List<Rect>();
            List<float> confidences = new List<float>();
            List<int> classIds = new List<int>();

            // 遍历所有预测框 (OutputSize)
            // 数据布局: [4(box) + 80(classes)] * OutputSize
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
                    // 还原坐标到原图尺寸
                    int rx = (int)((cx - xOffset - 0.5 * w) * scale);
                    int ry = (int)((cy - yOffset - 0.5 * h) * scale);
                    int rw = (int)(w * scale);
                    int rh = (int)(h * scale);

                    boxes.Add(new Rect(rx, ry, rw, rh));
                    confidences.Add(maxConf);
                    classIds.Add(maxClassId);
                }
            }

            // 执行 NMS (旋转框 NMS)
            // OpenCV 的 NMSBoxes 支持 RotatedRect
            int[] indices;
            CvDnn.NMSBoxes(boxes, confidences, ConfThreshold, NmsThreshold, out indices);

            List<DetData> finalResults = new List<DetData>();
            foreach (int idx in indices)
            {
                finalResults.Add(new DetData
                {
                    index = classIds[idx],
                    score = confidences[idx],
                    box = boxes[idx]
                });
            }

            return finalResults;
        }


        public static void DrawTextLines(Mat img, string[] messages, int startX, int startY, double fontSize, Scalar color, int thickness)
        {
            // 估算行高：字号 * 25 是 HersheySimplex 字体的常见行距
            int lineGap = (int)(fontSize * 30);
            for (int i = 0; i < messages.Length; i++)
            {
                // 计算当前行的 Y 坐标
                int currentY = startY + (i * lineGap);

                Cv2.PutText(img, messages[i], new Point(startX, currentY),
                            HersheyFonts.HersheySimplex, fontSize, color, thickness);
            }
        }

        /// <summary>
        /// 绘制检测结果（水平矩形框）
        /// </summary>
        /// <param name="results">检测结果列表</param>
        /// <param name="image">原始图像</param>
        /// <returns>绘制后的图像</returns>
        public static Mat DrawDetResult(List<DetData> results, Mat image)
        {
            // 克隆图像以免修改原图
            Mat mat = image.Clone();

            foreach (var item in results)
            {
                // 1. 绘制矩形框
                // Rect 结构包含 X, Y, Width, Height
                Cv2.Rectangle(mat, item.box, new Scalar(0, 255, 0), thickness: 2);
                // 2. 准备标签文本 (类别ID - 置信度)
                string label = $"{item.index} - {item.score:F2}";
                // 3. 计算文本的尺寸，用于绘制背景
                int baseLine = 1;
                Size textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.6, 1, out baseLine);
                // 4. 绘制标签背景（半透明黑色矩形），防止文字与背景混淆
                // 位置：矩形左上角略微上移，或者直接贴着左上角
                Point labelPosition = new Point(item.box.X, item.box.Y - (int)textSize.Height - 5);

                // 确保标签不画出图像边界
                if (labelPosition.Y < 0) labelPosition.Y = item.box.Y + (int)textSize.Height + 5;
                Rect labelBgRect = new Rect(labelPosition.X,
                                            labelPosition.Y - (int)textSize.Height, // OpenCV GetTextSize 返回的高度是基线到底部的距离，需调整
                                            (int)textSize.Width,
                                            (int)textSize.Height + (int)baseLine);
                // 如果背景框也在图像范围内，则绘制
                // 注意：这里简化处理，直接画在框上方
                Cv2.Rectangle(mat,
                               new Point(item.box.X, item.box.Y - textSize.Height - 5),
                               new Point(item.box.X + textSize.Width, item.box.Y),
                               new Scalar(0, 255, 0),
                               thickness: -1); // -1 表示填充
                // 5. 绘制文本（白色文字）
                Cv2.PutText(mat,
                            label,
                            new Point(item.box.X, item.box.Y - 5),
                            HersheyFonts.HersheySimplex,
                            0.6,
                            new Scalar(0, 0, 0),
                            1);
            }
            return mat;
        }

        public class DetData
        {
            public int index;
            public float score;
            public Rect box;
        }
    }

    public class ImagePreprocessor : IDisposable
    {
        private readonly int _inputSize;
        private Mat _rgbImg;
        private Mat _resizedImg;
        private Mat _paddedImg;
        private Mat _floatImg;
        private float[] _chwData;
        public ImagePreprocessor(int inputSize)
        {
            _inputSize = inputSize;
            Initialize();
        }
        private void Initialize()
        {
            _rgbImg = new Mat();
            _resizedImg = new Mat();
            // 初始化黑色背景，只需一次
            _paddedImg = Mat.Zeros(_inputSize, _inputSize, MatType.CV_8UC3);
            _floatImg = new Mat();
            _chwData = new float[3 * _inputSize * _inputSize];
        }
        /// <summary>
        /// 优化的预处理方法
        /// </summary>
        public float[] PreProcess(Mat img, out float scale, out int xOffset, out int yOffset)
        {
            // 1. BGR -> RGB (复用 _rgbImg)
            Cv2.CvtColor(img, _rgbImg, ColorConversionCodes.BGR2RGB);
            // 2. 计算 Letterbox 参数
            int maxDim = Math.Max(_rgbImg.Width, _rgbImg.Height);
            scale = (float)maxDim / _inputSize;

            int newWidth = (int)(_rgbImg.Width / scale);
            int newHeight = (int)(_rgbImg.Height / scale);
            // 3. Resize (复用 _resizedImg)
            Cv2.Resize(_rgbImg, _resizedImg, new OpenCvSharp.Size(newWidth, newHeight));
            // 4. Padding (复用 _paddedImg)
            // 先清空画布 (比重新创建 Mat 快得多)
            _paddedImg.SetTo(Scalar.Black);

            xOffset = (_inputSize - newWidth) / 2;
            yOffset = (_inputSize - newHeight) / 2;
            // 使用 ROI 进行快速拷贝
            using (var roi = new Mat(_paddedImg, new Rect(xOffset, yOffset, newWidth, newHeight)))
            {
                _resizedImg.CopyTo(roi);
            }

            //Cv2.ImShow("Padded Image", _paddedImg);
            //Cv2.WaitKey(0);
            // 5. 归一化 & 类型转换 (复用 _floatImg)
            // 0-255 -> 0-1, 并转为 float
            _paddedImg.ConvertTo(_floatImg, MatType.CV_32FC3, 1.0 / 255.0);
            // 6. HWC 转换为 CHW (Safe + Parallel)
            ConvertHwcToChwSafeParallel(_floatImg);
            return _chwData;
        }
        /// <summary>
        /// 安全且并行化的 HWC -> CHW 转换
        /// </summary>
        private void ConvertHwcToChwSafeParallel(Mat src)
        {
            // 获取底层数据句柄
            IntPtr dataPtr = src.Data;

            int rw = _inputSize;
            int rh = _inputSize;
            int channelSize = rw * rh;


            GCHandle resultHandle = default;
            try
            {
                resultHandle = GCHandle.Alloc(_chwData, GCHandleType.Pinned);
                IntPtr resultPtr = resultHandle.AddrOfPinnedObject();
                Parallel.For(0, 3, i =>
                {
                    using Mat dest = Mat.FromPixelData(rh, rw, MatType.CV_32FC1, resultPtr + i * rh * rw * sizeof(float));
                    Cv2.ExtractChannel(src, dest, i);
                });
            }
            finally
            {
                resultHandle.Free();
            }
        }
        public void Dispose()
        {
            _rgbImg?.Dispose();
            _resizedImg?.Dispose();
            _paddedImg?.Dispose();
            _floatImg?.Dispose();
        }
    }
}
