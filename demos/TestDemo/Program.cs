using JYPPX.TensorRtSharp.Cuda;
using JYPPX.TensorRtSharp.Nvinfer;
using System.Diagnostics;
namespace TestDemo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            ulong memorySize = 17796680;//18385 * 100;
            Stopwatch sw = new Stopwatch();
            sw.Start();
            CudaPinnedMemory<float> cudaPinned = new CudaPinnedMemory<float>(memorySize);
            sw.Stop();
            Console.WriteLine($"pinned alloc time: {sw.ElapsedMilliseconds} ms");

            sw.Restart();
            CudaPinnedMemory<float> cudaPinned1 = new CudaPinnedMemory<float>(memorySize + 1);
            sw.Stop();
            Console.WriteLine($"pinned1 alloc time: {sw.ElapsedMilliseconds} ms");
                
            sw.Restart();

            Cuda1DMemory<float> cuda1DMemory = new Cuda1DMemory<float>((ulong)memorySize + 100);
            sw.Stop();
            Console.WriteLine($"1D alloc time: {sw.ElapsedMilliseconds} ms");

            CudaStream cudaStream = new CudaStream();
            float[] data = new float[memorySize - 10];
            data[0] = 1;
            data[1] = 2;
            data[2] = 3;
            data[memorySize - 11] = 100000;
            float[] data1 = new float[memorySize + 200];
            
            sw.Start();
            cudaPinned.CopyFrom(data);
            sw.Stop();
            Console.WriteLine($"pinned copy time: {sw.ElapsedMilliseconds} ms");
            sw.Restart();
            cuda1DMemory.copyFromHostAsync(data, cudaStream);
            cudaStream.Synchronize();   
            sw.Stop();
            Console.WriteLine($"1D copy time: {sw.ElapsedMilliseconds} ms");
            sw.Restart();
            cuda1DMemory.copyToHostAsync(cudaPinned1, cudaStream);
            cudaStream.Synchronize();
            sw.Stop();
            Console.WriteLine($"1D copy to pinned time: {sw.ElapsedMilliseconds} ms");
            sw.Restart();
            cudaPinned1.CopyTo(data1);
            sw.Stop();
            Console.WriteLine($"pinned copy to host time: {sw.ElapsedMilliseconds} ms");
            Console.WriteLine($"{data1[0]}, {data1[1]}, {data1[2]}, {data1[memorySize - 11]}");



            //// 指定默认使用的 GPU 设备索引。
            //// 在多 GPU 环境下，可以通过修改此变量来选择特定的显卡。
            //int device = 0;

            //// 记录日志，标记设备信息查询的开始
            //Logger.Instance.INFO("=== Device Information ===");
            //// 获取当前系统中可见的 NVIDIA GPU 数量
            //int nbDevices = CudaDevice.GetDeviceCount();
            //// 检查系统中是否存在可用的 GPU 设备
            //if (nbDevices <= 0)
            //{
            //    // 如果没有找到设备，记录错误日志
            //    Logger.Instance.ERROR("Cannot find any available devices (GPUs)!");

            //    // 退出程序。Environment.Exit(0) 表示正常终止程序，
            //    // 尽管这里是因为报错退出，但返回 0 表示程序逻辑已处理完毕。
            //    Environment.Exit(0);
            //}
            //// 打印所有可用设备的列表
            //Logger.Instance.INFO("Available Devices: ");

            //// 创建一个 CudaDeviceProp 对象，用于存储目标设备的详细属性
            //CudaDeviceProp properties = new CudaDeviceProp();

            //// 遍历系统中的每一个 GPU
            //for (int deviceIdx = 0; deviceIdx < nbDevices; ++deviceIdx)
            //{
            //    // 获取索引为 deviceIdx 的 GPU 的详细属性
            //    CudaDeviceProp tempProperties = CudaDevice.GetDeviceProperties(deviceIdx);
            //    // clang-format off
            //    // 打印设备 ID、设备名称 以及 UUID (唯一标识符)
            //    // GetUuidString 是一个自定义辅助方法，用于将字节数组转换为格式化的 UUID 字符串
            //    Logger.Instance.INFO("  Device " + deviceIdx + ": \"" + tempProperties.Name + "\" UUID: "
            //       + GetUuidString(tempProperties.Uuid));
            //    // clang-format on
            //    // 如果当前遍历到的设备 ID 是我们想要使用的目标设备 (device 变量)，
            //    // 则将该设备的属性保存下来，供后续使用。
            //    if (deviceIdx == device)
            //    {
            //        properties = tempProperties;
            //    }
            //}
            //// 安全检查：确保请求的目标设备 ID (device) 在有效范围内 [0, nbDevices - 1]
            //// 防止因用户指定的 device ID 过大或过小导致越界异常
            //if (device < 0 || device >= nbDevices)
            //{
            //    Logger.Instance.ERROR("Cannot find device ID " + device + "!");
            //    Environment.Exit(0);
            //}
            //// 将 CUDA 上下文设置到指定的 GPU 设备上。
            //// 之后的 CUDA 操作（如内存分配、核函数启动）都将在此设备上执行。
            //CudaDevice.SetDevice(device);
            //// clang-format off
            //// 打印选定设备的详细信息
            //Logger.Instance.INFO("Selected Device: " + properties.Name);
            //Logger.Instance.INFO("Selected Device ID: " + device);
            //Logger.Instance.INFO("Selected Device UUID: " + GetUuidString(properties.Uuid));

            //// 打印计算能力，格式为 Major.Minor (例如 8.6)
            //Logger.Instance.INFO("Compute Capability: " + properties.Major + "." + properties.Minor);

            //// 打印流多处理器 的数量，SM 是 GPU 的核心计算单元
            //Logger.Instance.INFO("SMs: " + properties.MultiProcessorCount);

            //// 打印显存总量
            //// properties.TotalGlobalMem 通常返回字节数，这里的 +20 看起来是原代码的特定处理逻辑，
            //// 建议标准做法是除以 (1024 * 1024) 转换为 MiB，或者保持原样如果是特定修正值。
            //Logger.Instance.INFO("Device Global Memory: " + (properties.TotalGlobalMem + 20) + " MiB");

            //// 打印每个 SM 的共享内存大小
            //// >> 10 等同于除以 1024，将字节转换为 KiB
            //Logger.Instance.INFO("Shared Memory per SM: " + (properties.SharedMemPerMultiprocessor >> 10) + " KiB");

            //// 打印显存位宽以及 ECC (Error Correcting Code) 状态
            //// ECC 是一种内存纠错技术，通常用于工作站或服务器显卡
            //Logger.Instance.INFO("Memory Bus Width: " + properties.MemoryBusWidth + " bits"
            //                    + " (ECC " + (properties.ECCEnabled != 0 ? "enabled" : "disabled") + ")");
            //// 获取并打印 GPU 核心时钟频率 (单位：KHz)
            //int clockRate = CudaDevice.GetAttribute(CudaDeviceAttr.ClockRate, device);

            //// 获取并打印显存时钟频率 (单位：KHz)
            //int memoryClockRate = CudaDevice.GetAttribute(CudaDeviceAttr.MemoryClockRate, device);

            //// 将 KHz 转换为 GHz (除以 1,000,000) 并打印
            //// 1000000.0F 表示单精度浮点数，确保结果为小数
            //Logger.Instance.INFO("Application Compute Clock Rate: " + clockRate / 1000000.0F + " GHz");
            //Logger.Instance.INFO("Application Memory Clock Rate: " + memoryClockRate / 1000000.0F + " GHz");

            //Logger.Instance.INFO("");

            //// 提示用户注意：这里获取的是 "Application Clock"（应用程序时钟），
            //// 即驱动程序报告的默认或锁定频率，不一定代表 GPU 当前因负载变化的实际运行频率（Boost 频率）。
            //Logger.Instance.INFO("Note: The application clock rates do not reflect the actual clock rates that the GPU is "
            //                                                                     + "currently running at.");
        }
        /// <summary>
        /// 辅助方法：将 CudaUUID 结构体转换为格式化的 GPU UUID 字符串。
        /// 格式通常为：GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        /// </summary>
        /// <param name="uuid">包含 UUID 字节的 CudaUUID 对象</param>
        /// <returns>格式化后的 UUID 字符串</returns>
        public static string GetUuidString(CudaUUID uuid)
        {
            // 获取 UUID 字节数组的长度 (通常为 16)
            int kUUID_SIZE = uuid.Bytes.Length;
            // 使用 StringBuilder 高效地构建字符串
            System.Text.StringBuilder ss = new System.Text.StringBuilder();

            // 定义 UUID 的分段点，用于插入连字符 "-"
            // 例如：索引 0-4 一段，4-6 一段，依此类推
            int[] splits = { 0, 4, 6, 8, 10, kUUID_SIZE };
            // 添加固定的 "GPU" 前缀
            ss.Append("GPU");

            // 遍历分段定义，格式化每一部分的字节
            for (int splitIdx = 0; splitIdx < splits.Length - 1; ++splitIdx)
            {
                // 在每一段前添加连字符
                ss.Append("-");

                // 遍历当前分段内的所有字节
                for (int byteIdx = splits[splitIdx]; byteIdx < splits[splitIdx + 1]; ++byteIdx)
                {
                    // {0:x2} 表示将字节转换为两位的十六进制字符串 (例如 0x0F 转换为 "0f")
                    ss.AppendFormat("{0:x2}", uuid.Bytes[byteIdx]);
                }
            }

            return ss.ToString();
        }
        //public static void inferModel() 
        //{
        //    // 文件路径
        //    string filePath = "yolo11s-obb.engine";
        //    FileStream fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        //    byte[] data = new byte[fileStream.Length];

        //    BinaryReader binaryReader = new BinaryReader(fileStream);

        //    data = binaryReader.ReadBytes((int)fileStream.Length); // 读取整个文件到byte数组
        //    Runtime runtime = new Runtime();

        //    CudaEngine cudaEngine = runtime.deserializeCudaEngineByBlob(data, (ulong)fileStream.Length);
        //    JYPPX.TensorRtSharp.Nvinfer.ExecutionContext executionContext = cudaEngine.createExecutionContext(TrtExecutionContextAllocationStrategy.kSTATIC);

        //    Dims dims = executionContext.getTensorShape("images");

        //    Cuda1DMemory<float> input = new Cuda1DMemory<float>(3*1024*1024);

        //    Cuda1DMemory<float> output = new Cuda1DMemory<float>(1 * 20 * 21504);

        //    float[] outputHost = new float[1 * 84 * 8400];
        //    executionContext.setInputTensorAddress("images", input.get());

        //    executionContext.setOutputTensorAddress("output0", output.get());

        //    Mat img = Cv2.ImRead("plane.png");
        //    float[] inputHost = preProcess(img, out float scales);



        //    input.copyFromHost(inputHost);
        //    CudaStream cudaStream = new CudaStream();
        //    executionContext.executeV3(cudaStream);
        //    output.copyToHost(outputHost);

        //    Stopwatch sw = new Stopwatch();
        //    sw.Start();
        //    for(int i = 0; i < 1; ++i) 
        //    {
        //        input.copyFromHost(inputHost);
        //        executionContext.executeV3(cudaStream);
        //        output.copyToHost(outputHost);
        //    }

        //    sw.Stop();
        //    Console.WriteLine($"inference time: {sw.ElapsedMilliseconds/1f} ms");

        //    List<ObbData> result = postprocess(outputHost, scales);
        //    Mat re = drawObbResult(result, img);
        //    Cv2.ImShow("www", re);
        //    Cv2.WaitKey(0); 
        //    Console.ReadLine();

        //}

        //public static void buildModel() 
        //{
        //    Builder build = new Builder();
        //    Console.WriteLine($"platformHasFastFp16: {build.platformHasFastFp16()}");
        //    Console.WriteLine($"platformHasFastInt8: {build.platformHasFastInt8()}");
        //    Console.WriteLine($"maxDLABatchSize: {build.maxDLABatchSize()}");


        //    NetworkDefinition networkDefinition = build.createNetworkV2(TrtNetworkDefinitionCreationFlag.kEXPLICIT_BATCH);

        //    BuilderConfig builderConfig = build.createBuilderConfig();

        //    OnnxParser onnxParser = new OnnxParser(networkDefinition);
        //    string modelpath = "yolo11s-obb.onnx";
        //    if (onnxParser.parseFromFile(modelpath, 2) == false)
        //    {
        //        Console.WriteLine($"parse onnx model failed");
        //        return;
        //    }

        //    builderConfig.setFlag(TrtBuilderFlag.kFP16);

        //    CudaStream cudaStream = new CudaStream();

        //    builderConfig.setProfileStream(cudaStream);

        //    HostMemory hostMemory = build.buildSerializedNetwork(networkDefinition, builderConfig);

        //    // 文件路径
        //    string filePath = "yolo11s-obb.engine";

        //    // 创建或打开文件
        //    using (FileStream fs = new FileStream(filePath, FileMode.Create, FileAccess.Write))
        //    {
        //        // 写入数据，例如写入一些字节数据
        //        fs.Write(hostMemory.getByteData(), 0, (int)hostMemory.Size);
        //    }

        //    Console.ReadLine();
        //}





        ///// <summary>
        ///// 模型输入尺寸(如果模型输入形状有变化，此处要修改，使用ONNX模型查看工具Netron进行查看)
        ///// </summary>
        //private static int inputSize = 1024;

        ///// <summary>
        ///// 模型输出尺寸(如果模型输入形状有变化，此处要修改，使用ONNX模型查看工具Netron进行查看)
        ///// </summary>
        //private static int outputSize = 21504;

        //private static int categNum = 15;

        //private static float[] preProcess(Mat img, out float scales)
        //{
        //    // 创建临时Mat对象并转换颜色空间（BGR→RGB）
        //    Mat mat = new Mat();
        //    Cv2.CvtColor(img, mat, ColorConversionCodes.BGR2RGB);

        //    // 根据图像长宽比计算缩放比例
        //    Rect roi = new Rect();
        //    if (img.Cols > img.Rows)  // 宽>高的情况
        //    {
        //        scales = (float)img.Cols / (float)inputSize;
        //        Cv2.Resize(mat, mat, new Size(inputSize, img.Rows / scales));
        //        roi = new Rect(0, 0, inputSize, (int)(img.Rows / scales));
        //    }
        //    else  // 高≥宽的情况
        //    {
        //        scales = (float)img.Rows / (float)inputSize;
        //        Cv2.Resize(mat, mat, new Size(img.Cols / scales, inputSize));
        //        roi = new Rect(0, 0, (int)(img.Cols / scales), inputSize);
        //    }

        //    // 创建640x640的黑色背景Mat
        //    Mat mat1 = Mat.Zeros(inputSize, inputSize, MatType.CV_8UC3);
        //    // 将缩放后的图像拷贝到中央
        //    mat.CopyTo(new Mat(mat1, roi));
        //    // 归一化到0-1范围（CV_32FC3类型）
        //    mat1.ConvertTo(mat1, MatType.CV_32FC3, 1.0 / 255.0);

        //    // 准备输出数组（CHW格式：3x640x640）
        //    float[] array = new float[inputSize * inputSize * 3];
        //    GCHandle gCHandle = default(GCHandle);
        //    try
        //    {
        //        // 固定内存地址用于高效数据拷贝
        //        gCHandle = GCHandle.Alloc(array, GCHandleType.Pinned);
        //        IntPtr intPtr = gCHandle.AddrOfPinnedObject();

        //        // 分离RGB三个通道到连续内存
        //        for (int i = 0; i < 3; i++)
        //        {
        //            Mat mat2 = Mat.FromPixelData(inputSize, inputSize, MatType.CV_32FC1,
        //                        intPtr + (i * inputSize * inputSize * 4), 0L);
        //            Cv2.ExtractChannel(mat1, mat2, i);
        //        }
        //        return array;
        //    }
        //    finally
        //    {
        //        gCHandle.Free();  // 释放固定内存
        //    }
        //}

        //public class ObbData
        //{

        //    public int index;

        //    public float score;

        //    public RotatedRect box;
        //}
        //private static List<ObbData> postprocess(float[] result, float scales)
        //{
        //    // 初始化存储容器
        //    List<RotatedRect> positionBoxes = new List<RotatedRect>();  // 矩形框
        //    List<int> classIds = new List<int>();             // 类别ID
        //    List<float> confidences = new List<float>();      // 置信度
        //    List<float> rotations = new List<float>();        // 旋转角度

        //    // 解析模型输出（8400个预测框）
        //    for (int i = 0; i < outputSize; i++)
        //    {
        //        for (int j = 4; j < (categNum + 4); j++)  // 遍历每个类别
        //        {
        //            float conf = result[outputSize * j + i];
        //            int label = j - 4;
        //            if (conf > 0.2)  // 置信度阈值过滤
        //            {
        //                // 解析中心点坐标、宽高和旋转角度
        //                float cx = result[outputSize * 0 + i];
        //                float cy = result[outputSize * 1 + i];
        //                float ow = result[outputSize * 2 + i];
        //                float oh = result[outputSize * 3 + i];
        //                //float rotation = 0;
        //                float rotation = result[outputSize * (categNum + 4) + i];

        //                // 创建旋转矩形框（考虑预处理时的缩放）
        //                RotatedRect box = new RotatedRect(
        //                    new Point2f(cx * scales, cy * scales),
        //                    new Size2f(ow * scales, oh * scales),
        //                    (float)(rotation * 180.0 / Math.PI));

        //                // 存储检测结果
        //                positionBoxes.Add(box);
        //                classIds.Add(label);
        //                confidences.Add(conf);
        //                rotations.Add(rotation);
        //            }
        //        }
        //    }

        //    // 执行非极大值抑制（NMS）
        //    int[] indexes = new int[positionBoxes.Count];
        //    CvDnn.NMSBoxes(positionBoxes, confidences, 0.5f, 0.3f, out indexes);

        //    // 封装最终结果
        //    List<ObbData> boxes = new List<ObbData>();
        //    for (int i = 0; i < indexes.Length; i++)
        //    {
        //        int index = indexes[i];
        //        boxes.Add(new ObbData
        //        {
        //            index = classIds[index],
        //            score = confidences[index],
        //            box = positionBoxes[index]
        //        });
        //    }
        //    return boxes;
        //}

        //public static Mat drawObbResult(List<ObbData> bresult, Mat image)
        //{
        //    for (int i = 0; i < bresult.Count; i++)
        //    {
        //        // 获取旋转框的四个角点
        //        Point2f[] array = bresult[i].box.Points();

        //        // 绘制四边形边框
        //        for (int j = 0; j < 4; j++)
        //        {
        //            Cv2.Line(image, (Point)array[j], (Point)array[(j + 1) % 4],
        //                    new Scalar(255.0, 100.0, 200.0), 2);
        //        }

        //        // 在左上角显示类别和置信度
        //        Cv2.PutText(image,
        //                    $"{bresult[i].index}-{bresult[i].score:0.00}",
        //                    (Point)array[0],
        //                    HersheyFonts.HersheySimplex,
        //                    0.8,
        //                    new Scalar(0.0, 0.0, 0.0),
        //                    2);
        //    }
        //    return image;
        //}

    }
}
