using JYPPX.TensorRtSharp.Cuda;
using JYPPX.TensorRtSharp.Nvinfer;
namespace OnnxToEngine
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // --- 配置 TensorRT 日志回调 ---
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



            // 1. 创建 TensorRT Builder (构建器)
            // Builder 是 TensorRT 的核心入口，用于创建网络、配置和构建 Engine。
            Builder build = new Builder();

            // 2. 创建网络定义 (Network Definition)
            // createNetworkV2: 创建一个空的网络结构。
            // 显式批处理 标志表示网络定义中显式包含批处理维度 (Batch Dimension, N)。
            // 这是 TensorRT 较新版本的标准做法，支持动态形状等高级特性。
            NetworkDefinition networkDefinition = build.createNetworkV2(TrtNetworkDefinitionCreationFlag.kEXPLICIT_BATCH);
            // 3. 创建构建器配置
            // 用于指定构建 Engine 时的各种参数，例如精度模式、最大工作空间大小等。
            BuilderConfig builderConfig = build.createBuilderConfig();
            // 4. 创建 ONNX 解析器
            // ONNXParser 负责读取 ONNX 模型文件，并将其填充到上面创建的 networkDefinition 中。
            OnnxParser onnxParser = new OnnxParser(networkDefinition);

            // 指定待转换的 ONNX 模型文件路径
            string modelpath = "yolo11s-obb.onnx";

            // 5. 解析 ONNX 模型文件
            // parseFromFile: 从指定路径加载 ONNX 模型并解析。
            // 参数 2: 日志级别 (1=ERROR, 2=WARNING, 3=INFO, 4=VERBOSE)。这里设置为 2，表示显示警告及以上级别的信息。
            if (onnxParser.parseFromFile(modelpath, 2) == false)
            {
                // 如果解析失败，打印错误信息并退出方法
                Console.WriteLine($"parse onnx model failed");
                return;
            }
            // 6. 设置构建精度标志
            // kFP16: 启用半精度 (FP16) 推理模式。
            // 在支持 FP16 的 GPU（如 RTX 系列、Tesla 系列）上，这可以显著减少显存占用并提高推理速度，
            // 同时通常不会造成明显的精度损失。
            builderConfig.setFlag(TrtBuilderFlag.kFP16);
            // 7. 创建 CUDA 流
            // CudaStream 用于在 GPU 上执行异步操作。
            CudaStream cudaStream = new CudaStream();
            // 8. 设置优化配置文件的流
            // 将创建的 CUDA 流传递给构建器配置。
            // 这允许 TensorRT 在构建 Engine 的某些阶段（如层性能分析）使用此流，以便更准确地测量实际运行环境下的性能。
            builderConfig.setProfileStream(cudaStream);
            // 9. 构建并序列化网络
            // buildSerializedNetwork: 根据网络定义和配置，构建优化后的推理引擎 (Engine)，并将其序列化为二进制数据。
            // 这是一个耗时较长的过程，因为 TensorRT 会在此时进行内核 自动调优、层融合等优化。
            HostMemory hostMemory = build.buildSerializedNetwork(networkDefinition, builderConfig);
            // 10. 保存 Engine 到磁盘
            // 指定输出的 Engine 文件路径。Engine 文件是特定于 GPU 架构和 TensorRT 版本的，不可跨平台通用。
            string filePath = "yolo11s-obb.engine";
            // 使用文件流将内存中的序列化数据写入磁盘
            // using 语句确保文件流在使用完毕后正确释放资源
            using (FileStream fs = new FileStream(filePath, FileMode.Create, FileAccess.Write))
            {
                // hostMemory.getByteData(): 获取包含 Engine 二进制数据的字节数组
                // 0: 起始偏移量
                // (int)hostMemory.Size: 要写入的字节数
                fs.Write(hostMemory.getByteData(), 0, (int)hostMemory.Size);
            }

        }
    }
}
