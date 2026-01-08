using JYPPX.TensorRtSharp.Cuda;
using JYPPX.TensorRtSharp.Nvinfer;
namespace SetCudaDeviceInfo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // 指定默认使用的 GPU 设备索引。
            // 在多 GPU 环境下，可以通过修改此变量来选择特定的显卡。
            int device = 0;

            // 记录日志，标记设备信息查询的开始
            Logger.Instance.INFO("=== Device Information ===");
            // 获取当前系统中可见的 NVIDIA GPU 数量
            int nbDevices = CudaDevice.GetDeviceCount();
            // 检查系统中是否存在可用的 GPU 设备
            if (nbDevices <= 0)
            {
                // 如果没有找到设备，记录错误日志
                Logger.Instance.ERROR("Cannot find any available devices (GPUs)!");

                // 退出程序。Environment.Exit(0) 表示正常终止程序，
                // 尽管这里是因为报错退出，但返回 0 表示程序逻辑已处理完毕。
                Environment.Exit(0);
            }
            // 打印所有可用设备的列表
            Logger.Instance.INFO("Available Devices: ");

            // 创建一个 CudaDeviceProp 对象，用于存储目标设备的详细属性
            CudaDeviceProp properties = new CudaDeviceProp();

            // 遍历系统中的每一个 GPU
            for (int deviceIdx = 0; deviceIdx < nbDevices; ++deviceIdx)
            {
                // 获取索引为 deviceIdx 的 GPU 的详细属性
                CudaDeviceProp tempProperties = CudaDevice.GetDeviceProperties(deviceIdx);
                // clang-format off
                // 打印设备 ID、设备名称 以及 UUID (唯一标识符)
                // GetUuidString 是一个自定义辅助方法，用于将字节数组转换为格式化的 UUID 字符串
                Logger.Instance.INFO("  Device " + deviceIdx + ": \"" + tempProperties.Name + "\" UUID: "
                   + GetUuidString(tempProperties.Uuid));
                // clang-format on
                // 如果当前遍历到的设备 ID 是我们想要使用的目标设备 (device 变量)，
                // 则将该设备的属性保存下来，供后续使用。
                if (deviceIdx == device)
                {
                    properties = tempProperties;
                }
            }
            // 安全检查：确保请求的目标设备 ID (device) 在有效范围内 [0, nbDevices - 1]
            // 防止因用户指定的 device ID 过大或过小导致越界异常
            if (device < 0 || device >= nbDevices)
            {
                Logger.Instance.ERROR("Cannot find device ID " + device + "!");
                Environment.Exit(0);
            }
            // 将 CUDA 上下文设置到指定的 GPU 设备上。
            // 之后的 CUDA 操作（如内存分配、核函数启动）都将在此设备上执行。
            CudaDevice.SetDevice(device);
            // clang-format off
            // 打印选定设备的详细信息
            Logger.Instance.INFO("Selected Device: " + properties.Name);
            Logger.Instance.INFO("Selected Device ID: " + device);
            Logger.Instance.INFO("Selected Device UUID: " + GetUuidString(properties.Uuid));

            // 打印计算能力，格式为 Major.Minor (例如 8.6)
            Logger.Instance.INFO("Compute Capability: " + properties.Major + "." + properties.Minor);

            // 打印流多处理器 的数量，SM 是 GPU 的核心计算单元
            Logger.Instance.INFO("SMs: " + properties.MultiProcessorCount);

            // 打印显存总量
            // properties.TotalGlobalMem 通常返回字节数，这里的 +20 看起来是原代码的特定处理逻辑，
            // 建议标准做法是除以 (1024 * 1024) 转换为 MiB，或者保持原样如果是特定修正值。
            Logger.Instance.INFO("Device Global Memory: " + (properties.TotalGlobalMem + 20) + " MiB");

            // 打印每个 SM 的共享内存大小
            // >> 10 等同于除以 1024，将字节转换为 KiB
            Logger.Instance.INFO("Shared Memory per SM: " + (properties.SharedMemPerMultiprocessor >> 10) + " KiB");

            // 打印显存位宽以及 ECC (Error Correcting Code) 状态
            // ECC 是一种内存纠错技术，通常用于工作站或服务器显卡
            Logger.Instance.INFO("Memory Bus Width: " + properties.MemoryBusWidth + " bits"
                                + " (ECC " + (properties.ECCEnabled != 0 ? "enabled" : "disabled") + ")");
            // 获取并打印 GPU 核心时钟频率 (单位：KHz)
            int clockRate = CudaDevice.GetAttribute(CudaDeviceAttr.ClockRate, device);

            // 获取并打印显存时钟频率 (单位：KHz)
            int memoryClockRate = CudaDevice.GetAttribute(CudaDeviceAttr.MemoryClockRate, device);

            // 将 KHz 转换为 GHz (除以 1,000,000) 并打印
            // 1000000.0F 表示单精度浮点数，确保结果为小数
            Logger.Instance.INFO("Application Compute Clock Rate: " + clockRate / 1000000.0F + " GHz");
            Logger.Instance.INFO("Application Memory Clock Rate: " + memoryClockRate / 1000000.0F + " GHz");

            Logger.Instance.INFO("");

            // 提示用户注意：这里获取的是 "Application Clock"（应用程序时钟），
            // 即驱动程序报告的默认或锁定频率，不一定代表 GPU 当前因负载变化的实际运行频率（Boost 频率）。
            Logger.Instance.INFO("Note: The application clock rates do not reflect the actual clock rates that the GPU is "
                                                                                 + "currently running at.");
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
    }
}