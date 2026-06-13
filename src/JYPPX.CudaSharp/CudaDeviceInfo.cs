namespace JYPPX.CudaSharp;

/// <summary>
/// Describes one CUDA device as reported by the bridge.
/// 描述桥接层返回的单个 CUDA 设备信息。
/// </summary>
public sealed class CudaDeviceInfo
{
    /// <summary>
    /// Creates a CUDA device information snapshot.
    /// 创建 CUDA 设备信息快照。
    /// </summary>
    public CudaDeviceInfo(
        int ordinal,
        string name,
        int major,
        int minor,
        int multiProcessorCount,
        int warpSize,
        int maxThreadsPerBlock,
        bool canMapHostMemory,
        bool integrated,
        ulong totalGlobalMemory)
    {
        Ordinal = ordinal;
        Name = name;
        Major = major;
        Minor = minor;
        MultiProcessorCount = multiProcessorCount;
        WarpSize = warpSize;
        MaxThreadsPerBlock = maxThreadsPerBlock;
        CanMapHostMemory = canMapHostMemory;
        Integrated = integrated;
        TotalGlobalMemory = totalGlobalMemory;
    }

    /// <summary>
    /// Gets the CUDA device ordinal.
    /// 获取 CUDA 设备序号。
    /// </summary>
    public int Ordinal { get; }
    /// <summary>
    /// Gets the CUDA device name.
    /// 获取 CUDA 设备名称。
    /// </summary>
    public string Name { get; }
    /// <summary>
    /// Gets the CUDA compute capability major version.
    /// 获取 CUDA 计算能力主版本号。
    /// </summary>
    public int Major { get; }
    /// <summary>
    /// Gets the CUDA compute capability minor version.
    /// 获取 CUDA 计算能力次版本号。
    /// </summary>
    public int Minor { get; }
    /// <summary>
    /// Gets the number of streaming multiprocessors.
    /// 获取流式多处理器数量。
    /// </summary>
    public int MultiProcessorCount { get; }
    /// <summary>
    /// Gets the warp size reported by CUDA.
    /// 获取 CUDA 报告的 warp 大小。
    /// </summary>
    public int WarpSize { get; }
    /// <summary>
    /// Gets the maximum threads per block.
    /// 获取每个 block 支持的最大线程数。
    /// </summary>
    public int MaxThreadsPerBlock { get; }
    /// <summary>
    /// Gets whether the device can map host memory.
    /// 获取设备是否支持映射 host memory。
    /// </summary>
    public bool CanMapHostMemory { get; }
    /// <summary>
    /// Gets whether the device is integrated.
    /// 获取设备是否为集成型 GPU。
    /// </summary>
    public bool Integrated { get; }
    /// <summary>
    /// Gets the total global memory in bytes.
    /// 获取总 global memory 大小，单位字节。
    /// </summary>
    public ulong TotalGlobalMemory { get; }
}
