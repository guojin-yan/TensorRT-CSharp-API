namespace JYPPX.CudaSharp;

/// <summary>
/// Captures CUDA memory pressure and device capability signals for deployment diagnostics.
/// 捕获 CUDA 内存压力和设备能力信号，用于模型部署诊断。
/// </summary>
public sealed class CudaMemoryPressureSnapshot
{
    /// <summary>
    /// Creates a CUDA memory pressure snapshot.
    /// 创建 CUDA 内存压力快照。
    /// </summary>
    public CudaMemoryPressureSnapshot(int deviceOrdinal, CudaMemoryInfo memoryInfo, CudaDeviceProperties deviceProperties)
    {
        DeviceOrdinal = deviceOrdinal;
        MemoryInfo = memoryInfo;
        DeviceProperties = deviceProperties;
    }

    /// <summary>
    /// Gets the CUDA device ordinal that produced this snapshot.
    /// 获取生成该快照的 CUDA 设备序号。
    /// </summary>
    public int DeviceOrdinal { get; }

    /// <summary>
    /// Gets the current CUDA memory information.
    /// 获取当前 CUDA 内存信息。
    /// </summary>
    public CudaMemoryInfo MemoryInfo { get; }

    /// <summary>
    /// Gets the deployment-oriented CUDA device property snapshot.
    /// 获取面向部署的 CUDA 设备属性快照。
    /// </summary>
    public CudaDeviceProperties DeviceProperties { get; }

    /// <summary>
    /// Gets the current free memory ratio in the range 0 to 1.
    /// 获取当前空闲显存比例，范围为 0 到 1。
    /// </summary>
    public double FreeRatio => MemoryInfo.TotalBytes == 0 ? 0 : (double)MemoryInfo.FreeBytes / MemoryInfo.TotalBytes;

    /// <summary>
    /// Gets the current used memory ratio in the range 0 to 1.
    /// 获取当前已使用显存比例，范围为 0 到 1。
    /// </summary>
    public double UsedRatio => MemoryInfo.TotalBytes == 0 ? 0 : (double)MemoryInfo.UsedBytes / MemoryInfo.TotalBytes;

    /// <summary>
    /// Returns whether free device memory is below the specified ratio threshold.
    /// 返回空闲显存是否低于指定比例阈值。
    /// </summary>
    /// <param name="minimumFreeRatio">Minimum expected free memory ratio. 期望的最小空闲显存比例。</param>
    /// <returns><see langword="true"/> when memory pressure is high. 内存压力较高时返回 <see langword="true"/>。</returns>
    public bool IsBelowFreeRatio(double minimumFreeRatio)
    {
        return FreeRatio < minimumFreeRatio;
    }

    /// <summary>
    /// Returns a compact deployment diagnostic string.
    /// 返回简短的部署诊断字符串。
    /// </summary>
    public override string ToString()
    {
        return $"Device={DeviceOrdinal} Free={MemoryInfo.FreeBytes} Total={MemoryInfo.TotalBytes} FreeRatio={FreeRatio:0.000} CC={DeviceProperties.ComputeCapabilityLabel}";
    }
}
