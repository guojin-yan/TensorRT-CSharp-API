namespace JYPPX.CudaSharp;

/// <summary>
/// Identifies CUDA graph-memory attributes exposed by <c>cudaDeviceGetGraphMemAttribute</c>.
/// 标识 <c>cudaDeviceGetGraphMemAttribute</c> 暴露的 CUDA graph memory 属性。
/// </summary>
public enum CudaGraphMemoryAttribute
{
    /// <summary>
    /// Bytes currently associated with CUDA graphs.
    /// 当前与 CUDA graphs 关联的字节数。
    /// </summary>
    UsedMemoryCurrent = 0,

    /// <summary>
    /// High watermark of bytes associated with CUDA graphs since the last reset.
    /// 自上次重置以来与 CUDA graphs 关联的字节数峰值。
    /// </summary>
    UsedMemoryHigh = 1,

    /// <summary>
    /// Bytes currently reserved by the CUDA graphs asynchronous allocator.
    /// CUDA graphs 异步分配器当前保留的字节数。
    /// </summary>
    ReservedMemoryCurrent = 2,

    /// <summary>
    /// High watermark of bytes reserved by the CUDA graphs asynchronous allocator.
    /// CUDA graphs 异步分配器保留字节数的峰值。
    /// </summary>
    ReservedMemoryHigh = 3
}

/// <summary>
/// Captures graph-memory allocator counters for one CUDA device.
/// 捕获单个 CUDA 设备的 graph memory 分配器计数。
/// </summary>
public sealed class CudaDeviceGraphMemoryInfo
{
    /// <summary>
    /// Creates a CUDA graph-memory snapshot.
    /// 创建 CUDA graph memory 快照。
    /// </summary>
    public CudaDeviceGraphMemoryInfo(
        int deviceOrdinal,
        ulong usedMemoryCurrentBytes,
        ulong usedMemoryHighBytes,
        ulong reservedMemoryCurrentBytes,
        ulong reservedMemoryHighBytes)
    {
        DeviceOrdinal = deviceOrdinal;
        UsedMemoryCurrentBytes = usedMemoryCurrentBytes;
        UsedMemoryHighBytes = usedMemoryHighBytes;
        ReservedMemoryCurrentBytes = reservedMemoryCurrentBytes;
        ReservedMemoryHighBytes = reservedMemoryHighBytes;
    }

    /// <summary>
    /// Gets the CUDA device ordinal.
    /// 获取 CUDA 设备序号。
    /// </summary>
    public int DeviceOrdinal { get; }

    /// <summary>
    /// Gets bytes currently associated with CUDA graphs.
    /// 获取当前与 CUDA graphs 关联的字节数。
    /// </summary>
    public ulong UsedMemoryCurrentBytes { get; }

    /// <summary>
    /// Gets the high watermark of bytes associated with CUDA graphs.
    /// 获取与 CUDA graphs 关联字节数的峰值。
    /// </summary>
    public ulong UsedMemoryHighBytes { get; }

    /// <summary>
    /// Gets bytes currently reserved by the CUDA graphs asynchronous allocator.
    /// 获取 CUDA graphs 异步分配器当前保留的字节数。
    /// </summary>
    public ulong ReservedMemoryCurrentBytes { get; }

    /// <summary>
    /// Gets the high watermark of reserved graph-memory bytes.
    /// 获取 graph memory 保留字节数的峰值。
    /// </summary>
    public ulong ReservedMemoryHighBytes { get; }

    /// <summary>
    /// Returns a compact graph-memory summary.
    /// 返回紧凑的 graph memory 摘要。
    /// </summary>
    public override string ToString()
    {
        return $"Device={DeviceOrdinal} UsedCurrent={UsedMemoryCurrentBytes} UsedHigh={UsedMemoryHighBytes} ReservedCurrent={ReservedMemoryCurrentBytes} ReservedHigh={ReservedMemoryHighBytes}";
    }
}
