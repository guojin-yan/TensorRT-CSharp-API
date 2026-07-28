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
    /// Converts this graph-memory snapshot into a compact pointer-free summary.
    /// 将当前 graph memory 快照转换为紧凑、无指针逃逸的摘要。
    /// </summary>
    /// <remarks>
    /// This method only reads copied scalar counters returned by <c>cudaDeviceGetGraphMemAttribute</c>.
    /// It does not expose native graph-memory allocator pointers and does not promote local diagnostics
    /// to runtime or package-consumer proof.
    /// 该方法只读取 <c>cudaDeviceGetGraphMemAttribute</c> 返回的已复制标量计数；不会暴露原生
    /// graph memory allocator 指针，也不会将本地诊断晋级为 runtime 或 package-consumer proof。
    /// </remarks>
    public CudaDeviceGraphMemorySummary ToSummary()
    {
        return new CudaDeviceGraphMemorySummary(
            DeviceOrdinal,
            UsedMemoryCurrentBytes,
            UsedMemoryHighBytes,
            ReservedMemoryCurrentBytes,
            ReservedMemoryHighBytes);
    }

    /// <summary>
    /// Returns a compact graph-memory diagnostic string.
    /// 返回紧凑的 graph memory 诊断字符串。
    /// </summary>
    public override string ToString()
    {
        return $"Device={DeviceOrdinal} UsedCurrent={UsedMemoryCurrentBytes} UsedHigh={UsedMemoryHighBytes} ReservedCurrent={ReservedMemoryCurrentBytes} ReservedHigh={ReservedMemoryHighBytes}";
    }
}

/// <summary>
/// Summarizes copied CUDA device graph-memory counters without exposing native pointers.
/// 汇总已复制的 CUDA device graph-memory 计数，不暴露原生指针。
/// </summary>
public sealed class CudaDeviceGraphMemorySummary
{
    internal CudaDeviceGraphMemorySummary(
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

    /// <summary>Gets the CUDA device ordinal. 获取 CUDA 设备序号。</summary>
    public int DeviceOrdinal { get; }

    /// <summary>Gets bytes currently associated with CUDA graphs. 获取当前与 CUDA graphs 关联的字节数。</summary>
    public ulong UsedMemoryCurrentBytes { get; }

    /// <summary>Gets the used-memory high watermark in bytes. 获取 used memory 峰值字节数。</summary>
    public ulong UsedMemoryHighBytes { get; }

    /// <summary>Gets bytes currently reserved by the graph asynchronous allocator. 获取 graph 异步分配器当前保留字节数。</summary>
    public ulong ReservedMemoryCurrentBytes { get; }

    /// <summary>Gets the reserved-memory high watermark in bytes. 获取 reserved memory 峰值字节数。</summary>
    public ulong ReservedMemoryHighBytes { get; }

    /// <summary>Gets the copied graph-memory scalar counter count. 获取已复制 graph-memory 标量计数数量。</summary>
    public int CopiedScalarCounterCount => 4;

    /// <summary>Gets whether current counters are within their copied high watermarks. 获取当前计数是否不超过已复制峰值。</summary>
    public bool CurrentCountersWithinHighWatermarks =>
        UsedMemoryCurrentBytes <= UsedMemoryHighBytes &&
        ReservedMemoryCurrentBytes <= ReservedMemoryHighBytes;

    /// <summary>Gets the runtime evidence kind represented by this copied summary. 获取该 copied summary 表示的 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind => "copied-readonly-summary";

    /// <summary>Gets whether this summary is runtime execution evidence. 获取该摘要是否为 runtime execution evidence。</summary>
    public bool IsRuntimeExecutionEvidence => false;

    /// <summary>Gets whether this summary is runtime execution proof. 获取该摘要是否为 runtime execution proof。</summary>
    public bool IsRuntimeExecutionProof => false;

    /// <summary>Gets whether this summary is copied and pointer-free. 获取该摘要是否为复制型且无指针逃逸。</summary>
    public bool PointerFreeCopiedSummary => true;

    /// <summary>Gets whether this summary can be promoted as runtime proof. 获取该摘要是否可晋级为 runtime proof。</summary>
    public bool CanPromoteRuntimeProof => false;

    /// <summary>Gets whether this summary can promote public release proof. 获取该摘要是否可晋级为 public release proof。</summary>
    public bool CanPromoteReleaseProof => false;

    /// <summary>Gets whether deferred history can be deleted because of this summary. 获取是否可因该摘要删除 deferred history。</summary>
    public bool CanDeleteDeferredRecord => false;

    /// <summary>Formats this summary for logs and smoke output. 将该摘要格式化为日志和 smoke 输出。</summary>
    public override string ToString()
    {
        return $"Device={DeviceOrdinal} Counters={CopiedScalarCounterCount} UsedCurrent={UsedMemoryCurrentBytes} UsedHigh={UsedMemoryHighBytes} ReservedCurrent={ReservedMemoryCurrentBytes} ReservedHigh={ReservedMemoryHighBytes} RuntimeProof={CanPromoteRuntimeProof}";
    }
}
