namespace JYPPX.CudaSharp;

/// <summary>
/// CUDA managed-memory range attributes that can be queried as scalar values.
/// 可作为标量查询的 CUDA managed memory range 属性。
/// </summary>
public enum CudaMemoryRangeAttribute
{
    /// <summary>
    /// Whether every page in the range has read-mostly advice enabled.
    /// 该范围内所有页面是否启用了 read-mostly 建议。
    /// </summary>
    ReadMostly = 1,

    /// <summary>
    /// Preferred location for every page in the range.
    /// 该范围内所有页面的首选位置。
    /// </summary>
    PreferredLocation = 2,

    /// <summary>
    /// Devices that have accessed-by advice for the whole range.
    /// 对整个范围设置 accessed-by 建议的设备。
    /// </summary>
    /// <remarks>This attribute returns an array in CUDA and is not accepted by the scalar query helpers. 该属性在 CUDA 中返回数组，标量查询辅助方法不接受它。</remarks>
    AccessedBy = 3,

    /// <summary>
    /// Last location to which every page in the range was explicitly prefetched.
    /// 该范围内所有页面最近一次被显式预取到的位置。
    /// </summary>
    LastPrefetchLocation = 4,

    /// <summary>
    /// Preferred location type for every page in the range. CUDA 12.3 or newer.
    /// 该范围内所有页面的首选位置类型。需要 CUDA 12.3 或更高版本。
    /// </summary>
    PreferredLocationType = 5,

    /// <summary>
    /// Preferred location id for every page in the range. CUDA 12.3 or newer.
    /// 该范围内所有页面的首选位置 id。需要 CUDA 12.3 或更高版本。
    /// </summary>
    PreferredLocationId = 6,

    /// <summary>
    /// Last prefetch location type for every page in the range. CUDA 12.3 or newer.
    /// 该范围内所有页面最近一次预取位置类型。需要 CUDA 12.3 或更高版本。
    /// </summary>
    LastPrefetchLocationType = 7,

    /// <summary>
    /// Last prefetch location id for every page in the range. CUDA 12.3 or newer.
    /// 该范围内所有页面最近一次预取位置 id。需要 CUDA 12.3 或更高版本。
    /// </summary>
    LastPrefetchLocationId = 8
}

/// <summary>
/// CUDA memory location types returned by CUDA 12.3+ range-location attributes.
/// CUDA 12.3+ range location 属性返回的位置类型。
/// </summary>
public enum CudaMemoryLocationType
{
    /// <summary>
    /// Invalid or unspecified location.
    /// 无效或未指定的位置。
    /// </summary>
    Invalid = 0,

    /// <summary>
    /// Device location.
    /// 设备位置。
    /// </summary>
    Device = 1,

    /// <summary>
    /// Host location.
    /// 主机位置。
    /// </summary>
    Host = 2,

    /// <summary>
    /// Host NUMA node location.
    /// 主机 NUMA 节点位置。
    /// </summary>
    HostNuma = 3,

    /// <summary>
    /// Host NUMA node closest to the current CPU thread.
    /// 最靠近当前 CPU 线程的主机 NUMA 节点。
    /// </summary>
    HostNumaCurrent = 4,

    /// <summary>
    /// Location is accessible but not visible.
    /// 位置可访问但不可见。
    /// </summary>
    Invisible = 5
}

/// <summary>
/// Managed snapshot of one CUDA memory range scalar attribute.
/// 一个 CUDA memory range 标量属性的托管快照。
/// </summary>
public readonly struct CudaMemoryRangeAttributeValue
{
    internal CudaMemoryRangeAttributeValue(CudaMemoryRangeAttribute attribute, int rawValue)
    {
        Attribute = attribute;
        RawValue = rawValue;
    }

    /// <summary>
    /// Gets the queried attribute.
    /// 获取查询的属性。
    /// </summary>
    public CudaMemoryRangeAttribute Attribute { get; }

    /// <summary>
    /// Gets the raw CUDA 32-bit value.
    /// 获取 CUDA 返回的原始 32 位值。
    /// </summary>
    public int RawValue { get; }

    /// <summary>
    /// Gets whether <see cref="CudaMemoryRangeAttribute.ReadMostly"/> is enabled.
    /// 获取 <see cref="CudaMemoryRangeAttribute.ReadMostly"/> 是否启用。
    /// </summary>
    public bool IsEnabled => RawValue != 0;

    /// <summary>
    /// Interprets the raw value as a CUDA memory location type.
    /// 将原始值解释为 CUDA memory location type。
    /// </summary>
    public CudaMemoryLocationType LocationType => (CudaMemoryLocationType)RawValue;

    /// <summary>
    /// Returns a readable diagnostic summary.
    /// 返回可读诊断摘要。
    /// </summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{Attribute}={RawValue}";
    }
}

/// <summary>
/// Summarizes copied CUDA managed-memory range diagnostics without exposing native pointers.
/// 汇总已复制的 CUDA managed-memory range 诊断信息，不暴露原生指针。
/// </summary>
public sealed class CudaMemoryRangeDiagnosticSummary
{
    internal CudaMemoryRangeDiagnosticSummary(
        int rangeSizeInBytes,
        int copiedScalarAttributeCount,
        int copiedAccessedByDeviceCount,
        bool adviceControlAttempted,
        bool prefetchControlAttempted)
    {
        RangeSizeInBytes = rangeSizeInBytes < 0 ? 0 : rangeSizeInBytes;
        CopiedScalarAttributeCount = copiedScalarAttributeCount < 0 ? 0 : copiedScalarAttributeCount;
        CopiedAccessedByDeviceCount = copiedAccessedByDeviceCount < 0 ? 0 : copiedAccessedByDeviceCount;
        AdviceControlAttempted = adviceControlAttempted;
        PrefetchControlAttempted = prefetchControlAttempted;
    }

    /// <summary>Gets the queried range size in bytes. 获取查询范围大小，单位为字节。</summary>
    public int RangeSizeInBytes { get; }

    /// <summary>Gets the copied scalar range attribute count. 获取已复制标量 range attribute 数量。</summary>
    public int CopiedScalarAttributeCount { get; }

    /// <summary>Gets the copied accessed-by device count. 获取已复制 accessed-by 设备数量。</summary>
    public int CopiedAccessedByDeviceCount { get; }

    /// <summary>Gets whether memory advice control was attempted before the snapshot. 获取快照前是否尝试过 memory advice 控制。</summary>
    public bool AdviceControlAttempted { get; }

    /// <summary>Gets whether memory prefetch control was attempted before the snapshot. 获取快照前是否尝试过 memory prefetch 控制。</summary>
    public bool PrefetchControlAttempted { get; }

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
        return $"RangeBytes={RangeSizeInBytes} ScalarAttributes={CopiedScalarAttributeCount} AccessedByDevices={CopiedAccessedByDeviceCount} Advice={AdviceControlAttempted} Prefetch={PrefetchControlAttempted} RuntimeProof={CanPromoteRuntimeProof}";
    }
}
