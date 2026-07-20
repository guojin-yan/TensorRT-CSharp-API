using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Identifies the CUDA device-resource union variant returned by CUDA 13.
/// 标识 CUDA 13 返回的设备资源 union 变体。
/// </summary>
public enum CudaDevResourceType
{
    /// <summary>No valid device-resource variant is present. 不包含有效的设备资源变体。</summary>
    Invalid = 0,
    /// <summary>Streaming-multiprocessor resource metadata. 流式多处理器资源元数据。</summary>
    Sm = 1,
    /// <summary>Workqueue configuration metadata. 工作队列配置元数据。</summary>
    WorkqueueConfig = 1000,
    /// <summary>Opaque workqueue-presence metadata. 不透明工作队列的存在性元数据。</summary>
    Workqueue = 10000
}

/// <summary>
/// Pointer-free copy of the observable fields in a CUDA device resource.
/// CUDA 设备资源可观察字段的无指针复制快照。
/// </summary>
/// <remarks>
/// CUDA keeps internal padding, opaque workqueue state, and a linked-resource pointer
/// inside <c>cudaDevResource</c>. This snapshot intentionally never exposes or reuses
/// those fields; it is suitable for capability and partition-planning diagnostics only.
/// CUDA 在 <c>cudaDevResource</c> 内维护内部填充、opaque workqueue 状态和链式资源指针；
/// 此快照有意不暴露或复用这些字段，只适合能力与分区规划诊断。
/// </remarks>
public readonly struct CudaDevResourceSnapshot : IEquatable<CudaDevResourceSnapshot>
{
    internal CudaDevResourceSnapshot(
        CudaDevResourceType type,
        bool isValid,
        int deviceOrdinal,
        uint smCount,
        uint minSmPartitionSize,
        uint smCoscheduledAlignment,
        uint smFlags,
        uint workqueueConcurrencyLimit,
        int workqueueSharingScope,
        bool hasOpaqueWorkqueue,
        bool hasNextResource)
    {
        Type = type;
        IsValid = isValid;
        DeviceOrdinal = deviceOrdinal;
        SmCount = smCount;
        MinSmPartitionSize = minSmPartitionSize;
        SmCoscheduledAlignment = smCoscheduledAlignment;
        SmFlags = smFlags;
        WorkqueueConcurrencyLimit = workqueueConcurrencyLimit;
        WorkqueueSharingScope = workqueueSharingScope;
        HasOpaqueWorkqueue = hasOpaqueWorkqueue;
        HasNextResource = hasNextResource;
    }

    internal CudaDevResourceSnapshot(Internal.Interop.NativeCudaDevResourceSnapshot native)
        : this(
            (CudaDevResourceType)native.Type,
            native.IsValid != 0,
            native.DeviceOrdinal,
            native.SmCount,
            native.MinSmPartitionSize,
            native.SmCoscheduledAlignment,
            native.SmFlags,
            native.WorkqueueConcurrencyLimit,
            native.WorkqueueSharingScope,
            native.HasOpaqueWorkqueue != 0,
            native.HasNextResource != 0)
    {
    }

    /// <summary>Gets the copied resource variant. 获取复制的资源变体。</summary>
    public CudaDevResourceType Type { get; }
    /// <summary>Gets whether CUDA reported a valid resource. 获取 CUDA 是否报告了有效资源。</summary>
    public bool IsValid { get; }
    /// <summary>Gets the associated CUDA device ordinal. 获取关联的 CUDA 设备序号。</summary>
    public int DeviceOrdinal { get; }
    /// <summary>Gets the available SM count for an SM resource. 获取 SM 资源的可用 SM 数量。</summary>
    public uint SmCount { get; }
    /// <summary>Gets the minimum SM partition size. 获取最小 SM 分区大小。</summary>
    public uint MinSmPartitionSize { get; }
    /// <summary>Gets the SM coscheduling alignment. 获取 SM 协同调度对齐值。</summary>
    public uint SmCoscheduledAlignment { get; }
    /// <summary>Gets the copied SM resource flags. 获取复制的 SM 资源标志。</summary>
    public uint SmFlags { get; }
    /// <summary>Gets the workqueue concurrency limit. 获取工作队列并发限制。</summary>
    public uint WorkqueueConcurrencyLimit { get; }
    /// <summary>Gets the copied workqueue sharing scope. 获取复制的工作队列共享范围。</summary>
    public int WorkqueueSharingScope { get; }
    /// <summary>Gets whether an opaque workqueue was present. 获取是否存在不透明工作队列。</summary>
    public bool HasOpaqueWorkqueue { get; }
    /// <summary>Gets whether CUDA reported another linked resource without exposing its pointer. 获取 CUDA 是否报告了另一个链式资源，但不暴露其指针。</summary>
    public bool HasNextResource { get; }

    /// <summary>Gets whether this snapshot describes an SM resource. 获取此快照是否描述 SM 资源。</summary>
    public bool IsSmResource => Type == CudaDevResourceType.Sm;
    /// <summary>Gets whether this snapshot describes workqueue configuration. 获取此快照是否描述工作队列配置。</summary>
    public bool IsWorkqueueConfig => Type == CudaDevResourceType.WorkqueueConfig;
    /// <summary>Gets whether this snapshot describes an opaque workqueue. 获取此快照是否描述不透明工作队列。</summary>
    public bool IsWorkqueue => Type == CudaDevResourceType.Workqueue;

    /// <summary>Compares this snapshot with another snapshot. 将此快照与另一个快照进行比较。</summary>
    public bool Equals(CudaDevResourceSnapshot other) =>
        Type == other.Type &&
        IsValid == other.IsValid &&
        DeviceOrdinal == other.DeviceOrdinal &&
        SmCount == other.SmCount &&
        MinSmPartitionSize == other.MinSmPartitionSize &&
        SmCoscheduledAlignment == other.SmCoscheduledAlignment &&
        SmFlags == other.SmFlags &&
        WorkqueueConcurrencyLimit == other.WorkqueueConcurrencyLimit &&
        WorkqueueSharingScope == other.WorkqueueSharingScope &&
        HasOpaqueWorkqueue == other.HasOpaqueWorkqueue &&
        HasNextResource == other.HasNextResource;

    /// <summary>Compares this snapshot with another object. 将此快照与另一个对象进行比较。</summary>
    public override bool Equals(object? obj) => obj is CudaDevResourceSnapshot other && Equals(other);
    /// <summary>Returns a hash code for the copied metadata. 返回复制型元数据的哈希码。</summary>
    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            hash = hash * 31 + (int)Type;
            hash = hash * 31 + (IsValid ? 1 : 0);
            hash = hash * 31 + DeviceOrdinal;
            hash = hash * 31 + (int)SmCount;
            hash = hash * 31 + (int)MinSmPartitionSize;
            hash = hash * 31 + (int)SmCoscheduledAlignment;
            hash = hash * 31 + (int)SmFlags;
            hash = hash * 31 + (int)WorkqueueConcurrencyLimit;
            hash = hash * 31 + WorkqueueSharingScope;
            hash = hash * 31 + (HasOpaqueWorkqueue ? 1 : 0);
            hash = hash * 31 + (HasNextResource ? 1 : 0);
            return hash;
        }
    }
    /// <summary>Formats the copied resource metadata for diagnostics. 格式化复制型资源元数据以供诊断。</summary>
    public override string ToString() => $"Type={Type} Valid={IsValid} Device={DeviceOrdinal} SM={SmCount}/{MinSmPartitionSize}/{SmCoscheduledAlignment} WqLimit={WorkqueueConcurrencyLimit} OpaqueWorkqueue={HasOpaqueWorkqueue} NextResourceOmitted={HasNextResource}";

    /// <summary>Tests two snapshots for equality. 测试两个快照是否相等。</summary>
    public static bool operator ==(CudaDevResourceSnapshot left, CudaDevResourceSnapshot right) => left.Equals(right);
    /// <summary>Tests two snapshots for inequality. 测试两个快照是否不相等。</summary>
    public static bool operator !=(CudaDevResourceSnapshot left, CudaDevResourceSnapshot right) => !left.Equals(right);
}
