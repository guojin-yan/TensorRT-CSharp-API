using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Identifies the CUDA device-resource union variant returned by CUDA 13.
/// 标识 CUDA 13 返回的设备资源 union 变体。
/// </summary>
public enum CudaDevResourceType
{
    Invalid = 0,
    Sm = 1,
    WorkqueueConfig = 1000,
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

    public CudaDevResourceType Type { get; }
    public bool IsValid { get; }
    public int DeviceOrdinal { get; }
    public uint SmCount { get; }
    public uint MinSmPartitionSize { get; }
    public uint SmCoscheduledAlignment { get; }
    public uint SmFlags { get; }
    public uint WorkqueueConcurrencyLimit { get; }
    public int WorkqueueSharingScope { get; }
    public bool HasOpaqueWorkqueue { get; }
    public bool HasNextResource { get; }

    public bool IsSmResource => Type == CudaDevResourceType.Sm;
    public bool IsWorkqueueConfig => Type == CudaDevResourceType.WorkqueueConfig;
    public bool IsWorkqueue => Type == CudaDevResourceType.Workqueue;

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

    public override bool Equals(object? obj) => obj is CudaDevResourceSnapshot other && Equals(other);
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
    public override string ToString() => $"Type={Type} Valid={IsValid} Device={DeviceOrdinal} SM={SmCount}/{MinSmPartitionSize}/{SmCoscheduledAlignment} WqLimit={WorkqueueConcurrencyLimit} OpaqueWorkqueue={HasOpaqueWorkqueue} NextResourceOmitted={HasNextResource}";

    public static bool operator ==(CudaDevResourceSnapshot left, CudaDevResourceSnapshot right) => left.Equals(right);
    public static bool operator !=(CudaDevResourceSnapshot left, CudaDevResourceSnapshot right) => !left.Equals(right);
}
