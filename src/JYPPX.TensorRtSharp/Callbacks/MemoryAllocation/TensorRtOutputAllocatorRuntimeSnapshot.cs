using System;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports pointer-free state copied from a native TensorRT output allocator owner.
/// 表示从 native TensorRT output allocator owner 复制出的无指针运行状态。
/// </summary>
public readonly struct TensorRtOutputAllocatorRuntimeSnapshot
{
    private readonly long[] _shapeDimensions;

    internal TensorRtOutputAllocatorRuntimeSnapshot(
        TensorRtApiLine line,
        ulong ownerId,
        ulong invocationCount,
        ulong notifyShapeCount,
        ulong reallocateOutputCount,
        ulong failureCount,
        ulong inFlightCallbackCount,
        ulong maxInFlightCallbackCount,
        ulong attachCount,
        ulong detachCount,
        ulong allocationCount,
        ulong reuseCount,
        ulong releaseCount,
        ulong liveAllocationCount,
        ulong liveAllocationBytes,
        ulong peakLiveAllocationBytes,
        ulong lastRequestedSize,
        ulong lastAlignment,
        BridgeStatusCode lastStatus,
        bool isAttached,
        bool lastCallbackSucceeded,
        bool lastAllocationSucceeded,
        bool lastHadCurrentMemory,
        bool lastHadStream,
        TensorRtOutputAllocatorCallbackKind lastCallbackKind,
        string tensorName,
        long[] shapeDimensions,
        string diagnostic)
    {
        Line = line;
        OwnerId = ownerId;
        InvocationCount = invocationCount;
        NotifyShapeCount = notifyShapeCount;
        ReallocateOutputCount = reallocateOutputCount;
        FailureCount = failureCount;
        InFlightCallbackCount = inFlightCallbackCount;
        MaxInFlightCallbackCount = maxInFlightCallbackCount;
        AttachCount = attachCount;
        DetachCount = detachCount;
        AllocationCount = allocationCount;
        ReuseCount = reuseCount;
        ReleaseCount = releaseCount;
        LiveAllocationCount = liveAllocationCount;
        LiveAllocationBytes = liveAllocationBytes;
        PeakLiveAllocationBytes = peakLiveAllocationBytes;
        LastRequestedSize = lastRequestedSize;
        LastAlignment = lastAlignment;
        LastStatus = lastStatus;
        IsAttached = isAttached;
        LastCallbackSucceeded = lastCallbackSucceeded;
        LastAllocationSucceeded = lastAllocationSucceeded;
        LastHadCurrentMemory = lastHadCurrentMemory;
        LastHadStream = lastHadStream;
        LastCallbackKind = lastCallbackKind;
        TensorName = tensorName ?? string.Empty;
        _shapeDimensions = shapeDimensions == null ? Array.Empty<long>() : (long[])shapeDimensions.Clone();
        Diagnostic = diagnostic ?? string.Empty;
    }

    /// <summary>Gets the TensorRT API line. 获取 TensorRT API 版本线。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the native owner id. 获取 native owner id。</summary>
    public ulong OwnerId { get; }

    /// <summary>Gets total native callback invocations. 获取 native callback 调用总数。</summary>
    public ulong InvocationCount { get; }

    /// <summary>Gets shape-notification count. 获取 shape 通知次数。</summary>
    public ulong NotifyShapeCount { get; }

    /// <summary>Gets output-reallocation request count. 获取输出重分配请求次数。</summary>
    public ulong ReallocateOutputCount { get; }

    /// <summary>Gets callback or allocation failure count. 获取 callback 或分配失败次数。</summary>
    public ulong FailureCount { get; }

    /// <summary>Gets callbacks currently in flight. 获取当前 in-flight callback 数量。</summary>
    public ulong InFlightCallbackCount { get; }

    /// <summary>Gets maximum callback concurrency. 获取最大 callback 并发数。</summary>
    public ulong MaxInFlightCallbackCount { get; }

    /// <summary>Gets successful attach count. 获取成功 attach 次数。</summary>
    public ulong AttachCount { get; }

    /// <summary>Gets successful detach count. 获取成功 detach 次数。</summary>
    public ulong DetachCount { get; }

    /// <summary>Gets native CUDA allocation count. 获取 native CUDA 分配次数。</summary>
    public ulong AllocationCount { get; }

    /// <summary>Gets owner allocation reuse count. 获取 owner 分配复用次数。</summary>
    public ulong ReuseCount { get; }

    /// <summary>Gets successfully released allocation count. 获取成功释放的分配数量。</summary>
    public ulong ReleaseCount { get; }

    /// <summary>Gets live owner allocation count. 获取 owner 当前存活分配数量。</summary>
    public ulong LiveAllocationCount { get; }

    /// <summary>Gets live native allocation bytes. 获取当前 native 分配字节数。</summary>
    public ulong LiveAllocationBytes { get; }

    /// <summary>Gets peak live native allocation bytes. 获取峰值 native 分配字节数。</summary>
    public ulong PeakLiveAllocationBytes { get; }

    /// <summary>Gets the last requested output size. 获取最近请求的输出大小。</summary>
    public ulong LastRequestedSize { get; }

    /// <summary>Gets the last requested alignment. 获取最近请求的对齐。</summary>
    public ulong LastAlignment { get; }

    /// <summary>Gets the last bridge status. 获取最近 bridge status。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets whether the native owner is attached. 获取 native owner 是否已绑定。</summary>
    public bool IsAttached { get; }

    /// <summary>Gets whether the last managed callback succeeded. 获取最近托管 callback 是否成功。</summary>
    public bool LastCallbackSucceeded { get; }

    /// <summary>Gets whether the last native allocation succeeded. 获取最近 native 分配是否成功。</summary>
    public bool LastAllocationSucceeded { get; }

    /// <summary>Gets whether TensorRT supplied current memory on the last request. 获取最近请求是否带有 current memory。</summary>
    public bool LastHadCurrentMemory { get; }

    /// <summary>Gets whether TensorRT supplied a CUDA stream on the last request. 获取最近请求是否带有 CUDA stream。</summary>
    public bool LastHadStream { get; }

    /// <summary>Gets the last callback operation. 获取最近 callback 操作。</summary>
    public TensorRtOutputAllocatorCallbackKind LastCallbackKind { get; }

    /// <summary>Gets the copied tensor name. 获取复制出的 tensor 名称。</summary>
    public string TensorName { get; }

    /// <summary>Gets copied output shape dimensions. 获取复制出的输出 shape。</summary>
    public ReadOnlyCollection<long> ShapeDimensions => Array.AsReadOnly(_shapeDimensions ?? Array.Empty<long>());

    /// <summary>Gets the copied native diagnostic. 获取复制出的 native 诊断。</summary>
    public string Diagnostic { get; }

    /// <summary>Gets whether any native pointer is exposed by this snapshot. 获取该快照是否暴露 native pointer。</summary>
    public bool NativePointerExposed => false;

    /// <summary>Gets whether the owner observed a successful real TensorRT allocation callback. 获取 owner 是否观察到成功的真实 TensorRT 分配 callback。</summary>
    public bool RealCallbackRuntime =>
        ReallocateOutputCount > 0 &&
        AllocationCount > 0 &&
        FailureCount == 0 &&
        InFlightCallbackCount == 0 &&
        LastCallbackSucceeded;

    /// <summary>Gets the runtime evidence kind. 获取运行证据类型。</summary>
    public string RuntimeEvidenceKind => RealCallbackRuntime ? "local-tensorrt-output-allocator-runtime" : "runtime-attempt";

    /// <summary>Returns a compact diagnostic representation. 返回紧凑诊断表示。</summary>
    public override string ToString()
    {
        return $"{Line}:owner={OwnerId}:attached={IsAttached}:callbacks={InvocationCount}:allocations={AllocationCount}:live={LiveAllocationCount}:failures={FailureCount}:pointerExposed={NativePointerExposed}";
    }
}
