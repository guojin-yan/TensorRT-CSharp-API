using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>Reports copied, pointer-free state from a native <c>IGpuAllocator</c> owner. 报告 native IGpuAllocator owner 复制出的无指针状态。</summary>
public readonly struct TensorRtGpuAllocatorRuntimeSnapshot
{
    internal TensorRtGpuAllocatorRuntimeSnapshot(
        TensorRtApiLine line,
        ulong ownerId,
        ulong invocationCount,
        ulong allocateCount,
        ulong reallocateCount,
        ulong deallocateCount,
        ulong allocateAsyncCount,
        ulong deallocateAsyncCount,
        ulong rejectedCount,
        ulong callbackFailureCount,
        ulong cudaFailureCount,
        ulong inFlightCallbackCount,
        ulong maxInFlightCallbackCount,
        ulong attachCount,
        ulong detachCount,
        ulong liveAllocationCount,
        ulong liveAllocationBytes,
        ulong peakLiveAllocationBytes,
        ulong lastRequestedSize,
        ulong lastAlignment,
        uint lastAllocatorFlags,
        BridgeStatusCode lastStatus,
        TensorRtGpuAllocatorAttachmentTarget attachmentTarget,
        TensorRtGpuAllocatorCallbackKind lastCallbackKind,
        bool isAttached,
        bool lastCallbackSucceeded,
        bool lastOperationSucceeded,
        bool lastHadCurrentMemory,
        bool lastHadStream,
        string diagnostic)
    {
        Line = line;
        OwnerId = ownerId;
        InvocationCount = invocationCount;
        AllocateCount = allocateCount;
        ReallocateCount = reallocateCount;
        DeallocateCount = deallocateCount;
        AllocateAsyncCount = allocateAsyncCount;
        DeallocateAsyncCount = deallocateAsyncCount;
        RejectedCount = rejectedCount;
        CallbackFailureCount = callbackFailureCount;
        CudaFailureCount = cudaFailureCount;
        InFlightCallbackCount = inFlightCallbackCount;
        MaxInFlightCallbackCount = maxInFlightCallbackCount;
        AttachCount = attachCount;
        DetachCount = detachCount;
        LiveAllocationCount = liveAllocationCount;
        LiveAllocationBytes = liveAllocationBytes;
        PeakLiveAllocationBytes = peakLiveAllocationBytes;
        LastRequestedSize = lastRequestedSize;
        LastAlignment = lastAlignment;
        LastAllocatorFlags = lastAllocatorFlags;
        LastStatus = lastStatus;
        AttachmentTarget = attachmentTarget;
        LastCallbackKind = lastCallbackKind;
        IsAttached = isAttached;
        LastCallbackSucceeded = lastCallbackSucceeded;
        LastOperationSucceeded = lastOperationSucceeded;
        LastHadCurrentMemory = lastHadCurrentMemory;
        LastHadStream = lastHadStream;
        Diagnostic = diagnostic ?? string.Empty;
    }

    /// <summary>Gets the TensorRT API line. 获取 TensorRT API 版本线。</summary>
    public TensorRtApiLine Line { get; }
    /// <summary>Gets the stable native owner id. 获取稳定的 native owner id。</summary>
    public ulong OwnerId { get; }
    /// <summary>Gets total callback invocations. 获取回调总次数。</summary>
    public ulong InvocationCount { get; }
    /// <summary>Gets synchronous allocation callback count. 获取同步分配回调次数。</summary>
    public ulong AllocateCount { get; }
    /// <summary>Gets reallocation callback count. 获取重分配回调次数。</summary>
    public ulong ReallocateCount { get; }
    /// <summary>Gets synchronous release callback count. 获取同步释放回调次数。</summary>
    public ulong DeallocateCount { get; }
    /// <summary>Gets stream-aware allocation callback count. 获取异步分配回调次数。</summary>
    public ulong AllocateAsyncCount { get; }
    /// <summary>Gets stream-aware release callback count. 获取异步释放回调次数。</summary>
    public ulong DeallocateAsyncCount { get; }
    /// <summary>Gets managed policy rejection count. 获取托管策略拒绝次数。</summary>
    public ulong RejectedCount { get; }
    /// <summary>Gets managed callback failure count. 获取托管回调失败次数。</summary>
    public ulong CallbackFailureCount { get; }
    /// <summary>Gets native CUDA failure count. 获取 native CUDA 失败次数。</summary>
    public ulong CudaFailureCount { get; }
    /// <summary>Gets callbacks currently in flight. 获取当前执行中的回调数。</summary>
    public ulong InFlightCallbackCount { get; }
    /// <summary>Gets peak callback concurrency. 获取回调峰值并发数。</summary>
    public ulong MaxInFlightCallbackCount { get; }
    /// <summary>Gets successful native attach count. 获取 native 挂载次数。</summary>
    public ulong AttachCount { get; }
    /// <summary>Gets successful native detach count. 获取 native 解除挂载次数。</summary>
    public ulong DetachCount { get; }
    /// <summary>Gets live owner-tracked allocations. 获取存活的 owner 台账分配数。</summary>
    public ulong LiveAllocationCount { get; }
    /// <summary>Gets live owner-tracked bytes. 获取存活的 owner 台账字节数。</summary>
    public ulong LiveAllocationBytes { get; }
    /// <summary>Gets peak live bytes. 获取峰值存活字节数。</summary>
    public ulong PeakLiveAllocationBytes { get; }
    /// <summary>Gets the last requested size. 获取最近请求大小。</summary>
    public ulong LastRequestedSize { get; }
    /// <summary>Gets the last requested alignment. 获取最近请求对齐。</summary>
    public ulong LastAlignment { get; }
    /// <summary>Gets the last allocator flags. 获取最近 allocator flags。</summary>
    public uint LastAllocatorFlags { get; }
    /// <summary>Gets the last bridge status. 获取最近 bridge 状态。</summary>
    public BridgeStatusCode LastStatus { get; }
    /// <summary>Gets the current attachment target. 获取当前挂载目标。</summary>
    public TensorRtGpuAllocatorAttachmentTarget AttachmentTarget { get; }
    /// <summary>Gets the last callback kind. 获取最近回调类型。</summary>
    public TensorRtGpuAllocatorCallbackKind LastCallbackKind { get; }
    /// <summary>Gets whether the owner is attached. 获取 owner 是否已挂载。</summary>
    public bool IsAttached { get; }
    /// <summary>Gets whether the last managed callback completed. 获取最近托管回调是否完成。</summary>
    public bool LastCallbackSucceeded { get; }
    /// <summary>Gets whether the last native operation succeeded. 获取最近 native 操作是否成功。</summary>
    public bool LastOperationSucceeded { get; }
    /// <summary>Gets whether existing memory was reported without exposing it. 获取是否报告已有显存但不暴露地址。</summary>
    public bool LastHadCurrentMemory { get; }
    /// <summary>Gets whether a CUDA stream was reported without exposing it. 获取是否报告 CUDA stream 但不暴露句柄。</summary>
    public bool LastHadStream { get; }
    /// <summary>Gets copied native diagnostics. 获取复制后的 native 诊断。</summary>
    public string Diagnostic { get; }
    /// <summary>Gets whether this public snapshot exposes any native pointer. 获取该公开快照是否暴露 native pointer。</summary>
    public bool NativePointerExposed => false;
    /// <summary>Gets whether a successful real TensorRT allocation callback was observed. 获取是否观察到成功的真实 TensorRT 分配回调。</summary>
    public bool RealCallbackRuntime =>
        AllocateCount + AllocateAsyncCount > 0 &&
        CallbackFailureCount == 0 &&
        CudaFailureCount == 0 &&
        RejectedCount == 0 &&
        InFlightCallbackCount == 0 &&
        PeakLiveAllocationBytes > 0;
    /// <summary>Gets the evidence kind. 获取证据类型。</summary>
    public string RuntimeEvidenceKind => RealCallbackRuntime ? "local-tensorrt-gpu-allocator-runtime" : "runtime-attempt";
    /// <summary>Returns a compact diagnostic representation. 返回紧凑诊断表示。</summary>
    public override string ToString() => $"{Line}:owner={OwnerId}:target={AttachmentTarget}:callbacks={InvocationCount}:live={LiveAllocationCount}:rejected={RejectedCount}:callbackFailures={CallbackFailureCount}:cudaFailures={CudaFailureCount}:pointerExposed={NativePointerExposed}";
}
