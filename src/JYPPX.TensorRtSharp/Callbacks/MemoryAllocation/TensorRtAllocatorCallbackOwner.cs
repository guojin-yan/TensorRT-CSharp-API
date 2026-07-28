using System;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes one managed allocator dry-run diagnostic request.
/// 描述一次托管 allocator dry-run 诊断请求。
/// </summary>
/// <remarks>
/// This value is diagnostic-only. It does not carry a TensorRT device pointer, CUDA stream, or native allocator ownership.
/// 该值仅用于诊断；它不携带 TensorRT device pointer、CUDA stream 或 native allocator 所有权。
/// </remarks>
public readonly struct TensorRtAllocatorDryRunRequest
{
    /// <summary>
    /// Creates an allocator dry-run request.
    /// 创建 allocator dry-run 请求。
    /// </summary>
    /// <param name="size">The requested allocation size in bytes. 请求分配的字节数。</param>
    /// <param name="alignment">The requested alignment in bytes. 请求的字节对齐。</param>
    /// <param name="reason">A diagnostic reason copied from the caller. 调用方提供的诊断原因。</param>
    public TensorRtAllocatorDryRunRequest(ulong size, ulong alignment, string reason = "")
    {
        if (alignment == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(alignment), "Allocator dry-run alignment must be greater than zero.");
        }

        Size = size;
        Alignment = alignment;
        Reason = reason ?? string.Empty;
    }

    /// <summary>
    /// Gets the requested allocation size in bytes.
    /// 获取请求分配的字节数。
    /// </summary>
    public ulong Size { get; }

    /// <summary>
    /// Gets the requested alignment in bytes.
    /// 获取请求的字节对齐。
    /// </summary>
    public ulong Alignment { get; }

    /// <summary>
    /// Gets the caller-provided diagnostic reason.
    /// 获取调用方提供的诊断原因。
    /// </summary>
    public string Reason { get; }

    /// <summary>
    /// Returns a compact diagnostic representation.
    /// 返回紧凑的诊断表示。
    /// </summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{Reason}:{Size}:{Alignment}";
    }
}

/// <summary>
/// Reports the result of an allocator dry-run diagnostic.
/// 表示 allocator dry-run 诊断结果。
/// </summary>
/// <remarks>
/// The result intentionally contains no pointer value. It is a managed readiness signal, not an allocation result.
/// 该结果故意不包含任何指针值；它是托管可用性信号，不是分配结果。
/// </remarks>
public readonly struct TensorRtAllocatorDryRunResult
{
    /// <summary>
    /// Creates a dry-run result.
    /// 创建 dry-run 结果。
    /// </summary>
    /// <param name="succeeded">Whether the diagnostic completed successfully. 诊断是否成功完成。</param>
    /// <param name="diagnostic">A copied diagnostic message. 复制出的诊断消息。</param>
    public TensorRtAllocatorDryRunResult(bool succeeded, string diagnostic)
    {
        Succeeded = succeeded;
        Diagnostic = diagnostic ?? string.Empty;
    }

    /// <summary>
    /// Gets whether the diagnostic completed successfully.
    /// 获取诊断是否成功完成。
    /// </summary>
    public bool Succeeded { get; }

    /// <summary>
    /// Gets a copied diagnostic message.
    /// 获取复制出的诊断消息。
    /// </summary>
    public string Diagnostic { get; }

    /// <summary>
    /// Creates a successful dry-run result.
    /// 创建成功的 dry-run 结果。
    /// </summary>
    /// <param name="diagnostic">A copied diagnostic message. 复制出的诊断消息。</param>
    /// <returns>A successful result. 成功结果。</returns>
    public static TensorRtAllocatorDryRunResult Success(string diagnostic = "OK")
    {
        return new TensorRtAllocatorDryRunResult(true, diagnostic);
    }

    /// <summary>
    /// Creates a failed dry-run result.
    /// 创建失败的 dry-run 结果。
    /// </summary>
    /// <param name="diagnostic">A copied diagnostic message. 复制出的诊断消息。</param>
    /// <returns>A failed result. 失败结果。</returns>
    public static TensorRtAllocatorDryRunResult Failure(string diagnostic)
    {
        return new TensorRtAllocatorDryRunResult(false, diagnostic);
    }

    /// <summary>
    /// Returns a compact diagnostic representation.
    /// 返回紧凑的诊断表示。
    /// </summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{Succeeded}:{Diagnostic}";
    }
}

/// <summary>
/// Handles allocator dry-run diagnostics.
/// 处理 allocator dry-run 诊断。
/// </summary>
/// <param name="request">The copied dry-run request. 复制后的 dry-run 请求。</param>
/// <returns>A copied dry-run result that never contains a device pointer. 不包含 device pointer 的 dry-run 结果副本。</returns>
public delegate TensorRtAllocatorDryRunResult TensorRtAllocatorDryRunHandler(TensorRtAllocatorDryRunRequest request);

/// <summary>
/// Reports copied diagnostics from the native allocator owner dry-run C ABI.
/// 表示从 native allocator owner dry-run C ABI 复制出的诊断信息。
/// </summary>
/// <remarks>
/// This result does not expose or retain the native owner handle. It does not represent an enabled TensorRT allocator
/// callback and never contains a device pointer.
/// 该结果不会暴露或保留 native owner 句柄；它不表示 TensorRT allocator callback 已启用，也不会包含 device pointer。
/// </remarks>
public readonly struct TensorRtAllocatorNativeDryRunResult
{
    internal TensorRtAllocatorNativeDryRunResult(
        TensorRtApiLine line,
        ulong invocationCount,
        ulong failureCount,
        BridgeStatusCode lastStatus,
        bool isAttached,
        ulong lastSize,
        ulong lastAlignment,
        string diagnostic)
    {
        Line = line;
        InvocationCount = invocationCount;
        FailureCount = failureCount;
        LastStatus = lastStatus;
        IsAttached = isAttached;
        LastSize = lastSize;
        LastAlignment = lastAlignment;
        Diagnostic = diagnostic ?? string.Empty;
    }

    /// <summary>
    /// Gets the TensorRT API line used by the native dry-run owner.
    /// 获取 native dry-run owner 使用的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the native dry-run invocation count copied from the owner.
    /// 获取从 native owner 复制出的 dry-run 调用次数。
    /// </summary>
    public ulong InvocationCount { get; }

    /// <summary>
    /// Gets the native dry-run failure count copied from the owner.
    /// 获取从 native owner 复制出的 dry-run 失败次数。
    /// </summary>
    public ulong FailureCount { get; }

    /// <summary>
    /// Gets the last bridge status recorded by the native dry-run owner.
    /// 获取 native dry-run owner 记录的最近一次 bridge status。
    /// </summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>
    /// Gets whether the native dry-run owner is attached to TensorRT.
    /// 获取 native dry-run owner 是否绑定到 TensorRT。
    /// </summary>
    /// <remarks>
    /// This is always <see langword="false"/> in the dry-run C ABI skeleton.
    /// 在 dry-run C ABI 骨架中，该值始终为 <see langword="false"/>。
    /// </remarks>
    public bool IsAttached { get; }

    /// <summary>
    /// Gets the last dry-run size copied from the native owner.
    /// 获取从 native owner 复制出的最近一次 dry-run size。
    /// </summary>
    public ulong LastSize { get; }

    /// <summary>
    /// Gets the last dry-run alignment copied from the native owner.
    /// 获取从 native owner 复制出的最近一次 dry-run alignment。
    /// </summary>
    public ulong LastAlignment { get; }

    /// <summary>
    /// Gets the copied native diagnostic message.
    /// 获取复制出的 native 诊断消息。
    /// </summary>
    public string Diagnostic { get; }

    /// <summary>
    /// Gets whether the native dry-run diagnostic completed successfully.
    /// 获取 native dry-run 诊断是否成功完成。
    /// </summary>
    public bool Succeeded => LastStatus == BridgeStatusCode.Ok && FailureCount == 0;

    /// <summary>
    /// Returns a compact diagnostic representation.
    /// 返回紧凑的诊断表示。
    /// </summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{Line}:status={LastStatus}:invocations={InvocationCount}:failures={FailureCount}:attached={IsAttached}:{Diagnostic}";
    }
}

/// <summary>
/// Reports copied native allocator owner state intent data from the dry-run C ABI.
/// 表示从 dry-run C ABI 复制出的 native allocator owner 状态意图数据。
/// </summary>
/// <remarks>
/// This result copies the synthetic owner state only. It does not expose any native handle, device pointer, or real
/// TensorRT allocator callback.
/// 该结果只复制合成的 owner 状态，不暴露任何 native handle、device pointer 或真实 TensorRT allocator callback。
/// </remarks>
public readonly struct TensorRtAllocatorOwnerStateDryRunResult
{
    internal TensorRtAllocatorOwnerStateDryRunResult(
        TensorRtApiLine line,
        ulong ownerId,
        ulong stateTransitionCount,
        ulong ledgerAllocationCount,
        ulong ledgerReleaseCount,
        ulong ledgerFailureCount,
        ulong lastAllocationId,
        ulong lastReleaseAllocationId,
        ulong lastSize,
        ulong lastAlignment,
        ulong lastStreamValue,
        int attachState,
        BridgeStatusCode lastStatus,
        bool isAttached,
        bool hasLiveAllocation,
        string lastOperation,
        string diagnostic)
    {
        Line = line;
        OwnerId = ownerId;
        StateTransitionCount = stateTransitionCount;
        LedgerAllocationCount = ledgerAllocationCount;
        LedgerReleaseCount = ledgerReleaseCount;
        LedgerFailureCount = ledgerFailureCount;
        LastAllocationId = lastAllocationId;
        LastReleaseAllocationId = lastReleaseAllocationId;
        LastSize = lastSize;
        LastAlignment = lastAlignment;
        LastStreamValue = lastStreamValue;
        AttachState = attachState;
        LastStatus = lastStatus;
        IsAttached = isAttached;
        HasLiveAllocation = hasLiveAllocation;
        LastOperation = lastOperation ?? string.Empty;
        Diagnostic = diagnostic ?? string.Empty;
    }

    /// <summary>Gets the TensorRT API line used by the native dry-run owner. 获取 native dry-run owner 使用的 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the synthetic owner id copied from native. 获取从 native 复制出的合成 owner id。</summary>
    public ulong OwnerId { get; }

    /// <summary>Gets the number of state transitions copied from native. 获取从 native 复制出的状态转移次数。</summary>
    public ulong StateTransitionCount { get; }

    /// <summary>Gets the number of recorded allocation intents. 获取记录的分配意图次数。</summary>
    public ulong LedgerAllocationCount { get; }

    /// <summary>Gets the number of recorded release intents. 获取记录的释放意图次数。</summary>
    public ulong LedgerReleaseCount { get; }

    /// <summary>Gets the number of ledger failures copied from native. 获取从 native 复制出的 ledger 失败次数。</summary>
    public ulong LedgerFailureCount { get; }

    /// <summary>Gets the last synthetic allocation id. 获取最近一次合成 allocation id。</summary>
    public ulong LastAllocationId { get; }

    /// <summary>Gets the last synthetic release allocation id. 获取最近一次释放意图的 allocation id。</summary>
    public ulong LastReleaseAllocationId { get; }

    /// <summary>Gets the last copied size. 获取最近一次复制出的 size。</summary>
    public ulong LastSize { get; }

    /// <summary>Gets the last copied alignment. 获取最近一次复制出的 alignment。</summary>
    public ulong LastAlignment { get; }

    /// <summary>Gets the last copied stream value. 获取最近一次复制出的 stream value。</summary>
    public ulong LastStreamValue { get; }

    /// <summary>Gets the native attach state. 获取 native attach state。</summary>
    public int AttachState { get; }

    /// <summary>Gets the last bridge status recorded by native. 获取 native 记录的最近一次 bridge status。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets whether the synthetic owner is attached. 获取合成 owner 是否处于 attached 状态。</summary>
    public bool IsAttached { get; }

    /// <summary>Gets whether the synthetic ledger still has a live allocation. 获取合成 ledger 是否仍有 live allocation。</summary>
    public bool HasLiveAllocation { get; }

    /// <summary>Gets the copied last operation name. 获取复制出的最近一次操作名。</summary>
    public string LastOperation { get; }

    /// <summary>Gets the copied native diagnostic string. 获取复制出的 native 诊断字符串。</summary>
    public string Diagnostic { get; }

    /// <summary>Gets whether the copied state indicates success. 获取复制出的状态是否表示成功。</summary>
    public bool Succeeded => LastStatus == BridgeStatusCode.Ok && LedgerFailureCount == 0;

    /// <summary>Returns a compact diagnostic representation. 返回紧凑的诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{Line}:owner={OwnerId}:transitions={StateTransitionCount}:allocations={LedgerAllocationCount}:releases={LedgerReleaseCount}:failures={LedgerFailureCount}:live={HasLiveAllocation}:{LastOperation}:{Diagnostic}";
    }
}

/// <summary>
/// Reports a copied, pointer-free lifecycle snapshot from <see cref="TensorRtAllocatorCallbackOwner"/>.
/// 表示从 <see cref="TensorRtAllocatorCallbackOwner"/> 复制出的、无 pointer 的生命周期快照。
/// </summary>
/// <remarks>
/// The snapshot exposes managed keep-alive, in-flight, exception-to-status, and release-hook diagnostics only. It does
/// not expose a native callback owner pointer, device pointer, allocator-owned pointer, or borrowed TensorRT object.
/// 该快照只暴露托管 keep-alive、in-flight、exception-to-status 和 release hook 诊断；不会暴露 native callback owner
/// pointer、device pointer、allocator-owned pointer 或 TensorRT borrowed object。
/// </remarks>
public readonly struct TensorRtAllocatorCallbackOwnerSnapshot
{
    internal TensorRtAllocatorCallbackOwnerSnapshot(TensorRtAllocatorInternalRuntimePrototypeResult prototype)
    {
        OwnerId = prototype.OwnerId;
        Operation = prototype.Operation;
        LastStatus = prototype.LastStatus;
        InvocationCount = prototype.InvocationCount;
        FailureCount = prototype.FailureCount;
        InFlightCallbackCount = prototype.InFlightCallbackCount;
        MaxInFlightCallbackCount = prototype.MaxInFlightCallbackCount;
        ActivePrototypeCallCount = prototype.ActivePrototypeCallCount;
        ReleaseHookCount = prototype.ReleaseHookCount;
        CallbackStatePinned = prototype.CallbackStatePinned;
        DelegatePinned = prototype.DelegatePinned;
        DisposeRequested = prototype.DisposeRequested;
        IsAttached = prototype.IsAttached;
        LastDiagnostic = prototype.LastDiagnostic;
        ReleaseDiagnostic = prototype.ReleaseDiagnostic;
    }

    /// <summary>Gets the marker used by readiness to identify this snapshot. 获取 readiness 用于识别该快照的 marker。</summary>
    public string EvidenceKind => "allocator-owner-internal-runtime-prototype";

    /// <summary>Gets the callback kind represented by this diagnostic snapshot. 获取该诊断快照代表的 callback 类型。</summary>
    public string CallbackKind => "sync-allocator-prototype";

    /// <summary>Gets the runtime evidence kind. 获取 runtime 证据类型。</summary>
    public string RuntimeEvidenceKind => "not-present";

    /// <summary>Gets whether this snapshot proves a real TensorRT callback runtime. 获取该快照是否证明真实 TensorRT callback runtime。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this snapshot as real callback runtime proof. 获取 readiness 是否可将该快照提升为真实 callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the copied owner id. 获取复制出的 owner id。</summary>
    public long OwnerId { get; }

    /// <summary>Gets the copied operation label. 获取复制出的操作标签。</summary>
    public string Operation { get; }

    /// <summary>Gets the last copied bridge status. 获取最近一次复制出的 bridge status。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets the copied invocation count. 获取复制出的调用次数。</summary>
    public long InvocationCount { get; }

    /// <summary>Gets the copied failure count. 获取复制出的失败次数。</summary>
    public long FailureCount { get; }

    /// <summary>Gets the copied in-flight callback count. 获取复制出的 in-flight callback 数量。</summary>
    public long InFlightCallbackCount { get; }

    /// <summary>Gets the copied maximum in-flight callback count. 获取复制出的最大 in-flight callback 数量。</summary>
    public long MaxInFlightCallbackCount { get; }

    /// <summary>Gets the copied active prototype call count. 获取复制出的 active prototype 调用数量。</summary>
    public int ActivePrototypeCallCount { get; }

    /// <summary>Gets the copied release hook count. 获取复制出的 release hook 次数。</summary>
    public long ReleaseHookCount { get; }

    /// <summary>Gets whether managed callback state remains pinned. 获取托管 callback state 是否仍被 pin 住。</summary>
    public bool CallbackStatePinned { get; }

    /// <summary>Gets whether the managed delegate remains pinned. 获取托管 delegate 是否仍被 pin 住。</summary>
    public bool DelegatePinned { get; }

    /// <summary>Gets whether dispose has been requested. 获取是否已请求释放。</summary>
    public bool DisposeRequested { get; }

    /// <summary>Gets whether this owner is attached to TensorRT. 获取该 owner 是否已绑定到 TensorRT。</summary>
    public bool IsAttached { get; }

    /// <summary>Gets whether the public API exposes a device pointer. 获取 public API 是否暴露 device pointer。</summary>
    public bool DevicePointerExposed => false;

    /// <summary>Gets whether the diagnostic produced a device pointer. 获取该诊断是否产生 device pointer。</summary>
    public bool DevicePointerProduced => false;

    /// <summary>Gets whether a borrowed pointer escaped through this public API. 获取 borrowed pointer 是否通过该 public API 逃逸。</summary>
    public bool BorrowedPointerEscaped => false;

    /// <summary>Gets the copied diagnostic message. 获取复制出的诊断消息。</summary>
    public string LastDiagnostic { get; }

    /// <summary>Gets the copied release diagnostic message. 获取复制出的释放诊断消息。</summary>
    public string ReleaseDiagnostic { get; }

    /// <summary>Gets whether managed keep-alive handles are still present for a pre-dispose owner. 获取释放前 owner 的托管 keep-alive 句柄是否仍存在。</summary>
    public bool ManagedKeepAliveReady => CallbackStatePinned && DelegatePinned && !DisposeRequested;

    /// <summary>Gets whether dispose released keep-alive handles after callbacks drained. 获取 Dispose 是否在 callback drain 后释放 keep-alive 句柄。</summary>
    public bool DisposeReleaseReady =>
        DisposeRequested &&
        !CallbackStatePinned &&
        !DelegatePinned &&
        InFlightCallbackCount == 0 &&
        ReleaseHookCount > 0;

    /// <summary>Gets whether the copied public surface is pointer-free. 获取复制出的 public surface 是否无 pointer。</summary>
    public bool PointerFreeSurfaceReady =>
        !DevicePointerExposed &&
        !DevicePointerProduced &&
        !BorrowedPointerEscaped;

    /// <summary>Gets whether this copied diagnostic completed without managed failures. 获取该复制诊断是否未出现托管失败。</summary>
    public bool Succeeded => LastStatus == BridgeStatusCode.Ok && FailureCount == 0 && InFlightCallbackCount == 0;

    /// <summary>Returns a compact diagnostic representation. 返回紧凑的诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:owner={OwnerId}:operation={Operation}:status={LastStatus}:invocations={InvocationCount}:failures={FailureCount}:inflight={InFlightCallbackCount}:releaseHooks={ReleaseHookCount}:proof={IsRealCallbackRuntimeProof}";
    }
}

/// <summary>
/// Managed skeleton and native dry-run diagnostic surface for a future TensorRT allocator callback owner.
/// 未来 TensorRT allocator callback owner 的托管骨架与 native dry-run 诊断表面。
/// </summary>
/// <remarks>
/// This owner is currently dry-run only. It keeps managed callback state alive, records invocation/failure diagnostics, and
/// swallows handler exceptions into <see cref="LastCallbackException"/> and a failed <see cref="TensorRtAllocatorDryRunResult"/>.
/// It can create a short-lived native diagnostic owner through <see cref="RunNativeDryRunDiagnostic"/>, but it is not
/// attached to TensorRT, does not call allocator callbacks, and does not return or own device pointers.
/// 当前 owner 仅用于 dry-run。它会保持托管回调状态存活、记录调用/失败诊断，并把 handler 异常吞吐到
/// <see cref="LastCallbackException"/> 与失败的 <see cref="TensorRtAllocatorDryRunResult"/> 中。它可以通过
/// <see cref="RunNativeDryRunDiagnostic"/> 创建短生命周期 native 诊断 owner，但不会绑定到 TensorRT，不会调用
/// allocator callback，也不会返回或拥有 device pointer。
/// </remarks>
public sealed class TensorRtAllocatorCallbackOwner : IDisposable
{
    private static long s_nextOwnerId;

    private readonly object _gate = new object();
    private readonly long _ownerId;
    private readonly CallbackState _callbackState;
    private readonly TensorRtAllocatorInternalRuntimePrototypeCallback _runtimePrototypeCallback;
    private GCHandle _callbackStateHandle;
    private GCHandle _runtimePrototypeCallbackHandle;
    private bool _hasCallbackStateHandle;
    private bool _hasRuntimePrototypeCallbackHandle;
    private bool _disposeRequested;
    private int _activePrototypeCallCount;

    /// <summary>
    /// Creates a dry-run allocator callback owner.
    /// 创建 dry-run allocator callback owner。
    /// </summary>
    /// <param name="handler">The managed dry-run handler. 托管 dry-run 处理器。</param>
    /// <remarks>
    /// The handler is kept alive by this owner until <see cref="Dispose"/>. No TensorRT object borrows this owner in the
    /// current stage, so <see cref="IsAttached"/> always remains <see langword="false"/>.
    /// handler 会由当前 owner 保持存活直到 <see cref="Dispose"/>。当前阶段没有 TensorRT 对象借用该 owner，
    /// 因此 <see cref="IsAttached"/> 始终为 <see langword="false"/>。
    /// </remarks>
    public TensorRtAllocatorCallbackOwner(TensorRtAllocatorDryRunHandler handler)
    {
        if (handler == null)
        {
            throw new ArgumentNullException(nameof(handler));
        }

        _ownerId = Interlocked.Increment(ref s_nextOwnerId);
        _callbackState = new CallbackState(handler);
        _runtimePrototypeCallback = InvokeInternalSyncAllocatorPrototype;
        _callbackStateHandle = GCHandle.Alloc(_callbackState);
        _hasCallbackStateHandle = true;

        try
        {
            _runtimePrototypeCallbackHandle = GCHandle.Alloc(_runtimePrototypeCallback);
            _hasRuntimePrototypeCallbackHandle = true;
        }
        catch
        {
            FreeCallbackState();
            throw;
        }
    }

    /// <summary>
    /// Releases callback owner resources if <see cref="Dispose()"/> was not called.
    /// 如果调用方未调用 <see cref="Dispose()"/>，则释放 callback owner 资源。
    /// </summary>
    ~TensorRtAllocatorCallbackOwner()
    {
        Dispose();
    }

    /// <summary>
    /// Gets whether this owner has been disposed.
    /// 获取当前 owner 是否已释放。
    /// </summary>
    public bool IsDisposed
    {
        get
        {
            lock (_gate)
            {
                return _disposeRequested;
            }
        }
    }

    /// <summary>
    /// Gets whether this owner is currently attached to a TensorRT object.
    /// 获取当前 owner 是否正绑定到 TensorRT 对象。
    /// </summary>
    /// <remarks>
    /// The dry-run skeleton never attaches to TensorRT. This property is present so later native owner work can keep the
    /// same diagnostics shape without exposing borrowed pointers.
    /// dry-run 骨架不会绑定到 TensorRT。保留该属性是为了后续 native owner 工作延续同一诊断形状，同时不暴露 borrowed pointer。
    /// </remarks>
    public bool IsAttached => false;

    /// <summary>
    /// Gets the number of dry-run callback invocations observed by this owner.
    /// 获取当前 owner 观察到的 dry-run 回调调用次数。
    /// </summary>
    public long CallbackInvocationCount => _callbackState.InvocationCount;

    /// <summary>
    /// Gets the number of managed handler exceptions swallowed by this owner.
    /// 获取当前 owner 吞吐的托管 handler 异常次数。
    /// </summary>
    public long CallbackFailureCount => _callbackState.FailureCount;

    /// <summary>
    /// Gets the last exception thrown by the dry-run handler, if any.
    /// 获取 dry-run handler 最近一次抛出的异常。
    /// </summary>
    public Exception? LastCallbackException => _callbackState.LastException;

    /// <summary>
    /// Gets the last diagnostic message observed by this owner.
    /// 获取当前 owner 最近一次观察到的诊断消息。
    /// </summary>
    public string LastDiagnostic => _callbackState.LastDiagnostic;

    /// <summary>
    /// Runs a pointer-free lifecycle diagnostic through the managed allocator callback owner prototype.
    /// 通过托管 allocator callback owner prototype 执行无 pointer 的生命周期诊断。
    /// </summary>
    /// <param name="request">The copied dry-run request. 复制出的 dry-run 请求。</param>
    /// <returns>A copied lifecycle snapshot with no native handle or device pointer. 不包含 native handle 或 device pointer 的生命周期快照。</returns>
    /// <remarks>
    /// This method exercises the managed keep-alive and exception-to-status path used by the allocator owner prototype.
    /// It does not register the owner with TensorRT, does not call <c>setGpuAllocator</c>, and does not implement
    /// <c>IGpuAllocator::allocate/free/deallocate/reallocate</c>.
    /// 该方法会触发 allocator owner prototype 使用的托管 keep-alive 与 exception-to-status 路径。它不会把 owner 注册到
    /// TensorRT，不会调用 <c>setGpuAllocator</c>，也不会实现 <c>IGpuAllocator::allocate/free/deallocate/reallocate</c>。
    /// </remarks>
    public TensorRtAllocatorCallbackOwnerSnapshot RunLifecycleDiagnostic(TensorRtAllocatorDryRunRequest request)
    {
        return new TensorRtAllocatorCallbackOwnerSnapshot(RunInternalSyncAllocatorRuntimePrototype(request));
    }

    /// <summary>
    /// Gets a copied lifecycle snapshot of the current allocator callback owner state.
    /// 获取当前 allocator callback owner 状态的复制生命周期快照。
    /// </summary>
    /// <param name="operation">A copied operation label. 复制出的操作标签。</param>
    /// <returns>A copied lifecycle snapshot with no native handle or device pointer. 不包含 native handle 或 device pointer 的生命周期快照。</returns>
    /// <remarks>
    /// This method can be called after <see cref="Dispose"/> to verify release-hook and keep-alive cleanup diagnostics.
    /// It is still not proof that TensorRT invoked an allocator callback.
    /// 该方法可在 <see cref="Dispose"/> 后调用，用于验证 release hook 与 keep-alive 清理诊断；它仍然不证明 TensorRT
    /// 已调用 allocator callback。
    /// </remarks>
    public TensorRtAllocatorCallbackOwnerSnapshot GetSnapshot(string operation = "snapshot")
    {
        return new TensorRtAllocatorCallbackOwnerSnapshot(GetInternalRuntimePrototypeSnapshot(operation));
    }

    internal TensorRtAllocatorInternalRuntimePrototypeResult RunInternalSyncAllocatorRuntimePrototype(TensorRtAllocatorDryRunRequest request)
    {
        IntPtr callbackState;
        TensorRtAllocatorInternalRuntimePrototypeCallback callback;
        lock (_gate)
        {
            if (_disposeRequested)
            {
                throw new ObjectDisposedException(nameof(TensorRtAllocatorCallbackOwner));
            }

            if (!_hasCallbackStateHandle)
            {
                throw new ObjectDisposedException(nameof(TensorRtAllocatorCallbackOwner));
            }

            checked
            {
                _activePrototypeCallCount++;
            }

            callbackState = GCHandle.ToIntPtr(_callbackStateHandle);
            callback = _runtimePrototypeCallback;
        }

        BridgeStatusCode status;
        try
        {
            using Utf8Interop.Utf8StringScope reasonUtf8 = Utf8Interop.ToNativeString(request.Reason);
            status = callback(request.Size, request.Alignment, reasonUtf8.Pointer, callbackState);
        }
        finally
        {
            bool releaseNow;
            lock (_gate)
            {
                _activePrototypeCallCount--;
                releaseNow = _activePrototypeCallCount == 0 && _disposeRequested;
            }

            if (releaseNow)
            {
                FreeCallbackState();
            }
        }

        return CreateInternalRuntimePrototypeResult(status, "invoke");
    }

    internal TensorRtAllocatorInternalRuntimePrototypeResult GetInternalRuntimePrototypeSnapshot(string operation = "snapshot")
    {
        return CreateInternalRuntimePrototypeResult(_callbackState.LastStatus, operation);
    }

    /// <summary>
    /// Runs the managed allocator dry-run diagnostic.
    /// 执行托管 allocator dry-run 诊断。
    /// </summary>
    /// <param name="request">The dry-run request. dry-run 请求。</param>
    /// <returns>A copied diagnostic result that never contains a device pointer. 不包含 device pointer 的诊断结果副本。</returns>
    /// <remarks>
    /// This method does not call TensorRT, CUDA, or a native allocator trampoline. It is intended to validate managed
    /// callback lifetime, exception capture, and package-consumer surface before real allocator callbacks are unlocked.
    /// 该方法不会调用 TensorRT、CUDA 或 native allocator trampoline。它用于在解锁真实 allocator callback 前验证托管
    /// 回调生命周期、异常捕获与 package-consumer API 表面。
    /// </remarks>
    public TensorRtAllocatorDryRunResult RunDryRunDiagnostic(TensorRtAllocatorDryRunRequest request)
    {
        ThrowIfDisposed();
        _callbackState.RecordInvocation();

        try
        {
            TensorRtAllocatorDryRunResult result = _callbackState.Handler(request);
            string diagnostic = string.IsNullOrWhiteSpace(result.Diagnostic)
                ? (result.Succeeded ? "OK" : "allocator dry-run handler returned failure without a diagnostic.")
                : result.Diagnostic;

            result = new TensorRtAllocatorDryRunResult(result.Succeeded, diagnostic);
            _callbackState.RecordDiagnostic(result.Diagnostic);
            return result;
        }
        catch (Exception exception)
        {
            string diagnostic = "allocator dry-run handler threw " + exception.GetType().Name + ": " + exception.Message;
            _callbackState.RecordFailure(exception, diagnostic);
            return TensorRtAllocatorDryRunResult.Failure(diagnostic);
        }
    }

    /// <summary>
    /// Runs the native allocator owner dry-run C ABI diagnostic.
    /// 执行 native allocator owner dry-run C ABI 诊断。
    /// </summary>
    /// <param name="line">The TensorRT API line to probe. 要探测的 TensorRT API line。</param>
    /// <param name="request">The dry-run request. dry-run 请求。</param>
    /// <returns>Copied native dry-run diagnostics without any native handle or device pointer. 不包含 native handle 或 device pointer 的诊断副本。</returns>
    /// <remarks>
    /// This method creates a short-lived native diagnostic owner, emits one dry-run diagnostic, copies counters/status
    /// back to managed memory, and immediately releases the native handle. It does not attach to TensorRT, does not call
    /// <c>setGpuAllocator</c>, and does not implement <c>IGpuAllocator::allocate/free/deallocate/reallocate</c>.
    /// 该方法会创建一个短生命周期 native 诊断 owner，发出一次 dry-run 诊断，将计数与状态复制回托管内存，然后立即释放
    /// native 句柄。它不会绑定 TensorRT，不会调用 <c>setGpuAllocator</c>，也不会实现
    /// <c>IGpuAllocator::allocate/free/deallocate/reallocate</c>。
    /// </remarks>
    public TensorRtAllocatorNativeDryRunResult RunNativeDryRunDiagnostic(TensorRtApiLine line, TensorRtAllocatorDryRunRequest request)
    {
        ThrowIfDisposed();

        using SafeTensorRtObjectHandle nativeOwner = NativeBridgeApi.CreateAllocatorOwnerDryRun(line);
        NativeTensorRtAllocatorOwnerDiagnosticInfo nativeInfo =
            NativeBridgeApi.EmitAllocatorOwnerDryRunDiagnostic(line, nativeOwner, request.Size, request.Alignment, request.Reason);
        return CreateNativeDryRunResult(nativeInfo);
    }

    /// <summary>
    /// Runs the synthetic native owner state ledger dry-run sequence.
    /// 执行 synthetic native owner 状态 ledger dry-run 序列。
    /// </summary>
    /// <param name="line">The TensorRT API line to probe. 要探测的 TensorRT API line。</param>
    /// <param name="request">The dry-run request. dry-run 请求。</param>
    /// <param name="targetKind">The copied target kind label. 复制的目标类型标签。</param>
    /// <param name="streamValue">The synthetic stream value copied into the ledger. 复制到 ledger 的合成 stream value。</param>
    /// <returns>A copied native owner state snapshot with no handle or device pointer. 不包含 handle 或 device pointer 的 native owner 状态副本。</returns>
    /// <remarks>
    /// This method creates a short-lived native diagnostic owner, records synthetic attach/allocation/release/detach
    /// intents, copies the final state snapshot, and immediately releases the native handle. It does not attach to
    /// TensorRT, does not call <c>setGpuAllocator</c>, and does not implement any allocator callback trampoline.
    /// 该方法会创建短生命周期 native 诊断 owner，记录合成的 attach/allocation/release/detach intent，复制最终状态快照，
    /// 然后立即释放 native 句柄。它不会绑定 TensorRT，不会调用 <c>setGpuAllocator</c>，也不会实现任何 allocator callback trampoline。
    /// </remarks>
    public TensorRtAllocatorOwnerStateDryRunResult RunNativeStateLedgerDryRunDiagnostic(
        TensorRtApiLine line,
        TensorRtAllocatorDryRunRequest request,
        string targetKind = "IGpuAllocator",
        ulong streamValue = 0)
    {
        ThrowIfDisposed();

        using SafeTensorRtObjectHandle nativeOwner = NativeBridgeApi.CreateAllocatorOwnerDryRun(line);
        NativeTensorRtAllocatorOwnerStateInfo state = NativeBridgeApi.GetAllocatorOwnerDryRunState(line, nativeOwner);
        state = NativeBridgeApi.AttachAllocatorOwnerDryRunIntent(line, nativeOwner, targetKind);
        state = NativeBridgeApi.RecordAllocatorOwnerDryRunAllocationIntent(line, nativeOwner, request.Size, request.Alignment, streamValue);
        state = NativeBridgeApi.RecordAllocatorOwnerDryRunReleaseIntent(line, nativeOwner, state.LastAllocationId, streamValue);
        state = NativeBridgeApi.DetachAllocatorOwnerDryRunIntent(line, nativeOwner, targetKind);
        return CreateStateDryRunResult(state);
    }

    /// <summary>
    /// Releases the managed callback state keep-alive handle.
    /// 释放托管回调状态的 keep-alive 句柄。
    /// </summary>
    public void Dispose()
    {
        bool releaseNow;
        lock (_gate)
        {
            if (_disposeRequested)
            {
                return;
            }

            _disposeRequested = true;
            releaseNow = _activePrototypeCallCount == 0;
        }

        if (releaseNow)
        {
            FreeCallbackState();
        }

        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        lock (_gate)
        {
            if (_disposeRequested)
            {
                throw new ObjectDisposedException(nameof(TensorRtAllocatorCallbackOwner));
            }
        }
    }

    private void FreeCallbackState()
    {
        bool released = false;
        if (_hasRuntimePrototypeCallbackHandle)
        {
            _runtimePrototypeCallbackHandle.Free();
            _hasRuntimePrototypeCallbackHandle = false;
            released = true;
        }

        if (_hasCallbackStateHandle)
        {
            _callbackStateHandle.Free();
            _hasCallbackStateHandle = false;
            released = true;
        }

        if (released)
        {
            _callbackState.RecordReleaseHook("allocator-owner-internal-runtime-prototype release hook released managed GCHandle and delegate keep-alive handles after callbacks drained.");
            GC.KeepAlive(_runtimePrototypeCallback);
        }
    }

    private TensorRtAllocatorInternalRuntimePrototypeResult CreateInternalRuntimePrototypeResult(BridgeStatusCode status, string operation)
    {
        bool callbackStatePinned;
        bool delegatePinned;
        bool disposeRequested;
        int activePrototypeCallCount;
        lock (_gate)
        {
            callbackStatePinned = _hasCallbackStateHandle;
            delegatePinned = _hasRuntimePrototypeCallbackHandle;
            disposeRequested = _disposeRequested;
            activePrototypeCallCount = _activePrototypeCallCount;
        }

        BridgeStatusCode lastStatus = _callbackState.LastStatus;
        if (lastStatus != status)
        {
            lastStatus = status;
        }

        return new TensorRtAllocatorInternalRuntimePrototypeResult(
            ownerId: _ownerId,
            operation: operation,
            lastStatus: lastStatus,
            invocationCount: _callbackState.InvocationCount,
            failureCount: _callbackState.FailureCount,
            inFlightCallbackCount: _callbackState.InFlightCallbackCount,
            maxInFlightCallbackCount: _callbackState.MaxInFlightCallbackCount,
            activePrototypeCallCount: activePrototypeCallCount,
            releaseHookCount: _callbackState.ReleaseHookCount,
            callbackStatePinned: callbackStatePinned,
            delegatePinned: delegatePinned,
            disposeRequested: disposeRequested,
            isAttached: IsAttached,
            lastDiagnostic: _callbackState.LastDiagnostic,
            releaseDiagnostic: _callbackState.LastReleaseDiagnostic);
    }

    private static BridgeStatusCode InvokeInternalSyncAllocatorPrototype(
        ulong size,
        ulong alignment,
        IntPtr reason,
        IntPtr userState)
    {
        CallbackState? state = null;
        try
        {
            if (userState == IntPtr.Zero)
            {
                return BridgeStatusCode.InvalidArgument;
            }

            state = GCHandle.FromIntPtr(userState).Target as CallbackState;
            if (state == null)
            {
                return BridgeStatusCode.InvalidState;
            }

            state.EnterCallback();
            state.RecordInvocation();

            if (alignment == 0UL)
            {
                const string alignmentDiagnostic = "allocator-owner-internal-runtime-prototype invalid alignment; no TensorRT allocator callback was invoked.";
                state.RecordReturnedFailure(alignmentDiagnostic, BridgeStatusCode.InvalidArgument);
                return BridgeStatusCode.InvalidArgument;
            }

            TensorRtAllocatorDryRunResult result = state.Handler(new TensorRtAllocatorDryRunRequest(size, alignment, Utf8Interop.ReadString(reason)));
            string diagnostic = string.IsNullOrWhiteSpace(result.Diagnostic)
                ? (result.Succeeded ? "OK" : "allocator internal runtime prototype handler returned failure without a diagnostic.")
                : result.Diagnostic;

            if (result.Succeeded)
            {
                state.RecordStatus(BridgeStatusCode.Ok, diagnostic);
                return BridgeStatusCode.Ok;
            }

            state.RecordReturnedFailure(diagnostic, BridgeStatusCode.InvalidState);
            return BridgeStatusCode.InvalidState;
        }
        catch (Exception exception)
        {
            string diagnostic = "allocator internal runtime prototype handler threw " + exception.GetType().Name + ": " + exception.Message;
            state?.RecordFailure(exception, diagnostic, BridgeStatusCode.InvalidState);
            return BridgeStatusCode.InvalidState;
        }
        finally
        {
            state?.ExitCallback();
        }
    }

    private static TensorRtAllocatorNativeDryRunResult CreateNativeDryRunResult(NativeTensorRtAllocatorOwnerDiagnosticInfo info)
    {
        return new TensorRtAllocatorNativeDryRunResult(
            line: (TensorRtApiLine)info.Line,
            invocationCount: info.InvocationCount,
            failureCount: info.FailureCount,
            lastStatus: (BridgeStatusCode)info.LastStatus,
            isAttached: info.IsAttached != 0,
            lastSize: info.LastSize,
            lastAlignment: info.LastAlignment,
            diagnostic: BridgeInfoMapper.ReadFixedUtf8(info.LastDiagnostic));
    }

    private static TensorRtAllocatorOwnerStateDryRunResult CreateStateDryRunResult(NativeTensorRtAllocatorOwnerStateInfo info)
    {
        return new TensorRtAllocatorOwnerStateDryRunResult(
            line: (TensorRtApiLine)info.Line,
            ownerId: info.OwnerId,
            stateTransitionCount: info.StateTransitionCount,
            ledgerAllocationCount: info.LedgerAllocationCount,
            ledgerReleaseCount: info.LedgerReleaseCount,
            ledgerFailureCount: info.LedgerFailureCount,
            lastAllocationId: info.LastAllocationId,
            lastReleaseAllocationId: info.LastReleaseAllocationId,
            lastSize: info.LastSize,
            lastAlignment: info.LastAlignment,
            lastStreamValue: info.LastStreamValue,
            attachState: info.AttachState,
            lastStatus: (BridgeStatusCode)info.LastStatus,
            isAttached: info.IsAttached != 0,
            hasLiveAllocation: info.HasLiveAllocation != 0,
            lastOperation: BridgeInfoMapper.ReadFixedUtf8(info.LastOperation),
            diagnostic: BridgeInfoMapper.ReadFixedUtf8(info.LastDiagnostic));
    }

    private sealed class CallbackState
    {
        private long _invocationCount;
        private long _failureCount;
        private long _inFlightCallbackCount;
        private long _maxInFlightCallbackCount;
        private long _releaseHookCount;
        private int _lastStatus;
        private Exception? _lastException;
        private string _lastDiagnostic = string.Empty;
        private string _lastReleaseDiagnostic = string.Empty;

        public CallbackState(TensorRtAllocatorDryRunHandler handler)
        {
            Handler = handler;
        }

        public TensorRtAllocatorDryRunHandler Handler { get; }

        public long InvocationCount => Interlocked.Read(ref _invocationCount);

        public long FailureCount => Interlocked.Read(ref _failureCount);

        public long InFlightCallbackCount => Interlocked.Read(ref _inFlightCallbackCount);

        public long MaxInFlightCallbackCount => Interlocked.Read(ref _maxInFlightCallbackCount);

        public long ReleaseHookCount => Interlocked.Read(ref _releaseHookCount);

        public BridgeStatusCode LastStatus => (BridgeStatusCode)Volatile.Read(ref _lastStatus);

        public Exception? LastException => Volatile.Read(ref _lastException);

        public string LastDiagnostic => Volatile.Read(ref _lastDiagnostic);

        public string LastReleaseDiagnostic => Volatile.Read(ref _lastReleaseDiagnostic);

        public void EnterCallback()
        {
            long current = Interlocked.Increment(ref _inFlightCallbackCount);
            while (true)
            {
                long observedMax = Interlocked.Read(ref _maxInFlightCallbackCount);
                if (current <= observedMax)
                {
                    return;
                }

                if (Interlocked.CompareExchange(ref _maxInFlightCallbackCount, current, observedMax) == observedMax)
                {
                    return;
                }
            }
        }

        public void ExitCallback()
        {
            Interlocked.Decrement(ref _inFlightCallbackCount);
        }

        public void RecordInvocation()
        {
            Interlocked.Increment(ref _invocationCount);
        }

        public void RecordDiagnostic(string diagnostic)
        {
            Volatile.Write(ref _lastDiagnostic, diagnostic ?? string.Empty);
        }

        public void RecordStatus(BridgeStatusCode status, string diagnostic)
        {
            Volatile.Write(ref _lastStatus, (int)status);
            RecordDiagnostic(diagnostic);
        }

        public void RecordReturnedFailure(string diagnostic, BridgeStatusCode status)
        {
            RecordStatus(status, diagnostic);
            Interlocked.Increment(ref _failureCount);
        }

        public void RecordFailure(Exception exception, string diagnostic)
        {
            RecordFailure(exception, diagnostic, BridgeStatusCode.InvalidState);
        }

        public void RecordFailure(Exception exception, string diagnostic, BridgeStatusCode status)
        {
            Volatile.Write(ref _lastException, exception);
            RecordStatus(status, diagnostic);
            Interlocked.Increment(ref _failureCount);
        }

        public void RecordReleaseHook(string diagnostic)
        {
            Volatile.Write(ref _lastReleaseDiagnostic, diagnostic ?? string.Empty);
            Interlocked.Increment(ref _releaseHookCount);
        }
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate BridgeStatusCode TensorRtAllocatorInternalRuntimePrototypeCallback(
        ulong size,
        ulong alignment,
        IntPtr reason,
        IntPtr userState);
}

internal readonly struct TensorRtAllocatorInternalRuntimePrototypeResult
{
    internal TensorRtAllocatorInternalRuntimePrototypeResult(
        long ownerId,
        string operation,
        BridgeStatusCode lastStatus,
        long invocationCount,
        long failureCount,
        long inFlightCallbackCount,
        long maxInFlightCallbackCount,
        int activePrototypeCallCount,
        long releaseHookCount,
        bool callbackStatePinned,
        bool delegatePinned,
        bool disposeRequested,
        bool isAttached,
        string lastDiagnostic,
        string releaseDiagnostic)
    {
        OwnerId = ownerId;
        Operation = operation ?? string.Empty;
        LastStatus = lastStatus;
        InvocationCount = invocationCount;
        FailureCount = failureCount;
        InFlightCallbackCount = inFlightCallbackCount;
        MaxInFlightCallbackCount = maxInFlightCallbackCount;
        ActivePrototypeCallCount = activePrototypeCallCount;
        ReleaseHookCount = releaseHookCount;
        CallbackStatePinned = callbackStatePinned;
        DelegatePinned = delegatePinned;
        DisposeRequested = disposeRequested;
        IsAttached = isAttached;
        LastDiagnostic = lastDiagnostic ?? string.Empty;
        ReleaseDiagnostic = releaseDiagnostic ?? string.Empty;
    }

    public bool RealCallbackRuntime => false;

    public string EvidenceKind => "allocator-owner-internal-runtime-prototype";

    public string CallbackKind => "sync-allocator-prototype";

    public long OwnerId { get; }

    public string Operation { get; }

    public BridgeStatusCode LastStatus { get; }

    public long InvocationCount { get; }

    public long FailureCount { get; }

    public long InFlightCallbackCount { get; }

    public long MaxInFlightCallbackCount { get; }

    public int ActivePrototypeCallCount { get; }

    public long ReleaseHookCount { get; }

    public bool CallbackStatePinned { get; }

    public bool DelegatePinned { get; }

    public bool DisposeRequested { get; }

    public bool IsAttached { get; }

    public string LastDiagnostic { get; }

    public string ReleaseDiagnostic { get; }

    public bool Succeeded => LastStatus == BridgeStatusCode.Ok && FailureCount == 0 && InFlightCallbackCount == 0;

    public override string ToString()
    {
        return $"{EvidenceKind}:owner={OwnerId}:status={LastStatus}:invocations={InvocationCount}:failures={FailureCount}:inflight={InFlightCallbackCount}:releaseHooks={ReleaseHookCount}:realRuntime={RealCallbackRuntime}";
    }
}
