using System;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes one diagnostic request for the future TensorRT output allocator callback owner.
/// 描述未来 TensorRT output allocator callback owner 的一次诊断请求。
/// </summary>
/// <remarks>
/// This request contains copied metadata only. It does not carry a TensorRT output buffer, CUDA stream handle, or device
/// pointer ownership.
/// 该请求只包含复制出的元数据；不携带 TensorRT output buffer、CUDA stream handle 或 device pointer 所有权。
/// </remarks>
public readonly struct TensorRtOutputAllocatorCallbackRequest
{
    private const int MaxShapeRank = 8;
    private readonly long[] _shapeDimensions;

    /// <summary>
    /// Creates an output allocator owner diagnostic request.
    /// 创建 output allocator owner 诊断请求。
    /// </summary>
    /// <param name="tensorName">The copied output tensor name. 复制出的输出 tensor 名称。</param>
    /// <param name="requestedSize">The requested output buffer size in bytes. 请求的输出缓冲区字节数。</param>
    /// <param name="alignment">The requested output buffer alignment in bytes. 请求的输出缓冲区字节对齐。</param>
    /// <param name="shapeDimensions">The copied output shape dimensions. 复制出的输出 shape 维度。</param>
    /// <param name="reason">A diagnostic reason copied from the caller. 调用方提供的诊断原因。</param>
    /// <param name="hasCurrentMemory">Whether TensorRT reported an existing current memory pointer. TensorRT 是否报告已有 current memory pointer。</param>
    public TensorRtOutputAllocatorCallbackRequest(
        string tensorName,
        ulong requestedSize,
        ulong alignment,
        long[]? shapeDimensions,
        string reason = "",
        bool hasCurrentMemory = false)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Output allocator tensor name must not be empty.", nameof(tensorName));
        }

        if (alignment == 0UL)
        {
            throw new ArgumentOutOfRangeException(nameof(alignment), "Output allocator alignment must be greater than zero.");
        }

        _shapeDimensions = shapeDimensions == null ? Array.Empty<long>() : (long[])shapeDimensions.Clone();
        if (_shapeDimensions.Length > MaxShapeRank)
        {
            throw new ArgumentOutOfRangeException(nameof(shapeDimensions), "Output allocator diagnostic shape rank must be 8 or less.");
        }

        TensorName = tensorName;
        RequestedSize = requestedSize;
        Alignment = alignment;
        Reason = reason ?? string.Empty;
        HasCurrentMemory = hasCurrentMemory;
    }

    /// <summary>Gets the copied output tensor name. 获取复制出的输出 tensor 名称。</summary>
    public string TensorName { get; }

    /// <summary>Gets the requested output buffer size in bytes. 获取请求的输出缓冲区字节数。</summary>
    public ulong RequestedSize { get; }

    /// <summary>Gets the requested output buffer alignment in bytes. 获取请求的输出缓冲区字节对齐。</summary>
    public ulong Alignment { get; }

    /// <summary>Gets the copied output shape rank. 获取复制出的输出 shape rank。</summary>
    public int ShapeRank => _shapeDimensions.Length;

    /// <summary>Gets the copied output shape dimensions. 获取复制出的输出 shape 维度。</summary>
    public ReadOnlyCollection<long> ShapeDimensions => Array.AsReadOnly(_shapeDimensions);

    /// <summary>Gets the copied diagnostic reason. 获取复制出的诊断原因。</summary>
    public string Reason { get; }

    /// <summary>Gets whether an existing current memory pointer was reported. 获取是否报告了已有 current memory pointer。</summary>
    public bool HasCurrentMemory { get; }

    internal long[] CopyShapeDimensions()
    {
        return (long[])_shapeDimensions.Clone();
    }
}

/// <summary>
/// Reports copied diagnostics for the output allocator callback owner design gate.
/// 表示 output allocator callback owner 设计门禁复制出的诊断信息。
/// </summary>
/// <remarks>
/// This snapshot intentionally exposes no native handle, callback owner pointer, output buffer pointer, or borrowed
/// TensorRT object pointer. It is not proof that TensorRT has invoked <c>IOutputAllocator::notifyShape</c> or
/// <c>IOutputAllocator::reallocateOutput</c>.
/// 该快照有意不暴露 native handle、callback owner pointer、output buffer pointer 或 borrowed TensorRT object pointer。
/// 它不证明 TensorRT 已调用 <c>IOutputAllocator::notifyShape</c> 或 <c>IOutputAllocator::reallocateOutput</c>。
/// </remarks>
public readonly struct TensorRtOutputAllocatorCallbackOwnerSnapshot
{
    internal TensorRtOutputAllocatorCallbackOwnerSnapshot(
        TensorRtApiLine line,
        TensorRtOutputAllocatorRuntimeGateResult gate,
        TensorRtAllocatorOwnerStateDryRunResult? nativeLedger,
        BridgeStatusCode nativeLedgerStatus,
        string nativeLedgerDiagnostic,
        bool nativeLedgerAvailable)
    {
        Line = line;
        OwnerId = gate.OwnerId;
        Operation = gate.Operation;
        TensorName = gate.TensorName;
        RequestedSize = gate.RequestedSize;
        Alignment = gate.Alignment;
        ShapeRank = gate.ShapeRank;
        ShapeSummary = gate.ShapeSummary;
        HasCurrentMemory = gate.HasCurrentMemory;
        RuntimeGateStatus = gate.LastStatus;
        NativeLedgerStatus = nativeLedgerStatus;
        LastStatus = nativeLedgerStatus == BridgeStatusCode.Ok ? gate.LastStatus : nativeLedgerStatus;
        InvocationCount = gate.InvocationCount;
        NotifyShapeCount = gate.NotifyShapeCount;
        ReallocateOutputCount = gate.ReallocateOutputCount;
        FailureCount = gate.FailureCount;
        InFlightCallbackCount = gate.InFlightCallbackCount;
        MaxInFlightCallbackCount = gate.MaxInFlightCallbackCount;
        ActiveGateCallCount = gate.ActiveGateCallCount;
        ReleaseHookCount = gate.ReleaseHookCount;
        CallbackStatePinned = gate.CallbackStatePinned;
        DelegatePinned = gate.DelegatePinned;
        DisposeRequested = gate.DisposeRequested;
        IsAttached = false;
        NativeLedgerAvailable = nativeLedgerAvailable;
        NativeOwnerId = nativeLedger?.OwnerId ?? 0UL;
        StateTransitionCount = nativeLedger?.StateTransitionCount ?? 0UL;
        LedgerAllocationCount = nativeLedger?.LedgerAllocationCount ?? 0UL;
        LedgerReleaseCount = nativeLedger?.LedgerReleaseCount ?? 0UL;
        LedgerFailureCount = nativeLedger?.LedgerFailureCount ?? (nativeLedgerStatus == BridgeStatusCode.Ok ? 0UL : 1UL);
        LastAllocationId = nativeLedger?.LastAllocationId ?? 0UL;
        LastReleaseAllocationId = nativeLedger?.LastReleaseAllocationId ?? 0UL;
        LastStreamValue = nativeLedger?.LastStreamValue ?? 0UL;
        HasLiveAllocation = nativeLedger?.HasLiveAllocation ?? false;
        NativeLastOperation = nativeLedger?.LastOperation ?? string.Empty;
        NativeLedgerDiagnostic = nativeLedger?.Diagnostic ?? nativeLedgerDiagnostic ?? string.Empty;
        LastDiagnostic = ComposeDiagnostic(gate.LastDiagnostic, NativeLedgerDiagnostic, nativeLedgerAvailable);
        ReleaseDiagnostic = gate.ReleaseDiagnostic;
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 获取 readiness 用于识别该门禁的 marker。</summary>
    public string EvidenceKind => "output-allocator-callback-owner-design";

    /// <summary>Gets the callback kind represented by this diagnostic snapshot. 获取该诊断快照代表的 callback 类型。</summary>
    public string CallbackKind => "output-allocator-prototype";

    /// <summary>Gets the runtime evidence kind. 获取 runtime 证据类型。</summary>
    public string RuntimeEvidenceKind => "not-present";

    /// <summary>Gets whether this snapshot proves a real TensorRT callback runtime. 获取该快照是否证明真实 TensorRT callback runtime。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this snapshot as real callback runtime proof. 获取 readiness 是否可将该快照提升为真实 callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line used for the native ledger diagnostic attempt. 获取 native ledger 诊断尝试使用的 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the managed output allocator owner id copied from the runtime gate. 获取从 runtime gate 复制出的托管 output allocator owner id。</summary>
    public long OwnerId { get; }

    /// <summary>Gets the synthetic native owner id copied from the ledger dry-run. 获取从 ledger dry-run 复制出的合成 native owner id。</summary>
    public ulong NativeOwnerId { get; }

    /// <summary>Gets the last copied operation. 获取最近一次复制出的操作。</summary>
    public string Operation { get; }

    /// <summary>Gets the copied output tensor name. 获取复制出的输出 tensor 名称。</summary>
    public string TensorName { get; }

    /// <summary>Gets the copied requested output buffer size. 获取复制出的请求输出缓冲区大小。</summary>
    public ulong RequestedSize { get; }

    /// <summary>Gets the copied requested output buffer alignment. 获取复制出的请求输出缓冲区对齐。</summary>
    public ulong Alignment { get; }

    /// <summary>Gets the copied output shape rank. 获取复制出的输出 shape rank。</summary>
    public int ShapeRank { get; }

    /// <summary>Gets a compact copied output shape summary. 获取紧凑的复制输出 shape 摘要。</summary>
    public string ShapeSummary { get; }

    /// <summary>Gets whether TensorRT reported an existing current memory pointer. 获取 TensorRT 是否报告已有 current memory pointer。</summary>
    public bool HasCurrentMemory { get; }

    /// <summary>Gets the combined status for the gate and ledger diagnostics. 获取 gate 与 ledger 诊断合并后的状态。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets the status returned by the managed runtime gate. 获取托管 runtime gate 返回的状态。</summary>
    public BridgeStatusCode RuntimeGateStatus { get; }

    /// <summary>Gets the status returned by the native ledger diagnostic attempt. 获取 native ledger 诊断尝试返回的状态。</summary>
    public BridgeStatusCode NativeLedgerStatus { get; }

    /// <summary>Gets whether the native ledger diagnostic completed and produced copied state. 获取 native ledger 诊断是否完成并产生复制状态。</summary>
    public bool NativeLedgerAvailable { get; }

    /// <summary>Gets the total copied runtime gate invocation count. 获取复制出的 runtime gate 调用次数。</summary>
    public long InvocationCount { get; }

    /// <summary>Gets the copied notifyShape diagnostic count. 获取复制出的 notifyShape 诊断次数。</summary>
    public long NotifyShapeCount { get; }

    /// <summary>Gets the copied reallocateOutput diagnostic count. 获取复制出的 reallocateOutput 诊断次数。</summary>
    public long ReallocateOutputCount { get; }

    /// <summary>Gets the copied managed callback failure count. 获取复制出的托管 callback 失败次数。</summary>
    public long FailureCount { get; }

    /// <summary>Gets the copied in-flight callback count. 获取复制出的 in-flight callback 数量。</summary>
    public long InFlightCallbackCount { get; }

    /// <summary>Gets the copied maximum in-flight callback count. 获取复制出的最大 in-flight callback 数量。</summary>
    public long MaxInFlightCallbackCount { get; }

    /// <summary>Gets the copied active gate call count. 获取复制出的 active gate 调用数量。</summary>
    public int ActiveGateCallCount { get; }

    /// <summary>Gets the copied release hook count. 获取复制出的 release hook 次数。</summary>
    public long ReleaseHookCount { get; }

    /// <summary>Gets whether managed callback state remains pinned. 获取托管 callback state 是否仍被 pin 住。</summary>
    public bool CallbackStatePinned { get; }

    /// <summary>Gets whether the managed delegate remains pinned. 获取托管 delegate 是否仍被 pin 住。</summary>
    public bool DelegatePinned { get; }

    /// <summary>Gets whether dispose has been requested. 获取是否已请求释放。</summary>
    public bool DisposeRequested { get; }

    /// <summary>Gets whether this design owner is attached to a TensorRT execution context. 获取该设计 owner 是否已绑定到 TensorRT execution context。</summary>
    public bool IsAttached { get; }

    /// <summary>Gets the copied native owner state transition count. 获取复制出的 native owner 状态转移次数。</summary>
    public ulong StateTransitionCount { get; }

    /// <summary>Gets the copied ledger allocation intent count. 获取复制出的 ledger allocation intent 次数。</summary>
    public ulong LedgerAllocationCount { get; }

    /// <summary>Gets the copied ledger release intent count. 获取复制出的 ledger release intent 次数。</summary>
    public ulong LedgerReleaseCount { get; }

    /// <summary>Gets the copied ledger failure count. 获取复制出的 ledger 失败次数。</summary>
    public ulong LedgerFailureCount { get; }

    /// <summary>Gets the last synthetic allocation id. 获取最近一次合成 allocation id。</summary>
    public ulong LastAllocationId { get; }

    /// <summary>Gets the last synthetic release allocation id. 获取最近一次合成 release allocation id。</summary>
    public ulong LastReleaseAllocationId { get; }

    /// <summary>Gets the last copied synthetic stream value. 获取最近一次复制出的合成 stream 值。</summary>
    public ulong LastStreamValue { get; }

    /// <summary>Gets whether the synthetic ledger still has a live allocation. 获取合成 ledger 是否仍有 live allocation。</summary>
    public bool HasLiveAllocation { get; }

    /// <summary>Gets whether an output buffer pointer is exposed by this public API. 获取该 public API 是否暴露 output buffer pointer。</summary>
    public bool OutputBufferPointerExposed => false;

    /// <summary>Gets whether an output buffer pointer was produced by this diagnostic gate. 获取该诊断门禁是否产生 output buffer pointer。</summary>
    public bool OutputBufferPointerProduced => false;

    /// <summary>Gets the copied native ledger last operation. 获取复制出的 native ledger 最近操作。</summary>
    public string NativeLastOperation { get; }

    /// <summary>Gets the copied combined diagnostic message. 获取复制出的合并诊断消息。</summary>
    public string LastDiagnostic { get; }

    /// <summary>Gets the copied native ledger diagnostic message. 获取复制出的 native ledger 诊断消息。</summary>
    public string NativeLedgerDiagnostic { get; }

    /// <summary>Gets the copied release diagnostic message. 获取复制出的释放诊断消息。</summary>
    public string ReleaseDiagnostic { get; }

    /// <summary>Gets whether the diagnostic completed without gate or ledger failures. 获取诊断是否未出现 gate 或 ledger 失败。</summary>
    public bool Succeeded =>
        RuntimeGateStatus == BridgeStatusCode.Ok &&
        NativeLedgerStatus == BridgeStatusCode.Ok &&
        FailureCount == 0 &&
        LedgerFailureCount == 0 &&
        InFlightCallbackCount == 0;

    /// <summary>Returns a compact diagnostic representation. 返回紧凑的诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:owner={OwnerId}:operation={Operation}:status={LastStatus}:notify={NotifyShapeCount}:reallocate={ReallocateOutputCount}:ledger={LedgerAllocationCount}/{LedgerReleaseCount}:proof={IsRealCallbackRuntimeProof}";
    }

    private static string ComposeDiagnostic(string gateDiagnostic, string nativeLedgerDiagnostic, bool nativeLedgerAvailable)
    {
        string ledgerPrefix = nativeLedgerAvailable ? "native-ledger=" : "native-ledger-unavailable=";
        return "output-allocator-callback-owner-design; " +
            (gateDiagnostic ?? string.Empty) +
            "; " +
            ledgerPrefix +
            (nativeLedgerDiagnostic ?? string.Empty) +
            "; RealCallbackRuntime=False; IsRealCallbackRuntimeProof=False.";
    }
}

/// <summary>
/// High-level diagnostic owner for the future TensorRT output allocator callback bridge.
/// 未来 TensorRT output allocator callback bridge 的高层诊断 owner。
/// </summary>
/// <remarks>
/// This class is a design gate. It composes the managed output allocator runtime gate with the native allocator owner
/// ledger dry-run so callers can validate copied metadata, attach/detach intent, exception-to-status behavior, and
/// pointer-free public API shape. It does not call TensorRT <c>setOutputAllocator</c> and does not unlock the
/// <c>IOutputAllocator::notifyShape</c> or <c>IOutputAllocator::reallocateOutput</c> deferred rows.
/// 该类是设计门禁。它把托管 output allocator runtime gate 与 native allocator owner ledger dry-run 组合起来，使调用方可以
/// 验证复制元数据、attach/detach intent、exception-to-status 行为和不暴露 pointer 的 public API 形状。它不会调用 TensorRT
/// <c>setOutputAllocator</c>，也不会解除 <c>IOutputAllocator::notifyShape</c> 或
/// <c>IOutputAllocator::reallocateOutput</c> 的 deferred rows。
/// </remarks>
public sealed class TensorRtOutputAllocatorCallbackOwner : IDisposable
{
    private readonly object _gate = new object();
    private readonly TensorRtOutputAllocatorRuntimeGate _runtimeGate = new TensorRtOutputAllocatorRuntimeGate();
    private readonly TensorRtAllocatorCallbackOwner _nativeLedgerOwner;
    private TensorRtAllocatorOwnerStateDryRunResult? _lastNativeLedgerState;
    private BridgeStatusCode _lastNativeLedgerStatus = BridgeStatusCode.Ok;
    private string _lastNativeLedgerDiagnostic = string.Empty;
    private TensorRtApiLine _lastLine = TensorRtApiLine.TensorRt11;
    private bool _lastNativeLedgerAvailable;
    private bool _disposed;

    /// <summary>
    /// Creates an output allocator callback owner design gate.
    /// 创建 output allocator callback owner 设计门禁。
    /// </summary>
    public TensorRtOutputAllocatorCallbackOwner()
    {
        _nativeLedgerOwner = new TensorRtAllocatorCallbackOwner(static request =>
            TensorRtAllocatorDryRunResult.Success("output-allocator-callback-owner-design managed ledger keep-alive " + request));
    }

    /// <summary>Gets whether dispose has been requested. 获取是否已请求释放。</summary>
    public bool IsDisposed
    {
        get
        {
            lock (_gate)
            {
                return _disposed;
            }
        }
    }

    /// <summary>Gets whether this owner is attached to a TensorRT execution context. 获取该 owner 是否已绑定到 TensorRT execution context。</summary>
    public bool IsAttached => false;

    /// <summary>
    /// Runs the output allocator owner design diagnostic.
    /// 执行 output allocator owner 设计诊断。
    /// </summary>
    /// <param name="line">The TensorRT API line to use for native ledger intent diagnostics. 用于 native ledger intent 诊断的 TensorRT API line。</param>
    /// <param name="request">The copied output allocator diagnostic request. 复制出的 output allocator 诊断请求。</param>
    /// <param name="streamValue">A synthetic stream value copied into the native ledger diagnostic. 复制到 native ledger 诊断中的合成 stream 值。</param>
    /// <returns>A pointer-free copied diagnostic snapshot. 不含 pointer 的复制诊断快照。</returns>
    /// <remarks>
    /// The method emits synthetic notify/reallocate diagnostics and a native attach/allocation/release/detach ledger
    /// intent when the native bridge is available. Native bridge load failures are copied into the returned snapshot
    /// instead of escaping as proof of API absence.
    /// 该方法会发出合成的 notify/reallocate 诊断，并在 native bridge 可用时记录 native attach/allocation/release/detach
    /// ledger intent。native bridge 加载失败会被复制到返回快照中，不会被冒泡成 API 缺失证据。
    /// </remarks>
    public TensorRtOutputAllocatorCallbackOwnerSnapshot RunDesignDiagnostic(
        TensorRtApiLine line,
        TensorRtOutputAllocatorCallbackRequest request,
        ulong streamValue = 0UL)
    {
        ThrowIfDisposed();

        TensorRtOutputAllocatorRuntimeGateRequest gateRequest = new TensorRtOutputAllocatorRuntimeGateRequest(
            request.TensorName,
            request.RequestedSize,
            request.Alignment,
            request.CopyShapeDimensions(),
            request.Reason,
            request.HasCurrentMemory);

        _runtimeGate.RunInternalNotifyShapeRuntimeGate(gateRequest);
        TensorRtOutputAllocatorRuntimeGateResult gate = _runtimeGate.RunInternalReallocateOutputRuntimeGate(gateRequest);

        TensorRtAllocatorOwnerStateDryRunResult? nativeLedger = null;
        BridgeStatusCode nativeStatus = BridgeStatusCode.Ok;
        string nativeDiagnostic;
        bool nativeAvailable = false;
        try
        {
            nativeLedger = _nativeLedgerOwner.RunNativeStateLedgerDryRunDiagnostic(
                line,
                new TensorRtAllocatorDryRunRequest(
                    request.RequestedSize,
                    request.Alignment,
                    "output-allocator-callback-owner-design:" + request.TensorName),
                "IOutputAllocator",
                streamValue);
            nativeDiagnostic = nativeLedger.Value.Diagnostic;
            nativeStatus = nativeLedger.Value.LastStatus;
            nativeAvailable = true;
        }
        catch (Exception exception)
        {
            nativeStatus = BridgeStatusCode.RuntimeError;
            nativeDiagnostic = "output-allocator-callback-owner-design native ledger diagnostic unavailable: " +
                exception.GetType().Name +
                ": " +
                exception.Message;
        }

        lock (_gate)
        {
            _lastLine = line;
            _lastNativeLedgerState = nativeLedger;
            _lastNativeLedgerStatus = nativeStatus;
            _lastNativeLedgerDiagnostic = nativeDiagnostic;
            _lastNativeLedgerAvailable = nativeAvailable;
        }

        return new TensorRtOutputAllocatorCallbackOwnerSnapshot(
            line,
            gate,
            nativeLedger,
            nativeStatus,
            nativeDiagnostic,
            nativeAvailable);
    }

    /// <summary>
    /// Gets a copied snapshot of the current design gate state.
    /// 获取当前设计门禁状态的复制快照。
    /// </summary>
    /// <param name="operation">A copied operation label. 复制出的操作标签。</param>
    /// <returns>A pointer-free copied diagnostic snapshot. 不含 pointer 的复制诊断快照。</returns>
    public TensorRtOutputAllocatorCallbackOwnerSnapshot GetSnapshot(string operation = "snapshot")
    {
        TensorRtAllocatorOwnerStateDryRunResult? nativeLedger;
        BridgeStatusCode nativeStatus;
        string nativeDiagnostic;
        bool nativeAvailable;
        TensorRtApiLine line;
        lock (_gate)
        {
            nativeLedger = _lastNativeLedgerState;
            nativeStatus = _lastNativeLedgerStatus;
            nativeDiagnostic = _lastNativeLedgerDiagnostic;
            nativeAvailable = _lastNativeLedgerAvailable;
            line = _lastLine;
        }

        return new TensorRtOutputAllocatorCallbackOwnerSnapshot(
            line,
            _runtimeGate.GetInternalRuntimeGateSnapshot(operation),
            nativeLedger,
            nativeStatus,
            nativeDiagnostic,
            nativeAvailable);
    }

    /// <summary>
    /// Releases managed keep-alive handles owned by the design gate.
    /// 释放该设计门禁持有的托管 keep-alive 句柄。
    /// </summary>
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
        }

        _runtimeGate.Dispose();
        _nativeLedgerOwner.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(TensorRtOutputAllocatorCallbackOwner));
            }
        }
    }
}
