using System;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtOutputAllocatorCallbackOwner
{
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

}
