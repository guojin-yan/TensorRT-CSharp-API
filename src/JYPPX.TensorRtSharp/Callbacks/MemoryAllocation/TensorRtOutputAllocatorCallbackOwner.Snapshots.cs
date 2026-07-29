using System;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtOutputAllocatorCallbackOwner
{
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

}
