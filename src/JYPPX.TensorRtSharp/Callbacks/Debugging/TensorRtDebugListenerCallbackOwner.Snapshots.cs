using System;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtDebugListenerCallbackOwner
{
    /// <summary>
    /// Gets a copied snapshot of the current design gate state.
    /// 获取当前设计门禁状态的复制快照。
    /// </summary>
    /// <param name="operation">A copied operation label. 复制出的操作标签。</param>
    /// <returns>A pointer-free copied diagnostic snapshot. 不含 pointer 的复制诊断快照。</returns>
    public TensorRtDebugListenerCallbackOwnerSnapshot GetSnapshot(string operation = "snapshot")
    {
        return CreateSnapshot(_callbackState.LastStatus, operation, _callbackState.LastLine);
    }

    private TensorRtDebugListenerCallbackOwnerSnapshot CreateSnapshot(BridgeStatusCode status, string operation, TensorRtApiLine line)
    {
        bool callbackStatePinned;
        bool delegatePinned;
        bool disposeRequested;
        int activeGateCallCount;
        lock (_gate)
        {
            callbackStatePinned = _hasCallbackStateHandle;
            delegatePinned = _hasCallbackHandle;
            disposeRequested = _disposeRequested;
            activeGateCallCount = _activeGateCallCount;
        }

        BridgeStatusCode lastStatus = _callbackState.LastStatus;
        if (lastStatus != status)
        {
            lastStatus = status;
        }

        return new TensorRtDebugListenerCallbackOwnerSnapshot(
            ownerId: _ownerId,
            operation: operation,
            line: line,
            lastStatus: lastStatus,
            tensorName: _callbackState.LastTensorName,
            dataType: _callbackState.LastDataType,
            location: _callbackState.LastLocation,
            shapeRank: _callbackState.LastShapeRank,
            shapeSummary: _callbackState.LastShapeSummary,
            isInput: _callbackState.LastIsInput,
            isOutput: _callbackState.LastIsOutput,
            isShapeTensor: _callbackState.LastIsShapeTensor,
            isExecutionTensor: _callbackState.LastIsExecutionTensor,
            invocationCount: _callbackState.InvocationCount,
            processDebugTensorCount: _callbackState.ProcessDebugTensorCount,
            failureCount: _callbackState.FailureCount,
            inFlightCallbackCount: _callbackState.InFlightCallbackCount,
            maxInFlightCallbackCount: _callbackState.MaxInFlightCallbackCount,
            activeGateCallCount: activeGateCallCount,
            releaseHookCount: _callbackState.ReleaseHookCount,
            callbackStatePinned: callbackStatePinned,
            delegatePinned: delegatePinned,
            disposeRequested: disposeRequested,
            lastDiagnostic: _callbackState.LastDiagnostic,
            releaseDiagnostic: _callbackState.LastReleaseDiagnostic);
    }

}
