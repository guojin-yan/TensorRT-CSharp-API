using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

internal sealed partial class TensorRtOutputAllocatorRuntimeGate
{
    internal TensorRtOutputAllocatorRuntimeGateResult GetInternalRuntimeGateSnapshot(string operation = "snapshot")
    {
        return CreateResult(_callbackState.LastStatus, operation);
    }

    private TensorRtOutputAllocatorRuntimeGateResult CreateResult(BridgeStatusCode status, string operation)
    {
        bool callbackStatePinned;
        bool delegatePinned;
        bool disposeRequested;
        int activeGateCallCount;
        lock (_gate)
        {
            callbackStatePinned = _hasCallbackStateHandle;
            delegatePinned = _hasRuntimeGateCallbackHandle;
            disposeRequested = _disposeRequested;
            activeGateCallCount = _activeGateCallCount;
        }

        BridgeStatusCode lastStatus = _callbackState.LastStatus;
        if (lastStatus != status)
        {
            lastStatus = status;
        }

        return new TensorRtOutputAllocatorRuntimeGateResult(
            ownerId: _ownerId,
            operation: operation,
            tensorName: _callbackState.LastTensorName,
            requestedSize: _callbackState.LastRequestedSize,
            alignment: _callbackState.LastAlignment,
            shapeRank: _callbackState.LastShapeRank,
            shapeSummary: _callbackState.LastShapeSummary,
            hasCurrentMemory: _callbackState.LastHasCurrentMemory,
            lastStatus: lastStatus,
            invocationCount: _callbackState.InvocationCount,
            notifyShapeCount: _callbackState.NotifyShapeCount,
            reallocateOutputCount: _callbackState.ReallocateOutputCount,
            failureCount: _callbackState.FailureCount,
            inFlightCallbackCount: _callbackState.InFlightCallbackCount,
            maxInFlightCallbackCount: _callbackState.MaxInFlightCallbackCount,
            activeGateCallCount: activeGateCallCount,
            releaseHookCount: _callbackState.ReleaseHookCount,
            callbackStatePinned: callbackStatePinned,
            delegatePinned: delegatePinned,
            disposeRequested: disposeRequested,
            isAttached: false,
            lastDiagnostic: _callbackState.LastDiagnostic,
            releaseDiagnostic: _callbackState.LastReleaseDiagnostic);
    }

}
