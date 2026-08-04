using System;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

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
