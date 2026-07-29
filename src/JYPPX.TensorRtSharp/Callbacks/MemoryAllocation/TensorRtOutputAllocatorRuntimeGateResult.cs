using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

internal readonly struct TensorRtOutputAllocatorRuntimeGateResult
{
    internal TensorRtOutputAllocatorRuntimeGateResult(
        long ownerId,
        string operation,
        string tensorName,
        ulong requestedSize,
        ulong alignment,
        int shapeRank,
        string shapeSummary,
        bool hasCurrentMemory,
        BridgeStatusCode lastStatus,
        long invocationCount,
        long notifyShapeCount,
        long reallocateOutputCount,
        long failureCount,
        long inFlightCallbackCount,
        long maxInFlightCallbackCount,
        int activeGateCallCount,
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
        TensorName = tensorName ?? string.Empty;
        RequestedSize = requestedSize;
        Alignment = alignment;
        ShapeRank = shapeRank;
        ShapeSummary = shapeSummary ?? "[]";
        HasCurrentMemory = hasCurrentMemory;
        LastStatus = lastStatus;
        InvocationCount = invocationCount;
        NotifyShapeCount = notifyShapeCount;
        ReallocateOutputCount = reallocateOutputCount;
        FailureCount = failureCount;
        InFlightCallbackCount = inFlightCallbackCount;
        MaxInFlightCallbackCount = maxInFlightCallbackCount;
        ActiveGateCallCount = activeGateCallCount;
        ReleaseHookCount = releaseHookCount;
        CallbackStatePinned = callbackStatePinned;
        DelegatePinned = delegatePinned;
        DisposeRequested = disposeRequested;
        IsAttached = isAttached;
        LastDiagnostic = lastDiagnostic ?? string.Empty;
        ReleaseDiagnostic = releaseDiagnostic ?? string.Empty;
    }

    public bool RealCallbackRuntime => false;

    public string EvidenceKind => "output-allocator-internal-runtime-gate";

    public string CallbackKind => "output-allocator-prototype";

    public long OwnerId { get; }

    public string Operation { get; }

    public string TensorName { get; }

    public ulong RequestedSize { get; }

    public ulong Alignment { get; }

    public int ShapeRank { get; }

    public string ShapeSummary { get; }

    public bool HasCurrentMemory { get; }

    public BridgeStatusCode LastStatus { get; }

    public long InvocationCount { get; }

    public long NotifyShapeCount { get; }

    public long ReallocateOutputCount { get; }

    public long FailureCount { get; }

    public long InFlightCallbackCount { get; }

    public long MaxInFlightCallbackCount { get; }

    public int ActiveGateCallCount { get; }

    public long ReleaseHookCount { get; }

    public bool CallbackStatePinned { get; }

    public bool DelegatePinned { get; }

    public bool DisposeRequested { get; }

    public bool IsAttached { get; }

    public bool OutputBufferPointerExposed => false;

    public bool OutputBufferPointerProduced => false;

    public string LastDiagnostic { get; }

    public string ReleaseDiagnostic { get; }

    public bool Succeeded => LastStatus == BridgeStatusCode.Ok && FailureCount == 0 && InFlightCallbackCount == 0;

    public override string ToString()
    {
        return $"{EvidenceKind}:owner={OwnerId}:operation={Operation}:status={LastStatus}:invocations={InvocationCount}:notify={NotifyShapeCount}:reallocate={ReallocateOutputCount}:failures={FailureCount}:realRuntime={RealCallbackRuntime}";
    }
}
