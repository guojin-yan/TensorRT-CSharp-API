using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

internal readonly struct TensorRtOutputAllocatorRuntimeGateRequest
{
    internal TensorRtOutputAllocatorRuntimeGateRequest(
        string tensorName,
        ulong requestedSize,
        ulong alignment,
        long[] shapeDimensions,
        string reason = "",
        bool hasCurrentMemory = false)
    {
        TensorName = tensorName ?? string.Empty;
        RequestedSize = requestedSize;
        Alignment = alignment;
        ShapeDimensions = shapeDimensions == null ? Array.Empty<long>() : (long[])shapeDimensions.Clone();
        Reason = reason ?? string.Empty;
        HasCurrentMemory = hasCurrentMemory;
    }

    public string TensorName { get; }

    public ulong RequestedSize { get; }

    public ulong Alignment { get; }

    public int ShapeRank => ShapeDimensions.Length;

    public string Reason { get; }

    public bool HasCurrentMemory { get; }

    internal long[] ShapeDimensions { get; }

    internal long GetDimension(int index)
    {
        return index >= 0 && index < ShapeDimensions.Length ? ShapeDimensions[index] : 0L;
    }
}

internal sealed class TensorRtOutputAllocatorRuntimeGate : IDisposable
{
    private const int MaxShapeRank = 8;
    private const int NotifyShapeOperation = 1;
    private const int ReallocateOutputOperation = 2;
    private static long s_nextOwnerId;

    private readonly object _gate = new object();
    private readonly long _ownerId;
    private readonly CallbackState _callbackState = new CallbackState();
    private readonly TensorRtOutputAllocatorInternalRuntimeGateCallback _runtimeGateCallback;
    private GCHandle _callbackStateHandle;
    private GCHandle _runtimeGateCallbackHandle;
    private bool _hasCallbackStateHandle;
    private bool _hasRuntimeGateCallbackHandle;
    private bool _disposeRequested;
    private int _activeGateCallCount;

    internal TensorRtOutputAllocatorRuntimeGate()
    {
        _ownerId = Interlocked.Increment(ref s_nextOwnerId);
        _runtimeGateCallback = InvokeOutputAllocatorRuntimeGate;
        _callbackStateHandle = GCHandle.Alloc(_callbackState);
        _runtimeGateCallbackHandle = GCHandle.Alloc(_runtimeGateCallback);
        _hasCallbackStateHandle = true;
        _hasRuntimeGateCallbackHandle = true;
    }

    internal TensorRtOutputAllocatorRuntimeGateResult RunInternalNotifyShapeRuntimeGate(TensorRtOutputAllocatorRuntimeGateRequest request)
    {
        return RunInternalRuntimeGate(NotifyShapeOperation, request);
    }

    internal TensorRtOutputAllocatorRuntimeGateResult RunInternalReallocateOutputRuntimeGate(TensorRtOutputAllocatorRuntimeGateRequest request)
    {
        return RunInternalRuntimeGate(ReallocateOutputOperation, request);
    }

    internal TensorRtOutputAllocatorRuntimeGateResult GetInternalRuntimeGateSnapshot(string operation = "snapshot")
    {
        return CreateResult(_callbackState.LastStatus, operation);
    }

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
            releaseNow = _activeGateCallCount == 0;
        }

        if (releaseNow)
        {
            FreeCallbackState();
        }

        GC.SuppressFinalize(this);
    }

    private TensorRtOutputAllocatorRuntimeGateResult RunInternalRuntimeGate(int operation, TensorRtOutputAllocatorRuntimeGateRequest request)
    {
        IntPtr callbackState;
        TensorRtOutputAllocatorInternalRuntimeGateCallback callback;
        lock (_gate)
        {
            if (_disposeRequested || !_hasCallbackStateHandle)
            {
                throw new ObjectDisposedException(nameof(TensorRtOutputAllocatorRuntimeGate));
            }

            checked
            {
                _activeGateCallCount++;
            }

            callbackState = GCHandle.ToIntPtr(_callbackStateHandle);
            callback = _runtimeGateCallback;
        }

        BridgeStatusCode status;
        try
        {
            using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(request.TensorName);
            using Utf8Interop.Utf8StringScope reasonUtf8 = Utf8Interop.ToNativeString(request.Reason);
            status = callback(
                operation,
                tensorNameUtf8.Pointer,
                request.RequestedSize,
                request.Alignment,
                request.HasCurrentMemory ? 1 : 0,
                request.ShapeRank,
                request.GetDimension(0),
                request.GetDimension(1),
                request.GetDimension(2),
                request.GetDimension(3),
                request.GetDimension(4),
                request.GetDimension(5),
                request.GetDimension(6),
                request.GetDimension(7),
                reasonUtf8.Pointer,
                callbackState);
        }
        finally
        {
            bool releaseNow;
            lock (_gate)
            {
                _activeGateCallCount--;
                releaseNow = _activeGateCallCount == 0 && _disposeRequested;
            }

            if (releaseNow)
            {
                FreeCallbackState();
            }
        }

        return CreateResult(status, OperationName(operation));
    }

    private void FreeCallbackState()
    {
        bool released = false;
        if (_hasRuntimeGateCallbackHandle)
        {
            _runtimeGateCallbackHandle.Free();
            _hasRuntimeGateCallbackHandle = false;
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
            _callbackState.RecordReleaseHook("output-allocator-internal-runtime-gate release hook released managed GCHandle and delegate keep-alive handles after callbacks drained.");
            GC.KeepAlive(_runtimeGateCallback);
        }
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

    private static BridgeStatusCode InvokeOutputAllocatorRuntimeGate(
        int operation,
        IntPtr tensorName,
        ulong requestedSize,
        ulong alignment,
        int hasCurrentMemory,
        int shapeRank,
        long dim0,
        long dim1,
        long dim2,
        long dim3,
        long dim4,
        long dim5,
        long dim6,
        long dim7,
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
            state.RecordInvocation(operation);

            string tensor = Utf8Interop.ReadString(tensorName);
            string gateReason = Utf8Interop.ReadString(reason);
            string operationName = OperationName(operation);
            string shapeSummary = FormatShape(shapeRank, dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7);
            state.RecordRequest(tensor, requestedSize, alignment, shapeRank, shapeSummary, hasCurrentMemory != 0);

            if (string.Equals(gateReason, "throw", StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException("synthetic output allocator runtime gate failure");
            }

            if (operation != NotifyShapeOperation && operation != ReallocateOutputOperation)
            {
                string diagnostic = "output-allocator-internal-runtime-gate invalid operation; no TensorRT output allocator callback was invoked.";
                state.RecordReturnedFailure(diagnostic, BridgeStatusCode.InvalidArgument);
                return BridgeStatusCode.InvalidArgument;
            }

            if (string.IsNullOrWhiteSpace(tensor))
            {
                string diagnostic = "output-allocator-internal-runtime-gate invalid tensor name; no TensorRT output allocator callback was invoked.";
                state.RecordReturnedFailure(diagnostic, BridgeStatusCode.InvalidArgument);
                return BridgeStatusCode.InvalidArgument;
            }

            if (shapeRank < 0 || shapeRank > MaxShapeRank)
            {
                string diagnostic = "output-allocator-internal-runtime-gate invalid shape rank; no TensorRT output allocator callback was invoked.";
                state.RecordReturnedFailure(diagnostic, BridgeStatusCode.InvalidArgument);
                return BridgeStatusCode.InvalidArgument;
            }

            if (operation == ReallocateOutputOperation && alignment == 0UL)
            {
                string diagnostic = "output-allocator-internal-runtime-gate invalid reallocateOutput alignment; no TensorRT output allocator callback was invoked.";
                state.RecordReturnedFailure(diagnostic, BridgeStatusCode.InvalidArgument);
                return BridgeStatusCode.InvalidArgument;
            }

            string successDiagnostic =
                "output-allocator-internal-runtime-gate " + operationName +
                " copied tensor=" + tensor +
                " size=" + requestedSize.ToString(CultureInfo.InvariantCulture) +
                " alignment=" + alignment.ToString(CultureInfo.InvariantCulture) +
                " shape=" + shapeSummary +
                " output-buffer-pointer-exposed=false; no TensorRT output allocator callback was invoked.";
            state.RecordStatus(BridgeStatusCode.Ok, successDiagnostic);
            return BridgeStatusCode.Ok;
        }
        catch (Exception exception)
        {
            string diagnostic = "output allocator internal runtime gate handler threw " + exception.GetType().Name + ": " + exception.Message;
            state?.RecordFailure(exception, diagnostic, BridgeStatusCode.InvalidState);
            return BridgeStatusCode.InvalidState;
        }
        finally
        {
            state?.ExitCallback();
        }
    }

    private static string OperationName(int operation)
    {
        return operation switch
        {
            NotifyShapeOperation => "notify-shape",
            ReallocateOutputOperation => "reallocate-output",
            _ => "unknown"
        };
    }

    private static string FormatShape(
        int rank,
        long dim0,
        long dim1,
        long dim2,
        long dim3,
        long dim4,
        long dim5,
        long dim6,
        long dim7)
    {
        if (rank <= 0)
        {
            return "[]";
        }

        long[] dimensions = new[] { dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7 };
        int copiedRank = Math.Min(rank, MaxShapeRank);
        string[] values = new string[copiedRank];
        for (int index = 0; index < copiedRank; index++)
        {
            values[index] = dimensions[index].ToString(CultureInfo.InvariantCulture);
        }

        return "[" + string.Join("x", values) + "]";
    }

    private sealed class CallbackState
    {
        private long _invocationCount;
        private long _notifyShapeCount;
        private long _reallocateOutputCount;
        private long _failureCount;
        private long _inFlightCallbackCount;
        private long _maxInFlightCallbackCount;
        private long _releaseHookCount;
        private int _lastStatus;
        private int _lastShapeRank;
        private int _lastHasCurrentMemory;
        private ulong _lastRequestedSize;
        private ulong _lastAlignment;
        private Exception? _lastException;
        private string _lastTensorName = string.Empty;
        private string _lastShapeSummary = "[]";
        private string _lastDiagnostic = string.Empty;
        private string _lastReleaseDiagnostic = string.Empty;

        public long InvocationCount => Interlocked.Read(ref _invocationCount);

        public long NotifyShapeCount => Interlocked.Read(ref _notifyShapeCount);

        public long ReallocateOutputCount => Interlocked.Read(ref _reallocateOutputCount);

        public long FailureCount => Interlocked.Read(ref _failureCount);

        public long InFlightCallbackCount => Interlocked.Read(ref _inFlightCallbackCount);

        public long MaxInFlightCallbackCount => Interlocked.Read(ref _maxInFlightCallbackCount);

        public long ReleaseHookCount => Interlocked.Read(ref _releaseHookCount);

        public BridgeStatusCode LastStatus => (BridgeStatusCode)Volatile.Read(ref _lastStatus);

        public Exception? LastException => Volatile.Read(ref _lastException);

        public string LastTensorName => Volatile.Read(ref _lastTensorName);

        public ulong LastRequestedSize => Volatile.Read(ref _lastRequestedSize);

        public ulong LastAlignment => Volatile.Read(ref _lastAlignment);

        public int LastShapeRank => Volatile.Read(ref _lastShapeRank);

        public string LastShapeSummary => Volatile.Read(ref _lastShapeSummary);

        public bool LastHasCurrentMemory => Volatile.Read(ref _lastHasCurrentMemory) != 0;

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

        public void RecordInvocation(int operation)
        {
            Interlocked.Increment(ref _invocationCount);
            if (operation == NotifyShapeOperation)
            {
                Interlocked.Increment(ref _notifyShapeCount);
            }
            else if (operation == ReallocateOutputOperation)
            {
                Interlocked.Increment(ref _reallocateOutputCount);
            }
        }

        public void RecordRequest(string tensorName, ulong requestedSize, ulong alignment, int shapeRank, string shapeSummary, bool hasCurrentMemory)
        {
            Volatile.Write(ref _lastTensorName, tensorName ?? string.Empty);
            Volatile.Write(ref _lastRequestedSize, requestedSize);
            Volatile.Write(ref _lastAlignment, alignment);
            Volatile.Write(ref _lastShapeRank, shapeRank);
            Volatile.Write(ref _lastShapeSummary, shapeSummary ?? "[]");
            Volatile.Write(ref _lastHasCurrentMemory, hasCurrentMemory ? 1 : 0);
        }

        public void RecordStatus(BridgeStatusCode status, string diagnostic)
        {
            Volatile.Write(ref _lastStatus, (int)status);
            Volatile.Write(ref _lastDiagnostic, diagnostic ?? string.Empty);
        }

        public void RecordReturnedFailure(string diagnostic, BridgeStatusCode status)
        {
            RecordStatus(status, diagnostic);
            Interlocked.Increment(ref _failureCount);
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
    private delegate BridgeStatusCode TensorRtOutputAllocatorInternalRuntimeGateCallback(
        int operation,
        IntPtr tensorName,
        ulong requestedSize,
        ulong alignment,
        int hasCurrentMemory,
        int shapeRank,
        long dim0,
        long dim1,
        long dim2,
        long dim3,
        long dim4,
        long dim5,
        long dim6,
        long dim7,
        IntPtr reason,
        IntPtr userState);
}

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
