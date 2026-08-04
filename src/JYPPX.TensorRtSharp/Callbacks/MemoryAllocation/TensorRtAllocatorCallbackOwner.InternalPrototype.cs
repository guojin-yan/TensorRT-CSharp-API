using System;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtAllocatorCallbackOwner
{
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
