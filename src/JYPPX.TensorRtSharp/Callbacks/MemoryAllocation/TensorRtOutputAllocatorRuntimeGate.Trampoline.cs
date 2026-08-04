using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

internal sealed partial class TensorRtOutputAllocatorRuntimeGate
{
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
