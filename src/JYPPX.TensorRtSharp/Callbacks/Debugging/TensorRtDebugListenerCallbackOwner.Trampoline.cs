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
    private static BridgeStatusCode InvokeDebugListenerDesignGate(
        int line,
        IntPtr tensorName,
        int dataType,
        int location,
        int isInput,
        int isOutput,
        int isShapeTensor,
        int isExecutionTensor,
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
            state.RecordInvocation();

            TensorRtApiLine apiLine = Enum.IsDefined(typeof(TensorRtApiLine), line)
                ? (TensorRtApiLine)line
                : TensorRtApiLine.TensorRt11;
            string tensor = Utf8Interop.ReadString(tensorName);
            string gateReason = Utf8Interop.ReadString(reason);
            TensorRtDataType copiedDataType = Enum.IsDefined(typeof(TensorRtDataType), dataType)
                ? (TensorRtDataType)dataType
                : TensorRtDataType.Unknown;
            TensorRtTensorLocation copiedLocation = Enum.IsDefined(typeof(TensorRtTensorLocation), location)
                ? (TensorRtTensorLocation)location
                : TensorRtTensorLocation.Device;
            string shapeSummary = FormatShape(shapeRank, dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7);
            state.RecordRequest(
                apiLine,
                tensor,
                copiedDataType,
                copiedLocation,
                shapeRank,
                shapeSummary,
                isInput != 0,
                isOutput != 0,
                isShapeTensor != 0,
                isExecutionTensor != 0);

            if (string.Equals(gateReason, "throw", StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException("synthetic debug listener callback owner design failure");
            }

            if (string.IsNullOrWhiteSpace(tensor))
            {
                const string diagnostic = "debug-listener-callback-owner-design invalid tensor name; no TensorRT debug listener callback was invoked.";
                state.RecordReturnedFailure(diagnostic, BridgeStatusCode.InvalidArgument);
                return BridgeStatusCode.InvalidArgument;
            }

            if (shapeRank < 0 || shapeRank > MaxShapeRank)
            {
                const string diagnostic = "debug-listener-callback-owner-design invalid shape rank; no TensorRT debug listener callback was invoked.";
                state.RecordReturnedFailure(diagnostic, BridgeStatusCode.InvalidArgument);
                return BridgeStatusCode.InvalidArgument;
            }

            string successDiagnostic =
                "debug-listener-callback-owner-design process-debug-tensor copied tensor=" + tensor +
                " dataType=" + copiedDataType +
                " location=" + copiedLocation +
                " shape=" + shapeSummary +
                " input=" + (isInput != 0).ToString(CultureInfo.InvariantCulture) +
                " output=" + (isOutput != 0).ToString(CultureInfo.InvariantCulture) +
                " shapeTensor=" + (isShapeTensor != 0).ToString(CultureInfo.InvariantCulture) +
                " executionTensor=" + (isExecutionTensor != 0).ToString(CultureInfo.InvariantCulture) +
                " debug-tensor-pointer-exposed=false; no TensorRT debug listener callback was invoked.";
            state.RecordStatus(BridgeStatusCode.Ok, successDiagnostic);
            return BridgeStatusCode.Ok;
        }
        catch (Exception exception)
        {
            string diagnostic = "debug listener callback owner design handler threw " + exception.GetType().Name + ": " + exception.Message;
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
        private long _processDebugTensorCount;
        private long _failureCount;
        private long _inFlightCallbackCount;
        private long _maxInFlightCallbackCount;
        private long _releaseHookCount;
        private int _lastStatus;
        private int _lastLine = (int)TensorRtApiLine.TensorRt11;
        private int _lastDataType = (int)TensorRtDataType.Unknown;
        private int _lastLocation = (int)TensorRtTensorLocation.Device;
        private int _lastShapeRank;
        private int _lastIsInput;
        private int _lastIsOutput;
        private int _lastIsShapeTensor;
        private int _lastIsExecutionTensor;
        private Exception? _lastException;
        private string _lastTensorName = string.Empty;
        private string _lastShapeSummary = "[]";
        private string _lastDiagnostic = string.Empty;
        private string _lastReleaseDiagnostic = string.Empty;

        public CallbackState(TensorRtDebugListenerHandler? handler)
        {
            Handler = handler;
        }

        public TensorRtDebugListenerHandler? Handler { get; }

        public long InvocationCount => Interlocked.Read(ref _invocationCount);

        public long ProcessDebugTensorCount => Interlocked.Read(ref _processDebugTensorCount);

        public long FailureCount => Interlocked.Read(ref _failureCount);

        public long InFlightCallbackCount => Interlocked.Read(ref _inFlightCallbackCount);

        public long MaxInFlightCallbackCount => Interlocked.Read(ref _maxInFlightCallbackCount);

        public long ReleaseHookCount => Interlocked.Read(ref _releaseHookCount);

        public BridgeStatusCode LastStatus => (BridgeStatusCode)Volatile.Read(ref _lastStatus);

        public TensorRtApiLine LastLine => (TensorRtApiLine)Volatile.Read(ref _lastLine);

        public TensorRtDataType LastDataType => (TensorRtDataType)Volatile.Read(ref _lastDataType);

        public TensorRtTensorLocation LastLocation => (TensorRtTensorLocation)Volatile.Read(ref _lastLocation);

        public int LastShapeRank => Volatile.Read(ref _lastShapeRank);

        public bool LastIsInput => Volatile.Read(ref _lastIsInput) != 0;

        public bool LastIsOutput => Volatile.Read(ref _lastIsOutput) != 0;

        public bool LastIsShapeTensor => Volatile.Read(ref _lastIsShapeTensor) != 0;

        public bool LastIsExecutionTensor => Volatile.Read(ref _lastIsExecutionTensor) != 0;

        public Exception? LastException => Volatile.Read(ref _lastException);

        public string LastTensorName => Volatile.Read(ref _lastTensorName);

        public string LastShapeSummary => Volatile.Read(ref _lastShapeSummary);

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
            Interlocked.Increment(ref _processDebugTensorCount);
        }

        public void RecordRequest(
            TensorRtApiLine line,
            string tensorName,
            TensorRtDataType dataType,
            TensorRtTensorLocation location,
            int shapeRank,
            string shapeSummary,
            bool isInput,
            bool isOutput,
            bool isShapeTensor,
            bool isExecutionTensor)
        {
            Volatile.Write(ref _lastLine, (int)line);
            Volatile.Write(ref _lastTensorName, tensorName ?? string.Empty);
            Volatile.Write(ref _lastDataType, (int)dataType);
            Volatile.Write(ref _lastLocation, (int)location);
            Volatile.Write(ref _lastShapeRank, shapeRank);
            Volatile.Write(ref _lastShapeSummary, shapeSummary ?? "[]");
            Volatile.Write(ref _lastIsInput, isInput ? 1 : 0);
            Volatile.Write(ref _lastIsOutput, isOutput ? 1 : 0);
            Volatile.Write(ref _lastIsShapeTensor, isShapeTensor ? 1 : 0);
            Volatile.Write(ref _lastIsExecutionTensor, isExecutionTensor ? 1 : 0);
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
    private delegate BridgeStatusCode TensorRtDebugListenerDesignGateCallback(
        int line,
        IntPtr tensorName,
        int dataType,
        int location,
        int isInput,
        int isOutput,
        int isShapeTensor,
        int isExecutionTensor,
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
