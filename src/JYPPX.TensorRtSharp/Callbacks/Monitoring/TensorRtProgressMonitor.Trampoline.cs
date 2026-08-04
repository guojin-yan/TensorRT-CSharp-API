using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtProgressMonitor
{
    private static BridgeStatusCode InvokeManagedProgressMonitor(
        int eventKind,
        IntPtr phaseName,
        UIntPtr phaseNameLength,
        IntPtr parentPhase,
        UIntPtr parentPhaseLength,
        int step,
        int stepCount,
        out int shouldContinue,
        IntPtr userState)
    {
        shouldContinue = 1;
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

            state.RecordInvocation();
            bool continueBuild = state.Handler(new TensorRtProgressMonitorEvent(
                (TensorRtProgressMonitorEventKind)eventKind,
                DecodeUtf8(phaseName, phaseNameLength),
                DecodeUtf8Nullable(parentPhase, parentPhaseLength),
                step,
                stepCount));
            shouldContinue = continueBuild ? 1 : 0;
            return BridgeStatusCode.Ok;
        }
        catch (Exception exception)
        {
            shouldContinue = 1;
            state?.RecordFailure(exception);
            return BridgeStatusCode.InvalidState;
        }
    }

    private static string DecodeUtf8(IntPtr value, UIntPtr length)
    {
        return DecodeUtf8Nullable(value, length) ?? string.Empty;
    }

    private static string? DecodeUtf8Nullable(IntPtr value, UIntPtr length)
    {
        if (value == IntPtr.Zero || length == UIntPtr.Zero)
        {
            return null;
        }

        ulong byteLength = length.ToUInt64();
        if (byteLength > int.MaxValue)
        {
            throw new InvalidOperationException("TensorRT progress monitor text is too large for a managed string.");
        }

        byte[] buffer = new byte[checked((int)byteLength)];
        Marshal.Copy(value, buffer, 0, buffer.Length);
        return Encoding.UTF8.GetString(buffer, 0, buffer.Length);
    }

    private sealed class CallbackState
    {
        private long _invocationCount;
        private long _failureCount;
        private Exception? _lastException;

        public CallbackState(TensorRtProgressMonitorHandler handler)
        {
            Handler = handler;
        }

        public TensorRtProgressMonitorHandler Handler { get; }

        public long InvocationCount => Interlocked.Read(ref _invocationCount);

        public long FailureCount => Interlocked.Read(ref _failureCount);

        public Exception? LastException => Volatile.Read(ref _lastException);

        public void RecordInvocation()
        {
            Interlocked.Increment(ref _invocationCount);
        }

        public void RecordFailure(Exception exception)
        {
            Volatile.Write(ref _lastException, exception);
            Interlocked.Increment(ref _failureCount);
        }
    }
}
