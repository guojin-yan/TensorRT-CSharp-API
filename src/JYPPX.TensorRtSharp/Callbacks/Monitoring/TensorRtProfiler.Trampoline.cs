using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtProfiler
{
    private static BridgeStatusCode InvokeManagedProfiler(IntPtr layerName, UIntPtr layerNameLength, float milliseconds, IntPtr userState)
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

            state.RecordInvocation();
            state.Handler(DecodeUtf8(layerName, layerNameLength), milliseconds);
            return BridgeStatusCode.Ok;
        }
        catch (Exception exception)
        {
            state?.RecordFailure(exception);
            return BridgeStatusCode.InvalidState;
        }
    }

    private static string DecodeUtf8(IntPtr value, UIntPtr length)
    {
        if (value == IntPtr.Zero || length == UIntPtr.Zero)
        {
            return string.Empty;
        }

        ulong byteLength = length.ToUInt64();
        if (byteLength > int.MaxValue)
        {
            throw new InvalidOperationException("TensorRT profiler layer name is too large for a managed string.");
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

        public CallbackState(TensorRtProfilerHandler handler)
        {
            Handler = handler;
        }

        public TensorRtProfilerHandler Handler { get; }

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
