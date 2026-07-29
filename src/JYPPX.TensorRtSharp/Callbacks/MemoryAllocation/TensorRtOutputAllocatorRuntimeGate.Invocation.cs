using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

internal sealed partial class TensorRtOutputAllocatorRuntimeGate
{
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

}
