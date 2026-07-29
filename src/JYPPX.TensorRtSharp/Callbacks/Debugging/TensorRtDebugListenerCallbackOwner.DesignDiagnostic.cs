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
    /// <summary>
    /// Runs the debug-listener owner design diagnostic.
    /// 执行 debug-listener owner 设计诊断。
    /// </summary>
    /// <param name="line">The TensorRT API line represented by this diagnostic. 该诊断代表的 TensorRT API line。</param>
    /// <param name="request">The copied debug tensor diagnostic request. 复制出的 debug tensor 诊断请求。</param>
    /// <returns>A pointer-free copied diagnostic snapshot. 不含 pointer 的复制诊断快照。</returns>
    public TensorRtDebugListenerCallbackOwnerSnapshot RunDesignDiagnostic(
        TensorRtApiLine line,
        TensorRtDebugListenerCallbackRequest request)
    {
        IntPtr callbackState;
        TensorRtDebugListenerDesignGateCallback callback;
        lock (_gate)
        {
            if (_disposeRequested || !_hasCallbackStateHandle)
            {
                throw new ObjectDisposedException(nameof(TensorRtDebugListenerCallbackOwner));
            }

            checked
            {
                _activeGateCallCount++;
            }

            callbackState = GCHandle.ToIntPtr(_callbackStateHandle);
            callback = _callback;
        }

        BridgeStatusCode status;
        try
        {
            long[] shape = request.CopyShapeDimensions();
            using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(request.TensorName);
            using Utf8Interop.Utf8StringScope reasonUtf8 = Utf8Interop.ToNativeString(request.Reason);
            status = callback(
                (int)line,
                tensorNameUtf8.Pointer,
                (int)request.DataType,
                (int)request.Location,
                request.IsInput ? 1 : 0,
                request.IsOutput ? 1 : 0,
                request.IsShapeTensor ? 1 : 0,
                request.IsExecutionTensor ? 1 : 0,
                request.ShapeRank,
                GetDimension(shape, 0),
                GetDimension(shape, 1),
                GetDimension(shape, 2),
                GetDimension(shape, 3),
                GetDimension(shape, 4),
                GetDimension(shape, 5),
                GetDimension(shape, 6),
                GetDimension(shape, 7),
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

        return CreateSnapshot(status, "process-debug-tensor", line);
    }

}
