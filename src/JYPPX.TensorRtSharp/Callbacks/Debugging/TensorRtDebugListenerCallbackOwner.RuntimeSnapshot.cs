using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtDebugListenerCallbackOwner
{
    /// <summary>
    /// Gets a pointer-free snapshot from the real native debug-listener owner.
    /// 获取真实 native debug-listener owner 的无指针快照。
    /// </summary>
    public TensorRtDebugListenerRuntimeSnapshot GetRuntimeSnapshot()
    {
        if (_runtimeLine == null || _nativeHandle == null)
        {
            throw new InvalidOperationException("This owner was created for design diagnostics only.");
        }

        lock (_gate)
        {
            if (_nativeHandleReleased)
            {
                throw new ObjectDisposedException(nameof(TensorRtDebugListenerCallbackOwner));
            }
        }

        NativeTensorRtDebugListenerOwnerInfo info =
            NativeBridgeApi.GetDebugListenerOwnerInfo(_runtimeLine.Value, _nativeHandle);
        int shapeRank = Math.Max(0, Math.Min(MaxShapeRank, info.LastShapeRank));
        long[] shape = new long[shapeRank];
        if (info.LastShape != null)
        {
            Array.Copy(info.LastShape, shape, Math.Min(shape.Length, info.LastShape.Length));
        }

        TensorRtDataType dataType = Enum.IsDefined(typeof(TensorRtDataType), info.LastDataType)
            ? (TensorRtDataType)info.LastDataType
            : TensorRtDataType.Unknown;
        TensorRtTensorLocation location = Enum.IsDefined(typeof(TensorRtTensorLocation), info.LastLocation)
            ? (TensorRtTensorLocation)info.LastLocation
            : TensorRtTensorLocation.Device;

        return new TensorRtDebugListenerRuntimeSnapshot(
            line: (TensorRtApiLine)info.Line,
            ownerId: info.OwnerId,
            invocationCount: info.InvocationCount,
            failureCount: info.FailureCount,
            inFlightCallbackCount: info.InFlightCallbackCount,
            maxInFlightCallbackCount: info.MaxInFlightCallbackCount,
            attachCount: info.AttachCount,
            detachCount: info.DetachCount,
            lastStatus: (BridgeStatusCode)info.LastStatus,
            isAttached: info.IsAttached != 0,
            lastCallbackSucceeded: info.LastCallbackSucceeded != 0,
            tensorName: BridgeInfoMapper.ReadFixedUtf8(info.LastTensorName),
            dataType: dataType,
            location: location,
            shapeDimensions: shape,
            diagnostic: BridgeInfoMapper.ReadFixedUtf8(info.LastDiagnostic));
    }
}
