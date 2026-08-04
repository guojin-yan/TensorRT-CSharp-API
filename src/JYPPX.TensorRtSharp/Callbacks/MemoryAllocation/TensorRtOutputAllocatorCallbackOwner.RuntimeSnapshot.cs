using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtOutputAllocatorCallbackOwner
{
    /// <summary>
    /// Gets a pointer-free snapshot from the real native output allocator owner.
    /// 获取真实 native output allocator owner 的无指针快照。
    /// </summary>
    /// <returns>Copied callback, allocation, and lifecycle state. 复制后的 callback、分配及生命周期状态。</returns>
    public TensorRtOutputAllocatorRuntimeSnapshot GetRuntimeSnapshot()
    {
        if (_runtimeLine == null || _nativeHandle == null)
        {
            throw new InvalidOperationException("This owner was created for design diagnostics only.");
        }

        lock (_gate)
        {
            if (_resourcesReleased)
            {
                throw new ObjectDisposedException(nameof(TensorRtOutputAllocatorCallbackOwner));
            }
        }

        NativeTensorRtOutputAllocatorOwnerInfo info =
            NativeBridgeApi.GetOutputAllocatorOwnerInfo(_runtimeLine.Value, _nativeHandle);
        int shapeRank = Math.Max(0, Math.Min(MaxRuntimeShapeRank, info.LastShapeRank));
        long[] shape = new long[shapeRank];
        if (info.LastShape != null)
        {
            Array.Copy(info.LastShape, shape, Math.Min(shape.Length, info.LastShape.Length));
        }

        TensorRtOutputAllocatorCallbackKind kind = Enum.IsDefined(
            typeof(TensorRtOutputAllocatorCallbackKind),
            info.LastCallbackKind)
            ? (TensorRtOutputAllocatorCallbackKind)info.LastCallbackKind
            : TensorRtOutputAllocatorCallbackKind.Unknown;

        return new TensorRtOutputAllocatorRuntimeSnapshot(
            line: (TensorRtApiLine)info.Line,
            ownerId: info.OwnerId,
            invocationCount: info.InvocationCount,
            notifyShapeCount: info.NotifyShapeCount,
            reallocateOutputCount: info.ReallocateOutputCount,
            failureCount: info.FailureCount,
            inFlightCallbackCount: info.InFlightCallbackCount,
            maxInFlightCallbackCount: info.MaxInFlightCallbackCount,
            attachCount: info.AttachCount,
            detachCount: info.DetachCount,
            allocationCount: info.AllocationCount,
            reuseCount: info.ReuseCount,
            releaseCount: info.ReleaseCount,
            liveAllocationCount: info.LiveAllocationCount,
            liveAllocationBytes: info.LiveAllocationBytes,
            peakLiveAllocationBytes: info.PeakLiveAllocationBytes,
            lastRequestedSize: info.LastRequestedSize,
            lastAlignment: info.LastAlignment,
            lastStatus: (BridgeStatusCode)info.LastStatus,
            isAttached: info.IsAttached != 0,
            lastCallbackSucceeded: info.LastCallbackSucceeded != 0,
            lastAllocationSucceeded: info.LastAllocationSucceeded != 0,
            lastHadCurrentMemory: info.LastHadCurrentMemory != 0,
            lastHadStream: info.LastHadStream != 0,
            lastCallbackKind: kind,
            tensorName: BridgeInfoMapper.ReadFixedUtf8(info.LastTensorName),
            shapeDimensions: shape,
            diagnostic: BridgeInfoMapper.ReadFixedUtf8(info.LastDiagnostic));
    }
}
