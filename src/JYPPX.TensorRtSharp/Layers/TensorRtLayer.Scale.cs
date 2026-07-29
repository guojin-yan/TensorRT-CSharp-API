using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Layer wrapper.
/// 表示托管 TensorRT Tensor Rt Layer 包装器。
/// </summary>
public sealed partial class TensorRtLayer
{
    /// <summary>
    /// Gets the Scale Mode value.
    /// 获取 Scale Mode 值。
    /// </summary>
    public TensorRtScaleMode GetScaleMode()
    {
        return NativeBridgeApi.GetScaleMode(Line, _handle);
    }

    /// <summary>
    /// Sets the Scale Mode value.
    /// 设置 Scale Mode 值。
    /// </summary>
    public void SetScaleMode(TensorRtScaleMode mode)
    {
        NativeBridgeApi.SetScaleMode(Line, _handle, mode);
    }

    /// <summary>
    /// Gets the Scale Channel Axis value.
    /// 获取 Scale Channel Axis 值。
    /// </summary>
    public int GetScaleChannelAxis()
    {
        return NativeBridgeApi.GetScaleChannelAxis(Line, _handle);
    }

    /// <summary>
    /// Sets the Scale Channel Axis value.
    /// 设置 Scale Channel Axis 值。
    /// </summary>
    public void SetScaleChannelAxis(int channelAxis)
    {
        NativeBridgeApi.SetScaleChannelAxis(Line, _handle, channelAxis);
    }

}
