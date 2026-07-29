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
    /// Sets the Soft Max Axes value.
    /// 设置 Soft Max Axes 值。
    /// </summary>
    public void SetSoftMaxAxes(uint axes)
    {
        if (axes == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axes), "SoftMax axes bitmask must not be zero.");
        }

        NativeBridgeApi.SetSoftMaxAxes(Line, _handle, axes);
    }

    /// <summary>
    /// Gets the Soft Max Axes value.
    /// 获取 Soft Max Axes 值。
    /// </summary>
    public uint GetSoftMaxAxes()
    {
        return NativeBridgeApi.GetSoftMaxAxes(Line, _handle);
    }

}
