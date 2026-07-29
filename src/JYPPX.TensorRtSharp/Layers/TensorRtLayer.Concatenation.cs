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
    /// Sets the Concatenation Axis value.
    /// 设置 Concatenation Axis 值。
    /// </summary>
    public void SetConcatenationAxis(int axis)
    {
        NativeBridgeApi.SetConcatenationAxis(Line, _handle, axis);
    }

    /// <summary>
    /// Gets the Concatenation Axis value.
    /// 获取 Concatenation Axis 值。
    /// </summary>
    public int GetConcatenationAxis()
    {
        return NativeBridgeApi.GetConcatenationAxis(Line, _handle);
    }

}
