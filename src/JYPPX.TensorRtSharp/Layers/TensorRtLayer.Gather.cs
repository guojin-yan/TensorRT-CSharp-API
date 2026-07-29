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
    /// Gets the Gather Axis value.
    /// 获取 Gather Axis 值。
    /// </summary>
    public int GetGatherAxis()
    {
        return NativeBridgeApi.GetGatherAxis(Line, _handle);
    }

    /// <summary>
    /// Sets the Gather Axis value.
    /// 设置 Gather Axis 值。
    /// </summary>
    public void SetGatherAxis(int axis)
    {
        NativeBridgeApi.SetGatherAxis(Line, _handle, axis);
    }

}
