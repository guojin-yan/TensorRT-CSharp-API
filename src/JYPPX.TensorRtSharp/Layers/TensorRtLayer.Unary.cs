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
    /// Gets the Unary Operation value.
    /// 获取 Unary Operation 值。
    /// </summary>
    public TensorRtUnaryOperation GetUnaryOperation()
    {
        return NativeBridgeApi.GetUnaryOperation(Line, _handle);
    }

    /// <summary>
    /// Sets the Unary Operation value.
    /// 设置 Unary Operation 值。
    /// </summary>
    public void SetUnaryOperation(TensorRtUnaryOperation operation)
    {
        NativeBridgeApi.SetUnaryOperation(Line, _handle, operation);
    }

}
