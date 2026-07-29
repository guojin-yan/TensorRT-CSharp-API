using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Network Definition wrapper.
/// 表示托管 TensorRT Tensor Rt Network Definition 包装器。
/// </summary>
public sealed partial class TensorRtNetworkDefinition
{
    /// <summary>
    /// Adds a Element Wise layer or object.
    /// 添加 Element Wise 层或对象。
    /// </summary>
    public TensorRtLayer AddElementWise(TensorRtTensor left, TensorRtTensor right, TensorRtElementWiseOperation operation)
    {
        if (left == null)
        {
            throw new ArgumentNullException(nameof(left));
        }

        if (right == null)
        {
            throw new ArgumentNullException(nameof(right));
        }

        if (left.Line != Line || right.Line != Line)
        {
            throw new ArgumentException("Input tensors must belong to the same TensorRT API line as the network.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddElementWiseLayer(Line, _handle, left.Handle, right.Handle, operation));
    }

}
