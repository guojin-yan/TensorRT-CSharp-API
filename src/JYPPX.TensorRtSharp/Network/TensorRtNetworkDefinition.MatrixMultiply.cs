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
    /// Adds a Matrix Multiply layer or object.
    /// 添加 Matrix Multiply 层或对象。
    /// </summary>
    public TensorRtLayer AddMatrixMultiply(
        TensorRtTensor left,
        TensorRtMatrixOperation leftOperation,
        TensorRtTensor right,
        TensorRtMatrixOperation rightOperation)
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

        return new TensorRtLayer(Line, NativeBridgeApi.AddMatrixMultiplyLayer(Line, _handle, left.Handle, leftOperation, right.Handle, rightOperation));
    }

}
