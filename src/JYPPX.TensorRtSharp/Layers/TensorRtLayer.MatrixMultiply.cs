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
    /// Sets the Matrix Multiply Operation value.
    /// 设置 Matrix Multiply Operation 值。
    /// </summary>
    public void SetMatrixMultiplyOperation(int inputIndex, TensorRtMatrixOperation operation)
    {
        if (inputIndex < 0 || inputIndex > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(inputIndex), "MatrixMultiply input index must be 0 or 1.");
        }

        NativeBridgeApi.SetMatrixMultiplyOperation(Line, _handle, inputIndex, operation);
    }

    /// <summary>
    /// Gets the Matrix Multiply Operation value.
    /// 获取 Matrix Multiply Operation 值。
    /// </summary>
    public TensorRtMatrixOperation GetMatrixMultiplyOperation(int inputIndex)
    {
        if (inputIndex < 0 || inputIndex > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(inputIndex), "MatrixMultiply input index must be 0 or 1.");
        }

        return NativeBridgeApi.GetMatrixMultiplyOperation(Line, _handle, inputIndex);
    }

}
