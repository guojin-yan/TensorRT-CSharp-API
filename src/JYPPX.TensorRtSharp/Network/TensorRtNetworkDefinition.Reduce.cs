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
    /// Adds a Reduce layer or object.
    /// 添加 Reduce 层或对象。
    /// </summary>
    public TensorRtLayer AddReduce(TensorRtTensor input, TensorRtReduceOperation operation, uint axes, bool keepDimensions)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (input.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.");
        }

        if (axes == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axes), "Reduce axes bitmask must not be zero.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddReduceLayer(Line, _handle, input.Handle, operation, axes, keepDimensions));
    }

}
