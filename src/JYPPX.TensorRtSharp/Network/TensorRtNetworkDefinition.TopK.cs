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
    /// Adds a Top K layer or object.
    /// 添加 Top K 层或对象。
    /// </summary>
    public TensorRtLayer AddTopK(TensorRtTensor input, TensorRtTopKOperation operation, int k, uint axes)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (input.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.");
        }

        if (k <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(k), "TopK k must be greater than zero.");
        }

        if (axes == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axes), "TopK axes bitmask must not be zero.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddTopKLayer(Line, _handle, input.Handle, operation, k, axes));
    }

}
