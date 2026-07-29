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
    /// Adds a Gather layer or object.
    /// 添加 Gather 层或对象。
    /// </summary>
    public TensorRtLayer AddGather(TensorRtTensor data, TensorRtTensor indices, int axis)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        if (indices == null)
        {
            throw new ArgumentNullException(nameof(indices));
        }

        if (data.Line != Line || indices.Line != Line)
        {
            throw new ArgumentException("Input tensors must belong to the same TensorRT API line as the network.");
        }

        if (axis < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axis), "Gather axis must be greater than or equal to zero.");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddGatherLayer(Line, _handle, data.Handle, indices.Handle, axis));
    }

}
