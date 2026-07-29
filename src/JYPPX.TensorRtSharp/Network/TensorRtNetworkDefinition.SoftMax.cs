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
    /// Adds a Soft Max layer or object.
    /// 添加 Soft Max 层或对象。
    /// </summary>
    public TensorRtLayer AddSoftMax(TensorRtTensor input, uint axes)
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
            throw new ArgumentOutOfRangeException(nameof(axes), "SoftMax axes bitmask must not be zero.");
        }

        TensorRtLayer layer = new TensorRtLayer(Line, NativeBridgeApi.AddSoftMaxLayer(Line, _handle, input.Handle));
        try
        {
            layer.SetSoftMaxAxes(axes);
            return layer;
        }
        catch
        {
            layer.Dispose();
            throw;
        }
    }

}
