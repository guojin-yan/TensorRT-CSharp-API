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
    /// Adds a Shuffle layer or object.
    /// 添加 Shuffle 层或对象。
    /// </summary>
    public TensorRtLayer AddShuffle(TensorRtTensor input, TensorRtDims reshapeDimensions)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (reshapeDimensions == null)
        {
            throw new ArgumentNullException(nameof(reshapeDimensions));
        }

        if (input.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.");
        }

        TensorRtLayer layer = new TensorRtLayer(Line, NativeBridgeApi.AddShuffleLayer(Line, _handle, input.Handle));
        try
        {
            layer.SetShuffleReshapeDimensions(reshapeDimensions);
            return layer;
        }
        catch
        {
            layer.Dispose();
            throw;
        }
    }

}
