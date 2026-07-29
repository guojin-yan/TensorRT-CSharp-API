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
    /// Adds a Resize layer or object.
    /// 添加 Resize 层或对象。
    /// </summary>
    public TensorRtLayer AddResize(TensorRtTensor input, TensorRtDims outputDimensions, TensorRtResizeMode resizeMode = TensorRtResizeMode.Nearest)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (outputDimensions == null)
        {
            throw new ArgumentNullException(nameof(outputDimensions));
        }

        if (input.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.");
        }

        TensorRtLayer layer = new TensorRtLayer(Line, NativeBridgeApi.AddResizeLayer(Line, _handle, input.Handle));
        try
        {
            layer.SetResizeMode(resizeMode);
            layer.SetResizeOutputDimensions(outputDimensions);
            return layer;
        }
        catch
        {
            layer.Dispose();
            throw;
        }
    }

}
