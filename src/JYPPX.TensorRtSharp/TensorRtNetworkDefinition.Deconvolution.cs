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
    /// Adds a Deconvolution layer or object.
    /// 添加 Deconvolution 层或对象。
    /// </summary>
    public TensorRtLayer AddDeconvolution(TensorRtTensor input, int outputMaps, TensorRtDims kernelSize, TensorRtWeights kernelWeights, TensorRtWeights? biasWeights = null)
    {
        ValidateInputTensor(input, nameof(input));
        if (kernelSize == null)
        {
            throw new ArgumentNullException(nameof(kernelSize));
        }

        if (kernelWeights == null)
        {
            throw new ArgumentNullException(nameof(kernelWeights));
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddDeconvolutionLayer(Line, _handle, input.Handle, outputMaps, kernelSize, kernelWeights, biasWeights));
    }
}
