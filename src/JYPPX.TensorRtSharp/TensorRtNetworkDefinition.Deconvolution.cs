using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtNetworkDefinition
{
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
