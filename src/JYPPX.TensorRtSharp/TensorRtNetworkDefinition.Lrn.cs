using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtNetworkDefinition
{
    public TensorRtLayer AddLrn(TensorRtTensor input, int windowSize, float alpha, float beta, float k)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (input.Line != Line)
        {
            throw new ArgumentException("Input tensor must belong to the same TensorRT API line as the network.");
        }

        if (windowSize < 1 || windowSize > 15 || windowSize % 2 == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(windowSize), "LRN window size must be an odd value in [1, 15].");
        }

        return new TensorRtLayer(Line, NativeBridgeApi.AddLrnLayer(Line, _handle, input.Handle, windowSize, alpha, beta, k));
    }
}
