using System;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    private static TensorRtWeights.PinnedScope? PinOptionalWeights(TensorRtWeights? weights)
    {
        if (weights == null || weights.IsEmpty)
        {
            return null;
        }

        return weights.Pin();
    }

    private static void ValidateOptionalWeightsDataType(TensorRtDataType expected, TensorRtWeights? weights, string argumentName)
    {
        if (weights == null || weights.IsEmpty)
        {
            return;
        }

        if (weights.DataType != expected)
        {
            throw new ArgumentException("All TensorRT weights passed to a single layer must use the same data type.", argumentName);
        }
    }
}
