using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddDeconvolutionLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        int outputMaps,
        TensorRtDims kernelSize,
        TensorRtWeights kernelWeights,
        TensorRtWeights? biasWeights)
    {
        if (kernelSize == null)
        {
            throw new ArgumentNullException(nameof(kernelSize));
        }

        if (kernelWeights == null)
        {
            throw new ArgumentNullException(nameof(kernelWeights));
        }

        if (kernelWeights.IsEmpty)
        {
            throw new ArgumentException("Deconvolution kernel weights must not be empty.", nameof(kernelWeights));
        }

        if (outputMaps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputMaps), "Deconvolution output maps must be greater than zero.");
        }

        ValidateOptionalWeightsDataType(kernelWeights.DataType, biasWeights, nameof(biasWeights));
        NativeTensorRtDims nativeKernelSize = kernelSize.ToNative();
        using TensorRtWeights.PinnedScope kernelPinned = kernelWeights.Pin();
        TensorRtWeights.PinnedScope? biasPinned = PinOptionalWeights(biasWeights);
        try
        {
            SafeTensorRtObjectHandle layer;
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_deconvolution_nd(network, input, outputMaps, ref nativeKernelSize, (int)kernelWeights.DataType, kernelPinned.Pointer, (UIntPtr)kernelWeights.ElementCount, biasPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(biasWeights?.ElementCount ?? 0), out layer),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_deconvolution_nd(network, input, outputMaps, ref nativeKernelSize, (int)kernelWeights.DataType, kernelPinned.Pointer, (UIntPtr)kernelWeights.ElementCount, biasPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(biasWeights?.ElementCount ?? 0), out layer),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_deconvolution_nd(network, input, outputMaps, ref nativeKernelSize, (int)kernelWeights.DataType, kernelPinned.Pointer, (UIntPtr)kernelWeights.ElementCount, biasPinned?.Pointer ?? IntPtr.Zero, (UIntPtr)(biasWeights?.ElementCount ?? 0), out layer),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            };

            NativeStatus.ThrowIfFailed(status);
            return layer;
        }
        finally
        {
            biasPinned?.Dispose();
        }
    }

}
