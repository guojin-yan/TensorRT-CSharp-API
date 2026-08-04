using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddIdentityLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_identity(network, input, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_identity(network, input, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_identity(network, input, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddConstantLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, TensorRtDims shape, TensorRtWeights weights)
    {
        if (shape == null)
        {
            throw new ArgumentNullException(nameof(shape));
        }

        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        NativeTensorRtDims nativeDims = shape.ToNative();
        using TensorRtWeights.PinnedScope pinnedWeights = weights.Pin();
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_constant(network, ref nativeDims, (int)weights.DataType, pinnedWeights.Pointer, (UIntPtr)weights.ElementCount, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_constant(network, ref nativeDims, (int)weights.DataType, pinnedWeights.Pointer, (UIntPtr)weights.ElementCount, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_constant(network, ref nativeDims, (int)weights.DataType, pinnedWeights.Pointer, (UIntPtr)weights.ElementCount, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

}
