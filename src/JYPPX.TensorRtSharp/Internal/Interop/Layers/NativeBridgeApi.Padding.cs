using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddPaddingLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input, TensorRtDims prePadding, TensorRtDims postPadding)
    {
        if (prePadding == null)
        {
            throw new ArgumentNullException(nameof(prePadding));
        }

        if (postPadding == null)
        {
            throw new ArgumentNullException(nameof(postPadding));
        }

        NativeTensorRtDims nativePrePadding = prePadding.ToNative();
        NativeTensorRtDims nativePostPadding = postPadding.ToNative();
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_padding_nd(network, input, ref nativePrePadding, ref nativePostPadding, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_padding_nd(network, input, ref nativePrePadding, ref nativePostPadding, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_padding_nd(network, input, ref nativePrePadding, ref nativePostPadding, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

}
