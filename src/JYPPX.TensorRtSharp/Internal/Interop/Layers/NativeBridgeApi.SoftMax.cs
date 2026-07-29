using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddSoftMaxLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_softmax(network, input, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_softmax(network, input, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_softmax(network, input, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static void SetSoftMaxAxes(TensorRtApiLine line, SafeTensorRtObjectHandle layer, uint axes)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_softmax_layer_set_axes(layer, axes),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_softmax_layer_set_axes(layer, axes),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_softmax_layer_set_axes(layer, axes),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static uint GetSoftMaxAxes(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        uint axes;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_softmax_layer_get_axes(layer, out axes),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_softmax_layer_get_axes(layer, out axes),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_softmax_layer_get_axes(layer, out axes),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return axes;
    }

}
