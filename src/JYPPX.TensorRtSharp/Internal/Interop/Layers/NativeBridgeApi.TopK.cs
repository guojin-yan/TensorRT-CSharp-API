using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddTopKLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        TensorRtTopKOperation operation,
        int k,
        uint axes)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_topk(network, input, (int)operation, k, axes, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_topk(network, input, (int)operation, k, axes, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_topk(network, input, (int)operation, k, axes, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static TensorRtTopKOperation GetTopKOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int operation;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_topk_layer_get_operation(layer, out operation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_topk_layer_get_operation(layer, out operation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_topk_layer_get_operation(layer, out operation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtTopKOperation)operation;
    }

    public static int GetTopKValue(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int k;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_topk_layer_get_k(layer, out k),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_topk_layer_get_k(layer, out k),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_topk_layer_get_k(layer, out k),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return k;
    }

    public static uint GetTopKAxes(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        uint axes;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_topk_layer_get_axes(layer, out axes),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_topk_layer_get_axes(layer, out axes),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_topk_layer_get_axes(layer, out axes),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return axes;
    }

}
