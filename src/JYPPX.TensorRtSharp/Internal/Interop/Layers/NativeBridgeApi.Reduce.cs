using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddReduceLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        TensorRtReduceOperation operation,
        uint axes,
        bool keepDimensions)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_reduce(network, input, (int)operation, axes, keepDimensions ? 1 : 0, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_reduce(network, input, (int)operation, axes, keepDimensions ? 1 : 0, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_reduce(network, input, (int)operation, axes, keepDimensions ? 1 : 0, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static TensorRtReduceOperation GetReduceOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int operation;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_reduce_layer_get_operation(layer, out operation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_reduce_layer_get_operation(layer, out operation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_reduce_layer_get_operation(layer, out operation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtReduceOperation)operation;
    }

    public static uint GetReduceAxes(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        uint axes;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_reduce_layer_get_axes(layer, out axes),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_reduce_layer_get_axes(layer, out axes),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_reduce_layer_get_axes(layer, out axes),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return axes;
    }

    public static bool GetReduceKeepDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int keepDimensions;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_reduce_layer_get_keep_dimensions(layer, out keepDimensions),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_reduce_layer_get_keep_dimensions(layer, out keepDimensions),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_reduce_layer_get_keep_dimensions(layer, out keepDimensions),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return keepDimensions != 0;
    }

}
