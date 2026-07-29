using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddQuantizeLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        SafeTensorRtObjectHandle scale)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_quantize(network, input, scale, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_quantize(network, input, scale, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_quantize(network, input, scale, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddDequantizeLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        SafeTensorRtObjectHandle scale)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_dequantize(network, input, scale, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_dequantize(network, input, scale, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_dequantize(network, input, scale, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddQuantizeV2Layer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        SafeTensorRtObjectHandle scale,
        TensorRtDataType outputType)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_quantize_v2(network, input, scale, (int)outputType, out layer),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(AddQuantizeV2Layer)} is available for the TensorRT 10 adapter."),
            TensorRtApiLine.TensorRt11 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "TensorRT 11 currently exposes AddQuantize with layer output-type controls; AddQuantizeV2 is not a separate TensorRT 11 bridge entry point."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static int GetQuantizeAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int axis;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_quantize_layer_get_axis(layer, out axis),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_quantize_layer_get_axis(layer, out axis),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_quantize_layer_get_axis(layer, out axis),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return axis;
    }

    public static void SetQuantizeAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int axis)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_quantize_layer_set_axis(layer, axis),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_quantize_layer_set_axis(layer, axis),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_quantize_layer_set_axis(layer, axis),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetDequantizeAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int axis;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_dequantize_layer_get_axis(layer, out axis),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_dequantize_layer_get_axis(layer, out axis),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_dequantize_layer_get_axis(layer, out axis),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return axis;
    }

    public static void SetDequantizeAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int axis)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_dequantize_layer_set_axis(layer, axis),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_dequantize_layer_set_axis(layer, axis),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_dequantize_layer_set_axis(layer, axis),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }
}
