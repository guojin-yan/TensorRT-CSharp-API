using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddLrnLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input, int windowSize, float alpha, float beta, float k)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_lrn(network, input, windowSize, alpha, beta, k, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_lrn(network, input, windowSize, alpha, beta, k, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_lrn(network, input, windowSize, alpha, beta, k, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static int GetLrnWindowSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int windowSize;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_lrn_layer_get_window_size(layer, out windowSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_lrn_layer_get_window_size(layer, out windowSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_lrn_layer_get_window_size(layer, out windowSize),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return windowSize;
    }

    public static void SetLrnWindowSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int windowSize)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_lrn_layer_set_window_size(layer, windowSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_lrn_layer_set_window_size(layer, windowSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_lrn_layer_set_window_size(layer, windowSize),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static float GetLrnAlpha(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        float alpha;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_lrn_layer_get_alpha(layer, out alpha),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_lrn_layer_get_alpha(layer, out alpha),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_lrn_layer_get_alpha(layer, out alpha),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return alpha;
    }

    public static void SetLrnAlpha(TensorRtApiLine line, SafeTensorRtObjectHandle layer, float alpha)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_lrn_layer_set_alpha(layer, alpha),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_lrn_layer_set_alpha(layer, alpha),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_lrn_layer_set_alpha(layer, alpha),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static float GetLrnBeta(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        float beta;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_lrn_layer_get_beta(layer, out beta),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_lrn_layer_get_beta(layer, out beta),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_lrn_layer_get_beta(layer, out beta),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return beta;
    }

    public static void SetLrnBeta(TensorRtApiLine line, SafeTensorRtObjectHandle layer, float beta)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_lrn_layer_set_beta(layer, beta),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_lrn_layer_set_beta(layer, beta),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_lrn_layer_set_beta(layer, beta),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static float GetLrnK(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        float k;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_lrn_layer_get_k(layer, out k),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_lrn_layer_get_k(layer, out k),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_lrn_layer_get_k(layer, out k),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return k;
    }

    public static void SetLrnK(TensorRtApiLine line, SafeTensorRtObjectHandle layer, float k)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_lrn_layer_set_k(layer, k),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_lrn_layer_set_k(layer, k),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_lrn_layer_set_k(layer, k),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

}
