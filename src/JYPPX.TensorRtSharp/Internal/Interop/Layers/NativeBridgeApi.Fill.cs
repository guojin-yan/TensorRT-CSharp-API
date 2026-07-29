using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddFillLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, TensorRtDims dimensions, TensorRtFillOperation operation)
    {
        if (dimensions == null)
        {
            throw new ArgumentNullException(nameof(dimensions));
        }

        NativeTensorRtDims nativeDimensions = dimensions.ToNative();
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_fill(network, ref nativeDimensions, (int)operation, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_fill(network, ref nativeDimensions, (int)operation, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_fill(network, ref nativeDimensions, (int)operation, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static void SetFillDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims dimensions)
    {
        if (dimensions == null)
        {
            throw new ArgumentNullException(nameof(dimensions));
        }

        NativeTensorRtDims nativeDimensions = dimensions.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_fill_layer_set_dimensions(layer, ref nativeDimensions),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_fill_layer_set_dimensions(layer, ref nativeDimensions),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_fill_layer_set_dimensions(layer, ref nativeDimensions),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetFillDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims dimensions;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_fill_layer_get_dimensions(layer, out dimensions),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_fill_layer_get_dimensions(layer, out dimensions),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_fill_layer_get_dimensions(layer, out dimensions),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(dimensions);
    }

    public static void SetFillOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtFillOperation operation)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_fill_layer_set_operation(layer, (int)operation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_fill_layer_set_operation(layer, (int)operation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_fill_layer_set_operation(layer, (int)operation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtFillOperation GetFillOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int operation;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_fill_layer_get_operation(layer, out operation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_fill_layer_get_operation(layer, out operation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_fill_layer_get_operation(layer, out operation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtFillOperation)operation;
    }

    public static void SetFillAlpha(TensorRtApiLine line, SafeTensorRtObjectHandle layer, double alpha)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_fill_layer_set_alpha(layer, alpha),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_fill_layer_set_alpha(layer, alpha),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_fill_layer_set_alpha(layer, alpha),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static double GetFillAlpha(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        double alpha;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_fill_layer_get_alpha(layer, out alpha),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_fill_layer_get_alpha(layer, out alpha),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_fill_layer_get_alpha(layer, out alpha),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return alpha;
    }

    public static void SetFillBeta(TensorRtApiLine line, SafeTensorRtObjectHandle layer, double beta)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_fill_layer_set_beta(layer, beta),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_fill_layer_set_beta(layer, beta),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_fill_layer_set_beta(layer, beta),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static double GetFillBeta(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        double beta;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_fill_layer_get_beta(layer, out beta),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_fill_layer_get_beta(layer, out beta),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_fill_layer_get_beta(layer, out beta),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return beta;
    }

}
