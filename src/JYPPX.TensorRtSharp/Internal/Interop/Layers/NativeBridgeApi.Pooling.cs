using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddPoolingLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input, TensorRtPoolingType poolingType, TensorRtDims windowSize)
    {
        if (windowSize == null)
        {
            throw new ArgumentNullException(nameof(windowSize));
        }

        NativeTensorRtDims nativeWindowSize = windowSize.ToNative();
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_pooling_nd(network, input, (int)poolingType, ref nativeWindowSize, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_pooling_nd(network, input, (int)poolingType, ref nativeWindowSize, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_pooling_nd(network, input, (int)poolingType, ref nativeWindowSize, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static TensorRtPoolingType GetPoolingType(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int poolingType;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_pooling_layer_get_type(layer, out poolingType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_pooling_layer_get_type(layer, out poolingType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_type(layer, out poolingType),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtPoolingType)poolingType;
    }

    public static void SetPoolingWindowSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims windowSize)
    {
        if (windowSize == null)
        {
            throw new ArgumentNullException(nameof(windowSize));
        }

        NativeTensorRtDims nativeWindowSize = windowSize.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_pooling_layer_set_window_size_nd(layer, ref nativeWindowSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_pooling_layer_set_window_size_nd(layer, ref nativeWindowSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_pooling_layer_set_window_size_nd(layer, ref nativeWindowSize),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetPoolingWindowSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims nativeDims;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_pooling_layer_get_window_size_nd(layer, out nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_pooling_layer_get_window_size_nd(layer, out nativeDims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_window_size_nd(layer, out nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(nativeDims);
    }

    public static void SetPoolingStride(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims stride)
    {
        if (stride == null)
        {
            throw new ArgumentNullException(nameof(stride));
        }

        NativeTensorRtDims nativeStride = stride.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_pooling_layer_set_stride_nd(layer, ref nativeStride),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_pooling_layer_set_stride_nd(layer, ref nativeStride),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_pooling_layer_set_stride_nd(layer, ref nativeStride),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetPoolingStride(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims nativeDims;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_pooling_layer_get_stride_nd(layer, out nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_pooling_layer_get_stride_nd(layer, out nativeDims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_stride_nd(layer, out nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(nativeDims);
    }

    public static void SetPoolingPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        if (padding == null)
        {
            throw new ArgumentNullException(nameof(padding));
        }

        NativeTensorRtDims nativePadding = padding.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_pooling_layer_set_padding_nd(layer, ref nativePadding),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_pooling_layer_set_padding_nd(layer, ref nativePadding),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_pooling_layer_set_padding_nd(layer, ref nativePadding),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetPoolingPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims nativeDims;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_pooling_layer_get_padding_nd(layer, out nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_pooling_layer_get_padding_nd(layer, out nativeDims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_padding_nd(layer, out nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(nativeDims);
    }

}
