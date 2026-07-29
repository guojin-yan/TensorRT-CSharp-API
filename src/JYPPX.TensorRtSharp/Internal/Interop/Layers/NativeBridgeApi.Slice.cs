using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddSliceLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        TensorRtDims start,
        TensorRtDims size,
        TensorRtDims stride)
    {
        if (start == null)
        {
            throw new ArgumentNullException(nameof(start));
        }

        if (size == null)
        {
            throw new ArgumentNullException(nameof(size));
        }

        if (stride == null)
        {
            throw new ArgumentNullException(nameof(stride));
        }

        NativeTensorRtDims nativeStart = start.ToNative();
        NativeTensorRtDims nativeSize = size.ToNative();
        NativeTensorRtDims nativeStride = stride.ToNative();
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_slice(network, input, ref nativeStart, ref nativeSize, ref nativeStride, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_slice(network, input, ref nativeStart, ref nativeSize, ref nativeStride, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_slice(network, input, ref nativeStart, ref nativeSize, ref nativeStride, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static void SetSliceStart(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims start)
    {
        if (start == null)
        {
            throw new ArgumentNullException(nameof(start));
        }

        NativeTensorRtDims nativeStart = start.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_slice_layer_set_start(layer, ref nativeStart),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_slice_layer_set_start(layer, ref nativeStart),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_slice_layer_set_start(layer, ref nativeStart),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetSliceStart(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims start;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_slice_layer_get_start(layer, out start),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_slice_layer_get_start(layer, out start),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_slice_layer_get_start(layer, out start),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(start);
    }

    public static void SetSliceSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims size)
    {
        if (size == null)
        {
            throw new ArgumentNullException(nameof(size));
        }

        NativeTensorRtDims nativeSize = size.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_slice_layer_set_size(layer, ref nativeSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_slice_layer_set_size(layer, ref nativeSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_slice_layer_set_size(layer, ref nativeSize),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetSliceSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims size;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_slice_layer_get_size(layer, out size),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_slice_layer_get_size(layer, out size),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_slice_layer_get_size(layer, out size),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(size);
    }

    public static void SetSliceStride(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims stride)
    {
        if (stride == null)
        {
            throw new ArgumentNullException(nameof(stride));
        }

        NativeTensorRtDims nativeStride = stride.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_slice_layer_set_stride(layer, ref nativeStride),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_slice_layer_set_stride(layer, ref nativeStride),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_slice_layer_set_stride(layer, ref nativeStride),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetSliceStride(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims stride;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_slice_layer_get_stride(layer, out stride),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_slice_layer_get_stride(layer, out stride),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_slice_layer_get_stride(layer, out stride),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(stride);
    }

    public static void SetSliceMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtSliceMode mode)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_slice_layer_set_mode(layer, (int)mode),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_slice_layer_set_mode(layer, (int)mode),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_slice_layer_set_mode(layer, (int)mode),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtSliceMode GetSliceMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int mode;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_slice_layer_get_mode(layer, out mode),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_slice_layer_get_mode(layer, out mode),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_slice_layer_get_mode(layer, out mode),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtSliceMode)mode;
    }

}
