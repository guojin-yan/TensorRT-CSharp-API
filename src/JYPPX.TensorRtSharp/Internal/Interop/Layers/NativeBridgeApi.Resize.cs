using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddResizeLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_resize(network, input, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_resize(network, input, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_resize(network, input, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static void SetResizeOutputDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims dimensions)
    {
        if (dimensions == null)
        {
            throw new ArgumentNullException(nameof(dimensions));
        }

        NativeTensorRtDims nativeDimensions = dimensions.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_resize_layer_set_output_dimensions(layer, ref nativeDimensions),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_resize_layer_set_output_dimensions(layer, ref nativeDimensions),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_resize_layer_set_output_dimensions(layer, ref nativeDimensions),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetResizeOutputDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims nativeDims;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_resize_layer_get_output_dimensions(layer, out nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_resize_layer_get_output_dimensions(layer, out nativeDims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_resize_layer_get_output_dimensions(layer, out nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(nativeDims);
    }

    public static void SetResizeMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtResizeMode resizeMode)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_resize_layer_set_mode(layer, (int)resizeMode),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_resize_layer_set_mode(layer, (int)resizeMode),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_resize_layer_set_mode(layer, (int)resizeMode),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtResizeMode GetResizeMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int resizeMode;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_resize_layer_get_mode(layer, out resizeMode),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_resize_layer_get_mode(layer, out resizeMode),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_resize_layer_get_mode(layer, out resizeMode),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtResizeMode)resizeMode;
    }

    public static void SetResizeScales(TensorRtApiLine line, SafeTensorRtObjectHandle layer, float[] scales)
    {
        if (scales == null)
        {
            throw new ArgumentNullException(nameof(scales));
        }

        if (scales.Length == 0 || scales.Length > NativeTensorRtDims.MaxDimensionCount)
        {
            throw new ArgumentOutOfRangeException(nameof(scales), $"Resize scales must contain 1 to {NativeTensorRtDims.MaxDimensionCount} values.");
        }

        GCHandle handle = GCHandle.Alloc(scales, GCHandleType.Pinned);
        try
        {
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_resize_layer_set_scales(layer, handle.AddrOfPinnedObject(), scales.Length),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_resize_layer_set_scales(layer, handle.AddrOfPinnedObject(), scales.Length),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_resize_layer_set_scales(layer, handle.AddrOfPinnedObject(), scales.Length),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            };

            NativeStatus.ThrowIfFailed(status);
        }
        finally
        {
            handle.Free();
        }
    }

    public static float[] GetResizeScales(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        float[] scales = new float[NativeTensorRtDims.MaxDimensionCount];
        GCHandle handle = GCHandle.Alloc(scales, GCHandleType.Pinned);
        try
        {
            int scaleCount;
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_resize_layer_get_scales(layer, handle.AddrOfPinnedObject(), scales.Length, out scaleCount),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_resize_layer_get_scales(layer, handle.AddrOfPinnedObject(), scales.Length, out scaleCount),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_resize_layer_get_scales(layer, handle.AddrOfPinnedObject(), scales.Length, out scaleCount),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            };

            NativeStatus.ThrowIfFailed(status);
            if (scaleCount <= 0)
            {
                return Array.Empty<float>();
            }

            float[] result = new float[scaleCount];
            Array.Copy(scales, result, scaleCount);
            return result;
        }
        finally
        {
            handle.Free();
        }
    }

}
