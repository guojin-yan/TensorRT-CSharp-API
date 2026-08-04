using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddShuffleLayer(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle input)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_shuffle(network, input, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_shuffle(network, input, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_shuffle(network, input, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static void SetShuffleReshapeDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims dims)
    {
        if (dims == null)
        {
            throw new ArgumentNullException(nameof(dims));
        }

        NativeTensorRtDims nativeDims = dims.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_shuffle_layer_set_reshape_dimensions(layer, ref nativeDims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_shuffle_layer_set_reshape_dimensions(layer, ref nativeDims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_shuffle_layer_set_reshape_dimensions(layer, ref nativeDims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetShuffleReshapeDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims dims;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_shuffle_layer_get_reshape_dimensions(layer, out dims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_shuffle_layer_get_reshape_dimensions(layer, out dims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_shuffle_layer_get_reshape_dimensions(layer, out dims),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(dims);
    }

    public static void SetShuffleFirstTranspose(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims permutation)
    {
        if (permutation == null)
        {
            throw new ArgumentNullException(nameof(permutation));
        }

        NativeTensorRtDims nativePermutation = permutation.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_shuffle_layer_set_first_transpose(layer, ref nativePermutation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_shuffle_layer_set_first_transpose(layer, ref nativePermutation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_shuffle_layer_set_first_transpose(layer, ref nativePermutation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetShuffleFirstTranspose(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims permutation;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_shuffle_layer_get_first_transpose(layer, out permutation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_shuffle_layer_get_first_transpose(layer, out permutation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_shuffle_layer_get_first_transpose(layer, out permutation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(permutation);
    }

    public static void SetShuffleSecondTranspose(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims permutation)
    {
        if (permutation == null)
        {
            throw new ArgumentNullException(nameof(permutation));
        }

        NativeTensorRtDims nativePermutation = permutation.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_shuffle_layer_set_second_transpose(layer, ref nativePermutation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_shuffle_layer_set_second_transpose(layer, ref nativePermutation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_shuffle_layer_set_second_transpose(layer, ref nativePermutation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDims GetShuffleSecondTranspose(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        NativeTensorRtDims permutation;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_shuffle_layer_get_second_transpose(layer, out permutation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_shuffle_layer_get_second_transpose(layer, out permutation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_shuffle_layer_get_second_transpose(layer, out permutation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(permutation);
    }

    public static void SetShuffleZeroIsPlaceholder(TensorRtApiLine line, SafeTensorRtObjectHandle layer, bool zeroIsPlaceholder)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_shuffle_layer_set_zero_is_placeholder(layer, zeroIsPlaceholder ? 1 : 0),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_shuffle_layer_set_zero_is_placeholder(layer, zeroIsPlaceholder ? 1 : 0),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_shuffle_layer_set_zero_is_placeholder(layer, zeroIsPlaceholder ? 1 : 0),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static bool GetShuffleZeroIsPlaceholder(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        int enabled;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_shuffle_layer_get_zero_is_placeholder(layer, out enabled),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_shuffle_layer_get_zero_is_placeholder(layer, out enabled),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_shuffle_layer_get_zero_is_placeholder(layer, out enabled),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return enabled != 0;
    }

}
