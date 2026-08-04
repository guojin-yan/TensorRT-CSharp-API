using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    private delegate BridgeStatusCode LayerDims64Getter(SafeTensorRtObjectHandle layer, out NativeTensorRtDims64 value);
    private delegate BridgeStatusCode LayerSlotDims64Getter(SafeTensorRtObjectHandle layer, int index, out NativeTensorRtDims64 value);
    private delegate BridgeStatusCode LayerSlotDimensionInt64Getter(SafeTensorRtObjectHandle layer, int index, int dimensionIndex, out long value);

    public static TensorRtDims64 GetLayerInputTensorShape64(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotDims64(line, layer, index, nameof(GetLayerInputTensorShape64), NativeMethodsTensorRt.jyppx_trt11_layer_input_get_tensor_shape64);
    }

    public static TensorRtDims64 GetLayerOutputTensorShape64(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index)
    {
        return GetLayerSlotDims64(line, layer, index, nameof(GetLayerOutputTensorShape64), NativeMethodsTensorRt.jyppx_trt11_layer_output_get_tensor_shape64);
    }

    public static long GetLayerInputTensorDimensionExtent64(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, int dimensionIndex)
    {
        return GetLayerSlotDimensionInt64(line, layer, index, dimensionIndex, nameof(GetLayerInputTensorDimensionExtent64), NativeMethodsTensorRt.jyppx_trt11_layer_input_get_tensor_dimension_extent64);
    }

    public static long GetLayerOutputTensorDimensionExtent64(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, int dimensionIndex)
    {
        return GetLayerSlotDimensionInt64(line, layer, index, dimensionIndex, nameof(GetLayerOutputTensorDimensionExtent64), NativeMethodsTensorRt.jyppx_trt11_layer_output_get_tensor_dimension_extent64);
    }

    public static TensorRtDims64 GetShuffleReshapeDimensions64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetShuffleReshapeDimensions64), NativeMethodsTensorRt.jyppx_trt11_shuffle_layer_get_reshape_dimensions64);

    public static TensorRtDims64 GetSliceStart64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetSliceStart64), NativeMethodsTensorRt.jyppx_trt11_slice_layer_get_start64);

    public static TensorRtDims64 GetSliceSize64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetSliceSize64), NativeMethodsTensorRt.jyppx_trt11_slice_layer_get_size64);

    public static TensorRtDims64 GetSliceStride64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetSliceStride64), NativeMethodsTensorRt.jyppx_trt11_slice_layer_get_stride64);

    public static TensorRtDims64 GetDynamicQuantizeBlockShape64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetDynamicQuantizeBlockShape64), NativeMethodsTensorRt.jyppx_trt11_dynamic_quantize_layer_get_block_shape64);

    public static TensorRtDims64 GetPoolingWindowSize64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetPoolingWindowSize64), NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_window_size_nd64);

    public static TensorRtDims64 GetPoolingStride64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetPoolingStride64), NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_stride_nd64);

    public static TensorRtDims64 GetPoolingPadding64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetPoolingPadding64), NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_padding_nd64);

    public static TensorRtDims64 GetPoolingPrePadding64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetPoolingPrePadding64), NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_pre_padding64);

    public static TensorRtDims64 GetPoolingPostPadding64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetPoolingPostPadding64), NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_post_padding64);

    public static TensorRtDims64 GetResizeOutputDimensions64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetResizeOutputDimensions64), NativeMethodsTensorRt.jyppx_trt11_resize_layer_get_output_dimensions64);

    public static TensorRtDims64 GetFillDimensions64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetFillDimensions64), NativeMethodsTensorRt.jyppx_trt11_fill_layer_get_dimensions64);

    public static TensorRtDims64 GetConvolutionStride64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetConvolutionStride64), NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_stride_nd64);

    public static TensorRtDims64 GetConvolutionPrePadding64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetConvolutionPrePadding64), NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_pre_padding64);

    public static TensorRtDims64 GetConvolutionPostPadding64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetConvolutionPostPadding64), NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_post_padding64);

    public static TensorRtDims64 GetConvolutionDilation64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetConvolutionDilation64), NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_dilation_nd64);

    public static TensorRtDims64 GetDeconvolutionKernelSize64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetDeconvolutionKernelSize64), NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_kernel_size_nd64);

    public static TensorRtDims64 GetDeconvolutionStride64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetDeconvolutionStride64), NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_stride_nd64);

    public static TensorRtDims64 GetDeconvolutionPrePadding64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetDeconvolutionPrePadding64), NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_pre_padding64);

    public static TensorRtDims64 GetDeconvolutionPostPadding64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetDeconvolutionPostPadding64), NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_post_padding64);

    public static TensorRtDims64 GetDeconvolutionDilation64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetDeconvolutionDilation64), NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_dilation_nd64);

    public static TensorRtDims64 GetPaddingPrePadding64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetPaddingPrePadding64), NativeMethodsTensorRt.jyppx_trt11_padding_layer_get_pre_padding_nd64);

    public static TensorRtDims64 GetPaddingPostPadding64(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
        => GetLayerDims64(line, layer, nameof(GetPaddingPostPadding64), NativeMethodsTensorRt.jyppx_trt11_padding_layer_get_post_padding_nd64);

    private static TensorRtDims64 GetLayerSlotDims64(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, string apiName, LayerSlotDims64Getter getter)
    {
        EnsureTensorRt11LayerSlotApi(line, index, apiName);
        BridgeStatusCode status = getter(layer, index, out NativeTensorRtDims64 value);
        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims64.FromNative(value);
    }

    private static long GetLayerSlotDimensionInt64(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int index, int dimensionIndex, string apiName, LayerSlotDimensionInt64Getter getter)
    {
        EnsureTensorRt11LayerSlotDimensionApi(line, index, dimensionIndex, apiName);
        BridgeStatusCode status = getter(layer, index, dimensionIndex, out long value);
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    private static TensorRtDims64 GetLayerDims64(TensorRtApiLine line, SafeTensorRtObjectHandle layer, string apiName, LayerDims64Getter getter)
    {
        EnsureTensorRt11DeploymentApi(line, apiName);
        BridgeStatusCode status = getter(layer, out NativeTensorRtDims64 value);
        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims64.FromNative(value);
    }
}
