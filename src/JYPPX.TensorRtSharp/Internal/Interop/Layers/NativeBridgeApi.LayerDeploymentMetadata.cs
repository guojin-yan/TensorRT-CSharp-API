using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static void SetReduceOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtReduceOperation operation)
    {
        SetLayerInt(line, layer, (int)operation, NativeMethodsTensorRt.jyppx_trt8_reduce_layer_set_operation, NativeMethodsTensorRt.jyppx_trt10_reduce_layer_set_operation, NativeMethodsTensorRt.jyppx_trt11_reduce_layer_set_operation);
    }

    public static void SetReduceAxes(TensorRtApiLine line, SafeTensorRtObjectHandle layer, uint axes)
    {
        SetLayerUInt(line, layer, axes, NativeMethodsTensorRt.jyppx_trt8_reduce_layer_set_axes, NativeMethodsTensorRt.jyppx_trt10_reduce_layer_set_axes, NativeMethodsTensorRt.jyppx_trt11_reduce_layer_set_axes);
    }

    public static void SetReduceKeepDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer, bool keepDimensions)
    {
        SetLayerInt(line, layer, keepDimensions ? 1 : 0, NativeMethodsTensorRt.jyppx_trt8_reduce_layer_set_keep_dimensions, NativeMethodsTensorRt.jyppx_trt10_reduce_layer_set_keep_dimensions, NativeMethodsTensorRt.jyppx_trt11_reduce_layer_set_keep_dimensions);
    }

    public static void SetUnaryOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtUnaryOperation operation)
    {
        SetLayerInt(line, layer, (int)operation, NativeMethodsTensorRt.jyppx_trt8_unary_layer_set_operation, NativeMethodsTensorRt.jyppx_trt10_unary_layer_set_operation, NativeMethodsTensorRt.jyppx_trt11_unary_layer_set_operation);
    }

    public static void SetTopKOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtTopKOperation operation)
    {
        SetLayerInt(line, layer, (int)operation, NativeMethodsTensorRt.jyppx_trt8_topk_layer_set_operation, NativeMethodsTensorRt.jyppx_trt10_topk_layer_set_operation, NativeMethodsTensorRt.jyppx_trt11_topk_layer_set_operation);
    }

    public static void SetTopKValue(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int k)
    {
        SetLayerInt(line, layer, k, NativeMethodsTensorRt.jyppx_trt8_topk_layer_set_k, NativeMethodsTensorRt.jyppx_trt10_topk_layer_set_k, NativeMethodsTensorRt.jyppx_trt11_topk_layer_set_k);
    }

    public static void SetTopKAxes(TensorRtApiLine line, SafeTensorRtObjectHandle layer, uint axes)
    {
        SetLayerUInt(line, layer, axes, NativeMethodsTensorRt.jyppx_trt8_topk_layer_set_axes, NativeMethodsTensorRt.jyppx_trt10_topk_layer_set_axes, NativeMethodsTensorRt.jyppx_trt11_topk_layer_set_axes);
    }

    public static void SetGatherAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int axis)
    {
        SetLayerInt(line, layer, axis, NativeMethodsTensorRt.jyppx_trt8_gather_layer_set_axis, NativeMethodsTensorRt.jyppx_trt10_gather_layer_set_axis, NativeMethodsTensorRt.jyppx_trt11_gather_layer_set_axis);
    }

    public static TensorRtElementWiseOperation GetElementWiseOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtElementWiseOperation)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_elementwise_layer_get_operation, NativeMethodsTensorRt.jyppx_trt10_elementwise_layer_get_operation, NativeMethodsTensorRt.jyppx_trt11_elementwise_layer_get_operation);
    }

    public static void SetElementWiseOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtElementWiseOperation operation)
    {
        SetLayerInt(line, layer, (int)operation, NativeMethodsTensorRt.jyppx_trt8_elementwise_layer_set_operation, NativeMethodsTensorRt.jyppx_trt10_elementwise_layer_set_operation, NativeMethodsTensorRt.jyppx_trt11_elementwise_layer_set_operation);
    }

    public static void SetActivationType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtActivationType activationType)
    {
        SetLayerInt(line, layer, (int)activationType, NativeMethodsTensorRt.jyppx_trt8_activation_layer_set_type, NativeMethodsTensorRt.jyppx_trt10_activation_layer_set_type, NativeMethodsTensorRt.jyppx_trt11_activation_layer_set_type);
    }

    public static double GetActivationAlpha(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDouble(line, layer, NativeMethodsTensorRt.jyppx_trt8_activation_layer_get_alpha, NativeMethodsTensorRt.jyppx_trt10_activation_layer_get_alpha, NativeMethodsTensorRt.jyppx_trt11_activation_layer_get_alpha);
    }

    public static void SetActivationAlpha(TensorRtApiLine line, SafeTensorRtObjectHandle layer, double alpha)
    {
        SetLayerDouble(line, layer, alpha, NativeMethodsTensorRt.jyppx_trt8_activation_layer_set_alpha, NativeMethodsTensorRt.jyppx_trt10_activation_layer_set_alpha, NativeMethodsTensorRt.jyppx_trt11_activation_layer_set_alpha);
    }

    public static double GetActivationBeta(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDouble(line, layer, NativeMethodsTensorRt.jyppx_trt8_activation_layer_get_beta, NativeMethodsTensorRt.jyppx_trt10_activation_layer_get_beta, NativeMethodsTensorRt.jyppx_trt11_activation_layer_get_beta);
    }

    public static void SetActivationBeta(TensorRtApiLine line, SafeTensorRtObjectHandle layer, double beta)
    {
        SetLayerDouble(line, layer, beta, NativeMethodsTensorRt.jyppx_trt8_activation_layer_set_beta, NativeMethodsTensorRt.jyppx_trt10_activation_layer_set_beta, NativeMethodsTensorRt.jyppx_trt11_activation_layer_set_beta);
    }

    public static void SetPoolingType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtPoolingType poolingType)
    {
        SetLayerInt(line, layer, (int)poolingType, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_set_type, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_set_type, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_set_type);
    }

    public static double GetPoolingBlendFactor(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDouble(line, layer, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_get_blend_factor, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_get_blend_factor, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_blend_factor);
    }

    public static void SetPoolingBlendFactor(TensorRtApiLine line, SafeTensorRtObjectHandle layer, double blendFactor)
    {
        SetLayerDouble(line, layer, blendFactor, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_set_blend_factor, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_set_blend_factor, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_set_blend_factor);
    }

    public static bool GetPoolingAverageCountExcludesPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_get_average_count_excludes_padding, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_get_average_count_excludes_padding, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_average_count_excludes_padding) != 0;
    }

    public static void SetPoolingAverageCountExcludesPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, bool excludesPadding)
    {
        SetLayerInt(line, layer, excludesPadding ? 1 : 0, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_set_average_count_excludes_padding, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_set_average_count_excludes_padding, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_set_average_count_excludes_padding);
    }

    public static TensorRtDims GetPoolingPrePadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_get_pre_padding, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_get_pre_padding, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_pre_padding);
    }

    public static void SetPoolingPrePadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        SetLayerDims(line, layer, padding, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_set_pre_padding, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_set_pre_padding, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_set_pre_padding);
    }

    public static TensorRtDims GetPoolingPostPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_get_post_padding, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_get_post_padding, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_post_padding);
    }

    public static void SetPoolingPostPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        SetLayerDims(line, layer, padding, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_set_post_padding, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_set_post_padding, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_set_post_padding);
    }

    public static TensorRtPaddingMode GetPoolingPaddingMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtPaddingMode)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_get_padding_mode, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_get_padding_mode, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_padding_mode);
    }

    public static void SetPoolingPaddingMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtPaddingMode paddingMode)
    {
        SetLayerInt(line, layer, (int)paddingMode, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_set_padding_mode, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_set_padding_mode, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_set_padding_mode);
    }

    public static int GetConvolutionOutputMaps(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_get_nb_output_maps, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_get_nb_output_maps, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_nb_output_maps);
    }

    public static void SetConvolutionOutputMaps(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int outputMaps)
    {
        SetLayerInt(line, layer, outputMaps, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_set_nb_output_maps, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_set_nb_output_maps, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_set_nb_output_maps);
    }

    public static int GetConvolutionGroups(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_get_nb_groups, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_get_nb_groups, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_nb_groups);
    }

    public static void SetConvolutionGroups(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int groups)
    {
        SetLayerInt(line, layer, groups, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_set_nb_groups, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_set_nb_groups, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_set_nb_groups);
    }

    public static TensorRtDims GetConvolutionStride(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_get_stride_nd, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_get_stride_nd, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_stride_nd);
    }

    public static void SetConvolutionStride(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims stride)
    {
        SetLayerDims(line, layer, stride, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_set_stride_nd, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_set_stride_nd, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_set_stride_nd);
    }

    public static TensorRtDims GetConvolutionPrePadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_get_pre_padding, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_get_pre_padding, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_pre_padding);
    }

    public static void SetConvolutionPrePadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        SetLayerDims(line, layer, padding, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_set_pre_padding, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_set_pre_padding, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_set_pre_padding);
    }

    public static TensorRtDims GetConvolutionPostPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_get_post_padding, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_get_post_padding, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_post_padding);
    }

    public static void SetConvolutionPostPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        SetLayerDims(line, layer, padding, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_set_post_padding, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_set_post_padding, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_set_post_padding);
    }

    public static TensorRtDims GetConvolutionDilation(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_get_dilation_nd, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_get_dilation_nd, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_dilation_nd);
    }

    public static void SetConvolutionDilation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims dilation)
    {
        SetLayerDims(line, layer, dilation, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_set_dilation_nd, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_set_dilation_nd, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_set_dilation_nd);
    }

    public static TensorRtPaddingMode GetConvolutionPaddingMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtPaddingMode)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_get_padding_mode, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_get_padding_mode, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_padding_mode);
    }

    public static void SetConvolutionPaddingMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtPaddingMode paddingMode)
    {
        SetLayerInt(line, layer, (int)paddingMode, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_set_padding_mode, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_set_padding_mode, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_set_padding_mode);
    }

    public static int GetDeconvolutionOutputMaps(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_nb_output_maps, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_nb_output_maps, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_nb_output_maps);
    }

    public static void SetDeconvolutionOutputMaps(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int outputMaps)
    {
        SetLayerInt(line, layer, outputMaps, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_set_nb_output_maps, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_set_nb_output_maps, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_set_nb_output_maps);
    }

    public static int GetDeconvolutionGroups(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_nb_groups, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_nb_groups, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_nb_groups);
    }

    public static void SetDeconvolutionGroups(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int groups)
    {
        SetLayerInt(line, layer, groups, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_set_nb_groups, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_set_nb_groups, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_set_nb_groups);
    }

    public static TensorRtDims GetDeconvolutionKernelSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_kernel_size_nd, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_kernel_size_nd, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_kernel_size_nd);
    }

    public static void SetDeconvolutionKernelSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims kernelSize)
    {
        SetLayerDims(line, layer, kernelSize, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_set_kernel_size_nd, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_set_kernel_size_nd, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_set_kernel_size_nd);
    }

    public static TensorRtDims GetDeconvolutionStride(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_stride_nd, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_stride_nd, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_stride_nd);
    }

    public static void SetDeconvolutionStride(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims stride)
    {
        SetLayerDims(line, layer, stride, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_set_stride_nd, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_set_stride_nd, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_set_stride_nd);
    }

    public static TensorRtDims GetDeconvolutionPrePadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_pre_padding, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_pre_padding, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_pre_padding);
    }

    public static void SetDeconvolutionPrePadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        SetLayerDims(line, layer, padding, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_set_pre_padding, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_set_pre_padding, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_set_pre_padding);
    }

    public static TensorRtDims GetDeconvolutionPostPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_post_padding, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_post_padding, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_post_padding);
    }

    public static void SetDeconvolutionPostPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        SetLayerDims(line, layer, padding, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_set_post_padding, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_set_post_padding, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_set_post_padding);
    }

    public static TensorRtDims GetDeconvolutionDilation(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_dilation_nd, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_dilation_nd, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_dilation_nd);
    }

    public static void SetDeconvolutionDilation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims dilation)
    {
        SetLayerDims(line, layer, dilation, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_set_dilation_nd, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_set_dilation_nd, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_set_dilation_nd);
    }

    public static TensorRtPaddingMode GetDeconvolutionPaddingMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtPaddingMode)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_padding_mode, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_padding_mode, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_padding_mode);
    }

    public static void SetDeconvolutionPaddingMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtPaddingMode paddingMode)
    {
        SetLayerInt(line, layer, (int)paddingMode, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_set_padding_mode, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_set_padding_mode, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_set_padding_mode);
    }

    public static TensorRtScaleMode GetScaleMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtScaleMode)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_scale_layer_get_mode, NativeMethodsTensorRt.jyppx_trt10_scale_layer_get_mode, NativeMethodsTensorRt.jyppx_trt11_scale_layer_get_mode);
    }

    public static void SetScaleMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtScaleMode mode)
    {
        SetLayerInt(line, layer, (int)mode, NativeMethodsTensorRt.jyppx_trt8_scale_layer_set_mode, NativeMethodsTensorRt.jyppx_trt10_scale_layer_set_mode, NativeMethodsTensorRt.jyppx_trt11_scale_layer_set_mode);
    }

    public static int GetScaleChannelAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_scale_layer_get_channel_axis, NativeMethodsTensorRt.jyppx_trt10_scale_layer_get_channel_axis, NativeMethodsTensorRt.jyppx_trt11_scale_layer_get_channel_axis);
    }

    public static void SetScaleChannelAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int channelAxis)
    {
        SetLayerInt(line, layer, channelAxis, NativeMethodsTensorRt.jyppx_trt8_scale_layer_set_channel_axis, NativeMethodsTensorRt.jyppx_trt10_scale_layer_set_channel_axis, NativeMethodsTensorRt.jyppx_trt11_scale_layer_set_channel_axis);
    }

    public static TensorRtDims GetPaddingPrePadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_padding_layer_get_pre_padding_nd, NativeMethodsTensorRt.jyppx_trt10_padding_layer_get_pre_padding_nd, NativeMethodsTensorRt.jyppx_trt11_padding_layer_get_pre_padding_nd);
    }

    public static void SetPaddingPrePadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        SetLayerDims(line, layer, padding, NativeMethodsTensorRt.jyppx_trt8_padding_layer_set_pre_padding_nd, NativeMethodsTensorRt.jyppx_trt10_padding_layer_set_pre_padding_nd, NativeMethodsTensorRt.jyppx_trt11_padding_layer_set_pre_padding_nd);
    }

    public static TensorRtDims GetPaddingPostPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_padding_layer_get_post_padding_nd, NativeMethodsTensorRt.jyppx_trt10_padding_layer_get_post_padding_nd, NativeMethodsTensorRt.jyppx_trt11_padding_layer_get_post_padding_nd);
    }

    public static void SetPaddingPostPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        SetLayerDims(line, layer, padding, NativeMethodsTensorRt.jyppx_trt8_padding_layer_set_post_padding_nd, NativeMethodsTensorRt.jyppx_trt10_padding_layer_set_post_padding_nd, NativeMethodsTensorRt.jyppx_trt11_padding_layer_set_post_padding_nd);
    }

    public static TensorRtResizeCoordinateTransformation GetResizeCoordinateTransformation(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtResizeCoordinateTransformation)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_resize_layer_get_coordinate_transformation, NativeMethodsTensorRt.jyppx_trt10_resize_layer_get_coordinate_transformation, NativeMethodsTensorRt.jyppx_trt11_resize_layer_get_coordinate_transformation);
    }

    public static void SetResizeCoordinateTransformation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtResizeCoordinateTransformation value)
    {
        SetLayerInt(line, layer, (int)value, NativeMethodsTensorRt.jyppx_trt8_resize_layer_set_coordinate_transformation, NativeMethodsTensorRt.jyppx_trt10_resize_layer_set_coordinate_transformation, NativeMethodsTensorRt.jyppx_trt11_resize_layer_set_coordinate_transformation);
    }

    public static TensorRtResizeSelector GetResizeSelectorForSinglePixel(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtResizeSelector)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_resize_layer_get_selector_for_single_pixel, NativeMethodsTensorRt.jyppx_trt10_resize_layer_get_selector_for_single_pixel, NativeMethodsTensorRt.jyppx_trt11_resize_layer_get_selector_for_single_pixel);
    }

    public static void SetResizeSelectorForSinglePixel(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtResizeSelector value)
    {
        SetLayerInt(line, layer, (int)value, NativeMethodsTensorRt.jyppx_trt8_resize_layer_set_selector_for_single_pixel, NativeMethodsTensorRt.jyppx_trt10_resize_layer_set_selector_for_single_pixel, NativeMethodsTensorRt.jyppx_trt11_resize_layer_set_selector_for_single_pixel);
    }

    public static TensorRtResizeRoundMode GetResizeNearestRounding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtResizeRoundMode)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_resize_layer_get_nearest_rounding, NativeMethodsTensorRt.jyppx_trt10_resize_layer_get_nearest_rounding, NativeMethodsTensorRt.jyppx_trt11_resize_layer_get_nearest_rounding);
    }

    public static void SetResizeNearestRounding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtResizeRoundMode value)
    {
        SetLayerInt(line, layer, (int)value, NativeMethodsTensorRt.jyppx_trt8_resize_layer_set_nearest_rounding, NativeMethodsTensorRt.jyppx_trt10_resize_layer_set_nearest_rounding, NativeMethodsTensorRt.jyppx_trt11_resize_layer_set_nearest_rounding);
    }

    public static double GetResizeCubicCoefficient(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDouble(line, layer, NativeMethodsTensorRt.jyppx_trt8_resize_layer_get_cubic_coeff, NativeMethodsTensorRt.jyppx_trt10_resize_layer_get_cubic_coeff, NativeMethodsTensorRt.jyppx_trt11_resize_layer_get_cubic_coeff);
    }

    public static void SetResizeCubicCoefficient(TensorRtApiLine line, SafeTensorRtObjectHandle layer, double value)
    {
        SetLayerDouble(line, layer, value, NativeMethodsTensorRt.jyppx_trt8_resize_layer_set_cubic_coeff, NativeMethodsTensorRt.jyppx_trt10_resize_layer_set_cubic_coeff, NativeMethodsTensorRt.jyppx_trt11_resize_layer_set_cubic_coeff);
    }

    public static bool GetResizeExcludeOutside(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_resize_layer_get_exclude_outside, NativeMethodsTensorRt.jyppx_trt10_resize_layer_get_exclude_outside, NativeMethodsTensorRt.jyppx_trt11_resize_layer_get_exclude_outside) != 0;
    }

    public static void SetResizeExcludeOutside(TensorRtApiLine line, SafeTensorRtObjectHandle layer, bool value)
    {
        SetLayerInt(line, layer, value ? 1 : 0, NativeMethodsTensorRt.jyppx_trt8_resize_layer_set_exclude_outside, NativeMethodsTensorRt.jyppx_trt10_resize_layer_set_exclude_outside, NativeMethodsTensorRt.jyppx_trt11_resize_layer_set_exclude_outside);
    }

}
