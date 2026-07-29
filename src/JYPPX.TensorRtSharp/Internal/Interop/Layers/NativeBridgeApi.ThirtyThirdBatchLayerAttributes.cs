using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static TensorRtDims GetConvolutionPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_get_padding_nd, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_get_padding_nd, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_padding_nd);
    }

    public static void SetConvolutionPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        SetLayerDims(line, layer, padding, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_set_padding_nd, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_set_padding_nd, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_set_padding_nd);
    }

    public static TensorRtDims GetDeconvolutionPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_padding_nd, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_padding_nd, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_padding_nd);
    }

    public static void SetDeconvolutionPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        SetLayerDims(line, layer, padding, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_set_padding_nd, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_set_padding_nd, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_set_padding_nd);
    }

    public static TensorRtDims GetSliceAxes(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt10Or11(line, nameof(GetSliceAxes));
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt10_slice_layer_get_axes, NativeMethodsTensorRt.jyppx_trt10_slice_layer_get_axes, NativeMethodsTensorRt.jyppx_trt11_slice_layer_get_axes);
    }

    public static void SetSliceAxes(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims axes)
    {
        EnsureTensorRt10Or11(line, nameof(SetSliceAxes));
        SetLayerDims(line, layer, axes, NativeMethodsTensorRt.jyppx_trt10_slice_layer_set_axes, NativeMethodsTensorRt.jyppx_trt10_slice_layer_set_axes, NativeMethodsTensorRt.jyppx_trt11_slice_layer_set_axes);
    }

    public static TensorRtDataType GetNormalizationComputePrecision(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt8Or10(line, nameof(GetNormalizationComputePrecision));
        return (TensorRtDataType)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_normalization_layer_get_compute_precision, NativeMethodsTensorRt.jyppx_trt10_normalization_layer_get_compute_precision);
    }

    public static void SetNormalizationComputePrecision(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDataType dataType)
    {
        EnsureTensorRt8Or10(line, nameof(SetNormalizationComputePrecision));
        SetLayerInt(line, layer, (int)dataType, NativeMethodsTensorRt.jyppx_trt8_normalization_layer_set_compute_precision, NativeMethodsTensorRt.jyppx_trt10_normalization_layer_set_compute_precision);
    }

    public static bool GetResizeAlignCorners(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt8Only(line, nameof(GetResizeAlignCorners));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt8_resize_layer_get_align_corners(layer, out int alignCorners);
        NativeStatus.ThrowIfFailed(status);
        return alignCorners != 0;
    }

    public static void SetResizeAlignCorners(TensorRtApiLine line, SafeTensorRtObjectHandle layer, bool alignCorners)
    {
        EnsureTensorRt8Only(line, nameof(SetResizeAlignCorners));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt8_resize_layer_set_align_corners(layer, alignCorners ? 1 : 0);
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtDataType GetTopKIndicesType(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetTopKIndicesType));
        return (TensorRtDataType)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt11_topk_layer_get_indices_type, NativeMethodsTensorRt.jyppx_trt11_topk_layer_get_indices_type, NativeMethodsTensorRt.jyppx_trt11_topk_layer_get_indices_type);
    }

    public static bool SetTopKIndicesType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDataType dataType)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetTopKIndicesType));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_topk_layer_set_indices_type(layer, (int)dataType, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    public static TensorRtDims GetDequantizeBlockShape(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetDequantizeBlockShape));
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt11_dequantize_layer_get_block_shape, NativeMethodsTensorRt.jyppx_trt11_dequantize_layer_get_block_shape, NativeMethodsTensorRt.jyppx_trt11_dequantize_layer_get_block_shape);
    }

    public static bool SetDequantizeBlockShape(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims blockShape)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetDequantizeBlockShape));
        if (blockShape == null)
        {
            throw new ArgumentNullException(nameof(blockShape));
        }

        NativeTensorRtDims nativeBlockShape = blockShape.ToNative();
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_dequantize_layer_set_block_shape(layer, ref nativeBlockShape, out int success);
        NativeStatus.ThrowIfFailed(status);
        return success != 0;
    }

    private static void EnsureTensorRt10Or11(TensorRtApiLine line, string apiName)
    {
        if (line == TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{apiName} is available for TensorRT 10 and TensorRT 11 adapters.");
        }

        if (line != TensorRtApiLine.TensorRt10 && line != TensorRtApiLine.TensorRt11)
        {
            throw UnsupportedLine();
        }
    }

    private static void EnsureTensorRt8Or10(TensorRtApiLine line, string apiName)
    {
        if (line == TensorRtApiLine.TensorRt11)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{apiName} is available for TensorRT 8 and TensorRT 10 adapters.");
        }

        if (line != TensorRtApiLine.TensorRt8 && line != TensorRtApiLine.TensorRt10)
        {
            throw UnsupportedLine();
        }
    }

    private static void EnsureTensorRt8Only(TensorRtApiLine line, string apiName)
    {
        if (line == TensorRtApiLine.TensorRt8)
        {
            return;
        }

        if (line == TensorRtApiLine.TensorRt10 || line == TensorRtApiLine.TensorRt11)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{apiName} is available for the TensorRT 8 adapter.");
        }

        throw UnsupportedLine();
    }
}
