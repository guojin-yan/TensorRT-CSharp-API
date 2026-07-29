using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    private delegate BridgeStatusCode LayerWeightsInfoGetter(SafeTensorRtObjectHandle layer, out NativeTensorRtWeightsInfo info);

    public static NativeTensorRtWeightsInfo GetConstantLayerWeightsInfo(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerWeightsInfo(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt8_constant_layer_get_weights_info,
            NativeMethodsTensorRt.jyppx_trt10_constant_layer_get_weights_info,
            NativeMethodsTensorRt.jyppx_trt11_constant_layer_get_weights_info);
    }

    public static NativeTensorRtWeightsInfo GetConvolutionLayerKernelWeightsInfo(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerWeightsInfo(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt8_convolution_layer_get_kernel_weights_info,
            NativeMethodsTensorRt.jyppx_trt10_convolution_layer_get_kernel_weights_info,
            NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_kernel_weights_info);
    }

    public static NativeTensorRtWeightsInfo GetConvolutionLayerBiasWeightsInfo(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerWeightsInfo(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt8_convolution_layer_get_bias_weights_info,
            NativeMethodsTensorRt.jyppx_trt10_convolution_layer_get_bias_weights_info,
            NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_bias_weights_info);
    }

    public static NativeTensorRtWeightsInfo GetDeconvolutionLayerKernelWeightsInfo(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerWeightsInfo(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_kernel_weights_info,
            NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_kernel_weights_info,
            NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_kernel_weights_info);
    }

    public static NativeTensorRtWeightsInfo GetDeconvolutionLayerBiasWeightsInfo(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerWeightsInfo(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_bias_weights_info,
            NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_bias_weights_info,
            NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_bias_weights_info);
    }

    public static NativeTensorRtWeightsInfo GetScaleLayerShiftWeightsInfo(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerWeightsInfo(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt8_scale_layer_get_shift_weights_info,
            NativeMethodsTensorRt.jyppx_trt10_scale_layer_get_shift_weights_info,
            NativeMethodsTensorRt.jyppx_trt11_scale_layer_get_shift_weights_info);
    }

    public static NativeTensorRtWeightsInfo GetScaleLayerScaleWeightsInfo(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerWeightsInfo(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt8_scale_layer_get_scale_weights_info,
            NativeMethodsTensorRt.jyppx_trt10_scale_layer_get_scale_weights_info,
            NativeMethodsTensorRt.jyppx_trt11_scale_layer_get_scale_weights_info);
    }

    public static NativeTensorRtWeightsInfo GetScaleLayerPowerWeightsInfo(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerWeightsInfo(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt8_scale_layer_get_power_weights_info,
            NativeMethodsTensorRt.jyppx_trt10_scale_layer_get_power_weights_info,
            NativeMethodsTensorRt.jyppx_trt11_scale_layer_get_power_weights_info);
    }

    private static NativeTensorRtWeightsInfo GetLayerWeightsInfo(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        LayerWeightsInfoGetter trt8,
        LayerWeightsInfoGetter trt10,
        LayerWeightsInfoGetter trt11)
    {
        NativeTensorRtWeightsInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(layer, out info),
            TensorRtApiLine.TensorRt10 => trt10(layer, out info),
            TensorRtApiLine.TensorRt11 => trt11(layer, out info),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return info;
    }
}
