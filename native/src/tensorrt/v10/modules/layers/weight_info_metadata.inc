#if JYPPX_HAS_TENSORRT
namespace
{
static void jyppx_trt10_copy_layer_weights_info(const nvinfer1::Weights& weights, JYPPX_TensorRtWeightsInfo* out_info)
{
    out_info->data_type = static_cast<int32_t>(weights.type);
    out_info->count = weights.count;
    out_info->has_values = weights.values != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
}
}

#define JYPPX_TRT10_LAYER_WEIGHTS_INFO_GET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtWeightsInfo* out_info) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        out_info->data_type = 0; \
        out_info->count = 0; \
        out_info->has_values = JYPPX_FALSE; \
        LAYER_TYPE* typed_layer = nullptr; \
        status = jyppx_trt10_get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        jyppx_trt10_copy_layer_weights_info(typed_layer->METHOD(), out_info); \
        return JYPPX_STATUS_OK; \
    }

JYPPX_TRT10_LAYER_WEIGHTS_INFO_GET(jyppx_trt10_constant_layer_get_weights_info, nvinfer1::IConstantLayer, nvinfer1::LayerType::kCONSTANT, "constant", getWeights)
JYPPX_TRT10_LAYER_WEIGHTS_INFO_GET(jyppx_trt10_convolution_layer_get_kernel_weights_info, nvinfer1::IConvolutionLayer, nvinfer1::LayerType::kCONVOLUTION, "convolution", getKernelWeights)
JYPPX_TRT10_LAYER_WEIGHTS_INFO_GET(jyppx_trt10_convolution_layer_get_bias_weights_info, nvinfer1::IConvolutionLayer, nvinfer1::LayerType::kCONVOLUTION, "convolution", getBiasWeights)
JYPPX_TRT10_LAYER_WEIGHTS_INFO_GET(jyppx_trt10_deconvolution_layer_get_kernel_weights_info, nvinfer1::IDeconvolutionLayer, nvinfer1::LayerType::kDECONVOLUTION, "deconvolution", getKernelWeights)
JYPPX_TRT10_LAYER_WEIGHTS_INFO_GET(jyppx_trt10_deconvolution_layer_get_bias_weights_info, nvinfer1::IDeconvolutionLayer, nvinfer1::LayerType::kDECONVOLUTION, "deconvolution", getBiasWeights)
JYPPX_TRT10_LAYER_WEIGHTS_INFO_GET(jyppx_trt10_scale_layer_get_shift_weights_info, nvinfer1::IScaleLayer, nvinfer1::LayerType::kSCALE, "scale", getShift)
JYPPX_TRT10_LAYER_WEIGHTS_INFO_GET(jyppx_trt10_scale_layer_get_scale_weights_info, nvinfer1::IScaleLayer, nvinfer1::LayerType::kSCALE, "scale", getScale)
JYPPX_TRT10_LAYER_WEIGHTS_INFO_GET(jyppx_trt10_scale_layer_get_power_weights_info, nvinfer1::IScaleLayer, nvinfer1::LayerType::kSCALE, "scale", getPower)

#undef JYPPX_TRT10_LAYER_WEIGHTS_INFO_GET
#else
#define JYPPX_TRT10_VENDOR_MISSING_GET_WEIGHTS_INFO(FUNCTION_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, JYPPX_TensorRtWeightsInfo* out_info) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        out_info->data_type = 0; \
        out_info->count = 0; \
        out_info->has_values = JYPPX_FALSE; \
        return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); \
    }

JYPPX_TRT10_VENDOR_MISSING_GET_WEIGHTS_INFO(jyppx_trt10_constant_layer_get_weights_info, "constant layer weights info query")
JYPPX_TRT10_VENDOR_MISSING_GET_WEIGHTS_INFO(jyppx_trt10_convolution_layer_get_kernel_weights_info, "convolution kernel weights info query")
JYPPX_TRT10_VENDOR_MISSING_GET_WEIGHTS_INFO(jyppx_trt10_convolution_layer_get_bias_weights_info, "convolution bias weights info query")
JYPPX_TRT10_VENDOR_MISSING_GET_WEIGHTS_INFO(jyppx_trt10_deconvolution_layer_get_kernel_weights_info, "deconvolution kernel weights info query")
JYPPX_TRT10_VENDOR_MISSING_GET_WEIGHTS_INFO(jyppx_trt10_deconvolution_layer_get_bias_weights_info, "deconvolution bias weights info query")
JYPPX_TRT10_VENDOR_MISSING_GET_WEIGHTS_INFO(jyppx_trt10_scale_layer_get_shift_weights_info, "scale shift weights info query")
JYPPX_TRT10_VENDOR_MISSING_GET_WEIGHTS_INFO(jyppx_trt10_scale_layer_get_scale_weights_info, "scale scale weights info query")
JYPPX_TRT10_VENDOR_MISSING_GET_WEIGHTS_INFO(jyppx_trt10_scale_layer_get_power_weights_info, "scale power weights info query")

#undef JYPPX_TRT10_VENDOR_MISSING_GET_WEIGHTS_INFO
#endif
