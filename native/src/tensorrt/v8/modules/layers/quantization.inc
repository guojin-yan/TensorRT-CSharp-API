JYPPX_StatusCode jyppx_trt8_network_add_quantize(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* scale_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(input_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(scale_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "scale_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* input_payload = get_payload<nvinfer1::ITensor>(input_tensor);
    auto* scale_payload = get_payload<nvinfer1::ITensor>(scale_tensor);
    if (network_payload == nullptr || input_payload == nullptr || scale_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network, input tensor, or scale tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IQuantizeLayer* layer = network_payload->addQuantize(*input_payload, *scale_payload);
    return create_layer_reference_handle(layer, out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)scale_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add quantize");
#endif
}

JYPPX_StatusCode jyppx_trt8_network_add_dequantize(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* scale_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(input_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(scale_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "scale_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* input_payload = get_payload<nvinfer1::ITensor>(input_tensor);
    auto* scale_payload = get_payload<nvinfer1::ITensor>(scale_tensor);
    if (network_payload == nullptr || input_payload == nullptr || scale_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network, input tensor, or scale tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IDequantizeLayer* layer = network_payload->addDequantize(*input_payload, *scale_payload);
    return create_layer_reference_handle(layer, out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)scale_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add dequantize");
#endif
}

#define JYPPX_TRT8_QDQ_AXIS_ACCESSOR(FUNCTION_GET, FUNCTION_SET, GET_PAYLOAD, FEATURE) \
    JYPPX_StatusCode FUNCTION_GET(JYPPX_TensorRtLayer* layer, int32_t* out_axis) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(out_axis, "out_axis"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *out_axis = -1; \
        status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        auto* qdq_layer = GET_PAYLOAD(layer); \
        if (qdq_layer == nullptr) { jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT " FEATURE " layer."); return JYPPX_STATUS_INVALID_STATE; } \
        *out_axis = qdq_layer->getAxis(); \
        return JYPPX_STATUS_OK; \
    } \
    JYPPX_StatusCode FUNCTION_SET(JYPPX_TensorRtLayer* layer, int32_t axis) \
    { \
        if (axis < -1) { jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, FEATURE " axis must be -1 or a non-negative tensor dimension index."); return JYPPX_STATUS_INVALID_ARGUMENT; } \
        auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer"); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        auto* qdq_layer = GET_PAYLOAD(layer); \
        if (qdq_layer == nullptr) { jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT " FEATURE " layer."); return JYPPX_STATUS_INVALID_STATE; } \
        qdq_layer->setAxis(axis); \
        return JYPPX_STATUS_OK; \
    }

#if JYPPX_HAS_TENSORRT
JYPPX_TRT8_QDQ_AXIS_ACCESSOR(jyppx_trt8_quantize_layer_get_axis, jyppx_trt8_quantize_layer_set_axis, get_quantize_layer_payload, "quantize")
JYPPX_TRT8_QDQ_AXIS_ACCESSOR(jyppx_trt8_dequantize_layer_get_axis, jyppx_trt8_dequantize_layer_set_axis, get_dequantize_layer_payload, "dequantize")
#else
JYPPX_StatusCode jyppx_trt8_quantize_layer_get_axis(JYPPX_TensorRtLayer*, int32_t* out_axis) { auto status = jyppx::tensorrt::validate_output_pointer(out_axis, "out_axis"); if (status != JYPPX_STATUS_OK) { return status; } *out_axis = -1; return jyppx::tensorrt::report_vendor_missing(kLine, "quantize layer axis query"); }
JYPPX_StatusCode jyppx_trt8_quantize_layer_set_axis(JYPPX_TensorRtLayer*, int32_t) { return jyppx::tensorrt::report_vendor_missing(kLine, "quantize layer axis set"); }
JYPPX_StatusCode jyppx_trt8_dequantize_layer_get_axis(JYPPX_TensorRtLayer*, int32_t* out_axis) { auto status = jyppx::tensorrt::validate_output_pointer(out_axis, "out_axis"); if (status != JYPPX_STATUS_OK) { return status; } *out_axis = -1; return jyppx::tensorrt::report_vendor_missing(kLine, "dequantize layer axis query"); }
JYPPX_StatusCode jyppx_trt8_dequantize_layer_set_axis(JYPPX_TensorRtLayer*, int32_t) { return jyppx::tensorrt::report_vendor_missing(kLine, "dequantize layer axis set"); }
#endif
#undef JYPPX_TRT8_QDQ_AXIS_ACCESSOR
