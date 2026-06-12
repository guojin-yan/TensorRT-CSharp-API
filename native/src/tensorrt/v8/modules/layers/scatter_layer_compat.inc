#if JYPPX_HAS_TENSORRT
JYPPX_StatusCode jyppx_trt8_network_add_scatter(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* data_tensor, JYPPX_TensorRtTensor* indices_tensor, JYPPX_TensorRtTensor* updates_tensor, int32_t mode, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (mode < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* data = nullptr;
    nvinfer1::ITensor* indices = nullptr;
    nvinfer1::ITensor* updates = nullptr;
    status = jyppx_trt8_get_network_payload_for_api(network, &network_payload, "network scatter layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt8_get_tensor_payload_for_api(data_tensor, &data, "data_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt8_get_tensor_payload_for_api(indices_tensor, &indices, "indices_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt8_get_tensor_payload_for_api(updates_tensor, &updates, "updates_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addScatter(*data, *indices, *updates, static_cast<nvinfer1::ScatterMode>(mode)), out_layer);
}

#define JYPPX_TRT8_SCATTER_LAYER_INT_SET(FUNCTION_NAME, METHOD, VALUE_TYPE) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int32_t value) \
    { \
        if (value < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; } \
        nvinfer1::IScatterLayer* typed_layer = nullptr; \
        auto status = jyppx_trt8_get_typed_layer_for_api(layer, nvinfer1::LayerType::kSCATTER, "scatter", #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        typed_layer->METHOD(static_cast<VALUE_TYPE>(value)); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT8_SCATTER_LAYER_INT_GET(FUNCTION_NAME, METHOD, OUT_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int32_t* OUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUT_NAME, #OUT_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUT_NAME = 0; \
        nvinfer1::IScatterLayer* typed_layer = nullptr; \
        status = jyppx_trt8_get_typed_layer_for_api(layer, nvinfer1::LayerType::kSCATTER, "scatter", #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUT_NAME = static_cast<int32_t>(typed_layer->METHOD()); \
        return JYPPX_STATUS_OK; \
    }

JYPPX_TRT8_SCATTER_LAYER_INT_SET(jyppx_trt8_scatter_layer_set_mode, setMode, nvinfer1::ScatterMode)
JYPPX_TRT8_SCATTER_LAYER_INT_GET(jyppx_trt8_scatter_layer_get_mode, getMode, out_mode)
JYPPX_TRT8_SCATTER_LAYER_INT_SET(jyppx_trt8_scatter_layer_set_axis, setAxis, int32_t)
JYPPX_TRT8_SCATTER_LAYER_INT_GET(jyppx_trt8_scatter_layer_get_axis, getAxis, out_axis)

#undef JYPPX_TRT8_SCATTER_LAYER_INT_SET
#undef JYPPX_TRT8_SCATTER_LAYER_INT_GET
#else
#define JYPPX_TRT8_VENDOR_MISSING_GET_INT(FUNCTION_NAME, PARAM_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, int32_t* PARAM_NAME) { auto status = jyppx::tensorrt::validate_output_pointer(PARAM_NAME, #PARAM_NAME); if (status != JYPPX_STATUS_OK) { return status; } *PARAM_NAME = 0; return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }
#define JYPPX_TRT8_VENDOR_MISSING_SET_INT(FUNCTION_NAME, PARAM_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, int32_t PARAM_NAME) { (void)PARAM_NAME; return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }

JYPPX_StatusCode jyppx_trt8_network_add_scatter(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, int32_t, JYPPX_TensorRtLayer** out_layer) { auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer"); if (status != JYPPX_STATUS_OK) { return status; } *out_layer = nullptr; return jyppx::tensorrt::report_vendor_missing(kLine, "network scatter layer creation"); }
JYPPX_TRT8_VENDOR_MISSING_SET_INT(jyppx_trt8_scatter_layer_set_mode, mode, "scatter mode set")
JYPPX_TRT8_VENDOR_MISSING_GET_INT(jyppx_trt8_scatter_layer_get_mode, out_mode, "scatter mode query")
JYPPX_TRT8_VENDOR_MISSING_SET_INT(jyppx_trt8_scatter_layer_set_axis, axis, "scatter axis set")
JYPPX_TRT8_VENDOR_MISSING_GET_INT(jyppx_trt8_scatter_layer_get_axis, out_axis, "scatter axis query")

#undef JYPPX_TRT8_VENDOR_MISSING_GET_INT
#undef JYPPX_TRT8_VENDOR_MISSING_SET_INT
#endif
