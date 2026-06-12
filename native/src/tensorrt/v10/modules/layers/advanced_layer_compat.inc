#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
#define JYPPX_TRT10_ADV_LAYER_INT_SET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, VALUE_TYPE) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int32_t value) \
    { \
        if (value < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; } \
        LAYER_TYPE* typed_layer = nullptr; \
        auto status = jyppx_trt10_get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        typed_layer->METHOD(static_cast<VALUE_TYPE>(value)); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT10_ADV_LAYER_INT_GET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, OUT_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int32_t* OUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUT_NAME, #OUT_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUT_NAME = 0; \
        LAYER_TYPE* typed_layer = nullptr; \
        status = jyppx_trt10_get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUT_NAME = static_cast<int32_t>(typed_layer->METHOD()); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT10_ADV_LAYER_BOOL_SET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, PARAM_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, JYPPX_Boolean PARAM_NAME) \
    { \
        LAYER_TYPE* typed_layer = nullptr; \
        auto status = jyppx_trt10_get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        typed_layer->METHOD(PARAM_NAME != JYPPX_FALSE); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT10_ADV_LAYER_BOOL_GET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, OUT_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* OUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUT_NAME, #OUT_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUT_NAME = JYPPX_FALSE; \
        LAYER_TYPE* typed_layer = nullptr; \
        status = jyppx_trt10_get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUT_NAME = typed_layer->METHOD() ? JYPPX_TRUE : JYPPX_FALSE; \
        return JYPPX_STATUS_OK; \
    }

JYPPX_StatusCode jyppx_trt10_network_add_scatter(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* data_tensor, JYPPX_TensorRtTensor* indices_tensor, JYPPX_TensorRtTensor* updates_tensor, int32_t mode, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (mode < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* data = nullptr;
    nvinfer1::ITensor* indices = nullptr;
    nvinfer1::ITensor* updates = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network scatter layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(data_tensor, &data, "data_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(indices_tensor, &indices, "indices_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(updates_tensor, &updates, "updates_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addScatter(*data, *indices, *updates, static_cast<nvinfer1::ScatterMode>(mode)), out_layer);
}

JYPPX_StatusCode jyppx_trt10_network_add_dynamic_quantize(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, int32_t axis, int32_t block_size, int32_t output_type, int32_t scale_type, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (output_type < 0 || scale_type < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network dynamic quantize layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(
        network_payload->addDynamicQuantize(*input, axis, block_size, static_cast<nvinfer1::DataType>(output_type), static_cast<nvinfer1::DataType>(scale_type)),
        out_layer);
}

JYPPX_StatusCode jyppx_trt10_network_add_cumulative(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* axis_tensor, int32_t operation, JYPPX_Boolean exclusive, JYPPX_Boolean reverse, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (operation < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* axis = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network cumulative layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(axis_tensor, &axis, "axis_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addCumulative(*input, *axis, static_cast<nvinfer1::CumulativeOperation>(operation), exclusive != JYPPX_FALSE, reverse != JYPPX_FALSE), out_layer);
}

JYPPX_TRT10_ADV_LAYER_INT_SET(jyppx_trt10_scatter_layer_set_mode, nvinfer1::IScatterLayer, nvinfer1::LayerType::kSCATTER, "scatter", setMode, nvinfer1::ScatterMode)
JYPPX_TRT10_ADV_LAYER_INT_GET(jyppx_trt10_scatter_layer_get_mode, nvinfer1::IScatterLayer, nvinfer1::LayerType::kSCATTER, "scatter", getMode, out_mode)
JYPPX_TRT10_ADV_LAYER_INT_SET(jyppx_trt10_scatter_layer_set_axis, nvinfer1::IScatterLayer, nvinfer1::LayerType::kSCATTER, "scatter", setAxis, int32_t)
JYPPX_TRT10_ADV_LAYER_INT_GET(jyppx_trt10_scatter_layer_get_axis, nvinfer1::IScatterLayer, nvinfer1::LayerType::kSCATTER, "scatter", getAxis, out_axis)
JYPPX_TRT10_ADV_LAYER_INT_SET(jyppx_trt10_quantize_layer_set_to_type, nvinfer1::IQuantizeLayer, nvinfer1::LayerType::kQUANTIZE, "quantize", setToType, nvinfer1::DataType)
JYPPX_TRT10_ADV_LAYER_INT_GET(jyppx_trt10_quantize_layer_get_to_type, nvinfer1::IQuantizeLayer, nvinfer1::LayerType::kQUANTIZE, "quantize", getToType, out_data_type)
JYPPX_TRT10_ADV_LAYER_INT_SET(jyppx_trt10_dequantize_layer_set_to_type, nvinfer1::IDequantizeLayer, nvinfer1::LayerType::kDEQUANTIZE, "dequantize", setToType, nvinfer1::DataType)
JYPPX_TRT10_ADV_LAYER_INT_GET(jyppx_trt10_dequantize_layer_get_to_type, nvinfer1::IDequantizeLayer, nvinfer1::LayerType::kDEQUANTIZE, "dequantize", getToType, out_data_type)
JYPPX_TRT10_ADV_LAYER_INT_SET(jyppx_trt10_dynamic_quantize_layer_set_to_type, nvinfer1::IDynamicQuantizeLayer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", setToType, nvinfer1::DataType)
JYPPX_TRT10_ADV_LAYER_INT_GET(jyppx_trt10_dynamic_quantize_layer_get_to_type, nvinfer1::IDynamicQuantizeLayer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", getToType, out_data_type)
JYPPX_TRT10_ADV_LAYER_INT_SET(jyppx_trt10_dynamic_quantize_layer_set_scale_type, nvinfer1::IDynamicQuantizeLayer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", setScaleType, nvinfer1::DataType)
JYPPX_TRT10_ADV_LAYER_INT_GET(jyppx_trt10_dynamic_quantize_layer_get_scale_type, nvinfer1::IDynamicQuantizeLayer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", getScaleType, out_data_type)
JYPPX_TRT10_ADV_LAYER_INT_SET(jyppx_trt10_dynamic_quantize_layer_set_axis, nvinfer1::IDynamicQuantizeLayer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", setAxis, int32_t)
JYPPX_TRT10_ADV_LAYER_INT_GET(jyppx_trt10_dynamic_quantize_layer_get_axis, nvinfer1::IDynamicQuantizeLayer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", getAxis, out_axis)
JYPPX_TRT10_ADV_LAYER_INT_SET(jyppx_trt10_dynamic_quantize_layer_set_block_size, nvinfer1::IDynamicQuantizeLayer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", setBlockSize, int32_t)
JYPPX_TRT10_ADV_LAYER_INT_GET(jyppx_trt10_dynamic_quantize_layer_get_block_size, nvinfer1::IDynamicQuantizeLayer, nvinfer1::LayerType::kDYNAMIC_QUANTIZE, "dynamic quantize", getBlockSize, out_block_size)
JYPPX_TRT10_ADV_LAYER_INT_SET(jyppx_trt10_fill_layer_set_to_type, nvinfer1::IFillLayer, nvinfer1::LayerType::kFILL, "fill", setToType, nvinfer1::DataType)
JYPPX_TRT10_ADV_LAYER_INT_GET(jyppx_trt10_fill_layer_get_to_type, nvinfer1::IFillLayer, nvinfer1::LayerType::kFILL, "fill", getToType, out_data_type)

JYPPX_StatusCode jyppx_trt10_fill_layer_set_alpha_int64(JYPPX_TensorRtLayer* layer, int64_t alpha)
{
    nvinfer1::IFillLayer* typed_layer = nullptr;
    auto status = jyppx_trt10_get_typed_layer_for_api(layer, nvinfer1::LayerType::kFILL, "fill", "fill alpha int64 set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setAlphaInt64(alpha);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt10_fill_layer_get_alpha_int64(JYPPX_TensorRtLayer* layer, int64_t* out_alpha)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_alpha, "out_alpha");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_alpha = 0;
    nvinfer1::IFillLayer* typed_layer = nullptr;
    status = jyppx_trt10_get_typed_layer_for_api(layer, nvinfer1::LayerType::kFILL, "fill", "fill alpha int64 query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_alpha = typed_layer->getAlphaInt64();
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt10_fill_layer_set_beta_int64(JYPPX_TensorRtLayer* layer, int64_t beta)
{
    nvinfer1::IFillLayer* typed_layer = nullptr;
    auto status = jyppx_trt10_get_typed_layer_for_api(layer, nvinfer1::LayerType::kFILL, "fill", "fill beta int64 set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setBetaInt64(beta);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt10_fill_layer_get_beta_int64(JYPPX_TensorRtLayer* layer, int64_t* out_beta)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_beta, "out_beta");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_beta = 0;
    nvinfer1::IFillLayer* typed_layer = nullptr;
    status = jyppx_trt10_get_typed_layer_for_api(layer, nvinfer1::LayerType::kFILL, "fill", "fill beta int64 query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_beta = typed_layer->getBetaInt64();
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt10_fill_layer_is_alpha_beta_int64(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_is_int64)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_int64, "out_is_int64");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_int64 = JYPPX_FALSE;
    nvinfer1::IFillLayer* typed_layer = nullptr;
    status = jyppx_trt10_get_typed_layer_for_api(layer, nvinfer1::LayerType::kFILL, "fill", "fill alpha/beta type query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_int64 = typed_layer->isAlphaBetaInt64() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt10_cumulative_layer_set_operation(JYPPX_TensorRtLayer* layer, int32_t operation, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
    if (operation < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::ICumulativeLayer* typed_layer = nullptr;
    status = jyppx_trt10_get_typed_layer_for_api(layer, nvinfer1::LayerType::kCUMULATIVE, "cumulative", "cumulative operation set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = typed_layer->setOperation(static_cast<nvinfer1::CumulativeOperation>(operation)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_TRT10_ADV_LAYER_INT_GET(jyppx_trt10_cumulative_layer_get_operation, nvinfer1::ICumulativeLayer, nvinfer1::LayerType::kCUMULATIVE, "cumulative", getOperation, out_operation)
JYPPX_TRT10_ADV_LAYER_BOOL_SET(jyppx_trt10_cumulative_layer_set_exclusive, nvinfer1::ICumulativeLayer, nvinfer1::LayerType::kCUMULATIVE, "cumulative", setExclusive, exclusive)
JYPPX_TRT10_ADV_LAYER_BOOL_GET(jyppx_trt10_cumulative_layer_get_exclusive, nvinfer1::ICumulativeLayer, nvinfer1::LayerType::kCUMULATIVE, "cumulative", getExclusive, out_exclusive)
JYPPX_TRT10_ADV_LAYER_BOOL_SET(jyppx_trt10_cumulative_layer_set_reverse, nvinfer1::ICumulativeLayer, nvinfer1::LayerType::kCUMULATIVE, "cumulative", setReverse, reverse)
JYPPX_TRT10_ADV_LAYER_BOOL_GET(jyppx_trt10_cumulative_layer_get_reverse, nvinfer1::ICumulativeLayer, nvinfer1::LayerType::kCUMULATIVE, "cumulative", getReverse, out_reverse)

#undef JYPPX_TRT10_ADV_LAYER_INT_SET
#undef JYPPX_TRT10_ADV_LAYER_INT_GET
#undef JYPPX_TRT10_ADV_LAYER_BOOL_SET
#undef JYPPX_TRT10_ADV_LAYER_BOOL_GET
#else
#define JYPPX_TRT10_VENDOR_MISSING_LAYER_OUT(FUNCTION_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, int32_t, JYPPX_TensorRtLayer** out_layer) { auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer"); if (status != JYPPX_STATUS_OK) { return status; } *out_layer = nullptr; return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }
#define JYPPX_TRT10_VENDOR_MISSING_GET_INT(FUNCTION_NAME, PARAM_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, int32_t* PARAM_NAME) { auto status = jyppx::tensorrt::validate_output_pointer(PARAM_NAME, #PARAM_NAME); if (status != JYPPX_STATUS_OK) { return status; } *PARAM_NAME = 0; return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }
#define JYPPX_TRT10_VENDOR_MISSING_SET_INT(FUNCTION_NAME, PARAM_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, int32_t PARAM_NAME) { (void)PARAM_NAME; return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }
#define JYPPX_TRT10_VENDOR_MISSING_GET_BOOL(FUNCTION_NAME, PARAM_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, JYPPX_Boolean* PARAM_NAME) { auto status = jyppx::tensorrt::validate_output_pointer(PARAM_NAME, #PARAM_NAME); if (status != JYPPX_STATUS_OK) { return status; } *PARAM_NAME = JYPPX_FALSE; return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }
#define JYPPX_TRT10_VENDOR_MISSING_SET_BOOL(FUNCTION_NAME, PARAM_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, JYPPX_Boolean PARAM_NAME) { (void)PARAM_NAME; return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }

JYPPX_TRT10_VENDOR_MISSING_LAYER_OUT(jyppx_trt10_network_add_scatter, "network scatter layer creation")
JYPPX_StatusCode jyppx_trt10_network_add_dynamic_quantize(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, int32_t, int32_t, int32_t, int32_t, JYPPX_TensorRtLayer** out_layer) { auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer"); if (status != JYPPX_STATUS_OK) { return status; } *out_layer = nullptr; return jyppx::tensorrt::report_vendor_missing(kLine, "network dynamic quantize layer creation"); }
JYPPX_StatusCode jyppx_trt10_network_add_cumulative(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, int32_t, JYPPX_Boolean, JYPPX_Boolean, JYPPX_TensorRtLayer** out_layer) { auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer"); if (status != JYPPX_STATUS_OK) { return status; } *out_layer = nullptr; return jyppx::tensorrt::report_vendor_missing(kLine, "network cumulative layer creation"); }
JYPPX_TRT10_VENDOR_MISSING_SET_INT(jyppx_trt10_scatter_layer_set_mode, mode, "scatter mode set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_scatter_layer_get_mode, out_mode, "scatter mode query")
JYPPX_TRT10_VENDOR_MISSING_SET_INT(jyppx_trt10_scatter_layer_set_axis, axis, "scatter axis set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_scatter_layer_get_axis, out_axis, "scatter axis query")
JYPPX_TRT10_VENDOR_MISSING_SET_INT(jyppx_trt10_quantize_layer_set_to_type, data_type, "quantize output type set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_quantize_layer_get_to_type, out_data_type, "quantize output type query")
JYPPX_TRT10_VENDOR_MISSING_SET_INT(jyppx_trt10_dequantize_layer_set_to_type, data_type, "dequantize output type set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_dequantize_layer_get_to_type, out_data_type, "dequantize output type query")
JYPPX_TRT10_VENDOR_MISSING_SET_INT(jyppx_trt10_dynamic_quantize_layer_set_to_type, data_type, "dynamic quantize output type set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_dynamic_quantize_layer_get_to_type, out_data_type, "dynamic quantize output type query")
JYPPX_TRT10_VENDOR_MISSING_SET_INT(jyppx_trt10_dynamic_quantize_layer_set_scale_type, data_type, "dynamic quantize scale type set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_dynamic_quantize_layer_get_scale_type, out_data_type, "dynamic quantize scale type query")
JYPPX_TRT10_VENDOR_MISSING_SET_INT(jyppx_trt10_dynamic_quantize_layer_set_axis, axis, "dynamic quantize axis set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_dynamic_quantize_layer_get_axis, out_axis, "dynamic quantize axis query")
JYPPX_TRT10_VENDOR_MISSING_SET_INT(jyppx_trt10_dynamic_quantize_layer_set_block_size, block_size, "dynamic quantize block size set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_dynamic_quantize_layer_get_block_size, out_block_size, "dynamic quantize block size query")
JYPPX_TRT10_VENDOR_MISSING_SET_INT(jyppx_trt10_fill_layer_set_to_type, data_type, "fill output type set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_fill_layer_get_to_type, out_data_type, "fill output type query")
JYPPX_StatusCode jyppx_trt10_fill_layer_set_alpha_int64(JYPPX_TensorRtLayer*, int64_t alpha) { (void)alpha; return jyppx::tensorrt::report_vendor_missing(kLine, "fill alpha int64 set"); }
JYPPX_StatusCode jyppx_trt10_fill_layer_get_alpha_int64(JYPPX_TensorRtLayer*, int64_t* out_alpha) { auto status = jyppx::tensorrt::validate_output_pointer(out_alpha, "out_alpha"); if (status != JYPPX_STATUS_OK) { return status; } *out_alpha = 0; return jyppx::tensorrt::report_vendor_missing(kLine, "fill alpha int64 query"); }
JYPPX_StatusCode jyppx_trt10_fill_layer_set_beta_int64(JYPPX_TensorRtLayer*, int64_t beta) { (void)beta; return jyppx::tensorrt::report_vendor_missing(kLine, "fill beta int64 set"); }
JYPPX_StatusCode jyppx_trt10_fill_layer_get_beta_int64(JYPPX_TensorRtLayer*, int64_t* out_beta) { auto status = jyppx::tensorrt::validate_output_pointer(out_beta, "out_beta"); if (status != JYPPX_STATUS_OK) { return status; } *out_beta = 0; return jyppx::tensorrt::report_vendor_missing(kLine, "fill beta int64 query"); }
JYPPX_StatusCode jyppx_trt10_fill_layer_is_alpha_beta_int64(JYPPX_TensorRtLayer*, JYPPX_Boolean* out_is_int64) { auto status = jyppx::tensorrt::validate_output_pointer(out_is_int64, "out_is_int64"); if (status != JYPPX_STATUS_OK) { return status; } *out_is_int64 = JYPPX_FALSE; return jyppx::tensorrt::report_vendor_missing(kLine, "fill alpha/beta type query"); }
JYPPX_StatusCode jyppx_trt10_cumulative_layer_set_operation(JYPPX_TensorRtLayer*, int32_t operation, JYPPX_Boolean* out_success) { (void)operation; auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success"); if (status != JYPPX_STATUS_OK) { return status; } *out_success = JYPPX_FALSE; return jyppx::tensorrt::report_vendor_missing(kLine, "cumulative operation set"); }
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_cumulative_layer_get_operation, out_operation, "cumulative operation query")
JYPPX_TRT10_VENDOR_MISSING_SET_BOOL(jyppx_trt10_cumulative_layer_set_exclusive, exclusive, "cumulative exclusive set")
JYPPX_TRT10_VENDOR_MISSING_GET_BOOL(jyppx_trt10_cumulative_layer_get_exclusive, out_exclusive, "cumulative exclusive query")
JYPPX_TRT10_VENDOR_MISSING_SET_BOOL(jyppx_trt10_cumulative_layer_set_reverse, reverse, "cumulative reverse set")
JYPPX_TRT10_VENDOR_MISSING_GET_BOOL(jyppx_trt10_cumulative_layer_get_reverse, out_reverse, "cumulative reverse query")

#undef JYPPX_TRT10_VENDOR_MISSING_LAYER_OUT
#undef JYPPX_TRT10_VENDOR_MISSING_GET_INT
#undef JYPPX_TRT10_VENDOR_MISSING_SET_INT
#undef JYPPX_TRT10_VENDOR_MISSING_GET_BOOL
#undef JYPPX_TRT10_VENDOR_MISSING_SET_BOOL
#endif
