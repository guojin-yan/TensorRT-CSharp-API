#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
JYPPX_StatusCode jyppx_trt10_network_add_gather_v2(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* data_tensor, JYPPX_TensorRtTensor* indices_tensor, int32_t mode, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (mode < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* data = nullptr;
    nvinfer1::ITensor* indices = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network gather v2 layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(data_tensor, &data, "data_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(indices_tensor, &indices, "indices_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addGatherV2(*data, *indices, static_cast<nvinfer1::GatherMode>(mode)), out_layer);
}

JYPPX_StatusCode jyppx_trt10_network_add_ragged_softmax(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* bounds_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* bounds = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network ragged softmax layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(bounds_tensor, &bounds, "bounds_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addRaggedSoftMax(*input, *bounds), out_layer);
}

JYPPX_StatusCode jyppx_trt10_network_add_fill_v2(JYPPX_TensorRtNetworkDefinition* network, const JYPPX_TensorRtDims* dimensions, int32_t operation, int32_t output_type, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (operation < 0 || output_type < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::Dims native_dimensions{};
    status = make_dims(dimensions, &native_dimensions, "dimensions");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network fill v2 layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(
        network_payload->addFill(native_dimensions, static_cast<nvinfer1::FillOperation>(operation), static_cast<nvinfer1::DataType>(output_type)),
        out_layer);
}

JYPPX_StatusCode jyppx_trt10_network_add_quantize_v2(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* scale_tensor, int32_t output_type, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (output_type < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* scale = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network quantize v2 layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(scale_tensor, &scale, "scale_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addQuantize(*input, *scale, static_cast<nvinfer1::DataType>(output_type)), out_layer);
}

JYPPX_StatusCode jyppx_trt10_network_add_dequantize_v2(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* scale_tensor, int32_t output_type, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (output_type < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* scale = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network dequantize v2 layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(scale_tensor, &scale, "scale_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addDequantize(*input, *scale, static_cast<nvinfer1::DataType>(output_type)), out_layer);
}

JYPPX_StatusCode jyppx_trt10_network_add_parametric_relu(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* slope_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* slope = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network parametric ReLU layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(slope_tensor, &slope, "slope_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addParametricReLU(*input, *slope), out_layer);
}

JYPPX_StatusCode jyppx_trt10_network_add_non_zero(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network non-zero layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addNonZero(*input), out_layer);
}

JYPPX_StatusCode jyppx_trt10_network_add_squeeze(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* axes_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* axes = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network squeeze layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(axes_tensor, &axes, "axes_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addSqueeze(*input, *axes), out_layer);
}

JYPPX_StatusCode jyppx_trt10_network_add_unsqueeze(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* axes_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* axes = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network unsqueeze layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(axes_tensor, &axes, "axes_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addUnsqueeze(*input, *axes), out_layer);
}

JYPPX_StatusCode jyppx_trt10_network_mark_debug(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_marked)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_marked, "out_marked");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = JYPPX_FALSE;
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network debug tensor mark");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(tensor, &tensor_payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = network_payload->markDebug(*tensor_payload) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt10_network_unmark_debug(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_unmarked)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_unmarked, "out_unmarked");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_unmarked = JYPPX_FALSE;
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network debug tensor unmark");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(tensor, &tensor_payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_unmarked = network_payload->unmarkDebug(*tensor_payload) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt10_network_is_debug_tensor(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_is_debug)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_debug, "out_is_debug");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_debug = JYPPX_FALSE;
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network debug tensor query");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(tensor, &tensor_payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_debug = network_payload->isDebugTensor(*tensor_payload) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt10_network_mark_output_for_shapes(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_marked)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_marked, "out_marked");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = JYPPX_FALSE;
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network shape output mark");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(tensor, &tensor_payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = network_payload->markOutputForShapes(*tensor_payload) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt10_network_unmark_output_for_shapes(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_unmarked)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_unmarked, "out_unmarked");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_unmarked = JYPPX_FALSE;
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network shape output unmark");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(tensor, &tensor_payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_unmarked = network_payload->unmarkOutputForShapes(*tensor_payload) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt10_network_mark_weights_refittable(JYPPX_TensorRtNetworkDefinition* network, const char* weights_name, JYPPX_Boolean* out_marked)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_marked, "out_marked");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = JYPPX_FALSE;
    status = validate_c_string(weights_name, "weights_name");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network refittable weights mark");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = network_payload->markWeightsRefittable(weights_name) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt10_network_unmark_weights_refittable(JYPPX_TensorRtNetworkDefinition* network, const char* weights_name, JYPPX_Boolean* out_unmarked)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_unmarked, "out_unmarked");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_unmarked = JYPPX_FALSE;
    status = validate_c_string(weights_name, "weights_name");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network refittable weights unmark");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_unmarked = network_payload->unmarkWeightsRefittable(weights_name) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt10_network_are_weights_marked_refittable(JYPPX_TensorRtNetworkDefinition* network, const char* weights_name, JYPPX_Boolean* out_marked)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_marked, "out_marked");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = JYPPX_FALSE;
    status = validate_c_string(weights_name, "weights_name");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network refittable weights query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = network_payload->areWeightsMarkedRefittable(weights_name) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt10_network_set_weights_name(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtLayer* constant_layer, const char* weights_name, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;
    status = validate_c_string(weights_name, "weights_name");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::IConstantLayer* constant_payload = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network set weights name");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_typed_layer_for_api(constant_layer, nvinfer1::LayerType::kCONSTANT, "constant", "network set weights name constant layer", &constant_payload);
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::Weights weights = constant_payload->getWeights();
    if (weights.values == nullptr || weights.count <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT constant layer weights are null or empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_set = network_payload->setWeightsName(weights, weights_name) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}
#else
static JYPPX_StatusCode trt10_report_missing_layer_out(JYPPX_TensorRtLayer** out_layer, const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    return jyppx::tensorrt::report_vendor_missing(kLine, feature_name);
}

static JYPPX_StatusCode trt10_report_missing_bool_out(JYPPX_Boolean* out_value, const char* out_name, const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_value, out_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_value = JYPPX_FALSE;
    return jyppx::tensorrt::report_vendor_missing(kLine, feature_name);
}

JYPPX_StatusCode jyppx_trt10_network_add_gather_v2(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, int32_t, JYPPX_TensorRtLayer** out_layer)
{
    return trt10_report_missing_layer_out(out_layer, "network gather v2 layer creation");
}

JYPPX_StatusCode jyppx_trt10_network_add_ragged_softmax(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, JYPPX_TensorRtLayer** out_layer)
{
    return trt10_report_missing_layer_out(out_layer, "network ragged softmax layer creation");
}

JYPPX_StatusCode jyppx_trt10_network_add_fill_v2(JYPPX_TensorRtNetworkDefinition*, const JYPPX_TensorRtDims*, int32_t, int32_t, JYPPX_TensorRtLayer** out_layer)
{
    return trt10_report_missing_layer_out(out_layer, "network fill v2 layer creation");
}

JYPPX_StatusCode jyppx_trt10_network_add_quantize_v2(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, int32_t, JYPPX_TensorRtLayer** out_layer)
{
    return trt10_report_missing_layer_out(out_layer, "network quantize v2 layer creation");
}

JYPPX_StatusCode jyppx_trt10_network_add_dequantize_v2(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, int32_t, JYPPX_TensorRtLayer** out_layer)
{
    return trt10_report_missing_layer_out(out_layer, "network dequantize v2 layer creation");
}

JYPPX_StatusCode jyppx_trt10_network_add_parametric_relu(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, JYPPX_TensorRtLayer** out_layer)
{
    return trt10_report_missing_layer_out(out_layer, "network parametric ReLU layer creation");
}

JYPPX_StatusCode jyppx_trt10_network_add_non_zero(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_TensorRtLayer** out_layer)
{
    return trt10_report_missing_layer_out(out_layer, "network non-zero layer creation");
}

JYPPX_StatusCode jyppx_trt10_network_add_squeeze(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, JYPPX_TensorRtLayer** out_layer)
{
    return trt10_report_missing_layer_out(out_layer, "network squeeze layer creation");
}

JYPPX_StatusCode jyppx_trt10_network_add_unsqueeze(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, JYPPX_TensorRtLayer** out_layer)
{
    return trt10_report_missing_layer_out(out_layer, "network unsqueeze layer creation");
}

JYPPX_StatusCode jyppx_trt10_network_mark_debug(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_Boolean* out_marked)
{
    return trt10_report_missing_bool_out(out_marked, "out_marked", "network debug tensor mark");
}

JYPPX_StatusCode jyppx_trt10_network_unmark_debug(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_Boolean* out_unmarked)
{
    return trt10_report_missing_bool_out(out_unmarked, "out_unmarked", "network debug tensor unmark");
}

JYPPX_StatusCode jyppx_trt10_network_is_debug_tensor(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_Boolean* out_is_debug)
{
    return trt10_report_missing_bool_out(out_is_debug, "out_is_debug", "network debug tensor query");
}

JYPPX_StatusCode jyppx_trt10_network_mark_output_for_shapes(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_Boolean* out_marked)
{
    return trt10_report_missing_bool_out(out_marked, "out_marked", "network shape output mark");
}

JYPPX_StatusCode jyppx_trt10_network_unmark_output_for_shapes(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_Boolean* out_unmarked)
{
    return trt10_report_missing_bool_out(out_unmarked, "out_unmarked", "network shape output unmark");
}

JYPPX_StatusCode jyppx_trt10_network_mark_weights_refittable(JYPPX_TensorRtNetworkDefinition*, const char*, JYPPX_Boolean* out_marked)
{
    return trt10_report_missing_bool_out(out_marked, "out_marked", "network refittable weights mark");
}

JYPPX_StatusCode jyppx_trt10_network_unmark_weights_refittable(JYPPX_TensorRtNetworkDefinition*, const char*, JYPPX_Boolean* out_unmarked)
{
    return trt10_report_missing_bool_out(out_unmarked, "out_unmarked", "network refittable weights unmark");
}

JYPPX_StatusCode jyppx_trt10_network_are_weights_marked_refittable(JYPPX_TensorRtNetworkDefinition*, const char*, JYPPX_Boolean* out_marked)
{
    return trt10_report_missing_bool_out(out_marked, "out_marked", "network refittable weights query");
}

JYPPX_StatusCode jyppx_trt10_network_set_weights_name(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtLayer*, const char*, JYPPX_Boolean* out_set)
{
    return trt10_report_missing_bool_out(out_set, "out_set", "network set weights name");
}
#endif

JYPPX_StatusCode jyppx_trt10_network_add_plugin_v2_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "INetworkDefinition::addPluginV2 deferred: plugin object ownership and pointer-array lifetime require a dedicated safe plugin bridge");
}

JYPPX_StatusCode jyppx_trt10_network_add_plugin_v3_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "INetworkDefinition::addPluginV3 deferred: plugin capability object ownership requires a dedicated safe plugin bridge");
}

JYPPX_StatusCode jyppx_trt10_network_get_error_recorder_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "INetworkDefinition::getErrorRecorder deferred: borrowed callback ownership is not safe across the C ABI");
}

JYPPX_StatusCode jyppx_trt10_network_set_error_recorder_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "INetworkDefinition::setErrorRecorder deferred: callback reference counting and lifetime require a managed recorder bridge");
}

JYPPX_StatusCode jyppx_trt10_plugin_v2_layer_get_plugin_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginV2Layer::getPlugin deferred: borrowed plugin pointer ownership is not safe across the C ABI");
}

JYPPX_StatusCode jyppx_trt10_plugin_v3_layer_get_plugin_deferred(void)
{
    return jyppx::tensorrt::report_not_implemented(kLine, "IPluginV3Layer::getPlugin deferred: borrowed plugin pointer ownership is not safe across the C ABI");
}
