static JYPPX_StatusCode jyppx_trt8_report_legacy_network_api_unavailable(const char* feature_name)
{
#if JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, feature_name);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, feature_name);
#endif
}

JYPPX_StatusCode jyppx_trt8_network_add_fully_connected(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* input_tensor,
    int32_t nb_outputs,
    int32_t data_type,
    const void* kernel_values,
    size_t kernel_value_count,
    const void* bias_values,
    size_t bias_value_count,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (nb_outputs <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Fully connected output channel count must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(input_tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    std::vector<uint8_t> kernel_blob;
    status = copy_optional_weights(data_type, kernel_values, kernel_value_count, "Fully connected kernel", true, &kernel_blob);
    if (status != JYPPX_STATUS_OK) { return status; }

    std::vector<uint8_t> bias_blob;
    status = copy_optional_weights(data_type, bias_values, bias_value_count, "Fully connected bias", false, &bias_blob);
    if (status != JYPPX_STATUS_OK) { return status; }

    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(input_tensor);
    if (network_payload == nullptr || tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Network or tensor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::Weights kernel_weights{static_cast<nvinfer1::DataType>(data_type), kernel_blob.data(), static_cast<int64_t>(kernel_value_count)};
    nvinfer1::Weights bias_weights{static_cast<nvinfer1::DataType>(data_type), bias_blob.empty() ? nullptr : bias_blob.data(), static_cast<int64_t>(bias_value_count)};
    nvinfer1::IFullyConnectedLayer* layer = network_payload->addFullyConnected(*tensor_payload, nb_outputs, kernel_weights, bias_weights);

    std::vector<std::vector<uint8_t>> owned_blobs;
    owned_blobs.push_back(std::move(kernel_blob));
    if (!bias_blob.empty()) { owned_blobs.push_back(std::move(bias_blob)); }
    return create_layer_reference_handle(layer, out_layer, {}, -1, 0, 0, 0, -1, -1, 0, 0, 0, std::move(owned_blobs));
#else
    (void)network;
    (void)input_tensor;
    (void)data_type;
    (void)kernel_values;
    (void)kernel_value_count;
    (void)bias_values;
    (void)bias_value_count;
    return jyppx_trt8_report_legacy_network_api_unavailable("network add fully connected");
#endif
}

JYPPX_StatusCode jyppx_trt8_network_add_gather_v2(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* data_tensor,
    JYPPX_TensorRtTensor* indices_tensor,
    int32_t mode,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (mode < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }

#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* data = nullptr;
    nvinfer1::ITensor* indices = nullptr;
    status = jyppx_trt8_get_network_payload_for_api(network, &network_payload, "network gather v2 layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt8_get_tensor_payload_for_api(data_tensor, &data, "data_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt8_get_tensor_payload_for_api(indices_tensor, &indices, "indices_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addGatherV2(*data, *indices, static_cast<nvinfer1::GatherMode>(mode)), out_layer);
#else
    (void)network;
    (void)data_tensor;
    (void)indices_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network gather v2 layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt8_network_add_ragged_softmax(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* input_tensor,
    JYPPX_TensorRtTensor* bounds_tensor,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* bounds = nullptr;
    status = jyppx_trt8_get_network_payload_for_api(network, &network_payload, "network ragged softmax layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt8_get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt8_get_tensor_payload_for_api(bounds_tensor, &bounds, "bounds_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addRaggedSoftMax(*input, *bounds), out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)bounds_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network ragged softmax layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt8_network_add_parametric_relu(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* input_tensor,
    JYPPX_TensorRtTensor* slope_tensor,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* slope = nullptr;
    status = jyppx_trt8_get_network_payload_for_api(network, &network_payload, "network parametric ReLU layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt8_get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt8_get_tensor_payload_for_api(slope_tensor, &slope, "slope_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addParametricReLU(*input, *slope), out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)slope_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network parametric ReLU layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt8_network_add_non_zero(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* input_tensor,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;

#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = jyppx_trt8_get_network_payload_for_api(network, &network_payload, "network non-zero layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt8_get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addNonZero(*input), out_layer);
#else
    (void)network;
    (void)input_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network non-zero layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt8_network_mark_output_for_shapes(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* tensor,
    JYPPX_Boolean* out_marked)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_marked, "out_marked");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = jyppx_trt8_get_network_payload_for_api(network, &network_payload, "network shape output mark");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt8_get_tensor_payload_for_api(tensor, &tensor_payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = network_payload->markOutputForShapes(*tensor_payload) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)network;
    (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network shape output mark");
#endif
}

JYPPX_StatusCode jyppx_trt8_network_unmark_output_for_shapes(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* tensor,
    JYPPX_Boolean* out_unmarked)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_unmarked, "out_unmarked");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_unmarked = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = jyppx_trt8_get_network_payload_for_api(network, &network_payload, "network shape output unmark");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt8_get_tensor_payload_for_api(tensor, &tensor_payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_unmarked = network_payload->unmarkOutputForShapes(*tensor_payload) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)network;
    (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network shape output unmark");
#endif
}

JYPPX_StatusCode jyppx_trt8_network_has_explicit_precision(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_Boolean* out_has_explicit_precision)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_explicit_precision, "out_has_explicit_precision");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_explicit_precision = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = jyppx_trt8_get_network_payload_for_api(network, &network_payload, "network explicit precision query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_explicit_precision = network_payload->hasExplicitPrecision() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)network;
    return jyppx_trt8_report_legacy_network_api_unavailable("network explicit precision query");
#endif
}

JYPPX_StatusCode jyppx_trt8_network_set_weights_name(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtLayer* constant_layer,
    const char* weights_name,
    JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;
    status = validate_c_string(weights_name, "weights_name");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::IConstantLayer* constant_payload = nullptr;
    status = jyppx_trt8_get_network_payload_for_api(network, &network_payload, "network set weights name");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt8_get_typed_layer_for_api(constant_layer, nvinfer1::LayerType::kCONSTANT, "constant", "network set weights name constant layer", &constant_payload);
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::Weights weights = constant_payload->getWeights();
    if (weights.values == nullptr || weights.count <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT constant layer weights are null or empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_set = network_payload->setWeightsName(weights, weights_name) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)network;
    (void)constant_layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network set weights name");
#endif
}

JYPPX_StatusCode jyppx_trt8_fully_connected_layer_get_nb_output_channels(
    JYPPX_TensorRtLayer* layer,
    int32_t* out_channels)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_channels, "out_channels");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_channels = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IFullyConnectedLayer* fully_connected_layer = nullptr;
    status = jyppx_trt8_get_typed_layer_for_api(layer, nvinfer1::LayerType::kFULLY_CONNECTED, "fully connected", "fully connected output channel count query", &fully_connected_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_channels = fully_connected_layer->getNbOutputChannels();
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx_trt8_report_legacy_network_api_unavailable("fully connected output channel count query");
#endif
}

JYPPX_StatusCode jyppx_trt8_fully_connected_layer_set_nb_output_channels(
    JYPPX_TensorRtLayer* layer,
    int32_t channels)
{
    if (channels <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Fully connected output channel count must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IFullyConnectedLayer* fully_connected_layer = nullptr;
    auto status = jyppx_trt8_get_typed_layer_for_api(layer, nvinfer1::LayerType::kFULLY_CONNECTED, "fully connected", "fully connected output channel count set", &fully_connected_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    fully_connected_layer->setNbOutputChannels(channels);
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx_trt8_report_legacy_network_api_unavailable("fully connected output channel count set");
#endif
}

JYPPX_StatusCode jyppx_trt8_fully_connected_layer_get_kernel_weights_info(
    JYPPX_TensorRtLayer* layer,
    JYPPX_TensorRtWeightsInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IFullyConnectedLayer* fully_connected_layer = nullptr;
    status = jyppx_trt8_get_typed_layer_for_api(layer, nvinfer1::LayerType::kFULLY_CONNECTED, "fully connected", "fully connected kernel weights info query", &fully_connected_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    jyppx_trt8_copy_layer_weights_info(fully_connected_layer->getKernelWeights(), out_info);
    return JYPPX_STATUS_OK;
#else
    out_info->data_type = -1;
    out_info->count = 0;
    out_info->has_values = JYPPX_FALSE;
    (void)layer;
    return jyppx_trt8_report_legacy_network_api_unavailable("fully connected kernel weights info query");
#endif
}

JYPPX_StatusCode jyppx_trt8_fully_connected_layer_get_bias_weights_info(
    JYPPX_TensorRtLayer* layer,
    JYPPX_TensorRtWeightsInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IFullyConnectedLayer* fully_connected_layer = nullptr;
    status = jyppx_trt8_get_typed_layer_for_api(layer, nvinfer1::LayerType::kFULLY_CONNECTED, "fully connected", "fully connected bias weights info query", &fully_connected_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    jyppx_trt8_copy_layer_weights_info(fully_connected_layer->getBiasWeights(), out_info);
    return JYPPX_STATUS_OK;
#else
    out_info->data_type = -1;
    out_info->count = 0;
    out_info->has_values = JYPPX_FALSE;
    (void)layer;
    return jyppx_trt8_report_legacy_network_api_unavailable("fully connected bias weights info query");
#endif
}

static JYPPX_StatusCode jyppx_trt8_set_fully_connected_weights(
    JYPPX_TensorRtLayer* layer,
    int32_t data_type,
    const void* values,
    size_t value_count,
    const bool required,
    const char* weights_name)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    std::vector<uint8_t> blob;
    auto status = copy_optional_weights(data_type, values, value_count, weights_name, required, &blob);
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::IFullyConnectedLayer* fully_connected_layer = nullptr;
    status = jyppx_trt8_get_typed_layer_for_api(layer, nvinfer1::LayerType::kFULLY_CONNECTED, "fully connected", weights_name, &fully_connected_layer);
    if (status != JYPPX_STATUS_OK) { return status; }

    auto* payload = get_layer_reference_payload(layer);
    if (payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle does not carry the expected TensorRT layer reference payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::Weights weights{static_cast<nvinfer1::DataType>(data_type), blob.empty() ? nullptr : blob.data(), static_cast<int64_t>(value_count)};
    if (required)
    {
        fully_connected_layer->setKernelWeights(weights);
    }
    else
    {
        fully_connected_layer->setBiasWeights(weights);
    }

    if (!blob.empty()) { payload->owned_weight_blobs.push_back(std::move(blob)); }
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    (void)data_type;
    (void)values;
    (void)value_count;
    (void)required;
    return jyppx_trt8_report_legacy_network_api_unavailable(weights_name);
#endif
}

JYPPX_StatusCode jyppx_trt8_fully_connected_layer_set_kernel_weights(
    JYPPX_TensorRtLayer* layer,
    int32_t data_type,
    const void* values,
    size_t value_count)
{
    return jyppx_trt8_set_fully_connected_weights(layer, data_type, values, value_count, true, "fully connected kernel weights set");
}

JYPPX_StatusCode jyppx_trt8_fully_connected_layer_set_bias_weights(
    JYPPX_TensorRtLayer* layer,
    int32_t data_type,
    const void* values,
    size_t value_count)
{
    return jyppx_trt8_set_fully_connected_weights(layer, data_type, values, value_count, false, "fully connected bias weights set");
}
