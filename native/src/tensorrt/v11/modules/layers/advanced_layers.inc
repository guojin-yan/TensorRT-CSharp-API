JYPPX_StatusCode jyppx_trt11_network_mark_weights_refittable(
    JYPPX_TensorRtNetworkDefinition* network,
    const char* weights_name,
    JYPPX_Boolean* out_marked)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_marked, "out_marked");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = JYPPX_FALSE;
    status = validate_c_string(weights_name, "weights_name");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network refittable weights mark");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = network_payload->markWeightsRefittable(weights_name) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)network;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network refittable weights mark");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_unmark_weights_refittable(
    JYPPX_TensorRtNetworkDefinition* network,
    const char* weights_name,
    JYPPX_Boolean* out_unmarked)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_unmarked, "out_unmarked");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_unmarked = JYPPX_FALSE;
    status = validate_c_string(weights_name, "weights_name");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network refittable weights unmark");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_unmarked = network_payload->unmarkWeightsRefittable(weights_name) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)network;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network refittable weights unmark");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_are_weights_marked_refittable(
    JYPPX_TensorRtNetworkDefinition* network,
    const char* weights_name,
    JYPPX_Boolean* out_marked)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_marked, "out_marked");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = JYPPX_FALSE;
    status = validate_c_string(weights_name, "weights_name");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network refittable weights query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = network_payload->areWeightsMarkedRefittable(weights_name) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)network;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network refittable weights query");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_set_weights_name(
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
    nvinfer1::ILayer* layer_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network set weights name");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_layer_payload_for_api(constant_layer, &layer_payload, "network set weights name constant layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (layer_payload->getType() != nvinfer1::LayerType::kCONSTANT)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT network set weights name requires a constant layer.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* constant_payload = static_cast<nvinfer1::IConstantLayer*>(layer_payload);
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

JYPPX_StatusCode jyppx_trt11_network_add_cast(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* input_tensor,
    int32_t to_type,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network cast layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addCast(*input, static_cast<nvinfer1::DataType>(to_type)), out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)to_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network cast layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_non_zero(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* input_tensor,
    int32_t indices_type,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network non-zero layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addNonZero(*input, static_cast<nvinfer1::DataType>(indices_type)), out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)indices_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network non-zero layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_ragged_softmax(
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
    status = get_network_payload_for_api(network, &network_payload, "network ragged softmax layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(bounds_tensor, &bounds, "bounds_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addRaggedSoftMax(*input, *bounds), out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)bounds_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network ragged softmax layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_nms(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* boxes_tensor,
    JYPPX_TensorRtTensor* scores_tensor,
    JYPPX_TensorRtTensor* max_output_boxes_per_class_tensor,
    int32_t indices_type,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* boxes = nullptr;
    nvinfer1::ITensor* scores = nullptr;
    nvinfer1::ITensor* max_output = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network NMS layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(boxes_tensor, &boxes, "boxes_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(scores_tensor, &scores, "scores_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(max_output_boxes_per_class_tensor, &max_output, "max_output_boxes_per_class_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(
        network_payload->addNMS(*boxes, *scores, *max_output, static_cast<nvinfer1::DataType>(indices_type)),
        out_layer);
#else
    (void)network;
    (void)boxes_tensor;
    (void)scores_tensor;
    (void)max_output_boxes_per_class_tensor;
    (void)indices_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network NMS layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_reverse_sequence(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* input_tensor,
    JYPPX_TensorRtTensor* sequence_lengths_tensor,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* sequence_lengths = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network reverse sequence layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(sequence_lengths_tensor, &sequence_lengths, "sequence_lengths_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addReverseSequence(*input, *sequence_lengths), out_layer);
#else
    (void)network;
    (void)input_tensor;
    (void)sequence_lengths_tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network reverse sequence layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_einsum(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor** input_tensors,
    int32_t input_count,
    const char* equation,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (input_tensors == nullptr || input_count <= 0 || input_count > 16)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Einsum input tensor list must contain between 1 and 16 tensors.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    status = validate_c_string(equation, "equation");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network einsum layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::vector<nvinfer1::ITensor*> inputs;
    inputs.reserve(static_cast<size_t>(input_count));
    for (int32_t i = 0; i < input_count; ++i)
    {
        nvinfer1::ITensor* input = nullptr;
        status = get_tensor_payload_for_api(input_tensors[i], &input, "input_tensors");
        if (status != JYPPX_STATUS_OK) { return status; }
        inputs.push_back(input);
    }
    return create_layer_reference_handle(network_payload->addEinsum(inputs.data(), input_count, equation), out_layer);
#else
    (void)network;
    (void)input_tensors;
    (void)input_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network einsum layer creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_layer_set_input(JYPPX_TensorRtLayer* layer, int32_t index, JYPPX_TensorRtTensor* tensor)
{
    if (index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer input index must be non-negative.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILayer* payload = nullptr;
    auto status = get_layer_payload_for_api(layer, &payload, "layer input set");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::ITensor* input = nullptr;
    status = get_tensor_payload_for_api(tensor, &input, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (index > payload->getNbInputs())
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer input index must not exceed the current input count.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    payload->setInput(index, *input);
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer input set");
#endif
}

JYPPX_StatusCode jyppx_trt11_layer_set_metadata(JYPPX_TensorRtLayer* layer, const char* metadata)
{
    if (metadata == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer metadata must not be null. Use an empty string to clear metadata.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILayer* payload = nullptr;
    auto status = get_layer_payload_for_api(layer, &payload, "layer metadata set");
    if (status != JYPPX_STATUS_OK) { return status; }
    payload->setMetadata(metadata);
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer metadata set");
#endif
}

JYPPX_StatusCode jyppx_trt11_layer_get_metadata(
    JYPPX_TensorRtLayer* layer,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILayer* payload = nullptr;
    status = get_layer_payload_for_api(layer, &payload, "layer metadata query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(payload->getMetadata(), output_buffer, output_buffer_size, out_required_size);
#else
    (void)layer;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer metadata query");
#endif
}

JYPPX_StatusCode jyppx_trt11_layer_set_nb_ranks(JYPPX_TensorRtLayer* layer, int32_t rank_count, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
    if (rank_count <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer rank count must be positive.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILayer* payload = nullptr;
    status = get_layer_payload_for_api(layer, &payload, "layer rank count set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = payload->setNbRanks(rank_count) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    (void)rank_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer rank count set");
#endif
}

JYPPX_StatusCode jyppx_trt11_layer_get_nb_ranks(JYPPX_TensorRtLayer* layer, int32_t* out_rank_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_rank_count, "out_rank_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_rank_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILayer* payload = nullptr;
    status = get_layer_payload_for_api(layer, &payload, "layer rank count query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_rank_count = payload->getNbRanks();
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer rank count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_cast_layer_set_to_type(JYPPX_TensorRtLayer* layer, int32_t data_type)
{
    nvinfer1::ICastLayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kCAST, "cast", "cast layer output type set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setToType(static_cast<nvinfer1::DataType>(data_type));
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_cast_layer_get_to_type(JYPPX_TensorRtLayer* layer, int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = 0;
    nvinfer1::ICastLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kCAST, "cast", "cast layer output type query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = static_cast<int32_t>(typed_layer->getToType());
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_non_zero_layer_set_indices_type(JYPPX_TensorRtLayer* layer, int32_t data_type, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
    nvinfer1::INonZeroLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kNON_ZERO, "non-zero", "non-zero layer indices type set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = typed_layer->setIndicesType(static_cast<nvinfer1::DataType>(data_type)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_non_zero_layer_get_indices_type(JYPPX_TensorRtLayer* layer, int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = 0;
    nvinfer1::INonZeroLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kNON_ZERO, "non-zero", "non-zero layer indices type query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = static_cast<int32_t>(typed_layer->getIndicesType());
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_nms_layer_set_bounding_box_format(JYPPX_TensorRtLayer* layer, int32_t format)
{
    nvinfer1::INMSLayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kNMS, "NMS", "NMS layer bounding box format set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setBoundingBoxFormat(static_cast<nvinfer1::BoundingBoxFormat>(format));
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_nms_layer_get_bounding_box_format(JYPPX_TensorRtLayer* layer, int32_t* out_format)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_format, "out_format");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_format = 0;
    nvinfer1::INMSLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kNMS, "NMS", "NMS layer bounding box format query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_format = static_cast<int32_t>(typed_layer->getBoundingBoxFormat());
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_nms_layer_set_topk_box_limit(JYPPX_TensorRtLayer* layer, int32_t limit)
{
    if (limit <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "NMS TopK box limit must be positive.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    nvinfer1::INMSLayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kNMS, "NMS", "NMS layer TopK box limit set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setTopKBoxLimit(limit);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_nms_layer_get_topk_box_limit(JYPPX_TensorRtLayer* layer, int32_t* out_limit)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_limit, "out_limit");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_limit = 0;
    nvinfer1::INMSLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kNMS, "NMS", "NMS layer TopK box limit query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_limit = typed_layer->getTopKBoxLimit();
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode set_nms_tensor_input(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtTensor* tensor, int32_t index, const char* feature_name)
{
    nvinfer1::INMSLayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kNMS, "NMS", feature_name, &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::ITensor* input = nullptr;
    status = get_tensor_payload_for_api(tensor, &input, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setInput(index, *input);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_nms_layer_set_iou_threshold_tensor(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtTensor* tensor)
{
    return set_nms_tensor_input(layer, tensor, 3, "NMS layer IoU threshold tensor set");
}

JYPPX_StatusCode jyppx_trt11_nms_layer_set_score_threshold_tensor(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtTensor* tensor)
{
    return set_nms_tensor_input(layer, tensor, 4, "NMS layer score threshold tensor set");
}

JYPPX_StatusCode jyppx_trt11_nms_layer_set_indices_type(JYPPX_TensorRtLayer* layer, int32_t data_type, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
    nvinfer1::INMSLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kNMS, "NMS", "NMS layer indices type set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = typed_layer->setIndicesType(static_cast<nvinfer1::DataType>(data_type)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_nms_layer_get_indices_type(JYPPX_TensorRtLayer* layer, int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = 0;
    nvinfer1::INMSLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kNMS, "NMS", "NMS layer indices type query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = static_cast<int32_t>(typed_layer->getIndicesType());
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_reverse_sequence_layer_set_batch_axis(JYPPX_TensorRtLayer* layer, int32_t axis)
{
    if (axis < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::IReverseSequenceLayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kREVERSE_SEQUENCE, "reverse sequence", "reverse sequence layer batch axis set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setBatchAxis(axis);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_reverse_sequence_layer_get_batch_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_axis, "out_axis");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_axis = 0;
    nvinfer1::IReverseSequenceLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kREVERSE_SEQUENCE, "reverse sequence", "reverse sequence layer batch axis query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_axis = typed_layer->getBatchAxis();
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_reverse_sequence_layer_set_sequence_axis(JYPPX_TensorRtLayer* layer, int32_t axis)
{
    if (axis < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::IReverseSequenceLayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kREVERSE_SEQUENCE, "reverse sequence", "reverse sequence layer sequence axis set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setSequenceAxis(axis);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt11_reverse_sequence_layer_get_sequence_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_axis, "out_axis");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_axis = 0;
    nvinfer1::IReverseSequenceLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kREVERSE_SEQUENCE, "reverse sequence", "reverse sequence layer sequence axis query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_axis = typed_layer->getSequenceAxis();
    return JYPPX_STATUS_OK;
}
