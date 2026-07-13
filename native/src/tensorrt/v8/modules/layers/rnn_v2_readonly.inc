#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
static JYPPX_StatusCode create_optional_rnn_v2_tensor_reference(
    nvinfer1::ITensor* tensor,
    JYPPX_TensorRtTensor** out_tensor)
{
    *out_tensor = nullptr;
    return tensor == nullptr
        ? JYPPX_STATUS_OK
        : create_tensor_reference_handle(tensor, out_tensor);
}

static JYPPX_StatusCode validate_rnn_v2_gate_request(
    nvinfer1::IRNNv2Layer* rnn_layer,
    const int32_t layer_index,
    const int32_t gate,
    const JYPPX_Boolean is_w)
{
    if (layer_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "RNNv2 gate layer index must be non-negative.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const int32_t direction_multiplier =
        rnn_layer->getDirection() == nvinfer1::RNNDirection::kBIDIRECTION ? 2 : 1;
    const int32_t layer_count = rnn_layer->getLayerCount();
    if (layer_count < 0 || layer_count > INT32_MAX / direction_multiplier)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "RNNv2 gate layer count is invalid.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    if (layer_index >= layer_count * direction_multiplier)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "RNNv2 gate layer index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (gate < static_cast<int32_t>(nvinfer1::RNNGateType::kINPUT) ||
        gate > static_cast<int32_t>(nvinfer1::RNNGateType::kHIDDEN))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "RNNv2 gate value is outside the supported enum range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (is_w != JYPPX_FALSE && is_w != JYPPX_TRUE)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "RNNv2 is_w must be a valid bridge boolean.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode copy_rnn_v2_gate_weights(
    const nvinfer1::Weights& weights,
    JYPPX_TensorRtWeightsInfo* out_info,
    uint8_t* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size)
{
    out_info->data_type = static_cast<int32_t>(weights.type);
    out_info->count = weights.count;
    out_info->has_values = weights.values != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    *out_required_size = 0;

    if (weights.count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "RNNv2 gate weights returned a negative element count.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    if (weights.count == 0)
    {
        return JYPPX_STATUS_OK;
    }

    if (weights.values == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "RNNv2 gate weights returned a null value pointer for a non-empty result.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const size_t element_size = get_data_type_size(static_cast<int32_t>(weights.type));
    if (element_size == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "RNNv2 gate weights returned an unsupported data type.");
        return JYPPX_STATUS_NOT_SUPPORTED;
    }

    const uint64_t element_count = static_cast<uint64_t>(weights.count);
    if (element_count > static_cast<uint64_t>(SIZE_MAX / element_size))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "RNNv2 gate weight byte count overflow.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const size_t required_size = static_cast<size_t>(element_count) * element_size;
    *out_required_size = required_size;
    if (output_buffer == nullptr || output_buffer_size == 0)
    {
        return JYPPX_STATUS_OK;
    }

    if (output_buffer_size < required_size)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "RNNv2 gate weight output buffer is too small.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }

    std::memcpy(output_buffer, weights.values, required_size);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_layer_count(JYPPX_TensorRtLayer* layer, int32_t* out_layer_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer_count, "out_layer_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer_count = 0;
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* rnn_layer = get_rnn_v2_layer_payload(layer);
    if (rnn_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT RNNv2 layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_layer_count = rnn_layer->getLayerCount();
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_hidden_size(JYPPX_TensorRtLayer* layer, int32_t* out_hidden_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_hidden_size, "out_hidden_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_hidden_size = 0;
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* rnn_layer = get_rnn_v2_layer_payload(layer);
    if (rnn_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT RNNv2 layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_hidden_size = rnn_layer->getHiddenSize();
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_data_length(JYPPX_TensorRtLayer* layer, int32_t* out_data_length)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_length, "out_data_length");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_length = 0;
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* rnn_layer = get_rnn_v2_layer_payload(layer);
    if (rnn_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT RNNv2 layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_data_length = rnn_layer->getDataLength();
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_max_seq_length(JYPPX_TensorRtLayer* layer, int32_t* out_max_seq_length)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_max_seq_length, "out_max_seq_length");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_max_seq_length = 0;
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* rnn_layer = get_rnn_v2_layer_payload(layer);
    if (rnn_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT RNNv2 layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_max_seq_length = rnn_layer->getMaxSeqLength();
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_operation(JYPPX_TensorRtLayer* layer, int32_t* out_operation)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_operation, "out_operation");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_operation = 0;
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* rnn_layer = get_rnn_v2_layer_payload(layer);
    if (rnn_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT RNNv2 layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_operation = static_cast<int32_t>(rnn_layer->getOperation());
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_direction(JYPPX_TensorRtLayer* layer, int32_t* out_direction)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_direction, "out_direction");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_direction = 0;
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* rnn_layer = get_rnn_v2_layer_payload(layer);
    if (rnn_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT RNNv2 layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_direction = static_cast<int32_t>(rnn_layer->getDirection());
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_input_mode(JYPPX_TensorRtLayer* layer, int32_t* out_input_mode)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_input_mode, "out_input_mode");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_input_mode = 0;
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* rnn_layer = get_rnn_v2_layer_payload(layer);
    if (rnn_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT RNNv2 layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_input_mode = static_cast<int32_t>(rnn_layer->getInputMode());
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_cell_state(
    JYPPX_TensorRtLayer* layer,
    JYPPX_TensorRtTensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = nullptr;
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* rnn_layer = get_rnn_v2_layer_payload(layer);
    if (rnn_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT RNNv2 layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return create_optional_rnn_v2_tensor_reference(rnn_layer->getCellState(), out_tensor);
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_hidden_state(
    JYPPX_TensorRtLayer* layer,
    JYPPX_TensorRtTensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = nullptr;
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* rnn_layer = get_rnn_v2_layer_payload(layer);
    if (rnn_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT RNNv2 layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return create_optional_rnn_v2_tensor_reference(rnn_layer->getHiddenState(), out_tensor);
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_sequence_lengths(
    JYPPX_TensorRtLayer* layer,
    JYPPX_TensorRtTensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = nullptr;
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* rnn_layer = get_rnn_v2_layer_payload(layer);
    if (rnn_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT RNNv2 layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return create_optional_rnn_v2_tensor_reference(rnn_layer->getSequenceLengths(), out_tensor);
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_weights_for_gate_copy(
    JYPPX_TensorRtLayer* layer,
    const int32_t layer_index,
    const int32_t gate,
    const JYPPX_Boolean is_w,
    JYPPX_TensorRtWeightsInfo* out_info,
    uint8_t* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    out_info->data_type = -1;
    out_info->count = 0;
    out_info->has_values = JYPPX_FALSE;
    *out_required_size = 0;
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* rnn_layer = get_rnn_v2_layer_payload(layer);
    if (rnn_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT RNNv2 layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    status = validate_rnn_v2_gate_request(rnn_layer, layer_index, gate, is_w);
    if (status != JYPPX_STATUS_OK) { return status; }

    const nvinfer1::Weights weights = rnn_layer->getWeightsForGate(
        layer_index,
        static_cast<nvinfer1::RNNGateType>(gate),
        is_w == JYPPX_TRUE);
    return copy_rnn_v2_gate_weights(weights, out_info, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_bias_for_gate_copy(
    JYPPX_TensorRtLayer* layer,
    const int32_t layer_index,
    const int32_t gate,
    const JYPPX_Boolean is_w,
    JYPPX_TensorRtWeightsInfo* out_info,
    uint8_t* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    out_info->data_type = -1;
    out_info->count = 0;
    out_info->has_values = JYPPX_FALSE;
    *out_required_size = 0;
    status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* rnn_layer = get_rnn_v2_layer_payload(layer);
    if (rnn_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT RNNv2 layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    status = validate_rnn_v2_gate_request(rnn_layer, layer_index, gate, is_w);
    if (status != JYPPX_STATUS_OK) { return status; }

    const nvinfer1::Weights weights = rnn_layer->getBiasForGate(
        layer_index,
        static_cast<nvinfer1::RNNGateType>(gate),
        is_w == JYPPX_TRUE);
    return copy_rnn_v2_gate_weights(weights, out_info, output_buffer, output_buffer_size, out_required_size);
}
#else
JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_layer_count(JYPPX_TensorRtLayer* layer, int32_t* out_layer_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer_count, "out_layer_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer_count = 0;
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "RNNv2 layer count query");
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_hidden_size(JYPPX_TensorRtLayer* layer, int32_t* out_hidden_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_hidden_size, "out_hidden_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_hidden_size = 0;
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "RNNv2 hidden size query");
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_data_length(JYPPX_TensorRtLayer* layer, int32_t* out_data_length)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_length, "out_data_length");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_length = 0;
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "RNNv2 data length query");
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_max_seq_length(JYPPX_TensorRtLayer* layer, int32_t* out_max_seq_length)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_max_seq_length, "out_max_seq_length");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_max_seq_length = 0;
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "RNNv2 max sequence length query");
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_operation(JYPPX_TensorRtLayer* layer, int32_t* out_operation)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_operation, "out_operation");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_operation = 0;
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "RNNv2 operation query");
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_direction(JYPPX_TensorRtLayer* layer, int32_t* out_direction)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_direction, "out_direction");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_direction = 0;
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "RNNv2 direction query");
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_input_mode(JYPPX_TensorRtLayer* layer, int32_t* out_input_mode)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_input_mode, "out_input_mode");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_input_mode = 0;
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "RNNv2 input mode query");
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_cell_state(
    JYPPX_TensorRtLayer* layer,
    JYPPX_TensorRtTensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = nullptr;
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "RNNv2 cell state query");
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_hidden_state(
    JYPPX_TensorRtLayer* layer,
    JYPPX_TensorRtTensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = nullptr;
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "RNNv2 hidden state query");
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_sequence_lengths(
    JYPPX_TensorRtLayer* layer,
    JYPPX_TensorRtTensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor, "out_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = nullptr;
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "RNNv2 sequence lengths query");
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_weights_for_gate_copy(
    JYPPX_TensorRtLayer* layer,
    int32_t layer_index,
    int32_t gate,
    JYPPX_Boolean is_w,
    JYPPX_TensorRtWeightsInfo* out_info,
    uint8_t* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    out_info->data_type = -1;
    out_info->count = 0;
    out_info->has_values = JYPPX_FALSE;
    *out_required_size = 0;
    (void)layer;
    (void)layer_index;
    (void)gate;
    (void)is_w;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "RNNv2 gate weights copy");
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_get_bias_for_gate_copy(
    JYPPX_TensorRtLayer* layer,
    int32_t layer_index,
    int32_t gate,
    JYPPX_Boolean is_w,
    JYPPX_TensorRtWeightsInfo* out_info,
    uint8_t* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    out_info->data_type = -1;
    out_info->count = 0;
    out_info->has_values = JYPPX_FALSE;
    *out_required_size = 0;
    (void)layer;
    (void)layer_index;
    (void)gate;
    (void)is_w;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "RNNv2 gate bias copy");
}
#endif
