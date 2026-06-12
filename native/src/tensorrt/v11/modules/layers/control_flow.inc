namespace
{
#if JYPPX_HAS_TENSORRT
JYPPX_StatusCode get_loop_payload_for_api(JYPPX_TensorRtLoop* loop, nvinfer1::ILoop** out_loop, const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(loop, kLine, JYPPX_TENSORRT_OBJECT_KIND_LOOP, "loop");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* payload = get_payload<LoopReferencePayload>(loop);
    if (payload == nullptr || payload->loop == nullptr)
    {
        std::ostringstream builder;
        builder << "Loop handle does not carry the expected TensorRT payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_loop = payload->loop;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode get_if_conditional_payload_for_api(JYPPX_TensorRtIfConditional* conditional, nvinfer1::IIfConditional** out_conditional, const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(conditional, kLine, JYPPX_TENSORRT_OBJECT_KIND_IF_CONDITIONAL, "conditional");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* payload = get_payload<IfConditionalReferencePayload>(conditional);
    if (payload == nullptr || payload->conditional == nullptr)
    {
        std::ostringstream builder;
        builder << "If-conditional handle does not carry the expected TensorRT payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_conditional = payload->conditional;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode get_tensor_payload_for_control_flow(JYPPX_TensorRtTensor* tensor, nvinfer1::ITensor** out_tensor, const char* parameter_name)
{
    auto status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, parameter_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* tensor_payload = get_payload<nvinfer1::ITensor>(tensor);
    if (tensor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry the expected TensorRT tensor payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_tensor = tensor_payload;
    return JYPPX_STATUS_OK;
}

bool is_loop_boundary_layer_type(nvinfer1::LayerType type)
{
    return type == nvinfer1::LayerType::kTRIP_LIMIT
        || type == nvinfer1::LayerType::kRECURRENCE
        || type == nvinfer1::LayerType::kITERATOR
        || type == nvinfer1::LayerType::kLOOP_OUTPUT;
}

bool is_if_conditional_boundary_layer_type(nvinfer1::LayerType type)
{
    return type == nvinfer1::LayerType::kCONDITION
        || type == nvinfer1::LayerType::kCONDITIONAL_INPUT
        || type == nvinfer1::LayerType::kCONDITIONAL_OUTPUT;
}

JYPPX_StatusCode get_loop_boundary_layer_for_api(JYPPX_TensorRtLayer* layer, nvinfer1::ILoopBoundaryLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr || !is_loop_boundary_layer_type(layer_payload->getType()))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer is not a TensorRT loop boundary layer.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_layer = static_cast<nvinfer1::ILoopBoundaryLayer*>(layer_payload);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode get_if_conditional_boundary_layer_for_api(JYPPX_TensorRtLayer* layer, nvinfer1::IIfConditionalBoundaryLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr || !is_if_conditional_boundary_layer_type(layer_payload->getType()))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer is not a TensorRT if-conditional boundary layer.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_layer = static_cast<nvinfer1::IIfConditionalBoundaryLayer*>(layer_payload);
    return JYPPX_STATUS_OK;
}
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_loop(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtLoop** out_loop)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_loop, "out_loop");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_loop = nullptr;
    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    return create_loop_reference_handle(network_payload->addLoop(), out_loop);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add loop");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_if_conditional(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtIfConditional** out_conditional)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_conditional, "out_conditional");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_conditional = nullptr;
    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    return create_if_conditional_reference_handle(network_payload->addIfConditional(), out_conditional);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "network add if-conditional");
#endif
}

JYPPX_StatusCode jyppx_trt11_loop_set_name(JYPPX_TensorRtLoop* loop, const char* name)
{
    auto status = validate_c_string(name, "name");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILoop* loop_payload = nullptr;
    status = get_loop_payload_for_api(loop, &loop_payload, "loop name set");
    if (status != JYPPX_STATUS_OK) { return status; }
    loop_payload->setName(name);
    return JYPPX_STATUS_OK;
#else
    (void)loop;
    return jyppx::tensorrt::report_vendor_missing(kLine, "loop name set");
#endif
}

JYPPX_StatusCode jyppx_trt11_loop_get_name(JYPPX_TensorRtLoop* loop, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILoop* loop_payload = nullptr;
    status = get_loop_payload_for_api(loop, &loop_payload, "loop name query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(loop_payload->getName(), output_buffer, output_buffer_size, out_required_size);
#else
    (void)loop;
    return jyppx::tensorrt::report_vendor_missing(kLine, "loop name query");
#endif
}

JYPPX_StatusCode jyppx_trt11_loop_add_recurrence(JYPPX_TensorRtLoop* loop, JYPPX_TensorRtTensor* initial_value, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILoop* loop_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = get_loop_payload_for_api(loop, &loop_payload, "loop recurrence add");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_control_flow(initial_value, &tensor_payload, "initial_value");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(loop_payload->addRecurrence(*tensor_payload), out_layer);
#else
    (void)loop;
    (void)initial_value;
    return jyppx::tensorrt::report_vendor_missing(kLine, "loop recurrence add");
#endif
}

JYPPX_StatusCode jyppx_trt11_loop_add_trip_limit(JYPPX_TensorRtLoop* loop, JYPPX_TensorRtTensor* tensor, int32_t limit, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (limit < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILoop* loop_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = get_loop_payload_for_api(loop, &loop_payload, "loop trip limit add");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_control_flow(tensor, &tensor_payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(loop_payload->addTripLimit(*tensor_payload, static_cast<nvinfer1::TripLimit>(limit)), out_layer);
#else
    (void)loop;
    (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "loop trip limit add");
#endif
}

JYPPX_StatusCode jyppx_trt11_loop_add_iterator(JYPPX_TensorRtLoop* loop, JYPPX_TensorRtTensor* tensor, int32_t axis, JYPPX_Boolean reverse, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILoop* loop_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = get_loop_payload_for_api(loop, &loop_payload, "loop iterator add");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_control_flow(tensor, &tensor_payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(loop_payload->addIterator(*tensor_payload, axis, reverse != JYPPX_FALSE), out_layer);
#else
    (void)loop;
    (void)tensor;
    (void)axis;
    (void)reverse;
    return jyppx::tensorrt::report_vendor_missing(kLine, "loop iterator add");
#endif
}

JYPPX_StatusCode jyppx_trt11_loop_add_output(JYPPX_TensorRtLoop* loop, JYPPX_TensorRtTensor* tensor, int32_t output_kind, int32_t axis, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (output_kind < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILoop* loop_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = get_loop_payload_for_api(loop, &loop_payload, "loop output add");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_control_flow(tensor, &tensor_payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(loop_payload->addLoopOutput(*tensor_payload, static_cast<nvinfer1::LoopOutput>(output_kind), axis), out_layer);
#else
    (void)loop;
    (void)tensor;
    (void)axis;
    return jyppx::tensorrt::report_vendor_missing(kLine, "loop output add");
#endif
}

JYPPX_StatusCode jyppx_trt11_if_conditional_set_name(JYPPX_TensorRtIfConditional* conditional, const char* name)
{
    auto status = validate_c_string(name, "name");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IIfConditional* conditional_payload = nullptr;
    status = get_if_conditional_payload_for_api(conditional, &conditional_payload, "if-conditional name set");
    if (status != JYPPX_STATUS_OK) { return status; }
    conditional_payload->setName(name);
    return JYPPX_STATUS_OK;
#else
    (void)conditional;
    return jyppx::tensorrt::report_vendor_missing(kLine, "if-conditional name set");
#endif
}

JYPPX_StatusCode jyppx_trt11_if_conditional_get_name(JYPPX_TensorRtIfConditional* conditional, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IIfConditional* conditional_payload = nullptr;
    status = get_if_conditional_payload_for_api(conditional, &conditional_payload, "if-conditional name query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(conditional_payload->getName(), output_buffer, output_buffer_size, out_required_size);
#else
    (void)conditional;
    return jyppx::tensorrt::report_vendor_missing(kLine, "if-conditional name query");
#endif
}

JYPPX_StatusCode jyppx_trt11_if_conditional_set_condition(JYPPX_TensorRtIfConditional* conditional, JYPPX_TensorRtTensor* condition, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IIfConditional* conditional_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = get_if_conditional_payload_for_api(conditional, &conditional_payload, "if-conditional condition set");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_control_flow(condition, &tensor_payload, "condition");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(conditional_payload->setCondition(*tensor_payload), out_layer);
#else
    (void)conditional;
    (void)condition;
    return jyppx::tensorrt::report_vendor_missing(kLine, "if-conditional condition set");
#endif
}

JYPPX_StatusCode jyppx_trt11_if_conditional_add_input(JYPPX_TensorRtIfConditional* conditional, JYPPX_TensorRtTensor* input, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IIfConditional* conditional_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = get_if_conditional_payload_for_api(conditional, &conditional_payload, "if-conditional input add");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_control_flow(input, &tensor_payload, "input");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(conditional_payload->addInput(*tensor_payload), out_layer);
#else
    (void)conditional;
    (void)input;
    return jyppx::tensorrt::report_vendor_missing(kLine, "if-conditional input add");
#endif
}

JYPPX_StatusCode jyppx_trt11_if_conditional_add_output(JYPPX_TensorRtIfConditional* conditional, JYPPX_TensorRtTensor* true_output, JYPPX_TensorRtTensor* false_output, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IIfConditional* conditional_payload = nullptr;
    nvinfer1::ITensor* true_tensor_payload = nullptr;
    nvinfer1::ITensor* false_tensor_payload = nullptr;
    status = get_if_conditional_payload_for_api(conditional, &conditional_payload, "if-conditional output add");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_control_flow(true_output, &true_tensor_payload, "true_output");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_control_flow(false_output, &false_tensor_payload, "false_output");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(conditional_payload->addOutput(*true_tensor_payload, *false_tensor_payload), out_layer);
#else
    (void)conditional;
    (void)true_output;
    (void)false_output;
    return jyppx::tensorrt::report_vendor_missing(kLine, "if-conditional output add");
#endif
}

JYPPX_StatusCode jyppx_trt11_loop_boundary_layer_get_loop_name(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILoopBoundaryLayer* boundary = nullptr;
    status = get_loop_boundary_layer_for_api(layer, &boundary);
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::ILoop* loop = boundary->getLoop();
    return copy_string_to_buffer(loop != nullptr ? loop->getName() : "", output_buffer, output_buffer_size, out_required_size);
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "loop boundary name query");
#endif
}

JYPPX_StatusCode jyppx_trt11_if_conditional_boundary_layer_get_conditional_name(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IIfConditionalBoundaryLayer* boundary = nullptr;
    status = get_if_conditional_boundary_layer_for_api(layer, &boundary);
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::IIfConditional* conditional = boundary->getConditional();
    return copy_string_to_buffer(conditional != nullptr ? conditional->getName() : "", output_buffer, output_buffer_size, out_required_size);
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "if-conditional boundary name query");
#endif
}

JYPPX_StatusCode jyppx_trt11_loop_output_layer_get_kind(JYPPX_TensorRtLayer* layer, int32_t* out_kind)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_kind, "out_kind");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_kind = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILoopOutputLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kLOOP_OUTPUT, "loop output", "loop output kind query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_kind = static_cast<int32_t>(typed_layer->getLoopOutput());
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "loop output kind query");
#endif
}

JYPPX_StatusCode jyppx_trt11_loop_output_layer_set_axis(JYPPX_TensorRtLayer* layer, int32_t axis)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILoopOutputLayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kLOOP_OUTPUT, "loop output", "loop output axis set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setAxis(axis);
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    (void)axis;
    return jyppx::tensorrt::report_vendor_missing(kLine, "loop output axis set");
#endif
}

JYPPX_StatusCode jyppx_trt11_loop_output_layer_get_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_axis, "out_axis");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_axis = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ILoopOutputLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kLOOP_OUTPUT, "loop output", "loop output axis query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_axis = typed_layer->getAxis();
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "loop output axis query");
#endif
}

JYPPX_StatusCode jyppx_trt11_trip_limit_layer_get_kind(JYPPX_TensorRtLayer* layer, int32_t* out_kind)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_kind, "out_kind");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_kind = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ITripLimitLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kTRIP_LIMIT, "trip limit", "trip limit kind query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_kind = static_cast<int32_t>(typed_layer->getTripLimit());
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "trip limit kind query");
#endif
}

JYPPX_StatusCode jyppx_trt11_iterator_layer_set_axis(JYPPX_TensorRtLayer* layer, int32_t axis)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IIteratorLayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kITERATOR, "iterator", "iterator axis set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setAxis(axis);
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    (void)axis;
    return jyppx::tensorrt::report_vendor_missing(kLine, "iterator axis set");
#endif
}

JYPPX_StatusCode jyppx_trt11_iterator_layer_get_axis(JYPPX_TensorRtLayer* layer, int32_t* out_axis)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_axis, "out_axis");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_axis = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IIteratorLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kITERATOR, "iterator", "iterator axis query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_axis = typed_layer->getAxis();
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "iterator axis query");
#endif
}

JYPPX_StatusCode jyppx_trt11_iterator_layer_set_reverse(JYPPX_TensorRtLayer* layer, JYPPX_Boolean reverse)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IIteratorLayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kITERATOR, "iterator", "iterator reverse set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setReverse(reverse != JYPPX_FALSE);
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    (void)reverse;
    return jyppx::tensorrt::report_vendor_missing(kLine, "iterator reverse set");
#endif
}

JYPPX_StatusCode jyppx_trt11_iterator_layer_get_reverse(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_reverse)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_reverse, "out_reverse");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_reverse = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IIteratorLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kITERATOR, "iterator", "iterator reverse query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_reverse = typed_layer->getReverse() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "iterator reverse query");
#endif
}

JYPPX_StatusCode jyppx_trt11_fill_layer_set_alpha_int64(JYPPX_TensorRtLayer* layer, int64_t alpha)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IFillLayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kFILL, "fill", "fill alpha int64 set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setAlphaInt64(alpha);
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    (void)alpha;
    return jyppx::tensorrt::report_vendor_missing(kLine, "fill alpha int64 set");
#endif
}

JYPPX_StatusCode jyppx_trt11_fill_layer_get_alpha_int64(JYPPX_TensorRtLayer* layer, int64_t* out_alpha)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_alpha, "out_alpha");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_alpha = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IFillLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kFILL, "fill", "fill alpha int64 query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_alpha = typed_layer->getAlphaInt64();
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "fill alpha int64 query");
#endif
}

JYPPX_StatusCode jyppx_trt11_fill_layer_set_beta_int64(JYPPX_TensorRtLayer* layer, int64_t beta)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IFillLayer* typed_layer = nullptr;
    auto status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kFILL, "fill", "fill beta int64 set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setBetaInt64(beta);
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    (void)beta;
    return jyppx::tensorrt::report_vendor_missing(kLine, "fill beta int64 set");
#endif
}

JYPPX_StatusCode jyppx_trt11_fill_layer_get_beta_int64(JYPPX_TensorRtLayer* layer, int64_t* out_beta)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_beta, "out_beta");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_beta = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IFillLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kFILL, "fill", "fill beta int64 query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_beta = typed_layer->getBetaInt64();
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "fill beta int64 query");
#endif
}

JYPPX_StatusCode jyppx_trt11_fill_layer_is_alpha_beta_int64(JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_is_int64)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_int64, "out_is_int64");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_int64 = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IFillLayer* typed_layer = nullptr;
    status = get_typed_layer_for_api(layer, nvinfer1::LayerType::kFILL, "fill", "fill alpha/beta type query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_int64 = typed_layer->isAlphaBetaInt64() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "fill alpha/beta type query");
#endif
}
