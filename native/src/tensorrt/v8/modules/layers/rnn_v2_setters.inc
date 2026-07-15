#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
static JYPPX_StatusCode get_trt8_rnn_v2_for_setter(JYPPX_TensorRtLayer* layer, nvinfer1::IRNNv2Layer** out_layer)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = get_rnn_v2_layer_payload(layer);
    if (*out_layer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Layer handle is not a TensorRT RNNv2 layer.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode get_trt8_tensor_for_rnn_v2_setter(JYPPX_TensorRtTensor* tensor, nvinfer1::ITensor** out_tensor)
{
    auto status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = get_payload<nvinfer1::ITensor>(tensor);
    if (*out_tensor == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Tensor handle does not carry a TensorRT tensor payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode set_trt8_rnn_v2_enum_with_seh_guard(
    nvinfer1::IRNNv2Layer* layer,
    const int32_t value,
    const int32_t selector,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        if (selector == 0) { layer->setOperation(static_cast<nvinfer1::RNNOperation>(value)); }
        else if (selector == 1) { layer->setDirection(static_cast<nvinfer1::RNNDirection>(value)); }
        else { layer->setInputMode(static_cast<nvinfer1::RNNInputMode>(value)); }
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    if (selector == 0) { layer->setOperation(static_cast<nvinfer1::RNNOperation>(value)); }
    else if (selector == 1) { layer->setDirection(static_cast<nvinfer1::RNNDirection>(value)); }
    else { layer->setInputMode(static_cast<nvinfer1::RNNInputMode>(value)); }
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode set_trt8_rnn_v2_tensor_with_seh_guard(
    nvinfer1::IRNNv2Layer* layer,
    nvinfer1::ITensor* tensor,
    const int32_t selector,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        if (selector == 0) { layer->setCellState(*tensor); }
        else if (selector == 1) { layer->setHiddenState(*tensor); }
        else { layer->setSequenceLengths(*tensor); }
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    if (selector == 0) { layer->setCellState(*tensor); }
    else if (selector == 1) { layer->setHiddenState(*tensor); }
    else { layer->setSequenceLengths(*tensor); }
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode set_trt8_rnn_v2_enum(
    JYPPX_TensorRtLayer* layer,
    const int32_t value,
    const int32_t minimum,
    const int32_t maximum,
    const int32_t selector,
    const char* feature_name)
{
    if (value < minimum || value > maximum)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 RNNv2 enum value is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    nvinfer1::IRNNv2Layer* payload = nullptr;
    auto status = get_trt8_rnn_v2_for_setter(layer, &payload);
    if (status != JYPPX_STATUS_OK) { return status; }
    try { return set_trt8_rnn_v2_enum_with_seh_guard(payload, value, selector, feature_name); }
    catch (const std::exception& exception) { return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what()); }
    catch (...) { return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception"); }
}

static JYPPX_StatusCode set_trt8_rnn_v2_tensor(
    JYPPX_TensorRtLayer* layer,
    JYPPX_TensorRtTensor* tensor,
    const int32_t selector,
    const char* feature_name)
{
    nvinfer1::IRNNv2Layer* layer_payload = nullptr;
    auto status = get_trt8_rnn_v2_for_setter(layer, &layer_payload);
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = get_trt8_tensor_for_rnn_v2_setter(tensor, &tensor_payload);
    if (status != JYPPX_STATUS_OK) { return status; }
    try { return set_trt8_rnn_v2_tensor_with_seh_guard(layer_payload, tensor_payload, selector, feature_name); }
    catch (const std::exception& exception) { return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what()); }
    catch (...) { return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception"); }
}
#endif

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_set_operation(JYPPX_TensorRtLayer* layer, const int32_t operation)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    return set_trt8_rnn_v2_enum(layer, operation, 0, 3, 0, "RNNv2 operation set");
#else
    (void)layer; (void)operation;
    return jyppx::tensorrt::report_vendor_missing(kLine, "RNNv2 operation set");
#endif
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_set_direction(JYPPX_TensorRtLayer* layer, const int32_t direction)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    return set_trt8_rnn_v2_enum(layer, direction, 0, 1, 1, "RNNv2 direction set");
#else
    (void)layer; (void)direction;
    return jyppx::tensorrt::report_vendor_missing(kLine, "RNNv2 direction set");
#endif
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_set_input_mode(JYPPX_TensorRtLayer* layer, const int32_t input_mode)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    return set_trt8_rnn_v2_enum(layer, input_mode, 0, 1, 2, "RNNv2 input mode set");
#else
    (void)layer; (void)input_mode;
    return jyppx::tensorrt::report_vendor_missing(kLine, "RNNv2 input mode set");
#endif
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_set_cell_state(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtTensor* tensor)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    return set_trt8_rnn_v2_tensor(layer, tensor, 0, "RNNv2 cell state set");
#else
    (void)layer; (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "RNNv2 cell state set");
#endif
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_set_hidden_state(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtTensor* tensor)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    return set_trt8_rnn_v2_tensor(layer, tensor, 1, "RNNv2 hidden state set");
#else
    (void)layer; (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "RNNv2 hidden state set");
#endif
}

JYPPX_StatusCode jyppx_trt8_rnn_v2_layer_set_sequence_lengths(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtTensor* tensor)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    return set_trt8_rnn_v2_tensor(layer, tensor, 2, "RNNv2 sequence lengths set");
#else
    (void)layer; (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "RNNv2 sequence lengths set");
#endif
}
