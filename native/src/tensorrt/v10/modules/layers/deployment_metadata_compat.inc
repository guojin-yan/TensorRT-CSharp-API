#if JYPPX_HAS_TENSORRT
static JYPPX_StatusCode jyppx_trt10_get_network_payload_for_api(
    JYPPX_TensorRtNetworkDefinition* network,
    nvinfer1::INetworkDefinition** out_network,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_network = get_payload<nvinfer1::INetworkDefinition>(network);
    if (*out_network == nullptr)
    {
        std::ostringstream builder;
        builder << "Network handle does not carry the expected TensorRT payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode jyppx_trt10_get_tensor_payload_for_api(
    JYPPX_TensorRtTensor* tensor,
    nvinfer1::ITensor** out_tensor,
    const char* parameter_name)
{
    auto status = jyppx::tensorrt::validate_handle(tensor, kLine, JYPPX_TENSORRT_OBJECT_KIND_TENSOR, parameter_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor = get_payload<nvinfer1::ITensor>(tensor);
    if (*out_tensor == nullptr)
    {
        std::ostringstream builder;
        builder << parameter_name << " does not carry the expected TensorRT tensor payload.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    return JYPPX_STATUS_OK;
}

template <typename TLayer>
static JYPPX_StatusCode jyppx_trt10_get_typed_layer_for_api(
    JYPPX_TensorRtLayer* layer,
    const nvinfer1::LayerType expected_type,
    const char* type_name,
    const char* feature_name,
    TLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::ILayer* base_layer = get_layer_payload(layer);
    if (base_layer == nullptr)
    {
        std::ostringstream builder;
        builder << "Layer handle does not carry the expected TensorRT payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    if (base_layer->getType() != expected_type)
    {
        std::ostringstream builder;
        builder << "Layer handle is not a TensorRT " << type_name << " layer.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_layer = static_cast<TLayer*>(base_layer);
    return JYPPX_STATUS_OK;
}

#define JYPPX_TRT10_LAYER_INT_SET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, VALUE_TYPE) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int32_t value) \
    { \
        if (value < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; } \
        LAYER_TYPE* typed_layer = nullptr; \
        auto status = jyppx_trt10_get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        typed_layer->METHOD(static_cast<VALUE_TYPE>(value)); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT10_LAYER_INT_GET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, OUT_NAME) \
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

#define JYPPX_TRT10_LAYER_UINT_SET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, PARAM_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, uint32_t PARAM_NAME) \
    { \
        if (PARAM_NAME == 0) { return JYPPX_STATUS_INVALID_ARGUMENT; } \
        LAYER_TYPE* typed_layer = nullptr; \
        auto status = jyppx_trt10_get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        typed_layer->METHOD(PARAM_NAME); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT10_LAYER_UINT_GET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, OUT_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, uint32_t* OUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUT_NAME, #OUT_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUT_NAME = 0; \
        LAYER_TYPE* typed_layer = nullptr; \
        status = jyppx_trt10_get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUT_NAME = static_cast<uint32_t>(typed_layer->METHOD()); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT10_LAYER_BOOL_SET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, PARAM_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, JYPPX_Boolean PARAM_NAME) \
    { \
        LAYER_TYPE* typed_layer = nullptr; \
        auto status = jyppx_trt10_get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        typed_layer->METHOD(PARAM_NAME != JYPPX_FALSE); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT10_LAYER_BOOL_GET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, OUT_NAME) \
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

#define JYPPX_TRT10_LAYER_DOUBLE_SET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, PARAM_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, double PARAM_NAME) \
    { \
        LAYER_TYPE* typed_layer = nullptr; \
        auto status = jyppx_trt10_get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        typed_layer->METHOD(static_cast<float>(PARAM_NAME)); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT10_LAYER_DOUBLE_GET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, OUT_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, double* OUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUT_NAME, #OUT_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUT_NAME = 0.0; \
        LAYER_TYPE* typed_layer = nullptr; \
        status = jyppx_trt10_get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUT_NAME = static_cast<double>(typed_layer->METHOD()); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT10_LAYER_INT64_SET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, PARAM_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int64_t PARAM_NAME) \
    { \
        if (PARAM_NAME <= 0) { return JYPPX_STATUS_INVALID_ARGUMENT; } \
        LAYER_TYPE* typed_layer = nullptr; \
        auto status = jyppx_trt10_get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        typed_layer->METHOD(PARAM_NAME); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT10_LAYER_INT64_GET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, OUT_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, int64_t* OUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUT_NAME, #OUT_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUT_NAME = 0; \
        LAYER_TYPE* typed_layer = nullptr; \
        status = jyppx_trt10_get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        *OUT_NAME = static_cast<int64_t>(typed_layer->METHOD()); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT10_LAYER_DIMS_SET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, PARAM_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, const JYPPX_TensorRtDims* PARAM_NAME) \
    { \
        nvinfer1::Dims native_dims{}; \
        auto status = make_dims(PARAM_NAME, &native_dims, #PARAM_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        LAYER_TYPE* typed_layer = nullptr; \
        status = jyppx_trt10_get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        typed_layer->METHOD(native_dims); \
        return JYPPX_STATUS_OK; \
    }

#define JYPPX_TRT10_LAYER_DIMS_GET(FUNCTION_NAME, LAYER_TYPE, ENUM_VALUE, TYPE_NAME, METHOD, OUT_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer* layer, JYPPX_TensorRtDims* OUT_NAME) \
    { \
        auto status = jyppx::tensorrt::validate_output_pointer(OUT_NAME, #OUT_NAME); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        std::memset(OUT_NAME, 0, sizeof(*OUT_NAME)); \
        LAYER_TYPE* typed_layer = nullptr; \
        status = jyppx_trt10_get_typed_layer_for_api(layer, ENUM_VALUE, TYPE_NAME, #FUNCTION_NAME, &typed_layer); \
        if (status != JYPPX_STATUS_OK) { return status; } \
        copy_dims(typed_layer->METHOD(), OUT_NAME); \
        return JYPPX_STATUS_OK; \
    }

JYPPX_StatusCode jyppx_trt10_network_add_cast(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* input_tensor,
    int32_t to_type,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network cast layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addCast(*input, static_cast<nvinfer1::DataType>(to_type)), out_layer);
}

JYPPX_StatusCode jyppx_trt10_network_add_nms(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* boxes_tensor,
    JYPPX_TensorRtTensor* scores_tensor,
    JYPPX_TensorRtTensor* max_output_boxes_per_class_tensor,
    int32_t indices_type,
    JYPPX_TensorRtLayer** out_layer)
{
    (void)indices_type;
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* boxes = nullptr;
    nvinfer1::ITensor* scores = nullptr;
    nvinfer1::ITensor* max_output = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network NMS layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(boxes_tensor, &boxes, "boxes_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(scores_tensor, &scores, "scores_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(max_output_boxes_per_class_tensor, &max_output, "max_output_boxes_per_class_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addNMS(*boxes, *scores, *max_output), out_layer);
}

JYPPX_StatusCode jyppx_trt10_network_add_einsum(
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
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network einsum layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::vector<nvinfer1::ITensor*> inputs;
    inputs.reserve(static_cast<size_t>(input_count));
    for (int32_t i = 0; i < input_count; ++i)
    {
        nvinfer1::ITensor* input = nullptr;
        status = jyppx_trt10_get_tensor_payload_for_api(input_tensors[i], &input, "input_tensors");
        if (status != JYPPX_STATUS_OK) { return status; }
        inputs.push_back(input);
    }
    return create_layer_reference_handle(network_payload->addEinsum(inputs.data(), input_count, equation), out_layer);
}

JYPPX_StatusCode jyppx_trt10_network_add_one_hot(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* indices_tensor, JYPPX_TensorRtTensor* values_tensor, JYPPX_TensorRtTensor* depth_tensor, int32_t axis, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* indices = nullptr;
    nvinfer1::ITensor* values = nullptr;
    nvinfer1::ITensor* depth = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network one-hot layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(indices_tensor, &indices, "indices_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(values_tensor, &values, "values_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(depth_tensor, &depth, "depth_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addOneHot(*indices, *values, *depth, axis), out_layer);
}

JYPPX_StatusCode jyppx_trt10_network_add_assertion(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* condition_tensor, const char* message, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    status = validate_c_string(message, "message");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* condition = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network assertion layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(condition_tensor, &condition, "condition_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addAssertion(*condition, message), out_layer);
}

JYPPX_StatusCode jyppx_trt10_network_add_grid_sample(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* input_tensor, JYPPX_TensorRtTensor* grid_tensor, JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* grid = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network grid sample layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(grid_tensor, &grid, "grid_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addGridSample(*input, *grid), out_layer);
}

JYPPX_StatusCode jyppx_trt10_network_add_normalization_v2(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* input_tensor,
    JYPPX_TensorRtTensor* scale_tensor,
    JYPPX_TensorRtTensor* bias_tensor,
    uint32_t axes,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (axes == 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    nvinfer1::ITensor* scale = nullptr;
    nvinfer1::ITensor* bias = nullptr;
    status = jyppx_trt10_get_network_payload_for_api(network, &network_payload, "network normalization layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(scale_tensor, &scale, "scale_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt10_get_tensor_payload_for_api(bias_tensor, &bias, "bias_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_layer_reference_handle(network_payload->addNormalization(*input, *scale, *bias, axes), out_layer);
}

JYPPX_TRT10_LAYER_INT_SET(jyppx_trt10_gather_layer_set_mode, nvinfer1::IGatherLayer, nvinfer1::LayerType::kGATHER, "gather", setMode, nvinfer1::GatherMode)
JYPPX_TRT10_LAYER_INT_GET(jyppx_trt10_gather_layer_get_mode, nvinfer1::IGatherLayer, nvinfer1::LayerType::kGATHER, "gather", getMode, out_mode)
JYPPX_TRT10_LAYER_INT_SET(jyppx_trt10_gather_layer_set_nb_elementwise_dims, nvinfer1::IGatherLayer, nvinfer1::LayerType::kGATHER, "gather", setNbElementWiseDims, int32_t)
JYPPX_TRT10_LAYER_INT_GET(jyppx_trt10_gather_layer_get_nb_elementwise_dims, nvinfer1::IGatherLayer, nvinfer1::LayerType::kGATHER, "gather", getNbElementWiseDims, out_elementwise_dims)

JYPPX_TRT10_LAYER_INT_SET(jyppx_trt10_one_hot_layer_set_axis, nvinfer1::IOneHotLayer, nvinfer1::LayerType::kONE_HOT, "one-hot", setAxis, int32_t)
JYPPX_TRT10_LAYER_INT_GET(jyppx_trt10_one_hot_layer_get_axis, nvinfer1::IOneHotLayer, nvinfer1::LayerType::kONE_HOT, "one-hot", getAxis, out_axis)

JYPPX_StatusCode jyppx_trt10_assertion_layer_set_message(JYPPX_TensorRtLayer* layer, const char* message)
{
    auto status = validate_c_string(message, "message");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::IAssertionLayer* typed_layer = nullptr;
    status = jyppx_trt10_get_typed_layer_for_api(layer, nvinfer1::LayerType::kASSERTION, "assertion", "assertion message set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setMessage(message);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt10_assertion_layer_get_message(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
    nvinfer1::IAssertionLayer* typed_layer = nullptr;
    status = jyppx_trt10_get_typed_layer_for_api(layer, nvinfer1::LayerType::kASSERTION, "assertion", "assertion message query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(typed_layer->getMessage(), output_buffer, output_buffer_size, out_required_size);
}

JYPPX_TRT10_LAYER_INT_SET(jyppx_trt10_grid_sample_layer_set_interpolation_mode, nvinfer1::IGridSampleLayer, nvinfer1::LayerType::kGRID_SAMPLE, "grid sample", setInterpolationMode, nvinfer1::InterpolationMode)
JYPPX_TRT10_LAYER_INT_GET(jyppx_trt10_grid_sample_layer_get_interpolation_mode, nvinfer1::IGridSampleLayer, nvinfer1::LayerType::kGRID_SAMPLE, "grid sample", getInterpolationMode, out_interpolation_mode)
JYPPX_TRT10_LAYER_BOOL_SET(jyppx_trt10_grid_sample_layer_set_align_corners, nvinfer1::IGridSampleLayer, nvinfer1::LayerType::kGRID_SAMPLE, "grid sample", setAlignCorners, align_corners)
JYPPX_TRT10_LAYER_BOOL_GET(jyppx_trt10_grid_sample_layer_get_align_corners, nvinfer1::IGridSampleLayer, nvinfer1::LayerType::kGRID_SAMPLE, "grid sample", getAlignCorners, out_align_corners)

JYPPX_StatusCode jyppx_trt10_grid_sample_layer_set_sample_mode(JYPPX_TensorRtLayer* layer, int32_t sample_mode)
{
    if (sample_mode < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    nvinfer1::IGridSampleLayer* typed_layer = nullptr;
    auto status = jyppx_trt10_get_typed_layer_for_api(layer, nvinfer1::LayerType::kGRID_SAMPLE, "grid sample", "grid sample mode set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    if (!typed_layer->setSampleMode(static_cast<nvinfer1::SampleMode>(sample_mode)))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IGridSampleLayer::setSampleMode returned false.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    return JYPPX_STATUS_OK;
}

JYPPX_TRT10_LAYER_INT_GET(jyppx_trt10_grid_sample_layer_get_sample_mode, nvinfer1::IGridSampleLayer, nvinfer1::LayerType::kGRID_SAMPLE, "grid sample", getSampleMode, out_sample_mode)

JYPPX_TRT10_LAYER_DOUBLE_SET(jyppx_trt10_normalization_layer_set_epsilon, nvinfer1::INormalizationLayer, nvinfer1::LayerType::kNORMALIZATION, "normalization", setEpsilon, epsilon)
JYPPX_TRT10_LAYER_DOUBLE_GET(jyppx_trt10_normalization_layer_get_epsilon, nvinfer1::INormalizationLayer, nvinfer1::LayerType::kNORMALIZATION, "normalization", getEpsilon, out_epsilon)
JYPPX_TRT10_LAYER_UINT_SET(jyppx_trt10_normalization_layer_set_axes, nvinfer1::INormalizationLayer, nvinfer1::LayerType::kNORMALIZATION, "normalization", setAxes, axes)
JYPPX_TRT10_LAYER_UINT_GET(jyppx_trt10_normalization_layer_get_axes, nvinfer1::INormalizationLayer, nvinfer1::LayerType::kNORMALIZATION, "normalization", getAxes, out_axes)
JYPPX_TRT10_LAYER_INT64_SET(jyppx_trt10_normalization_layer_set_nb_groups, nvinfer1::INormalizationLayer, nvinfer1::LayerType::kNORMALIZATION, "normalization", setNbGroups, group_count)
JYPPX_TRT10_LAYER_INT64_GET(jyppx_trt10_normalization_layer_get_nb_groups, nvinfer1::INormalizationLayer, nvinfer1::LayerType::kNORMALIZATION, "normalization", getNbGroups, out_group_count)
JYPPX_TRT10_LAYER_INT_SET(jyppx_trt10_normalization_layer_set_compute_precision, nvinfer1::INormalizationLayer, nvinfer1::LayerType::kNORMALIZATION, "normalization", setComputePrecision, nvinfer1::DataType)
JYPPX_TRT10_LAYER_INT_GET(jyppx_trt10_normalization_layer_get_compute_precision, nvinfer1::INormalizationLayer, nvinfer1::LayerType::kNORMALIZATION, "normalization", getComputePrecision, out_data_type)

JYPPX_StatusCode jyppx_trt10_cast_layer_set_to_type(JYPPX_TensorRtLayer* layer, int32_t data_type)
{
    nvinfer1::ICastLayer* typed_layer = nullptr;
    auto status = jyppx_trt10_get_typed_layer_for_api(layer, nvinfer1::LayerType::kCAST, "cast", "cast layer output type set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setToType(static_cast<nvinfer1::DataType>(data_type));
    return JYPPX_STATUS_OK;
}

JYPPX_TRT10_LAYER_INT_GET(jyppx_trt10_cast_layer_get_to_type, nvinfer1::ICastLayer, nvinfer1::LayerType::kCAST, "cast", getToType, out_data_type)

JYPPX_StatusCode jyppx_trt10_nms_layer_set_bounding_box_format(JYPPX_TensorRtLayer* layer, int32_t format)
{
    nvinfer1::INMSLayer* typed_layer = nullptr;
    auto status = jyppx_trt10_get_typed_layer_for_api(layer, nvinfer1::LayerType::kNMS, "NMS", "NMS layer bounding box format set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    typed_layer->setBoundingBoxFormat(static_cast<nvinfer1::BoundingBoxFormat>(format));
    return JYPPX_STATUS_OK;
}

JYPPX_TRT10_LAYER_INT_GET(jyppx_trt10_nms_layer_get_bounding_box_format, nvinfer1::INMSLayer, nvinfer1::LayerType::kNMS, "NMS", getBoundingBoxFormat, out_format)
JYPPX_TRT10_LAYER_INT_SET(jyppx_trt10_nms_layer_set_topk_box_limit, nvinfer1::INMSLayer, nvinfer1::LayerType::kNMS, "NMS", setTopKBoxLimit, int32_t)
JYPPX_TRT10_LAYER_INT_GET(jyppx_trt10_nms_layer_get_topk_box_limit, nvinfer1::INMSLayer, nvinfer1::LayerType::kNMS, "NMS", getTopKBoxLimit, out_limit)

JYPPX_TRT10_LAYER_DIMS_SET(jyppx_trt10_constant_layer_set_dimensions, nvinfer1::IConstantLayer, nvinfer1::LayerType::kCONSTANT, "constant", setDimensions, dimensions)
JYPPX_TRT10_LAYER_DIMS_GET(jyppx_trt10_constant_layer_get_dimensions, nvinfer1::IConstantLayer, nvinfer1::LayerType::kCONSTANT, "constant", getDimensions, out_dimensions)

JYPPX_StatusCode jyppx_trt10_einsum_layer_set_equation(JYPPX_TensorRtLayer* layer, const char* equation, JYPPX_Boolean* out_success)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = JYPPX_FALSE;
    status = validate_c_string(equation, "equation");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::IEinsumLayer* typed_layer = nullptr;
    status = jyppx_trt10_get_typed_layer_for_api(layer, nvinfer1::LayerType::kEINSUM, "einsum", "einsum equation set", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_success = typed_layer->setEquation(equation) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt10_einsum_layer_get_equation(JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
    nvinfer1::IEinsumLayer* typed_layer = nullptr;
    status = jyppx_trt10_get_typed_layer_for_api(layer, nvinfer1::LayerType::kEINSUM, "einsum", "einsum equation query", &typed_layer);
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(typed_layer->getEquation(), output_buffer, output_buffer_size, out_required_size);
}

#undef JYPPX_TRT10_LAYER_INT_SET
#undef JYPPX_TRT10_LAYER_INT_GET
#undef JYPPX_TRT10_LAYER_UINT_SET
#undef JYPPX_TRT10_LAYER_UINT_GET
#undef JYPPX_TRT10_LAYER_BOOL_SET
#undef JYPPX_TRT10_LAYER_BOOL_GET
#undef JYPPX_TRT10_LAYER_DOUBLE_SET
#undef JYPPX_TRT10_LAYER_DOUBLE_GET
#undef JYPPX_TRT10_LAYER_INT64_SET
#undef JYPPX_TRT10_LAYER_INT64_GET
#undef JYPPX_TRT10_LAYER_DIMS_SET
#undef JYPPX_TRT10_LAYER_DIMS_GET
#else
#define JYPPX_TRT10_VENDOR_MISSING_GET_INT(FUNCTION_NAME, PARAM_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, int32_t* PARAM_NAME) { auto status = jyppx::tensorrt::validate_output_pointer(PARAM_NAME, #PARAM_NAME); if (status != JYPPX_STATUS_OK) { return status; } *PARAM_NAME = 0; return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }
#define JYPPX_TRT10_VENDOR_MISSING_SET_INT(FUNCTION_NAME, PARAM_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, int32_t PARAM_NAME) { (void)PARAM_NAME; return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }
#define JYPPX_TRT10_VENDOR_MISSING_GET_UINT(FUNCTION_NAME, PARAM_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, uint32_t* PARAM_NAME) { auto status = jyppx::tensorrt::validate_output_pointer(PARAM_NAME, #PARAM_NAME); if (status != JYPPX_STATUS_OK) { return status; } *PARAM_NAME = 0; return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }
#define JYPPX_TRT10_VENDOR_MISSING_SET_UINT(FUNCTION_NAME, PARAM_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, uint32_t PARAM_NAME) { (void)PARAM_NAME; return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }
#define JYPPX_TRT10_VENDOR_MISSING_GET_DOUBLE(FUNCTION_NAME, PARAM_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, double* PARAM_NAME) { auto status = jyppx::tensorrt::validate_output_pointer(PARAM_NAME, #PARAM_NAME); if (status != JYPPX_STATUS_OK) { return status; } *PARAM_NAME = 0.0; return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }
#define JYPPX_TRT10_VENDOR_MISSING_SET_DOUBLE(FUNCTION_NAME, PARAM_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, double PARAM_NAME) { (void)PARAM_NAME; return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }
#define JYPPX_TRT10_VENDOR_MISSING_GET_INT64(FUNCTION_NAME, PARAM_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, int64_t* PARAM_NAME) { auto status = jyppx::tensorrt::validate_output_pointer(PARAM_NAME, #PARAM_NAME); if (status != JYPPX_STATUS_OK) { return status; } *PARAM_NAME = 0; return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }
#define JYPPX_TRT10_VENDOR_MISSING_SET_INT64(FUNCTION_NAME, PARAM_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, int64_t PARAM_NAME) { (void)PARAM_NAME; return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }
#define JYPPX_TRT10_VENDOR_MISSING_GET_DIMS(FUNCTION_NAME, PARAM_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, JYPPX_TensorRtDims* PARAM_NAME) { auto status = jyppx::tensorrt::validate_output_pointer(PARAM_NAME, #PARAM_NAME); if (status != JYPPX_STATUS_OK) { return status; } std::memset(PARAM_NAME, 0, sizeof(*PARAM_NAME)); return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }
#define JYPPX_TRT10_VENDOR_MISSING_SET_DIMS(FUNCTION_NAME, PARAM_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, const JYPPX_TensorRtDims* PARAM_NAME) { (void)PARAM_NAME; return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }
#define JYPPX_TRT10_VENDOR_MISSING_GET_BOOL(FUNCTION_NAME, PARAM_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, JYPPX_Boolean* PARAM_NAME) { auto status = jyppx::tensorrt::validate_output_pointer(PARAM_NAME, #PARAM_NAME); if (status != JYPPX_STATUS_OK) { return status; } *PARAM_NAME = JYPPX_FALSE; return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }
#define JYPPX_TRT10_VENDOR_MISSING_SET_BOOL(FUNCTION_NAME, PARAM_NAME, FEATURE_NAME) \
    JYPPX_StatusCode FUNCTION_NAME(JYPPX_TensorRtLayer*, JYPPX_Boolean PARAM_NAME) { (void)PARAM_NAME; return jyppx::tensorrt::report_vendor_missing(kLine, FEATURE_NAME); }

JYPPX_StatusCode jyppx_trt10_network_add_cast(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, int32_t, JYPPX_TensorRtLayer** out_layer) { auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer"); if (status != JYPPX_STATUS_OK) { return status; } *out_layer = nullptr; return jyppx::tensorrt::report_vendor_missing(kLine, "network cast layer creation"); }
JYPPX_StatusCode jyppx_trt10_network_add_nms(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, int32_t, JYPPX_TensorRtLayer** out_layer) { auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer"); if (status != JYPPX_STATUS_OK) { return status; } *out_layer = nullptr; return jyppx::tensorrt::report_vendor_missing(kLine, "network NMS layer creation"); }
JYPPX_StatusCode jyppx_trt10_network_add_einsum(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor**, int32_t, const char*, JYPPX_TensorRtLayer** out_layer) { auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer"); if (status != JYPPX_STATUS_OK) { return status; } *out_layer = nullptr; return jyppx::tensorrt::report_vendor_missing(kLine, "network einsum layer creation"); }
JYPPX_StatusCode jyppx_trt10_network_add_one_hot(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, int32_t, JYPPX_TensorRtLayer** out_layer) { auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer"); if (status != JYPPX_STATUS_OK) { return status; } *out_layer = nullptr; return jyppx::tensorrt::report_vendor_missing(kLine, "network one-hot layer creation"); }
JYPPX_StatusCode jyppx_trt10_network_add_assertion(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, const char*, JYPPX_TensorRtLayer** out_layer) { auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer"); if (status != JYPPX_STATUS_OK) { return status; } *out_layer = nullptr; return jyppx::tensorrt::report_vendor_missing(kLine, "network assertion layer creation"); }
JYPPX_StatusCode jyppx_trt10_network_add_grid_sample(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, JYPPX_TensorRtLayer** out_layer) { auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer"); if (status != JYPPX_STATUS_OK) { return status; } *out_layer = nullptr; return jyppx::tensorrt::report_vendor_missing(kLine, "network grid sample layer creation"); }
JYPPX_StatusCode jyppx_trt10_network_add_normalization_v2(JYPPX_TensorRtNetworkDefinition*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, JYPPX_TensorRtTensor*, uint32_t, JYPPX_TensorRtLayer** out_layer) { auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer"); if (status != JYPPX_STATUS_OK) { return status; } *out_layer = nullptr; return jyppx::tensorrt::report_vendor_missing(kLine, "network normalization layer creation"); }

JYPPX_TRT10_VENDOR_MISSING_SET_INT(jyppx_trt10_gather_layer_set_mode, mode, "gather mode set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_gather_layer_get_mode, out_mode, "gather mode query")
JYPPX_TRT10_VENDOR_MISSING_SET_INT(jyppx_trt10_gather_layer_set_nb_elementwise_dims, elementwise_dims, "gather elementwise dims set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_gather_layer_get_nb_elementwise_dims, out_elementwise_dims, "gather elementwise dims query")
JYPPX_TRT10_VENDOR_MISSING_SET_INT(jyppx_trt10_one_hot_layer_set_axis, axis, "one-hot axis set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_one_hot_layer_get_axis, out_axis, "one-hot axis query")
JYPPX_StatusCode jyppx_trt10_assertion_layer_set_message(JYPPX_TensorRtLayer*, const char*) { return jyppx::tensorrt::report_vendor_missing(kLine, "assertion message set"); }
JYPPX_StatusCode jyppx_trt10_assertion_layer_get_message(JYPPX_TensorRtLayer*, char*, size_t, size_t* out_required_size) { auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size"); if (status != JYPPX_STATUS_OK) { return status; } *out_required_size = 0; return jyppx::tensorrt::report_vendor_missing(kLine, "assertion message query"); }
JYPPX_TRT10_VENDOR_MISSING_SET_INT(jyppx_trt10_grid_sample_layer_set_interpolation_mode, interpolation_mode, "grid sample interpolation mode set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_grid_sample_layer_get_interpolation_mode, out_interpolation_mode, "grid sample interpolation mode query")
JYPPX_TRT10_VENDOR_MISSING_SET_BOOL(jyppx_trt10_grid_sample_layer_set_align_corners, align_corners, "grid sample align corners set")
JYPPX_TRT10_VENDOR_MISSING_GET_BOOL(jyppx_trt10_grid_sample_layer_get_align_corners, out_align_corners, "grid sample align corners query")
JYPPX_TRT10_VENDOR_MISSING_SET_INT(jyppx_trt10_grid_sample_layer_set_sample_mode, sample_mode, "grid sample mode set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_grid_sample_layer_get_sample_mode, out_sample_mode, "grid sample mode query")
JYPPX_TRT10_VENDOR_MISSING_SET_DOUBLE(jyppx_trt10_normalization_layer_set_epsilon, epsilon, "normalization epsilon set")
JYPPX_TRT10_VENDOR_MISSING_GET_DOUBLE(jyppx_trt10_normalization_layer_get_epsilon, out_epsilon, "normalization epsilon query")
JYPPX_TRT10_VENDOR_MISSING_SET_UINT(jyppx_trt10_normalization_layer_set_axes, axes, "normalization axes set")
JYPPX_TRT10_VENDOR_MISSING_GET_UINT(jyppx_trt10_normalization_layer_get_axes, out_axes, "normalization axes query")
JYPPX_TRT10_VENDOR_MISSING_SET_INT64(jyppx_trt10_normalization_layer_set_nb_groups, group_count, "normalization group count set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT64(jyppx_trt10_normalization_layer_get_nb_groups, out_group_count, "normalization group count query")
JYPPX_TRT10_VENDOR_MISSING_SET_INT(jyppx_trt10_normalization_layer_set_compute_precision, data_type, "normalization compute precision set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_normalization_layer_get_compute_precision, out_data_type, "normalization compute precision query")
JYPPX_TRT10_VENDOR_MISSING_SET_INT(jyppx_trt10_cast_layer_set_to_type, data_type, "cast output type set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_cast_layer_get_to_type, out_data_type, "cast output type query")
JYPPX_TRT10_VENDOR_MISSING_SET_INT(jyppx_trt10_nms_layer_set_bounding_box_format, format, "NMS bounding-box format set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_nms_layer_get_bounding_box_format, out_format, "NMS bounding-box format query")
JYPPX_TRT10_VENDOR_MISSING_SET_INT(jyppx_trt10_nms_layer_set_topk_box_limit, limit, "NMS TopK box limit set")
JYPPX_TRT10_VENDOR_MISSING_GET_INT(jyppx_trt10_nms_layer_get_topk_box_limit, out_limit, "NMS TopK box limit query")
JYPPX_TRT10_VENDOR_MISSING_SET_DIMS(jyppx_trt10_constant_layer_set_dimensions, dimensions, "constant dimensions set")
JYPPX_TRT10_VENDOR_MISSING_GET_DIMS(jyppx_trt10_constant_layer_get_dimensions, out_dimensions, "constant dimensions query")
JYPPX_StatusCode jyppx_trt10_einsum_layer_set_equation(JYPPX_TensorRtLayer*, const char*, JYPPX_Boolean* out_success) { auto status = jyppx::tensorrt::validate_output_pointer(out_success, "out_success"); if (status != JYPPX_STATUS_OK) { return status; } *out_success = JYPPX_FALSE; return jyppx::tensorrt::report_vendor_missing(kLine, "einsum equation set"); }
JYPPX_StatusCode jyppx_trt10_einsum_layer_get_equation(JYPPX_TensorRtLayer*, char*, size_t, size_t* out_required_size) { auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size"); if (status != JYPPX_STATUS_OK) { return status; } *out_required_size = 0; return jyppx::tensorrt::report_vendor_missing(kLine, "einsum equation query"); }

#undef JYPPX_TRT10_VENDOR_MISSING_GET_INT
#undef JYPPX_TRT10_VENDOR_MISSING_SET_INT
#undef JYPPX_TRT10_VENDOR_MISSING_GET_UINT
#undef JYPPX_TRT10_VENDOR_MISSING_SET_UINT
#undef JYPPX_TRT10_VENDOR_MISSING_GET_DOUBLE
#undef JYPPX_TRT10_VENDOR_MISSING_SET_DOUBLE
#undef JYPPX_TRT10_VENDOR_MISSING_GET_INT64
#undef JYPPX_TRT10_VENDOR_MISSING_SET_INT64
#undef JYPPX_TRT10_VENDOR_MISSING_GET_DIMS
#undef JYPPX_TRT10_VENDOR_MISSING_SET_DIMS
#undef JYPPX_TRT10_VENDOR_MISSING_GET_BOOL
#undef JYPPX_TRT10_VENDOR_MISSING_SET_BOOL
#endif
