namespace
{
JYPPX_StatusCode get_builder_payload_for_boundary(JYPPX_TensorRtBuilder* builder, nvinfer1::IBuilder** out_payload, const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK) { return status; }

    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    if (builder_payload == nullptr)
    {
        std::ostringstream message;
        message << "Builder handle does not carry the expected TensorRT payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, message.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_payload = builder_payload;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode validate_topk_indices_type(const int32_t indices_type)
{
#if JYPPX_HAS_TENSORRT
    const auto type = static_cast<nvinfer1::DataType>(indices_type);
    if (type == nvinfer1::DataType::kINT32 || type == nvinfer1::DataType::kINT64)
    {
        return JYPPX_STATUS_OK;
    }
#else
    (void)indices_type;
#endif

    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT TopK V2 indices type must be INT32 or INT64.");
    return JYPPX_STATUS_INVALID_ARGUMENT;
}
}

JYPPX_StatusCode jyppx_trt11_builder_get_max_dla_batch_size(JYPPX_TensorRtBuilder* builder, int32_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_size = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilder* builder_payload = nullptr;
    status = get_builder_payload_for_boundary(builder, &builder_payload, "builder maximum DLA batch size query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_size = builder_payload->getMaxDLABatchSize();
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder maximum DLA batch size query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_set_max_threads(JYPPX_TensorRtBuilder* builder, int32_t max_threads, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;
    if (max_threads < 1)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder max_threads must be greater than or equal to 1.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilder* builder_payload = nullptr;
    status = get_builder_payload_for_boundary(builder, &builder_payload, "builder max threads set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = builder_payload->setMaxThreads(max_threads) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    (void)max_threads;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder max threads set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_get_max_threads(JYPPX_TensorRtBuilder* builder, int32_t* out_max_threads)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_max_threads, "out_max_threads");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_max_threads = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilder* builder_payload = nullptr;
    status = get_builder_payload_for_boundary(builder, &builder_payload, "builder max threads query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_max_threads = builder_payload->getMaxThreads();
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder max threads query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_clear_gpu_allocator(JYPPX_TensorRtBuilder* builder)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilder* builder_payload = nullptr;
    auto status = get_builder_payload_for_boundary(builder, &builder_payload, "builder GPU allocator clear");
    if (status != JYPPX_STATUS_OK) { return status; }
    builder_payload->setGpuAllocator(nullptr);
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder GPU allocator clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_has_error_recorder(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilder* builder_payload = nullptr;
    status = get_builder_payload_for_boundary(builder, &builder_payload, "builder error recorder query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_recorder = builder_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder error recorder query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_clear_error_recorder(JYPPX_TensorRtBuilder* builder)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilder* builder_payload = nullptr;
    auto status = get_builder_payload_for_boundary(builder, &builder_payload, "builder error recorder clear");
    if (status != JYPPX_STATUS_OK) { return status; }
    builder_payload->setErrorRecorder(nullptr);
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder error recorder clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_reset(JYPPX_TensorRtBuilder* builder)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilder* builder_payload = nullptr;
    auto status = get_builder_payload_for_boundary(builder, &builder_payload, "builder reset");
    if (status != JYPPX_STATUS_OK) { return status; }
    builder_payload->reset();
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder reset");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_is_network_supported(
    JYPPX_TensorRtBuilder* builder,
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtBuilderConfig* config,
    JYPPX_Boolean* out_supported)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_supported, "out_supported");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_supported = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilder* builder_payload = nullptr;
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = get_builder_payload_for_boundary(builder, &builder_payload, "builder network support query");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_network_payload_for_api(network, &network_payload, "builder network support query");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_supported = builder_payload->isNetworkSupported(*network_payload, *config_payload) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    (void)network;
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder network support query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_has_error_recorder(JYPPX_TensorRtCudaEngine* engine, JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine error recorder query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_recorder = engine_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine error recorder query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_clear_error_recorder(JYPPX_TensorRtCudaEngine* engine)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    auto status = get_engine_payload_ext(engine, &engine_payload, "engine error recorder clear");
    if (status != JYPPX_STATUS_OK) { return status; }
    engine_payload->setErrorRecorder(nullptr);
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine error recorder clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_aliased_input_tensor(
    JYPPX_TensorRtCudaEngine* engine,
    const char* tensor_name,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine aliased input tensor query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(engine_payload->getAliasedInputTensor(tensor_name), output_buffer, output_buffer_size, out_required_size);
#else
    (void)engine;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine aliased input tensor query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_has_error_recorder(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context error recorder query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_recorder = context_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context error recorder query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_clear_error_recorder(JYPPX_TensorRtExecutionContext* context)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    auto status = get_context_payload_ext(context, &context_payload, "execution context error recorder clear");
    if (status != JYPPX_STATUS_OK) { return status; }
    context_payload->setErrorRecorder(nullptr);
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context error recorder clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_has_error_recorder(JYPPX_TensorRtNetworkDefinition* network, JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network error recorder query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_recorder = network_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)network;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network error recorder query");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_clear_error_recorder(JYPPX_TensorRtNetworkDefinition* network)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    auto status = get_network_payload_for_api(network, &network_payload, "network error recorder clear");
    if (status != JYPPX_STATUS_OK) { return status; }
    network_payload->setErrorRecorder(nullptr);
    return JYPPX_STATUS_OK;
#else
    (void)network;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network error recorder clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_remove_tensor(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    auto status = get_network_payload_for_api(network, &network_payload, "network tensor removal");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(tensor, &tensor_payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    network_payload->removeTensor(*tensor_payload);
    return JYPPX_STATUS_OK;
#else
    (void)network;
    (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network tensor removal");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_add_topk_v2(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtTensor* input_tensor,
    int32_t operation,
    int32_t k,
    uint32_t axes,
    int32_t indices_type,
    JYPPX_TensorRtLayer** out_layer)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_layer, "out_layer");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_layer = nullptr;
    if (k <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TopK k must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (axes == 0U)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TopK axes bitmask must not be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    status = validate_topk_indices_type(indices_type);
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* input = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network topk v2 layer creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(input_tensor, &input, "input_tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = create_layer_reference_handle(
        network_payload->addTopK(*input, static_cast<nvinfer1::TopKOperation>(operation), k, axes, static_cast<nvinfer1::DataType>(indices_type)),
        out_layer);
    if (status == JYPPX_STATUS_OK)
    {
        auto* payload = get_layer_reference_payload(*out_layer);
        if (payload != nullptr)
        {
            payload->topk_operation = operation;
            payload->topk_k = k;
            payload->topk_axes = axes;
        }
    }
    return status;
#else
    (void)network;
    (void)input_tensor;
    (void)operation;
    (void)indices_type;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network topk v2 layer creation");
#endif
}
