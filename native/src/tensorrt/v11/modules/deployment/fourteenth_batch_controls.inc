JYPPX_StatusCode jyppx_trt11_builder_build_engine_with_config(
    JYPPX_TensorRtBuilder* builder,
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtBuilderConfig* config,
    JYPPX_TensorRtCudaEngine** out_engine)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_engine, "out_engine");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_engine = nullptr;

#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilder* builder_payload = nullptr;
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = get_builder_payload_for_boundary(builder, &builder_payload, "builder direct engine build");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_network_payload_for_api(network, &network_payload, "builder direct engine build");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    status = ensure_minimal_identity_network(*network_payload);
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::ICudaEngine* engine = builder_payload->buildEngineWithConfig(*network_payload, *config_payload);
    if (engine == nullptr) { return report_null_vendor_object("buildEngineWithConfig"); }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE, engine, &destroy_payload<nvinfer1::ICudaEngine>);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_engine = reinterpret_cast<JYPPX_TensorRtCudaEngine*>(handle);
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    (void)network;
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder direct engine build");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_build_serialized_network_with_kernel_text(
    JYPPX_TensorRtBuilder* builder,
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_TensorRtBuilderConfig* config,
    JYPPX_TensorRtHostMemory** out_host_memory,
    JYPPX_TensorRtHostMemory** out_kernel_text)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_host_memory, "out_host_memory");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_kernel_text, "out_kernel_text");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_host_memory = nullptr;
    *out_kernel_text = nullptr;

#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilder* builder_payload = nullptr;
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = get_builder_payload_for_boundary(builder, &builder_payload, "serialized network build with kernel text");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_network_payload_for_api(network, &network_payload, "serialized network build with kernel text");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    status = ensure_minimal_identity_network(*network_payload);
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::IHostMemory* kernel_text = nullptr;
    nvinfer1::IHostMemory* host_memory = builder_payload->buildSerializedNetwork(*network_payload, *config_payload, kernel_text);
    if (host_memory == nullptr)
    {
        if (kernel_text != nullptr) { delete kernel_text; }
        return report_null_vendor_object("buildSerializedNetwork(kernelText)");
    }

    JYPPX_TensorRtObjectBase* host_handle = nullptr;
    status = create_handle_with_payload(&host_handle, JYPPX_TENSORRT_OBJECT_KIND_HOST_MEMORY, host_memory, &destroy_payload<nvinfer1::IHostMemory>);
    if (status != JYPPX_STATUS_OK)
    {
        if (kernel_text != nullptr) { delete kernel_text; }
        return status;
    }

    JYPPX_TensorRtObjectBase* kernel_handle = nullptr;
    if (kernel_text != nullptr)
    {
        status = create_handle_with_payload(&kernel_handle, JYPPX_TENSORRT_OBJECT_KIND_HOST_MEMORY, kernel_text, &destroy_payload<nvinfer1::IHostMemory>);
        if (status != JYPPX_STATUS_OK)
        {
            jyppx_trt_object_destroy(reinterpret_cast<JYPPX_TensorRtObjectBase*>(host_handle));
            return status;
        }
    }

    *out_host_memory = reinterpret_cast<JYPPX_TensorRtHostMemory*>(host_handle);
    *out_kernel_text = reinterpret_cast<JYPPX_TensorRtHostMemory*>(kernel_handle);
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    (void)network;
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "serialized network build with kernel text");
#endif
}

JYPPX_StatusCode jyppx_trt11_host_memory_get_type(JYPPX_TensorRtHostMemory* host_memory, int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = -1;
    status = jyppx::tensorrt::validate_handle(host_memory, kLine, JYPPX_TENSORRT_OBJECT_KIND_HOST_MEMORY, "host_memory");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* payload = get_payload<nvinfer1::IHostMemory>(host_memory);
    if (payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_data_type = static_cast<int32_t>(payload->type());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "host memory data type query");
#endif
}

JYPPX_StatusCode jyppx_trt11_optimization_profile_set_shape_values_v2(
    JYPPX_TensorRtOptimizationProfile* profile,
    const char* input_name,
    int32_t selector,
    const int64_t* values,
    int32_t value_count,
    JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;
    status = validate_c_string(input_name, "input_name");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (selector < 0 || selector > 2 || values == nullptr || value_count <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Shape-value V2 selector must be [0,2] and values must be non-empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT
    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    *out_set = profile_payload->setShapeValuesV2(input_name, static_cast<nvinfer1::OptProfileSelector>(selector), values, value_count)
        ? JYPPX_TRUE
        : JYPPX_FALSE;
    if (*out_set == JYPPX_FALSE)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IOptimizationProfile::setShapeValuesV2 returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile shape values V2 set");
#endif
}

JYPPX_StatusCode jyppx_trt11_optimization_profile_get_shape_value_count_v2(
    JYPPX_TensorRtOptimizationProfile* profile,
    const char* input_name,
    int32_t* out_count)
{
    return jyppx_trt11_optimization_profile_get_shape_value_count(profile, input_name, out_count);
}

JYPPX_StatusCode jyppx_trt11_optimization_profile_get_shape_values_v2(
    JYPPX_TensorRtOptimizationProfile* profile,
    const char* input_name,
    int32_t selector,
    int64_t* output_values,
    int32_t output_count,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
    status = validate_c_string(input_name, "input_name");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (selector < 0 || selector > 2 || output_count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Shape-value V2 selector must be [0,2] and output count must be non-negative.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    status = jyppx::tensorrt::validate_handle(profile, kLine, JYPPX_TENSORRT_OBJECT_KIND_OPTIMIZATION_PROFILE, "profile");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT
    auto* profile_payload = get_payload<nvinfer1::IOptimizationProfile>(profile);
    if (profile_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    const int32_t count = profile_payload->getNbShapeValues(input_name);
    *out_count = count;
    if (count <= 0) { return JYPPX_STATUS_OK; }
    if (output_values == nullptr || output_count < count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Shape-value V2 output buffer is too small.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }
    const int64_t* values = profile_payload->getShapeValuesV2(input_name, static_cast<nvinfer1::OptProfileSelector>(selector));
    if (values == nullptr) { return JYPPX_STATUS_OK; }
    for (int32_t i = 0; i < count; ++i) { output_values[i] = values[i]; }
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "optimization profile shape values V2 query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_clear_flag(JYPPX_TensorRtBuilderConfig* config, int32_t flag)
{
    if (flag < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    config_payload->clearFlag(static_cast<nvinfer1::BuilderFlag>(flag));
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config flag clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_plugins_to_serialize(
    JYPPX_TensorRtBuilderConfig* config,
    const char** paths,
    int32_t path_count,
    JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;
    if (path_count < 0 || (path_count > 0 && paths == nullptr))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Plugin serialization path array must be present when path_count is positive.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    for (int32_t i = 0; i < path_count; ++i)
    {
        status = validate_c_string(paths[i], "plugin_path");
        if (status != JYPPX_STATUS_OK) { return status; }
    }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    config_payload->setPluginsToSerialize(paths, path_count);
    *out_set = JYPPX_TRUE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config plugin serialization path set");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_clear_tensor_address(
    JYPPX_TensorRtExecutionContext* context,
    const char* tensor_name,
    JYPPX_Boolean* out_cleared)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_cleared, "out_cleared");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cleared = JYPPX_FALSE;
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context tensor address clear");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cleared = context_payload->setTensorAddress(tensor_name, nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context tensor address clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_clear_device_memory(JYPPX_TensorRtExecutionContext* context)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    auto status = get_context_payload_ext(context, &context_payload, "execution context device memory clear");
    if (status != JYPPX_STATUS_OK) { return status; }
    context_payload->setDeviceMemoryV2(nullptr, 0);
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context device memory clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_clear_input_tensor_address(
    JYPPX_TensorRtExecutionContext* context,
    const char* tensor_name,
    JYPPX_Boolean* out_cleared)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_cleared, "out_cleared");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cleared = JYPPX_FALSE;
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context input tensor address clear");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cleared = context_payload->setInputTensorAddress(tensor_name, nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context input tensor address clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_clear_output_tensor_address(
    JYPPX_TensorRtExecutionContext* context,
    const char* tensor_name,
    JYPPX_Boolean* out_cleared)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_cleared, "out_cleared");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cleared = JYPPX_FALSE;
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context output tensor address clear");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cleared = context_payload->setOutputTensorAddress(tensor_name, nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context output tensor address clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_clear_input_consumed_event(
    JYPPX_TensorRtExecutionContext* context,
    JYPPX_Boolean* out_cleared)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_cleared, "out_cleared");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cleared = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context input consumed event clear");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cleared = context_payload->setInputConsumedEvent(nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)context;
    return jyppx::cuda::report_cuda_dependency_missing("TensorRT execution context input consumed event clear");
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context input consumed event clear");
#endif
}
