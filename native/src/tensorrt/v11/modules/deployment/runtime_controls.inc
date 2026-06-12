JYPPX_StatusCode jyppx_trt11_engine_get_streamable_weights_size(JYPPX_TensorRtCudaEngine* engine, int64_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_size = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine streamable weights size query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_size = engine_payload->getStreamableWeightsSize();
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine streamable weights size query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_set_weight_streaming_budget_v2(JYPPX_TensorRtCudaEngine* engine, int64_t budget, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine weight streaming budget set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = engine_payload->setWeightStreamingBudgetV2(budget) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    (void)budget;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine weight streaming budget set");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_weight_streaming_budget_v2(JYPPX_TensorRtCudaEngine* engine, int64_t* out_budget)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_budget, "out_budget");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_budget = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine weight streaming budget query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_budget = engine_payload->getWeightStreamingBudgetV2();
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine weight streaming budget query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_weight_streaming_automatic_budget(JYPPX_TensorRtCudaEngine* engine, int64_t* out_budget)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_budget, "out_budget");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_budget = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine weight streaming automatic budget query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_budget = engine_payload->getWeightStreamingAutomaticBudget();
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine weight streaming automatic budget query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_weight_streaming_scratch_memory_size(JYPPX_TensorRtCudaEngine* engine, int64_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_size = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine weight streaming scratch memory query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_size = engine_payload->getWeightStreamingScratchMemorySize();
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine weight streaming scratch memory query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_engine_stat(JYPPX_TensorRtCudaEngine* engine, int32_t stat, int64_t* out_value)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_value, "out_value");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_value = -1;
    if (stat < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine stat query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_value = engine_payload->getEngineStat(static_cast<nvinfer1::EngineStat>(stat));
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine stat query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_get_hardware_compatibility_level(JYPPX_TensorRtCudaEngine* engine, int32_t* out_level)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_level, "out_level");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_level = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine hardware compatibility level query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_level = static_cast<int32_t>(engine_payload->getHardwareCompatibilityLevel());
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine hardware compatibility level query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_is_input_consumed_event_set(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_is_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_set, "out_is_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_set = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context input consumed event query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_set = context_payload->getInputConsumedEvent() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)context;
    return jyppx::cuda::report_cuda_dependency_missing("TensorRT execution context input consumed event query");
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context input consumed event query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_is_output_tensor_address_set(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_Boolean* out_bound)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_bound, "out_bound");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_bound = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context output tensor address query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_bound = context_payload->getOutputTensorAddress(tensor_name) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context output tensor address query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_has_output_allocator(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_Boolean* out_has_allocator)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_allocator, "out_has_allocator");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_allocator = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context output allocator query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_allocator = context_payload->getOutputAllocator(tensor_name) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context output allocator query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_has_temporary_storage_allocator(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_has_allocator)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_allocator, "out_has_allocator");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_allocator = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context temporary storage allocator query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_allocator = context_payload->getTemporaryStorageAllocator() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context temporary storage allocator query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_flags(JYPPX_TensorRtBuilderConfig* config, uint32_t flags)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    config_payload->setFlags(static_cast<nvinfer1::BuilderFlags>(flags));
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config flags set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_flags(JYPPX_TensorRtBuilderConfig* config, uint32_t* out_flags)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_flags, "out_flags");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_flags = 0;
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_flags = static_cast<uint32_t>(config_payload->getFlags());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config flags query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_default_device_type(JYPPX_TensorRtBuilderConfig* config, int32_t device_type)
{
    if (device_type < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    config_payload->setDefaultDeviceType(static_cast<nvinfer1::DeviceType>(device_type));
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config default device type set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_default_device_type(JYPPX_TensorRtBuilderConfig* config, int32_t* out_device_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_device_type, "out_device_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_device_type = 0;
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_device_type = static_cast<int32_t>(config_payload->getDefaultDeviceType());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config default device type query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_dla_core(JYPPX_TensorRtBuilderConfig* config, int32_t dla_core)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    config_payload->setDLACore(dla_core);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config DLA core set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_dla_core(JYPPX_TensorRtBuilderConfig* config, int32_t* out_dla_core)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_dla_core, "out_dla_core");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_dla_core = -1;
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_dla_core = config_payload->getDLACore();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config DLA core query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_tiling_optimization_level(JYPPX_TensorRtBuilderConfig* config, int32_t level, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;
    if (level < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_set = config_payload->setTilingOptimizationLevel(static_cast<nvinfer1::TilingOptimizationLevel>(level)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config tiling optimization level set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_tiling_optimization_level(JYPPX_TensorRtBuilderConfig* config, int32_t* out_level)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_level, "out_level");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_level = 0;
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_level = static_cast<int32_t>(config_payload->getTilingOptimizationLevel());
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config tiling optimization level query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_l2_limit_for_tiling(JYPPX_TensorRtBuilderConfig* config, int64_t bytes, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_set = config_payload->setL2LimitForTiling(bytes) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config L2 tiling limit set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_l2_limit_for_tiling(JYPPX_TensorRtBuilderConfig* config, int64_t* out_bytes)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_bytes, "out_bytes");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_bytes = 0;
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_bytes = config_payload->getL2LimitForTiling();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config L2 tiling limit query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_max_nb_tactics(JYPPX_TensorRtBuilderConfig* config, int32_t max_tactics)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    config_payload->setMaxNbTactics(max_tactics);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config max tactics set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_max_nb_tactics(JYPPX_TensorRtBuilderConfig* config, int32_t* out_max_tactics)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_max_tactics, "out_max_tactics");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_max_tactics = 0;
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_max_tactics = config_payload->getMaxNbTactics();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config max tactics query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_remote_auto_tuning_config(JYPPX_TensorRtBuilderConfig* config, const char* config_text, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_c_string(config_text, "config_text");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_set = config_payload->setRemoteAutoTuningConfig(config_text) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config remote auto tuning config set");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_remote_auto_tuning_config(JYPPX_TensorRtBuilderConfig* config, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    return copy_string_to_buffer(config_payload->getRemoteAutoTuningConfig(), output_buffer, output_buffer_size, out_required_size);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config remote auto tuning config query");
#endif
}
