JYPPX_StatusCode jyppx_trt11_engine_get_profile_tensor_values_v2(
    JYPPX_TensorRtCudaEngine* engine,
    const char* tensor_name,
    int32_t profile_index,
    int32_t selector,
    int64_t* output_values,
    int32_t output_count,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    if (profile_index < 0 || selector < 0 || selector > 2 || output_count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profile tensor values V2 require a non-negative profile index, selector in [0,2], and a non-negative output count.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine profile tensor values V2 query");
    if (status != JYPPX_STATUS_OK) { return status; }

    const int64_t* values = engine_payload->getProfileTensorValuesV2(
        tensor_name,
        profile_index,
        static_cast<nvinfer1::OptProfileSelector>(selector));
    if (values == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    if (output_count == 0 || output_values == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Profile tensor values V2 output buffer is required when TensorRT returns values.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }

    for (int32_t i = 0; i < output_count; ++i)
    {
        output_values[i] = values[i];
    }

    *out_count = output_count;
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    (void)profile_index;
    (void)selector;
    (void)output_values;
    (void)output_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine profile tensor values V2 query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_inspector_clear_error_recorder(JYPPX_TensorRtEngineInspector* inspector)
{
    auto status = jyppx::tensorrt::validate_handle(inspector, kLine, JYPPX_TENSORRT_OBJECT_KIND_ENGINE_INSPECTOR, "inspector");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* inspector_payload = get_payload<nvinfer1::IEngineInspector>(inspector);
    if (inspector_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    inspector_payload->setErrorRecorder(nullptr);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine inspector error recorder clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_input_consumed_event_value(JYPPX_TensorRtExecutionContext* context, uintptr_t* out_event)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_event, "out_event");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_event = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context input-consumed event value query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_event = reinterpret_cast<uintptr_t>(context_payload->getInputConsumedEvent());
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context input-consumed event value query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_runtime_config_allocation_strategy(JYPPX_TensorRtExecutionContext* context, int32_t* out_strategy)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_strategy, "out_strategy");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_strategy = -1;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context runtime config allocation strategy query");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::IRuntimeConfig* config = context_payload->getRuntimeConfig();
    if (config == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context does not expose a runtime config.");
        return JYPPX_STATUS_NOT_FOUND;
    }

    *out_strategy = static_cast<int32_t>(config->getExecutionContextAllocationStrategy());
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context runtime config allocation strategy query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_engine_name(
    JYPPX_TensorRtExecutionContext* context,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context engine name query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(context_payload->getEngine().getName(), output_buffer, output_buffer_size, out_required_size);
#else
    (void)context;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context engine name query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_engine_io_tensor_count(JYPPX_TensorRtExecutionContext* context, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context engine I/O tensor count query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = context_payload->getEngine().getNbIOTensors();
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context engine I/O tensor count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_engine_layer_count(JYPPX_TensorRtExecutionContext* context, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context engine layer count query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = context_payload->getEngine().getNbLayers();
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context engine layer count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_engine_optimization_profile_count(JYPPX_TensorRtExecutionContext* context, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context engine optimization profile count query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = context_payload->getEngine().getNbOptimizationProfiles();
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context engine optimization profile count query");
#endif
}
