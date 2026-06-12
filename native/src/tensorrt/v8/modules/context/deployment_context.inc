JYPPX_StatusCode jyppx_trt8_execution_context_set_device_memory(JYPPX_TensorRtExecutionContext* context, JYPPX_CudaMemory* memory)
{
    auto status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    auto* context_payload = get_payload<nvinfer1::IExecutionContext>(context);
    auto* memory_payload = reinterpret_cast<jyppx::cuda::MemoryObject*>(memory);
    if (context_payload == nullptr || memory_payload == nullptr || memory_payload->pointer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context or CUDA memory handle does not carry the expected payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    context_payload->setDeviceMemory(memory_payload->pointer);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::cuda::report_cuda_dependency_missing("TensorRT execution context device memory set");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context device memory set");
#endif
}

JYPPX_StatusCode jyppx_trt8_execution_context_get_max_output_size(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, int64_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_size = -1;
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context maximum output size query");
    if (status != JYPPX_STATUS_OK) { return status; }

    *out_size = context_payload->getMaxOutputSize(tensor_name);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)context;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context maximum output size query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context maximum output size query");
#endif
}

JYPPX_StatusCode jyppx_trt8_execution_context_set_tensor_debug_state(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_Boolean debug_state)
{
    auto status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    (void)debug_state;

    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 does not expose IExecutionContext tensor debug state APIs.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode jyppx_trt8_execution_context_get_tensor_debug_state(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_Boolean* out_debug_state)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_debug_state, "out_debug_state");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_debug_state = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }

    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 does not expose IExecutionContext tensor debug state APIs.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode jyppx_trt8_execution_context_set_all_tensors_debug_state(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean debug_state)
{
    auto status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }
    (void)debug_state;

    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 does not expose IExecutionContext tensor debug state APIs.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode jyppx_trt8_execution_context_get_device_memory_size(JYPPX_TensorRtExecutionContext* context, size_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_size = 0;
    status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 execution context does not expose context-scoped device memory size. Query engine device memory size instead.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode jyppx_trt8_execution_context_update_device_memory_size_for_shapes(JYPPX_TensorRtExecutionContext* context, size_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_size = 0;
    status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 does not expose updateDeviceMemorySizeForShapes.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode jyppx_trt8_execution_context_set_persistent_cache_limit(JYPPX_TensorRtExecutionContext* context, size_t cache_size)
{
    auto status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT
    auto* context_payload = get_payload<nvinfer1::IExecutionContext>(context);
    if (context_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    context_payload->setPersistentCacheLimit(cache_size);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context persistent cache limit set");
#endif
}

JYPPX_StatusCode jyppx_trt8_execution_context_get_persistent_cache_limit(JYPPX_TensorRtExecutionContext* context, size_t* out_cache_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_cache_size, "out_cache_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cache_size = 0;
    status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT
    auto* context_payload = get_payload<nvinfer1::IExecutionContext>(context);
    if (context_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    *out_cache_size = context_payload->getPersistentCacheLimit();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context persistent cache limit query");
#endif
}

JYPPX_StatusCode jyppx_trt8_execution_context_set_input_consumed_event(JYPPX_TensorRtExecutionContext* context, JYPPX_CudaEvent* event)
{
    auto status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_event(event, "event");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    auto* context_payload = get_payload<nvinfer1::IExecutionContext>(context);
    auto* event_payload = reinterpret_cast<jyppx::cuda::EventObject*>(event);
    if (context_payload == nullptr || event_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context or CUDA event handle does not carry the expected payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    const bool result = context_payload->setInputConsumedEvent(event_payload->handle);
    if (!result)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::setInputConsumedEvent returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::cuda::report_cuda_dependency_missing("TensorRT execution context input consumed event set");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context input consumed event set");
#endif
}
