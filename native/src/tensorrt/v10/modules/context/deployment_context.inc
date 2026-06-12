JYPPX_StatusCode jyppx_trt10_execution_context_set_device_memory(JYPPX_TensorRtExecutionContext* context, JYPPX_CudaMemory* memory)
{
    auto status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10 && JYPPX_HAS_CUDA_TOOLKIT
    auto* context_payload = get_payload<nvinfer1::IExecutionContext>(context);
    auto* memory_payload = reinterpret_cast<jyppx::cuda::MemoryObject*>(memory);
    if (context_payload == nullptr || memory_payload == nullptr || memory_payload->pointer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context or CUDA memory handle does not carry the expected payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    context_payload->setDeviceMemoryV2(memory_payload->pointer, static_cast<int64_t>(memory_payload->size));
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context device memory set");
#elif JYPPX_HAS_TENSORRT
    return jyppx::cuda::report_cuda_dependency_missing("TensorRT execution context device memory set");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context device memory set");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_get_max_output_size(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, int64_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_size = -1;
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
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

JYPPX_StatusCode jyppx_trt10_execution_context_set_tensor_debug_state(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_Boolean debug_state)
{
    auto status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context tensor debug state set");
    if (status != JYPPX_STATUS_OK) { return status; }

    if (!context_payload->setTensorDebugState(tensor_name, debug_state != JYPPX_FALSE))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::setTensorDebugState returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)context;
    (void)debug_state;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context tensor debug state set");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context tensor debug state set");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_get_tensor_debug_state(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_Boolean* out_debug_state)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_debug_state, "out_debug_state");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_debug_state = JYPPX_FALSE;
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context tensor debug state query");
    if (status != JYPPX_STATUS_OK) { return status; }

    *out_debug_state = context_payload->getDebugState(tensor_name) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)context;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context tensor debug state query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context tensor debug state query");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_set_all_tensors_debug_state(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean debug_state)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    nvinfer1::IExecutionContext* context_payload = nullptr;
    auto status = get_context_payload_ext(context, &context_payload, "execution context all tensors debug state set");
    if (status != JYPPX_STATUS_OK) { return status; }

    if (!context_payload->setAllTensorsDebugState(debug_state != JYPPX_FALSE))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::setAllTensorsDebugState returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)context;
    (void)debug_state;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context all tensors debug state set");
#else
    (void)context;
    (void)debug_state;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context all tensors debug state set");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_get_device_memory_size(JYPPX_TensorRtExecutionContext* context, size_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_size = 0;
    status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    jyppx::common::set_last_error(
        JYPPX_ERROR_CATEGORY_TENSORRT,
        "TensorRT 10 IExecutionContext does not expose a context-scoped device memory size query. "
        "Use engine device memory size or updateDeviceMemorySizeForShapes after shapes are specified.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context device memory size query");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context device memory size query");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_update_device_memory_size_for_shapes(JYPPX_TensorRtExecutionContext* context, size_t* out_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_size = 0;
    status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 10
    auto* context_payload = get_payload<nvinfer1::IExecutionContext>(context);
    if (context_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_size = context_payload->updateDeviceMemorySizeForShapes();
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context shape device memory size update");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context shape device memory size update");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_set_persistent_cache_limit(JYPPX_TensorRtExecutionContext* context, size_t cache_size)
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

JYPPX_StatusCode jyppx_trt10_execution_context_get_persistent_cache_limit(JYPPX_TensorRtExecutionContext* context, size_t* out_cache_size)
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

JYPPX_StatusCode jyppx_trt10_execution_context_set_input_consumed_event(JYPPX_TensorRtExecutionContext* context, JYPPX_CudaEvent* event)
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
