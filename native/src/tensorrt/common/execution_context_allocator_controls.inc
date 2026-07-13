#ifndef JYPPX_TRT_EXECUTION_CONTEXT_ALLOCATOR_API
#error "JYPPX_TRT_EXECUTION_CONTEXT_ALLOCATOR_API must be defined before including execution_context_allocator_controls.inc"
#endif

#if JYPPX_HAS_TENSORRT
static JYPPX_StatusCode jyppx_trt_execution_context_allocator_control_exception(
    const char* feature_name,
    const std::exception& exception)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
}

static JYPPX_StatusCode jyppx_trt_execution_context_allocator_control_unknown_exception(const char* feature_name)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
}

static JYPPX_StatusCode execution_context_has_output_allocator_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    const char* tensor_name,
    JYPPX_Boolean* out_has_allocator,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_allocator = context_payload->getOutputAllocator(tensor_name) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_allocator = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_allocator = context_payload->getOutputAllocator(tensor_name) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode execution_context_has_temporary_storage_allocator_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    JYPPX_Boolean* out_has_allocator,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_allocator = context_payload->getTemporaryStorageAllocator() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_allocator = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_allocator = context_payload->getTemporaryStorageAllocator() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode execution_context_clear_output_allocator_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    const char* tensor_name,
    JYPPX_Boolean* out_cleared,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_cleared = context_payload->setOutputAllocator(tensor_name, nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_cleared = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_cleared = context_payload->setOutputAllocator(tensor_name, nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode execution_context_clear_temporary_storage_allocator_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    JYPPX_Boolean* out_cleared,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_cleared = context_payload->setTemporaryStorageAllocator(nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_cleared = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_cleared = context_payload->setTemporaryStorageAllocator(nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}
#endif

JYPPX_StatusCode JYPPX_TRT_EXECUTION_CONTEXT_ALLOCATOR_API(execution_context_has_output_allocator)(
    JYPPX_TensorRtExecutionContext* context,
    const char* tensor_name,
    JYPPX_Boolean* out_has_allocator)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_allocator, "out_has_allocator");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_allocator = JYPPX_FALSE;
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context output allocator query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return execution_context_has_output_allocator_with_seh_guard(
            context_payload,
            tensor_name,
            out_has_allocator,
            "execution context output allocator query");
    }
    catch (const std::exception& exception)
    {
        *out_has_allocator = JYPPX_FALSE;
        return jyppx_trt_execution_context_allocator_control_exception("execution context output allocator query", exception);
    }
    catch (...)
    {
        *out_has_allocator = JYPPX_FALSE;
        return jyppx_trt_execution_context_allocator_control_unknown_exception("execution context output allocator query");
    }
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context output allocator query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_EXECUTION_CONTEXT_ALLOCATOR_API(execution_context_has_temporary_storage_allocator)(
    JYPPX_TensorRtExecutionContext* context,
    JYPPX_Boolean* out_has_allocator)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_allocator, "out_has_allocator");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_allocator = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context temporary storage allocator query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return execution_context_has_temporary_storage_allocator_with_seh_guard(
            context_payload,
            out_has_allocator,
            "execution context temporary storage allocator query");
    }
    catch (const std::exception& exception)
    {
        *out_has_allocator = JYPPX_FALSE;
        return jyppx_trt_execution_context_allocator_control_exception("execution context temporary storage allocator query", exception);
    }
    catch (...)
    {
        *out_has_allocator = JYPPX_FALSE;
        return jyppx_trt_execution_context_allocator_control_unknown_exception("execution context temporary storage allocator query");
    }
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context temporary storage allocator query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_EXECUTION_CONTEXT_ALLOCATOR_API(execution_context_clear_output_allocator)(
    JYPPX_TensorRtExecutionContext* context,
    const char* tensor_name,
    JYPPX_Boolean* out_cleared)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_cleared, "out_cleared");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_cleared = JYPPX_FALSE;
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context output allocator clear");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return execution_context_clear_output_allocator_with_seh_guard(
            context_payload,
            tensor_name,
            out_cleared,
            "execution context output allocator clear");
    }
    catch (const std::exception& exception)
    {
        *out_cleared = JYPPX_FALSE;
        return jyppx_trt_execution_context_allocator_control_exception("execution context output allocator clear", exception);
    }
    catch (...)
    {
        *out_cleared = JYPPX_FALSE;
        return jyppx_trt_execution_context_allocator_control_unknown_exception("execution context output allocator clear");
    }
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context output allocator clear");
#endif
}

JYPPX_StatusCode JYPPX_TRT_EXECUTION_CONTEXT_ALLOCATOR_API(execution_context_clear_temporary_storage_allocator)(
    JYPPX_TensorRtExecutionContext* context,
    JYPPX_Boolean* out_cleared)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_cleared, "out_cleared");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_cleared = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context temporary storage allocator clear");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return execution_context_clear_temporary_storage_allocator_with_seh_guard(
            context_payload,
            out_cleared,
            "execution context temporary storage allocator clear");
    }
    catch (const std::exception& exception)
    {
        *out_cleared = JYPPX_FALSE;
        return jyppx_trt_execution_context_allocator_control_exception("execution context temporary storage allocator clear", exception);
    }
    catch (...)
    {
        *out_cleared = JYPPX_FALSE;
        return jyppx_trt_execution_context_allocator_control_unknown_exception("execution context temporary storage allocator clear");
    }
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context temporary storage allocator clear");
#endif
}
