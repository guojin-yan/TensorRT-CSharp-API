#ifndef JYPPX_TRT_EXECUTION_CONTEXT_DEBUG_LISTENER_API
#error "JYPPX_TRT_EXECUTION_CONTEXT_DEBUG_LISTENER_API must be defined before including execution_context_debug_listener_controls.inc"
#endif

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
static JYPPX_StatusCode execution_context_debug_listener_control_exception(
    const char* feature_name,
    const std::exception& exception)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
}

static JYPPX_StatusCode execution_context_debug_listener_control_unknown_exception(const char* feature_name)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
}

static JYPPX_StatusCode execution_context_has_debug_listener_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    JYPPX_Boolean* out_has_listener,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_listener = context_payload->getDebugListener() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_listener = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_listener = context_payload->getDebugListener() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode execution_context_clear_debug_listener_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    JYPPX_Boolean* out_cleared,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_cleared = context_payload->setDebugListener(nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_cleared = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_cleared = context_payload->setDebugListener(nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}
#endif

JYPPX_StatusCode JYPPX_TRT_EXECUTION_CONTEXT_DEBUG_LISTENER_API(execution_context_has_debug_listener)(
    JYPPX_TensorRtExecutionContext* context,
    JYPPX_Boolean* out_has_listener)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_listener, "out_has_listener");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_listener = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context debug listener query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return execution_context_has_debug_listener_with_seh_guard(
            context_payload,
            out_has_listener,
            "execution context debug listener query");
    }
    catch (const std::exception& exception)
    {
        *out_has_listener = JYPPX_FALSE;
        return execution_context_debug_listener_control_exception("execution context debug listener query", exception);
    }
    catch (...)
    {
        *out_has_listener = JYPPX_FALSE;
        return execution_context_debug_listener_control_unknown_exception("execution context debug listener query");
    }
#elif JYPPX_HAS_TENSORRT
    (void)context;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context debug listener query");
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context debug listener query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_EXECUTION_CONTEXT_DEBUG_LISTENER_API(execution_context_clear_debug_listener)(
    JYPPX_TensorRtExecutionContext* context,
    JYPPX_Boolean* out_cleared)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_cleared, "out_cleared");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_cleared = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context debug listener clear");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return execution_context_clear_debug_listener_with_seh_guard(
            context_payload,
            out_cleared,
            "execution context debug listener clear");
    }
    catch (const std::exception& exception)
    {
        *out_cleared = JYPPX_FALSE;
        return execution_context_debug_listener_control_exception("execution context debug listener clear", exception);
    }
    catch (...)
    {
        *out_cleared = JYPPX_FALSE;
        return execution_context_debug_listener_control_unknown_exception("execution context debug listener clear");
    }
#elif JYPPX_HAS_TENSORRT
    (void)context;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context debug listener clear");
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context debug listener clear");
#endif
}
