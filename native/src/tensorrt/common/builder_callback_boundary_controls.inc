#ifndef JYPPX_TRT_BUILDER_CALLBACK_BOUNDARY_API
#error "JYPPX_TRT_BUILDER_CALLBACK_BOUNDARY_API must be defined before including builder_callback_boundary_controls.inc"
#endif

#if JYPPX_HAS_TENSORRT
static JYPPX_StatusCode builder_callback_boundary_control_exception(
    const char* feature_name,
    const std::exception& exception)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
}

static JYPPX_StatusCode builder_callback_boundary_control_unknown_exception(const char* feature_name)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
}

static JYPPX_StatusCode get_builder_payload_for_callback_boundary(
    JYPPX_TensorRtBuilder* builder,
    nvinfer1::IBuilder** out_payload,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

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

static JYPPX_StatusCode builder_clear_gpu_allocator_with_seh_guard(
    nvinfer1::IBuilder* builder_payload,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        builder_payload->setGpuAllocator(nullptr);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    builder_payload->setGpuAllocator(nullptr);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode builder_has_error_recorder_with_seh_guard(
    nvinfer1::IBuilder* builder_payload,
    JYPPX_Boolean* out_has_recorder,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_recorder = builder_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_recorder = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_recorder = builder_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode builder_has_logger_with_seh_guard(
    nvinfer1::IBuilder* builder_payload,
    JYPPX_Boolean* out_has_logger,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_logger = builder_payload->getLogger() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_logger = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_logger = builder_payload->getLogger() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode builder_clear_error_recorder_with_seh_guard(
    nvinfer1::IBuilder* builder_payload,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        builder_payload->setErrorRecorder(nullptr);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    builder_payload->setErrorRecorder(nullptr);
    return JYPPX_STATUS_OK;
#endif
}
#endif

JYPPX_StatusCode JYPPX_TRT_BUILDER_CALLBACK_BOUNDARY_API(builder_clear_gpu_allocator)(JYPPX_TensorRtBuilder* builder)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilder* builder_payload = nullptr;
    auto status = get_builder_payload_for_callback_boundary(builder, &builder_payload, "builder GPU allocator clear");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return builder_clear_gpu_allocator_with_seh_guard(
            builder_payload,
            "builder GPU allocator clear");
    }
    catch (const std::exception& exception)
    {
        return builder_callback_boundary_control_exception("builder GPU allocator clear", exception);
    }
    catch (...)
    {
        return builder_callback_boundary_control_unknown_exception("builder GPU allocator clear");
    }
#else
    (void)builder;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder GPU allocator clear");
#endif
}

JYPPX_StatusCode JYPPX_TRT_BUILDER_CALLBACK_BOUNDARY_API(builder_has_error_recorder)(
    JYPPX_TensorRtBuilder* builder,
    JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilder* builder_payload = nullptr;
    status = get_builder_payload_for_callback_boundary(builder, &builder_payload, "builder error recorder query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return builder_has_error_recorder_with_seh_guard(
            builder_payload,
            out_has_recorder,
            "builder error recorder query");
    }
    catch (const std::exception& exception)
    {
        *out_has_recorder = JYPPX_FALSE;
        return builder_callback_boundary_control_exception("builder error recorder query", exception);
    }
    catch (...)
    {
        *out_has_recorder = JYPPX_FALSE;
        return builder_callback_boundary_control_unknown_exception("builder error recorder query");
    }
#else
    (void)builder;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder error recorder query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_BUILDER_CALLBACK_BOUNDARY_API(builder_has_logger)(
    JYPPX_TensorRtBuilder* builder,
    JYPPX_Boolean* out_has_logger)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_logger, "out_has_logger");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_logger = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilder* builder_payload = nullptr;
    status = get_builder_payload_for_callback_boundary(builder, &builder_payload, "builder logger presence query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return builder_has_logger_with_seh_guard(
            builder_payload,
            out_has_logger,
            "builder logger presence query");
    }
    catch (const std::exception& exception)
    {
        *out_has_logger = JYPPX_FALSE;
        return builder_callback_boundary_control_exception("builder logger presence query", exception);
    }
    catch (...)
    {
        *out_has_logger = JYPPX_FALSE;
        return builder_callback_boundary_control_unknown_exception("builder logger presence query");
    }
#else
    (void)builder;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder logger presence query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_BUILDER_CALLBACK_BOUNDARY_API(builder_clear_error_recorder)(JYPPX_TensorRtBuilder* builder)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilder* builder_payload = nullptr;
    auto status = get_builder_payload_for_callback_boundary(builder, &builder_payload, "builder error recorder clear");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return builder_clear_error_recorder_with_seh_guard(
            builder_payload,
            "builder error recorder clear");
    }
    catch (const std::exception& exception)
    {
        return builder_callback_boundary_control_exception("builder error recorder clear", exception);
    }
    catch (...)
    {
        return builder_callback_boundary_control_unknown_exception("builder error recorder clear");
    }
#else
    (void)builder;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder error recorder clear");
#endif
}
