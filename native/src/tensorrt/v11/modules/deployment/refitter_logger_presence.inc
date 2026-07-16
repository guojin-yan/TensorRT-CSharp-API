namespace
{
#if JYPPX_HAS_TENSORRT
static JYPPX_StatusCode trt11_refitter_has_logger_with_seh_guard(
    nvinfer1::IRefitter* refitter_payload,
    JYPPX_Boolean* out_has_logger,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_logger = refitter_payload->getLogger() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_logger = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_logger = refitter_payload->getLogger() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}
#endif
}

JYPPX_StatusCode jyppx_trt11_refitter_has_logger(
    JYPPX_TensorRtRefitter* refitter,
    JYPPX_Boolean* out_has_logger)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_logger, "out_has_logger");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_logger = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_ext(refitter, &payload, "refitter logger presence query");
    if (status != JYPPX_STATUS_OK) { return status; }
    try
    {
        return trt11_refitter_has_logger_with_seh_guard(
            payload,
            out_has_logger,
            "refitter logger presence query");
    }
    catch (const std::exception& exception)
    {
        *out_has_logger = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_exception(
            kLine,
            "refitter logger presence query",
            exception.what());
    }
    catch (...)
    {
        *out_has_logger = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_exception(
            kLine,
            "refitter logger presence query",
            "unknown native exception");
    }
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter logger presence query");
#endif
}
