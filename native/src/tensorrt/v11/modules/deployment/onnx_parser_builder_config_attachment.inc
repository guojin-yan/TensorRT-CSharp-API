namespace
{
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11
static JYPPX_StatusCode trt11_onnx_parser_set_builder_config_with_seh_guard(
    nvonnxparser::IParser* parser,
    const nvinfer1::IBuilderConfig* config,
    JYPPX_Boolean* out_set,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_set = parser->setBuilderConfig(config) ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_set = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_set = parser->setBuilderConfig(config) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}
#endif
}

JYPPX_StatusCode jyppx_trt11_onnx_parser_set_builder_config_safe(
    JYPPX_TensorRtOnnxParser* parser,
    JYPPX_TensorRtBuilderConfig* config,
    JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;

    status = jyppx::tensorrt::validate_handle(
        parser,
        kLine,
        JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER,
        "parser");
    if (status != JYPPX_STATUS_OK) { return status; }

    status = jyppx::tensorrt::validate_handle(
        config,
        kLine,
        JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG,
        "config");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 11
    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (parser_payload == nullptr || config_payload == nullptr)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "ONNX parser builder-config attachment requires live parser and builder-config payloads.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    try
    {
        return trt11_onnx_parser_set_builder_config_with_seh_guard(
            parser_payload,
            config_payload,
            out_set,
            "ONNX parser builder-config attachment");
    }
    catch (const std::exception& exception)
    {
        *out_set = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_exception(
            kLine,
            "ONNX parser builder-config attachment",
            exception.what());
    }
    catch (...)
    {
        *out_set = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_exception(
            kLine,
            "ONNX parser builder-config attachment",
            "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(
        JYPPX_ERROR_CATEGORY_TENSORRT,
        "TensorRT 11 ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser builder-config attachment");
#endif
}
