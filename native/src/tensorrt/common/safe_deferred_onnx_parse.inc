#ifndef JYPPX_TRT_SAFE_ONNX_PREFIX
#error "JYPPX_TRT_SAFE_ONNX_PREFIX must be defined before including safe_deferred_onnx_parse.inc"
#endif

#define JYPPX_TRT_SAFE_ONNX_CONCAT2(prefix, name) prefix##name
#define JYPPX_TRT_SAFE_ONNX_CONCAT(prefix, name) JYPPX_TRT_SAFE_ONNX_CONCAT2(prefix, name)
#define JYPPX_TRT_SAFE_ONNX_FN(name) JYPPX_TRT_SAFE_ONNX_CONCAT(JYPPX_TRT_SAFE_ONNX_PREFIX, name)

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
static JYPPX_StatusCode JYPPX_TRT_SAFE_ONNX_FN(parse_with_seh_guard)(
    nvonnxparser::IParser* parser,
    const void* model_data,
    size_t model_size,
    JYPPX_Boolean* out_parsed)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_parsed = parser->parseWithWeightDescriptors(model_data, model_size) ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_parsed = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, "ONNX parser parseWithWeightDescriptors", seh_exception_code);
    }
#else
    *out_parsed = parser->parseWithWeightDescriptors(model_data, model_size) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}
#endif

JYPPX_StatusCode JYPPX_TRT_SAFE_ONNX_FN(onnx_parser_parse_with_weight_descriptors)(
    JYPPX_TensorRtOnnxParser* parser,
    const void* model_data,
    size_t model_size,
    JYPPX_Boolean* out_parsed)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_parsed, "out_parsed");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_parsed = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (model_data == nullptr || model_size == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX model data must not be null or empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != JYPPX_TRT_SAFE_ONNX_EXPECTED_MAJOR)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser parseWithWeightDescriptors");
    }

    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser handle does not carry the expected TensorRT parser payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    try
    {
        status = JYPPX_TRT_SAFE_ONNX_FN(parse_with_seh_guard)(parser_payload, model_data, model_size, out_parsed);
        if (status == JYPPX_STATUS_OK && *out_parsed == JYPPX_FALSE)
        {
            set_parser_failure_message(*parser_payload);
        }
        return status;
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX parser parseWithWeightDescriptors", exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, "ONNX parser parseWithWeightDescriptors", "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser parseWithWeightDescriptors");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser parseWithWeightDescriptors");
#endif
}

#undef JYPPX_TRT_SAFE_ONNX_FN
#undef JYPPX_TRT_SAFE_ONNX_CONCAT
#undef JYPPX_TRT_SAFE_ONNX_CONCAT2
