JYPPX_StatusCode jyppx_trt10_onnx_parser_create(JYPPX_TensorRtLogger* logger, JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtOnnxParser** out_parser)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_parser, "out_parser");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(logger, kLine, JYPPX_TENSORRT_OBJECT_KIND_LOGGER, "logger");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_parser = nullptr;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser creation");
    }

    auto* logger_payload = get_payload<ManagedLogger>(logger);
    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (logger_payload == nullptr || network_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Logger or network handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network_payload, *logger_payload);
    if (parser == nullptr)
    {
        return report_null_vendor_object("nvonnxparser::createParser");
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, parser, &destroy_parser_payload);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_parser = reinterpret_cast<JYPPX_TensorRtOnnxParser*>(handle);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser creation");
#endif
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_parse_from_file(JYPPX_TensorRtOnnxParser* parser, const char* file_path, int32_t verbosity, JYPPX_Boolean* out_parsed)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_parsed, "out_parsed");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(file_path, "file_path");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_parsed = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser parse from file");
    }

    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser handle does not carry the expected TensorRT parser payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    mark_onnx_parser_support_ready(parser_payload, false);
    const bool parsed = parser_payload->parseFromFile(file_path, verbosity);
    *out_parsed = parsed ? JYPPX_TRUE : JYPPX_FALSE;
    if (!parsed)
    {
        set_parser_failure_message(*parser_payload);
    }

    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser parse from file");
#endif
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_parse_from_memory(JYPPX_TensorRtOnnxParser* parser, const void* model_data, size_t model_size, const char* model_path, JYPPX_Boolean* out_parsed)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_parsed, "out_parsed");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

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

    *out_parsed = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser parse from memory");
    }

    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser handle does not carry the expected TensorRT parser payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    mark_onnx_parser_support_ready(parser_payload, false);
    const bool parsed = parser_payload->parse(model_data, model_size, model_path);
    *out_parsed = parsed ? JYPPX_TRUE : JYPPX_FALSE;
    if (!parsed)
    {
        set_parser_failure_message(*parser_payload);
    }

    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser parse from memory");
#endif
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_get_error_count(JYPPX_TensorRtOnnxParser* parser, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser error count query");
    }

    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser handle does not carry the expected TensorRT parser payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_count = parser_payload->getNbErrors();
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser error count query");
#endif
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_get_error(JYPPX_TensorRtOnnxParser* parser, int32_t index, JYPPX_TensorRtParserErrorInfo* out_error)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_error, "out_error");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Parser error index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    std::memset(out_error, 0, sizeof(*out_error));

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser error query");
    }

    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser handle does not carry the expected TensorRT parser payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const int32_t count = parser_payload->getNbErrors();
    if (index >= count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Parser error index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    fill_parser_error(parser_payload->getError(index), index, out_error);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser error query");
#endif
}

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
static JYPPX_StatusCode get_parser_error_for_diagnostic(JYPPX_TensorRtOnnxParser* parser, const int32_t index, const nvonnxparser::IParserError** out_error)
{
    *out_error = nullptr;

    auto status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Parser error index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser diagnostic query");
    }

    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser handle does not carry the expected TensorRT parser payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const int32_t count = parser_payload->getNbErrors();
    if (index >= count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Parser error index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const nvonnxparser::IParserError* parser_error = parser_payload->getError(index);
    if (parser_error == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT returned a null parser error object.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_error = parser_error;
    return JYPPX_STATUS_OK;
}

static const char* get_parser_error_string_field(const nvonnxparser::IParserError* parser_error, const int32_t field)
{
    switch (field)
    {
    case 0:
        return parser_error->desc();
    case 1:
        return parser_error->file();
    case 2:
        return parser_error->func();
    case 3:
#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
        return parser_error->nodeName();
#else
        return "";
#endif
    case 4:
#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
        return parser_error->nodeOperator();
#else
        return "";
#endif
    default:
        return "";
    }
}
#endif

static JYPPX_StatusCode get_parser_diagnostic_int(JYPPX_TensorRtOnnxParser* parser, const int32_t index, const int32_t field, int32_t* out_value)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_value, "out_value");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_value = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    const nvonnxparser::IParserError* parser_error = nullptr;
    status = get_parser_error_for_diagnostic(parser, index, &parser_error);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    switch (field)
    {
    case 0:
        *out_value = static_cast<int32_t>(parser_error->code());
        break;
    case 1:
        *out_value = parser_error->line();
        break;
    case 2:
        *out_value = parser_error->node();
        break;
    default:
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Unknown parser diagnostic integer field.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    (void)index;
    (void)field;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser diagnostic integer query");
#endif
}

static JYPPX_StatusCode get_parser_diagnostic_string(JYPPX_TensorRtOnnxParser* parser, const int32_t index, const int32_t field, char* output_buffer, const size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    const nvonnxparser::IParserError* parser_error = nullptr;
    status = get_parser_error_for_diagnostic(parser, index, &parser_error);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(get_parser_error_string_field(parser_error, field), output_buffer, output_buffer_size, out_required_size);
#elif JYPPX_HAS_TENSORRT
    (void)output_buffer;
    (void)output_buffer_size;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    (void)index;
    (void)field;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser diagnostic string query");
#endif
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_error_get_code(JYPPX_TensorRtOnnxParser* parser, int32_t index, int32_t* out_code)
{
    return get_parser_diagnostic_int(parser, index, 0, out_code);
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_error_get_line(JYPPX_TensorRtOnnxParser* parser, int32_t index, int32_t* out_line)
{
    return get_parser_diagnostic_int(parser, index, 1, out_line);
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_error_get_node(JYPPX_TensorRtOnnxParser* parser, int32_t index, int32_t* out_node)
{
    return get_parser_diagnostic_int(parser, index, 2, out_node);
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_error_get_description(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return get_parser_diagnostic_string(parser, index, 0, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_error_get_file(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return get_parser_diagnostic_string(parser, index, 1, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_error_get_function(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return get_parser_diagnostic_string(parser, index, 2, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_error_get_node_name(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return get_parser_diagnostic_string(parser, index, 3, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_error_get_node_operator(JYPPX_TensorRtOnnxParser* parser, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return get_parser_diagnostic_string(parser, index, 4, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_error_get_local_function_stack_size(JYPPX_TensorRtOnnxParser* parser, int32_t index, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    const nvonnxparser::IParserError* parser_error = nullptr;
    status = get_parser_error_for_diagnostic(parser, index, &parser_error);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = parser_error->localFunctionStackSize();
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    (void)parser;
    (void)index;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser local function stack size query");
#endif
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_error_get_local_function_stack_entry(JYPPX_TensorRtOnnxParser* parser, int32_t index, int32_t stack_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = 0;

    if (stack_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Parser local function stack index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    const nvonnxparser::IParserError* parser_error = nullptr;
    status = get_parser_error_for_diagnostic(parser, index, &parser_error);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int32_t stack_size = parser_error->localFunctionStackSize();
    if (stack_index >= stack_size)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Parser local function stack index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const char* const* stack = parser_error->localFunctionStack();
    const char* value = stack != nullptr ? stack[stack_index] : "";
    return copy_string_to_buffer(value, output_buffer, output_buffer_size, out_required_size);
#elif JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    (void)parser;
    (void)index;
    (void)output_buffer;
    (void)output_buffer_size;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser local function stack is not exposed by this TensorRT major line.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#elif JYPPX_HAS_TENSORRT
    (void)output_buffer;
    (void)output_buffer_size;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    (void)index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser local function stack entry query");
#endif
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_clear_errors(JYPPX_TensorRtOnnxParser* parser)
{
    auto status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser error clear");
    }

    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser handle does not carry the expected TensorRT parser payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    parser_payload->clearErrors();
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser error clear");
#endif
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_get_flags(JYPPX_TensorRtOnnxParser* parser, uint32_t* out_flags)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_flags, "out_flags");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_flags = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser flags query");
    }

    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser handle does not carry the expected TensorRT parser payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_flags = static_cast<uint32_t>(parser_payload->getFlags());
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser flags query");
#endif
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_set_flags(JYPPX_TensorRtOnnxParser* parser, uint32_t flags)
{
    auto status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser flags set");
    }

    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser handle does not carry the expected TensorRT parser payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    parser_payload->setFlags(static_cast<nvonnxparser::OnnxParserFlags>(flags));
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)flags;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)flags;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser flags set");
#endif
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_get_flag(JYPPX_TensorRtOnnxParser* parser, int32_t flag, JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (flag < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser flag must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_enabled = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser flag query");
    }

    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser handle does not carry the expected TensorRT parser payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_enabled = parser_payload->getFlag(static_cast<nvonnxparser::OnnxParserFlag>(flag)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser flag query");
#endif
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_set_flag(JYPPX_TensorRtOnnxParser* parser, int32_t flag)
{
    auto status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (flag < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser flag must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser flag set");
    }

    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser handle does not carry the expected TensorRT parser payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    parser_payload->setFlag(static_cast<nvonnxparser::OnnxParserFlag>(flag));
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser flag set");
#endif
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_clear_flag(JYPPX_TensorRtOnnxParser* parser, int32_t flag)
{
    auto status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (flag < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser flag must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser flag clear");
    }

    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser handle does not carry the expected TensorRT parser payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    parser_payload->clearFlag(static_cast<nvonnxparser::OnnxParserFlag>(flag));
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser flag clear");
#endif
}

JYPPX_StatusCode jyppx_trt10_onnx_parser_supports_operator(JYPPX_TensorRtOnnxParser* parser, const char* operator_name, JYPPX_Boolean* out_supported)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_supported, "out_supported");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(operator_name, "operator_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_supported = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser operator support query");
    }

    auto* parser_payload = get_payload<nvonnxparser::IParser>(parser);
    if (parser_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser handle does not carry the expected TensorRT parser payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_supported = parser_payload->supportsOperator(operator_name) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser operator support query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_create_inspector(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtEngineInspector** out_inspector)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_inspector, "out_inspector");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(engine, kLine, JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE, "engine");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_inspector = nullptr;

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine inspector creation");
    }

    auto* engine_payload = get_payload<nvinfer1::ICudaEngine>(engine);
    if (engine_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine handle does not carry a TensorRT engine payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IEngineInspector* inspector = engine_payload->createEngineInspector();
    if (inspector == nullptr)
    {
        return report_null_vendor_object("createEngineInspector");
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_ENGINE_INSPECTOR, inspector, &destroy_payload<nvinfer1::IEngineInspector>);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_inspector = reinterpret_cast<JYPPX_TensorRtEngineInspector*>(handle);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine inspector creation");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_inspector_get_engine_information(JYPPX_TensorRtEngineInspector* inspector, int32_t format, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_handle(inspector, kLine, JYPPX_TENSORRT_OBJECT_KIND_ENGINE_INSPECTOR, "inspector");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine inspector information query");
    }

    if (format < 0 || format > 1)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine inspector format must be 0 (oneline) or 1 (json).");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* inspector_payload = get_payload<nvinfer1::IEngineInspector>(inspector);
    if (inspector_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine inspector handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return copy_engine_information(inspector_payload, format, output_buffer, output_buffer_size, out_required_size);
#else
    if (out_required_size != nullptr)
    {
        *out_required_size = 0;
    }
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine inspector information query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_inspector_get_layer_information(JYPPX_TensorRtEngineInspector* inspector, int32_t layer_index, int32_t format, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = 0;

    status = jyppx::tensorrt::validate_handle(inspector, kLine, JYPPX_TENSORRT_OBJECT_KIND_ENGINE_INSPECTOR, "inspector");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (layer_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine inspector layer index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (format < 0 || format > 1)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine inspector format must be 0 (oneline) or 1 (json).");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine inspector layer information query");
    }

    auto* inspector_payload = get_payload<nvinfer1::IEngineInspector>(inspector);
    if (inspector_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine inspector handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const char* text = inspector_payload->getLayerInformation(layer_index, static_cast<nvinfer1::LayerInformationFormat>(format));
    if (text == nullptr)
    {
        return report_null_vendor_object("getLayerInformation");
    }

    return copy_string_to_buffer(text, output_buffer, output_buffer_size, out_required_size);
#else
    (void)layer_index;
    (void)format;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine inspector layer information query");
#endif
}

JYPPX_StatusCode jyppx_trt10_engine_inspector_set_execution_context(JYPPX_TensorRtEngineInspector* inspector, JYPPX_TensorRtExecutionContext* context)
{
    auto status = jyppx::tensorrt::validate_handle(inspector, kLine, JYPPX_TENSORRT_OBJECT_KIND_ENGINE_INSPECTOR, "inspector");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "engine inspector execution context binding");
    }

    auto* inspector_payload = get_payload<nvinfer1::IEngineInspector>(inspector);
    if (inspector_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine inspector handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    auto* context_payload = get_payload<nvinfer1::IExecutionContext>(context);
    if (context_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    inspector_payload->setExecutionContext(context_payload);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine inspector execution context binding");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_set_input_shape(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, const JYPPX_TensorRtDims* dims)
{
    auto status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(tensor_name, "tensor_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context input shape set");
    }

    nvinfer1::Dims native_dims{};
    status = make_dims(dims, &native_dims, "dims");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* context_payload = get_payload<nvinfer1::IExecutionContext>(context);
    if (context_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const bool result = context_payload->setInputShape(tensor_name, native_dims);
    if (!result)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::setInputShape returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context input shape set");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_set_tensor_address(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_CudaMemory* memory)
{
    auto status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(tensor_name, "tensor_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context tensor address set");
    }

    auto* context_payload = get_payload<nvinfer1::IExecutionContext>(context);
    auto* memory_payload = reinterpret_cast<jyppx::cuda::MemoryObject*>(memory);
    if (context_payload == nullptr || memory_payload == nullptr || memory_payload->pointer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context or CUDA memory handle does not carry the expected payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const bool result = context_payload->setTensorAddress(tensor_name, memory_payload->pointer);
    if (!result)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::setTensorAddress returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context tensor address set");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_set_input_tensor_address(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_CudaMemory* memory)
{
    auto status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_c_string(tensor_name, "tensor_name");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    auto* context_payload = get_payload<nvinfer1::IExecutionContext>(context);
    auto* memory_payload = reinterpret_cast<jyppx::cuda::MemoryObject*>(memory);
    if (context_payload == nullptr || memory_payload == nullptr || memory_payload->pointer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context or CUDA memory handle does not carry the expected payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const bool result = context_payload->setInputTensorAddress(tensor_name, memory_payload->pointer);
    if (!result)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::setInputTensorAddress returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context input tensor address set");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_set_output_tensor_address(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_CudaMemory* memory)
{
    auto status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_c_string(tensor_name, "tensor_name");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    auto* context_payload = get_payload<nvinfer1::IExecutionContext>(context);
    auto* memory_payload = reinterpret_cast<jyppx::cuda::MemoryObject*>(memory);
    if (context_payload == nullptr || memory_payload == nullptr || memory_payload->pointer == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context or CUDA memory handle does not carry the expected payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const bool result = context_payload->setOutputTensorAddress(tensor_name, memory_payload->pointer);
    if (!result)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::setOutputTensorAddress returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context output tensor address set");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context output tensor address set");
#endif
}

JYPPX_StatusCode jyppx_trt10_execution_context_enqueue_async(JYPPX_TensorRtExecutionContext* context, JYPPX_CudaStream* stream)
{
    auto status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context enqueue");
    }

    auto* context_payload = get_payload<nvinfer1::IExecutionContext>(context);
    auto* stream_payload = reinterpret_cast<jyppx::cuda::StreamObject*>(stream);
    if (context_payload == nullptr || stream_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context or CUDA stream handle does not carry the expected payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const bool result = context_payload->enqueueV3(stream_payload->handle);
    if (!result)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IExecutionContext::enqueueV3 returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }

    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    return jyppx::cuda::report_cuda_dependency_missing("TensorRT execution context enqueue");
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context enqueue");
#endif
}

#if JYPPX_HAS_TENSORRT
namespace
{
nvinfer1::IElementWiseLayer* get_elementwise_layer_payload_ext(const JYPPX_TensorRtObjectBase* object)
{
    auto* layer = get_layer_payload(object);
    if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kELEMENTWISE)
    {
        return nullptr;
    }

    return static_cast<nvinfer1::IElementWiseLayer*>(layer);
}

JYPPX_StatusCode get_engine_payload_ext(JYPPX_TensorRtCudaEngine* engine, nvinfer1::ICudaEngine** out_engine, const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(engine, kLine, JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE, "engine");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, feature_name);
    }

    *out_engine = get_payload<nvinfer1::ICudaEngine>(engine);
    if (*out_engine == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine handle does not carry a TensorRT engine payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode get_context_payload_ext(JYPPX_TensorRtExecutionContext* context, nvinfer1::IExecutionContext** out_context, const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != 10)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, feature_name);
    }

    *out_context = get_payload<nvinfer1::IExecutionContext>(context);
    if (*out_context == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return JYPPX_STATUS_OK;
}

#ifndef JYPPX_TRT10_VALIDATE_NAMED_TENSOR_DEFINED
JYPPX_StatusCode validate_named_tensor(const char* tensor_name)
{
    return validate_c_string(tensor_name, "tensor_name");
}
#endif

JYPPX_StatusCode validate_profile_selector(const int32_t selector)
{
    if (selector < 0 || selector > 2)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Optimization profile selector must be in the TensorRT enum range [0, 2].");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode get_layer_reference_ext(JYPPX_TensorRtLayer* layer, LayerReferencePayload** out_payload, const char* expected_description)
{
    auto status = jyppx::tensorrt::validate_handle(layer, kLine, JYPPX_TENSORRT_OBJECT_KIND_LAYER, "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_payload = get_layer_reference_payload(layer);
    if (*out_payload == nullptr || (*out_payload)->layer == nullptr)
    {
        std::ostringstream builder;
        builder << "Layer handle is not a TensorRT " << expected_description << " layer.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    return JYPPX_STATUS_OK;
}
}
#endif
