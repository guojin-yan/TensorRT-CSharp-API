#ifndef JYPPX_TRT_PARSER_REFITTER_API
#error "JYPPX_TRT_PARSER_REFITTER_API must be defined before including parser_refitter_diagnostics.inc."
#endif

#ifndef JYPPX_TRT_EXPECTED_MAJOR
#error "JYPPX_TRT_EXPECTED_MAJOR must be defined before including parser_refitter_diagnostics.inc."
#endif

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
static void fill_parser_refitter_error(const nvonnxparser::IParserError* parser_error, const int32_t index, JYPPX_TensorRtParserErrorInfo* out_error)
{
    std::memset(out_error, 0, sizeof(*out_error));
    out_error->index = index;
    if (parser_error == nullptr)
    {
        return;
    }

    out_error->code = static_cast<int32_t>(parser_error->code());
    out_error->line = parser_error->line();
    out_error->node = parser_error->node();
    copy_c_string(parser_error->desc(), out_error->description, sizeof(out_error->description));
    copy_c_string(parser_error->file(), out_error->file, sizeof(out_error->file));
    copy_c_string(parser_error->func(), out_error->function_name, sizeof(out_error->function_name));
    copy_c_string(parser_error->nodeName(), out_error->node_name, sizeof(out_error->node_name));
    copy_c_string(parser_error->nodeOperator(), out_error->node_operator, sizeof(out_error->node_operator));
}

static JYPPX_StatusCode get_parser_refitter_payload(
    JYPPX_TensorRtOnnxParserRefitter* parser_refitter,
    nvonnxparser::IParserRefitter** out_payload,
    const char* feature_name)
{
    *out_payload = nullptr;

    auto status = jyppx::tensorrt::validate_handle(
        parser_refitter,
        kLine,
        JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER_REFITTER,
        "parser_refitter");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != JYPPX_TRT_EXPECTED_MAJOR)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, feature_name);
    }

    auto* payload = get_payload<nvonnxparser::IParserRefitter>(parser_refitter);
    if (payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser refitter handle does not carry the expected TensorRT parser-refitter payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_payload = payload;
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode report_parser_refitter_native_exception(const char* feature_name, const std::exception& exception)
{
    std::ostringstream builder;
    builder << feature_name << " caught a native exception: " << exception.what();
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
    return JYPPX_STATUS_RUNTIME_ERROR;
}

static JYPPX_StatusCode report_parser_refitter_unknown_exception(const char* feature_name)
{
    std::ostringstream builder;
    builder << feature_name << " caught an unknown native exception.";
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
    return JYPPX_STATUS_RUNTIME_ERROR;
}

static JYPPX_StatusCode get_parser_refitter_error_for_diagnostic(
    JYPPX_TensorRtOnnxParserRefitter* parser_refitter,
    const int32_t index,
    const nvonnxparser::IParserError** out_error,
    const char* feature_name)
{
    *out_error = nullptr;

    if (index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Parser refitter error index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    nvonnxparser::IParserRefitter* payload = nullptr;
    auto status = get_parser_refitter_payload(parser_refitter, &payload, feature_name);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int32_t count = payload->getNbErrors();
    if (index >= count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Parser refitter error index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const nvonnxparser::IParserError* parser_error = payload->getError(index);
    if (parser_error == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT returned a null parser refitter error object.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_error = parser_error;
    return JYPPX_STATUS_OK;
}

static const char* get_parser_refitter_error_string_field(const nvonnxparser::IParserError* parser_error, const int32_t field)
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
        return parser_error->nodeName();
    case 4:
        return parser_error->nodeOperator();
    default:
        return "";
    }
}

static JYPPX_StatusCode get_parser_refitter_diagnostic_string(
    JYPPX_TensorRtOnnxParserRefitter* parser_refitter,
    const int32_t index,
    const int32_t field,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = 0;

    const nvonnxparser::IParserError* parser_error = nullptr;
    status = get_parser_refitter_error_for_diagnostic(parser_refitter, index, &parser_error, feature_name);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(
        get_parser_refitter_error_string_field(parser_error, field),
        output_buffer,
        output_buffer_size,
        out_required_size);
}
#endif

JYPPX_StatusCode JYPPX_TRT_PARSER_REFITTER_API(create)(
    JYPPX_TensorRtRefitter* refitter,
    JYPPX_TensorRtLogger* logger,
    JYPPX_TensorRtOnnxParserRefitter** out_parser_refitter)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_parser_refitter, "out_parser_refitter");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(refitter, kLine, JYPPX_TENSORRT_OBJECT_KIND_REFITTER, "refitter");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_handle(logger, kLine, JYPPX_TENSORRT_OBJECT_KIND_LOGGER, "logger");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_parser_refitter = nullptr;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != JYPPX_TRT_EXPECTED_MAJOR)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser refitter creation");
    }

    auto* refitter_payload = get_payload<nvinfer1::IRefitter>(refitter);
    auto* logger_payload = get_payload<ManagedLogger>(logger);
    if (refitter_payload == nullptr || logger_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter or logger handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvonnxparser::IParserRefitter* parser_refitter = nvonnxparser::createParserRefitter(*refitter_payload, *logger_payload);
    if (parser_refitter == nullptr)
    {
        return report_null_vendor_object("nvonnxparser::createParserRefitter");
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(
        &handle,
        JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER_REFITTER,
        parser_refitter,
        &destroy_payload<nvonnxparser::IParserRefitter>);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_parser_refitter = reinterpret_cast<JYPPX_TensorRtOnnxParserRefitter*>(handle);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser refitter creation");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PARSER_REFITTER_API(get_error_count)(JYPPX_TensorRtOnnxParserRefitter* parser_refitter, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvonnxparser::IParserRefitter* payload = nullptr;
    status = get_parser_refitter_payload(parser_refitter, &payload, "ONNX parser refitter error count query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = payload->getNbErrors();
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)parser_refitter;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser_refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser refitter error count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PARSER_REFITTER_API(get_error)(
    JYPPX_TensorRtOnnxParserRefitter* parser_refitter,
    int32_t index,
    JYPPX_TensorRtParserErrorInfo* out_error)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_error, "out_error");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_error, 0, sizeof(*out_error));

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    const nvonnxparser::IParserError* parser_error = nullptr;
    status = get_parser_refitter_error_for_diagnostic(parser_refitter, index, &parser_error, "ONNX parser refitter error query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    fill_parser_refitter_error(parser_error, index, out_error);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)parser_refitter;
    (void)index;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser_refitter;
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser refitter error query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PARSER_REFITTER_API(error_get_description)(
    JYPPX_TensorRtOnnxParserRefitter* parser_refitter,
    int32_t index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    return get_parser_refitter_diagnostic_string(parser_refitter, index, 0, output_buffer, output_buffer_size, out_required_size, "ONNX parser refitter error description query");
#elif JYPPX_HAS_TENSORRT
    (void)parser_refitter;
    (void)index;
    (void)output_buffer;
    (void)output_buffer_size;
    if (out_required_size != nullptr)
    {
        *out_required_size = 0;
    }
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser_refitter;
    (void)index;
    (void)output_buffer;
    (void)output_buffer_size;
    if (out_required_size != nullptr)
    {
        *out_required_size = 0;
    }
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser refitter error description query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PARSER_REFITTER_API(error_get_file)(
    JYPPX_TensorRtOnnxParserRefitter* parser_refitter,
    int32_t index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    return get_parser_refitter_diagnostic_string(parser_refitter, index, 1, output_buffer, output_buffer_size, out_required_size, "ONNX parser refitter error file query");
#elif JYPPX_HAS_TENSORRT
    (void)parser_refitter;
    (void)index;
    (void)output_buffer;
    (void)output_buffer_size;
    if (out_required_size != nullptr)
    {
        *out_required_size = 0;
    }
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser_refitter;
    (void)index;
    (void)output_buffer;
    (void)output_buffer_size;
    if (out_required_size != nullptr)
    {
        *out_required_size = 0;
    }
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser refitter error file query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PARSER_REFITTER_API(error_get_function)(
    JYPPX_TensorRtOnnxParserRefitter* parser_refitter,
    int32_t index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    return get_parser_refitter_diagnostic_string(parser_refitter, index, 2, output_buffer, output_buffer_size, out_required_size, "ONNX parser refitter error function query");
#elif JYPPX_HAS_TENSORRT
    (void)parser_refitter;
    (void)index;
    (void)output_buffer;
    (void)output_buffer_size;
    if (out_required_size != nullptr)
    {
        *out_required_size = 0;
    }
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser_refitter;
    (void)index;
    (void)output_buffer;
    (void)output_buffer_size;
    if (out_required_size != nullptr)
    {
        *out_required_size = 0;
    }
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser refitter error function query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PARSER_REFITTER_API(error_get_node_name)(
    JYPPX_TensorRtOnnxParserRefitter* parser_refitter,
    int32_t index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    return get_parser_refitter_diagnostic_string(parser_refitter, index, 3, output_buffer, output_buffer_size, out_required_size, "ONNX parser refitter error node name query");
#elif JYPPX_HAS_TENSORRT
    (void)parser_refitter;
    (void)index;
    (void)output_buffer;
    (void)output_buffer_size;
    if (out_required_size != nullptr)
    {
        *out_required_size = 0;
    }
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser_refitter;
    (void)index;
    (void)output_buffer;
    (void)output_buffer_size;
    if (out_required_size != nullptr)
    {
        *out_required_size = 0;
    }
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser refitter error node name query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PARSER_REFITTER_API(error_get_node_operator)(
    JYPPX_TensorRtOnnxParserRefitter* parser_refitter,
    int32_t index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    return get_parser_refitter_diagnostic_string(parser_refitter, index, 4, output_buffer, output_buffer_size, out_required_size, "ONNX parser refitter error node operator query");
#elif JYPPX_HAS_TENSORRT
    (void)parser_refitter;
    (void)index;
    (void)output_buffer;
    (void)output_buffer_size;
    if (out_required_size != nullptr)
    {
        *out_required_size = 0;
    }
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser_refitter;
    (void)index;
    (void)output_buffer;
    (void)output_buffer_size;
    if (out_required_size != nullptr)
    {
        *out_required_size = 0;
    }
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser refitter error node operator query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PARSER_REFITTER_API(error_get_local_function_stack_size)(
    JYPPX_TensorRtOnnxParserRefitter* parser_refitter,
    int32_t index,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    const nvonnxparser::IParserError* parser_error = nullptr;
    status = get_parser_refitter_error_for_diagnostic(parser_refitter, index, &parser_error, "ONNX parser refitter local function stack size query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = parser_error->localFunctionStackSize();
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)parser_refitter;
    (void)index;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser_refitter;
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser refitter local function stack size query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PARSER_REFITTER_API(error_get_local_function_stack_entry)(
    JYPPX_TensorRtOnnxParserRefitter* parser_refitter,
    int32_t index,
    int32_t stack_index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = 0;

    if (stack_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Parser refitter local function stack index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    const nvonnxparser::IParserError* parser_error = nullptr;
    status = get_parser_refitter_error_for_diagnostic(parser_refitter, index, &parser_error, "ONNX parser refitter local function stack entry query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int32_t stack_size = parser_error->localFunctionStackSize();
    if (stack_index >= stack_size)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Parser refitter local function stack index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const char* const* stack = parser_error->localFunctionStack();
    const char* value = stack != nullptr ? stack[stack_index] : "";
    return copy_string_to_buffer(value, output_buffer, output_buffer_size, out_required_size);
#elif JYPPX_HAS_TENSORRT
    (void)parser_refitter;
    (void)index;
    (void)output_buffer;
    (void)output_buffer_size;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser_refitter;
    (void)index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser refitter local function stack entry query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PARSER_REFITTER_API(clear_errors)(JYPPX_TensorRtOnnxParserRefitter* parser_refitter)
{
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvonnxparser::IParserRefitter* payload = nullptr;
    auto status = get_parser_refitter_payload(parser_refitter, &payload, "ONNX parser refitter error clear");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    payload->clearErrors();
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)parser_refitter;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser_refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser refitter error clear");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PARSER_REFITTER_API(refit_from_bytes)(
    JYPPX_TensorRtOnnxParserRefitter* parser_refitter,
    const void* model_data,
    size_t model_size,
    const char* model_path,
    JYPPX_Boolean* out_refitted)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_refitted, "out_refitted");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (model_data == nullptr || model_size == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX model data must not be null or empty for parser-refitter refitFromBytes.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_refitted = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvonnxparser::IParserRefitter* payload = nullptr;
    status = get_parser_refitter_payload(parser_refitter, &payload, "ONNX parser refitter refit from bytes");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        *out_refitted = payload->refitFromBytes(model_data, model_size, model_path) ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        return report_parser_refitter_native_exception("ONNX parser refitter refitFromBytes", exception);
    }
    catch (...)
    {
        return report_parser_refitter_unknown_exception("ONNX parser refitter refitFromBytes");
    }
#elif JYPPX_HAS_TENSORRT
    (void)parser_refitter;
    (void)model_path;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser_refitter;
    (void)model_path;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser refitter refit from bytes");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PARSER_REFITTER_API(refit_from_file)(
    JYPPX_TensorRtOnnxParserRefitter* parser_refitter,
    const char* file_path,
    JYPPX_Boolean* out_refitted)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_refitted, "out_refitted");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(file_path, "file_path");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_refitted = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvonnxparser::IParserRefitter* payload = nullptr;
    status = get_parser_refitter_payload(parser_refitter, &payload, "ONNX parser refitter refit from file");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        *out_refitted = payload->refitFromFile(file_path) ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        return report_parser_refitter_native_exception("ONNX parser refitter refitFromFile", exception);
    }
    catch (...)
    {
        return report_parser_refitter_unknown_exception("ONNX parser refitter refitFromFile");
    }
#elif JYPPX_HAS_TENSORRT
    (void)parser_refitter;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser_refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser refitter refit from file");
#endif
}

#if JYPPX_TRT_EXPECTED_MAJOR >= 11
JYPPX_StatusCode JYPPX_TRT_PARSER_REFITTER_API(load_model_proto)(
    JYPPX_TensorRtOnnxParserRefitter* parser_refitter,
    const void* model_data,
    size_t model_size,
    const char* model_path,
    JYPPX_Boolean* out_loaded)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_loaded, "out_loaded");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (model_data == nullptr || model_size == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX model proto data must not be null or empty for parser-refitter loadModelProto.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_loaded = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvonnxparser::IParserRefitter* payload = nullptr;
    status = get_parser_refitter_payload(parser_refitter, &payload, "ONNX parser refitter model proto load");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        *out_loaded = payload->loadModelProto(model_data, model_size, model_path) ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        return report_parser_refitter_native_exception("ONNX parser refitter loadModelProto", exception);
    }
    catch (...)
    {
        return report_parser_refitter_unknown_exception("ONNX parser refitter loadModelProto");
    }
#elif JYPPX_HAS_TENSORRT
    (void)parser_refitter;
    (void)model_path;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser_refitter;
    (void)model_path;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser refitter model proto load");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PARSER_REFITTER_API(load_initializer)(
    JYPPX_TensorRtOnnxParserRefitter* parser_refitter,
    const char* name,
    const void* data,
    size_t data_size,
    JYPPX_Boolean* out_loaded)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_loaded, "out_loaded");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(name, "name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (data == nullptr || data_size == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX initializer data must not be null or empty for parser-refitter loadInitializer.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_loaded = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvonnxparser::IParserRefitter* payload = nullptr;
    status = get_parser_refitter_payload(parser_refitter, &payload, "ONNX parser refitter initializer load");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        *out_loaded = payload->loadInitializer(name, data, data_size) ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        return report_parser_refitter_native_exception("ONNX parser refitter loadInitializer", exception);
    }
    catch (...)
    {
        return report_parser_refitter_unknown_exception("ONNX parser refitter loadInitializer");
    }
#elif JYPPX_HAS_TENSORRT
    (void)parser_refitter;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser_refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser refitter initializer load");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PARSER_REFITTER_API(refit_model_proto)(
    JYPPX_TensorRtOnnxParserRefitter* parser_refitter,
    JYPPX_Boolean* out_refitted)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_refitted, "out_refitted");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_refitted = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvonnxparser::IParserRefitter* payload = nullptr;
    status = get_parser_refitter_payload(parser_refitter, &payload, "ONNX parser refitter model proto refit");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        *out_refitted = payload->refitModelProto() ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        return report_parser_refitter_native_exception("ONNX parser refitter refitModelProto", exception);
    }
    catch (...)
    {
        return report_parser_refitter_unknown_exception("ONNX parser refitter refitModelProto");
    }
#elif JYPPX_HAS_TENSORRT
    (void)parser_refitter;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser_refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser refitter model proto refit");
#endif
}
#endif

#undef JYPPX_TRT_PARSER_REFITTER_API
#undef JYPPX_TRT_EXPECTED_MAJOR
