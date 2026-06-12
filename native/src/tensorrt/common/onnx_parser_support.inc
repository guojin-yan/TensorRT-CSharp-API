#ifndef JYPPX_TRT_ONNX_PARSER_API
#error "JYPPX_TRT_ONNX_PARSER_API must be defined before including onnx_parser_support.inc."
#endif

#ifndef JYPPX_TRT_EXPECTED_MAJOR
#error "JYPPX_TRT_EXPECTED_MAJOR must be defined before including onnx_parser_support.inc."
#endif

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
static JYPPX_StatusCode get_onnx_parser_payload_for_support(
    JYPPX_TensorRtOnnxParser* parser,
    nvonnxparser::IParser** out_parser,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(parser, kLine, JYPPX_TENSORRT_OBJECT_KIND_ONNX_PARSER, "parser");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (JYPPX_TENSORRT_VERSION_MAJOR_NUM != JYPPX_TRT_EXPECTED_MAJOR)
    {
        return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, feature_name);
    }

    *out_parser = get_payload<nvonnxparser::IParser>(parser);
    if (*out_parser == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser handle does not carry the expected TensorRT parser payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode ensure_onnx_parser_support_ready(const nvonnxparser::IParser* parser)
{
    if (is_onnx_parser_support_ready(parser))
    {
        return JYPPX_STATUS_OK;
    }

    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser subgraph support data is unavailable. Call supportsModelV2 successfully before querying subgraph support.");
    return JYPPX_STATUS_INVALID_STATE;
}

static JYPPX_StatusCode get_onnx_parser_subgraph_nodes_for_support(
    nvonnxparser::IParser* parser,
    const int64_t subgraph_index,
    int64_t** out_nodes,
    int64_t* out_count)
{
    *out_nodes = nullptr;
    *out_count = 0;

#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    auto status = ensure_onnx_parser_support_ready(parser);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (subgraph_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser subgraph index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const int64_t subgraph_count = parser->getNbSubgraphs();
    if (subgraph_count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT returned a negative ONNX parser subgraph count.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    if (subgraph_index >= subgraph_count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser subgraph index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    int64_t node_count = 0;
    int64_t* nodes = parser->getSubgraphNodes(subgraph_index, node_count);
    if (node_count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT returned a negative ONNX parser subgraph node count.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    if (node_count > 0 && nodes == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT returned null ONNX parser subgraph nodes for a non-empty subgraph.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_nodes = nodes;
    *out_count = node_count;
    return JYPPX_STATUS_OK;
#else
    (void)parser;
    (void)subgraph_index;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser subgraph node query");
#endif
}

static JYPPX_StatusCode count_onnx_parser_subgraphs_by_support(
    nvonnxparser::IParser* parser,
    const bool supported,
    int64_t* out_count)
{
#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    auto status = ensure_onnx_parser_support_ready(parser);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int64_t subgraph_count = parser->getNbSubgraphs();
    if (subgraph_count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT returned a negative ONNX parser subgraph count.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    int64_t match_count = 0;
    for (int64_t index = 0; index < subgraph_count; ++index)
    {
        if (parser->isSubgraphSupported(index) == supported)
        {
            ++match_count;
        }
    }

    *out_count = match_count;
    return JYPPX_STATUS_OK;
#else
    (void)parser;
    (void)supported;
    (void)out_count;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser subgraph support count query");
#endif
}
#endif

JYPPX_StatusCode JYPPX_TRT_ONNX_PARSER_API(get_used_vc_plugin_library_count)(JYPPX_TensorRtOnnxParser* parser, int64_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IParser* parser_payload = nullptr;
    status = get_onnx_parser_payload_for_support(parser, &parser_payload, "ONNX parser used VC plugin library count query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    int64_t count = 0;
    const char* const* libraries = parser_payload->getUsedVCPluginLibraries(count);
    if (count < 0 || (count > 0 && libraries == nullptr))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT returned invalid used VC plugin library metadata.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_count = count;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)parser;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser used VC plugin library count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ONNX_PARSER_API(get_used_vc_plugin_library)(JYPPX_TensorRtOnnxParser* parser, int64_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = 0;
    if (index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Used VC plugin library index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IParser* parser_payload = nullptr;
    status = get_onnx_parser_payload_for_support(parser, &parser_payload, "ONNX parser used VC plugin library query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    int64_t count = 0;
    const char* const* libraries = parser_payload->getUsedVCPluginLibraries(count);
    if (count < 0 || (count > 0 && libraries == nullptr))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT returned invalid used VC plugin library metadata.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    if (index >= count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Used VC plugin library index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return copy_string_to_buffer(libraries[index], output_buffer, output_buffer_size, out_required_size);
#elif JYPPX_HAS_TENSORRT
    (void)parser;
    (void)output_buffer;
    (void)output_buffer_size;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser used VC plugin library query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ONNX_PARSER_API(supports_model_v2)(JYPPX_TensorRtOnnxParser* parser, const void* model_data, size_t model_size, const char* model_path, JYPPX_Boolean* out_supported)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_supported, "out_supported");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (model_data == nullptr || model_size == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX model data must not be null or empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_supported = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IParser* parser_payload = nullptr;
    status = get_onnx_parser_payload_for_support(parser, &parser_payload, "ONNX parser model support query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    const bool supported = parser_payload->supportsModelV2(model_data, model_size, model_path);
    mark_onnx_parser_support_ready(parser_payload, true);
    *out_supported = supported ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)parser_payload;
    (void)model_path;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser model support query");
#endif
#elif JYPPX_HAS_TENSORRT
    (void)parser;
    (void)model_path;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    (void)model_path;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser model support query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ONNX_PARSER_API(get_subgraph_count)(JYPPX_TensorRtOnnxParser* parser, int64_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IParser* parser_payload = nullptr;
    status = get_onnx_parser_payload_for_support(parser, &parser_payload, "ONNX parser subgraph count query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    status = ensure_onnx_parser_support_ready(parser_payload);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int64_t count = parser_payload->getNbSubgraphs();
    if (count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT returned a negative ONNX parser subgraph count.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_count = count;
    return JYPPX_STATUS_OK;
#else
    (void)parser_payload;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser subgraph count query");
#endif
#elif JYPPX_HAS_TENSORRT
    (void)parser;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser subgraph count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ONNX_PARSER_API(is_subgraph_supported)(JYPPX_TensorRtOnnxParser* parser, int64_t index, JYPPX_Boolean* out_supported)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_supported, "out_supported");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_supported = JYPPX_FALSE;
    if (index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser subgraph index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IParser* parser_payload = nullptr;
    status = get_onnx_parser_payload_for_support(parser, &parser_payload, "ONNX parser subgraph support query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    status = ensure_onnx_parser_support_ready(parser_payload);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int64_t count = parser_payload->getNbSubgraphs();
    if (count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT returned a negative ONNX parser subgraph count.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    if (index >= count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser subgraph index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_supported = parser_payload->isSubgraphSupported(index) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)parser_payload;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser subgraph support query");
#endif
#elif JYPPX_HAS_TENSORRT
    (void)parser;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser subgraph support query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ONNX_PARSER_API(get_subgraph_node_count)(JYPPX_TensorRtOnnxParser* parser, int64_t index, int64_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IParser* parser_payload = nullptr;
    status = get_onnx_parser_payload_for_support(parser, &parser_payload, "ONNX parser subgraph node count query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    int64_t* nodes = nullptr;
    return get_onnx_parser_subgraph_nodes_for_support(parser_payload, index, &nodes, out_count);
#elif JYPPX_HAS_TENSORRT
    (void)parser;
    (void)index;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser subgraph node count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ONNX_PARSER_API(get_subgraph_node)(JYPPX_TensorRtOnnxParser* parser, int64_t subgraph_index, int64_t node_index, int64_t* out_node)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_node, "out_node");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_node = 0;
    if (node_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser subgraph node index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IParser* parser_payload = nullptr;
    status = get_onnx_parser_payload_for_support(parser, &parser_payload, "ONNX parser subgraph node query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    int64_t* nodes = nullptr;
    int64_t node_count = 0;
    status = get_onnx_parser_subgraph_nodes_for_support(parser_payload, subgraph_index, &nodes, &node_count);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (node_index >= node_count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser subgraph node index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_node = nodes[node_index];
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)parser;
    (void)subgraph_index;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    (void)subgraph_index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser subgraph node query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ONNX_PARSER_API(layer_output_tensor_exists)(JYPPX_TensorRtOnnxParser* parser, const char* layer_name, int64_t output_index, JYPPX_Boolean* out_exists)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_exists, "out_exists");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_c_string(layer_name, "layer_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (output_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "ONNX parser layer output index must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_exists = JYPPX_FALSE;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IParser* parser_payload = nullptr;
    status = get_onnx_parser_payload_for_support(parser, &parser_payload, "ONNX parser layer output tensor existence query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    *out_exists = parser_payload->getLayerOutputTensor(layer_name, output_index) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)parser_payload;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "ONNX parser layer output tensor existence query");
#endif
#elif JYPPX_HAS_TENSORRT
    (void)parser;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser layer output tensor existence query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ONNX_PARSER_API(get_supported_subgraph_count)(JYPPX_TensorRtOnnxParser* parser, int64_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IParser* parser_payload = nullptr;
    status = get_onnx_parser_payload_for_support(parser, &parser_payload, "ONNX parser supported subgraph count query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return count_onnx_parser_subgraphs_by_support(parser_payload, true, out_count);
#elif JYPPX_HAS_TENSORRT
    (void)parser;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser supported subgraph count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ONNX_PARSER_API(get_unsupported_subgraph_count)(JYPPX_TensorRtOnnxParser* parser, int64_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_HAS_TENSORRT_ONNXPARSER
    nvonnxparser::IParser* parser_payload = nullptr;
    status = get_onnx_parser_payload_for_support(parser, &parser_payload, "ONNX parser unsupported subgraph count query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return count_onnx_parser_subgraphs_by_support(parser_payload, false, out_count);
#elif JYPPX_HAS_TENSORRT
    (void)parser;
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser library was not found at build time.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
#else
    (void)parser;
    return jyppx::tensorrt::report_vendor_missing(kLine, "ONNX parser unsupported subgraph count query");
#endif
}

#undef JYPPX_TRT_ONNX_PARSER_API
#undef JYPPX_TRT_EXPECTED_MAJOR
