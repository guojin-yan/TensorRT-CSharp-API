namespace
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
using Trt8PluginSerializationBody = JYPPX_StatusCode (*)(void* state);

static JYPPX_StatusCode run_trt8_plugin_serialization_with_seh_guard(
    Trt8PluginSerializationBody body,
    void* state,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        return body(state);
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    return body(state);
#endif
}

static JYPPX_StatusCode run_trt8_plugin_serialization(
    Trt8PluginSerializationBody body,
    void* state,
    const char* feature_name)
{
    try
    {
        return run_trt8_plugin_serialization_with_seh_guard(body, state, feature_name);
    }
    catch (const std::exception& exception)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}

struct Trt8PluginSerializationPathQueryState
{
    nvinfer1::IBuilderConfig* config;
    int32_t index;
    char* output_buffer;
    size_t output_buffer_size;
    size_t* out_required_size;
};

static JYPPX_StatusCode trt8_plugin_serialization_path_query_body(void* raw_state)
{
    auto* state = static_cast<Trt8PluginSerializationPathQueryState*>(raw_state);
    const int32_t count = state->config->getNbPluginsToSerialize();
    if (state->index < 0 || state->index >= count)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "Serialized plugin path index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    return copy_string_to_buffer(
        state->config->getPluginToSerialize(state->index),
        state->output_buffer,
        state->output_buffer_size,
        state->out_required_size);
}

struct Trt8PluginSerializationPathSetState
{
    nvinfer1::IBuilderConfig* config;
    const char** paths;
    int32_t path_count;
};

static JYPPX_StatusCode trt8_plugin_serialization_path_set_body(void* raw_state)
{
    auto* state = static_cast<Trt8PluginSerializationPathSetState*>(raw_state);
    state->config->setPluginsToSerialize(state->paths, state->path_count);
    return JYPPX_STATUS_OK;
}
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_get_plugin_to_serialize(
    JYPPX_TensorRtBuilderConfig* config,
    int32_t index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
    status = jyppx::tensorrt::validate_handle(
        config,
        kLine,
        JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG,
        "config");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    auto* payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (payload == nullptr)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    Trt8PluginSerializationPathQueryState state{
        payload,
        index,
        output_buffer,
        output_buffer_size,
        out_required_size};
    return run_trt8_plugin_serialization(
        trt8_plugin_serialization_path_query_body,
        &state,
        "builder config serialized plugin path query");
#elif JYPPX_HAS_TENSORRT
    (void)index; (void)output_buffer; (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_mismatch(
        kLine,
        JYPPX_TENSORRT_VERSION_MAJOR_NUM,
        "builder config serialized plugin path query");
#else
    (void)index; (void)output_buffer; (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(
        kLine,
        "builder config serialized plugin path query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_config_set_plugins_to_serialize(
    JYPPX_TensorRtBuilderConfig* config,
    const char** paths,
    int32_t path_count,
    JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;
    if (path_count < 0 || path_count > 1000000 || (path_count > 0 && paths == nullptr))
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "Serialized plugin path count is outside the supported range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    for (int32_t index = 0; index < path_count; ++index)
    {
        status = validate_c_string(paths[index], "plugin_path");
        if (status != JYPPX_STATUS_OK) { return status; }
    }
    status = jyppx::tensorrt::validate_handle(
        config,
        kLine,
        JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG,
        "config");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    auto* payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (payload == nullptr)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "Builder config handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    Trt8PluginSerializationPathSetState state{payload, paths, path_count};
    status = run_trt8_plugin_serialization(
        trt8_plugin_serialization_path_set_body,
        &state,
        "builder config serialized plugin path set");
    if (status == JYPPX_STATUS_OK) { *out_set = JYPPX_TRUE; }
    return status;
#elif JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(
        kLine,
        JYPPX_TENSORRT_VERSION_MAJOR_NUM,
        "builder config serialized plugin path set");
#else
    return jyppx::tensorrt::report_vendor_missing(
        kLine,
        "builder config serialized plugin path set");
#endif
}
