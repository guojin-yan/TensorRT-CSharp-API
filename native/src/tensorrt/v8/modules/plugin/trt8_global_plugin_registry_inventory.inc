#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
namespace
{
constexpr int32_t kMaximumTrt8GlobalPluginFieldCount = 1000000;

static JYPPX_StatusCode get_trt8_global_plugin_registry_body(void* raw_registry)
{
    auto** out_registry = static_cast<nvinfer1::IPluginRegistry**>(raw_registry);
    *out_registry = ::getPluginRegistry();
    return JYPPX_STATUS_OK;
}

struct Trt8GlobalParentSearchState
{
    nvinfer1::IPluginRegistry* registry;
    bool enabled;
};

struct Trt8GlobalCreatorFieldCountState
{
    nvinfer1::IPluginCreator* creator;
    int32_t* out_count;
};

static JYPPX_StatusCode set_trt8_global_parent_search_body(void* raw_state)
{
    auto* state = static_cast<Trt8GlobalParentSearchState*>(raw_state);
    state->registry->setParentSearchEnabled(state->enabled);
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode get_trt8_global_creator_field_count_body(void* raw_state)
{
    auto* state = static_cast<Trt8GlobalCreatorFieldCountState*>(raw_state);
    const nvinfer1::PluginFieldCollection* fields = state->creator->getFieldNames();
    if (fields == nullptr)
    {
        *state->out_count = 0;
        return JYPPX_STATUS_OK;
    }

    const int32_t field_count = fields->nbFields;
    if (field_count < 0 || field_count > kMaximumTrt8GlobalPluginFieldCount)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "TensorRT 8 global plugin creator returned an invalid field count.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    if (field_count > 0 && fields->fields == nullptr)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "TensorRT 8 global plugin creator returned a non-empty field collection without storage.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *state->out_count = field_count;
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode run_trt8_global_registry_seh_guard(
    JYPPX_StatusCode (*body)(void*),
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

static JYPPX_StatusCode get_trt8_global_plugin_registry(nvinfer1::IPluginRegistry** out_registry)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_registry, "out_registry");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_registry = nullptr;
    try
    {
        status = run_trt8_global_registry_seh_guard(
            get_trt8_global_plugin_registry_body,
            out_registry,
            "TensorRT 8 global plugin registry query");
    }
    catch (const std::exception& exception)
    {
        return jyppx_trt8_report_plugin_inventory_exception("TensorRT 8 global plugin registry query", exception);
    }
    catch (...)
    {
        return jyppx_trt8_report_plugin_inventory_unknown_exception("TensorRT 8 global plugin registry query");
    }
    if (status != JYPPX_STATUS_OK) { return status; }
    if (*out_registry == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 global plugin registry is null.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode get_trt8_global_plugin_creator(int32_t creator_index, nvinfer1::IPluginCreator** out_creator)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_creator, "out_creator");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_creator = nullptr;
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = get_trt8_global_plugin_registry(&registry);
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::IPluginCreator* const* creators = nullptr;
    int32_t creator_count = 0;
    status = jyppx_trt8_get_inventory_registry_creators(
        registry,
        &creators,
        &creator_count,
        "TensorRT 8 global plugin creator lookup by index");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_index(creator_index, creator_count, "global_plugin_creator");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (creators == nullptr || creators[creator_index] == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 global plugin registry returned a null creator pointer.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    *out_creator = creators[creator_index];
    return JYPPX_STATUS_OK;
}
}
#endif

JYPPX_StatusCode jyppx_trt8_global_plugin_registry_exists(JYPPX_Boolean* out_exists)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_exists, "out_exists");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_exists = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = get_trt8_global_plugin_registry(&registry);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_exists = registry != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 global plugin registry existence query");
#endif
}

JYPPX_StatusCode jyppx_trt8_global_plugin_registry_get_creator_count(int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = get_trt8_global_plugin_registry(&registry);
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::IPluginCreator* const* creators = nullptr;
    return jyppx_trt8_get_inventory_registry_creators(
        registry,
        &creators,
        out_count,
        "TensorRT 8 global plugin registry creator count query");
#else
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 global plugin registry creator count query");
#endif
}

JYPPX_StatusCode jyppx_trt8_global_plugin_registry_has_error_recorder(JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = get_trt8_global_plugin_registry(&registry);
    if (status != JYPPX_STATUS_OK) { return status; }
    try
    {
        return jyppx_trt8_get_plugin_registry_has_error_recorder_with_seh_guard(
            registry,
            out_has_recorder,
            "TensorRT 8 global plugin registry error recorder query");
    }
    catch (const std::exception& exception)
    {
        return jyppx_trt8_report_plugin_inventory_exception("TensorRT 8 global plugin registry error recorder query", exception);
    }
    catch (...)
    {
        return jyppx_trt8_report_plugin_inventory_unknown_exception("TensorRT 8 global plugin registry error recorder query");
    }
#else
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 global plugin registry error recorder query");
#endif
}

JYPPX_StatusCode jyppx_trt8_global_plugin_registry_is_parent_search_enabled(JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_enabled = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = get_trt8_global_plugin_registry(&registry);
    if (status != JYPPX_STATUS_OK) { return status; }
    try
    {
        return jyppx_trt8_get_plugin_registry_parent_search_enabled_with_seh_guard(
            registry,
            out_enabled,
            "TensorRT 8 global plugin registry parent search query");
    }
    catch (const std::exception& exception)
    {
        return jyppx_trt8_report_plugin_inventory_exception("TensorRT 8 global plugin registry parent search query", exception);
    }
    catch (...)
    {
        return jyppx_trt8_report_plugin_inventory_unknown_exception("TensorRT 8 global plugin registry parent search query");
    }
#else
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 global plugin registry parent search query");
#endif
}

JYPPX_StatusCode jyppx_trt8_global_plugin_registry_set_parent_search_enabled(JYPPX_Boolean enabled)
{
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IPluginRegistry* registry = nullptr;
    auto status = get_trt8_global_plugin_registry(&registry);
    if (status != JYPPX_STATUS_OK) { return status; }
    Trt8GlobalParentSearchState state{registry, enabled != JYPPX_FALSE};
    try
    {
        return run_trt8_global_registry_seh_guard(
            set_trt8_global_parent_search_body,
            &state,
            "TensorRT 8 global plugin registry parent search set");
    }
    catch (const std::exception& exception)
    {
        return jyppx_trt8_report_plugin_inventory_exception("TensorRT 8 global plugin registry parent search set", exception);
    }
    catch (...)
    {
        return jyppx_trt8_report_plugin_inventory_unknown_exception("TensorRT 8 global plugin registry parent search set");
    }
#else
    (void)enabled;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 global plugin registry parent search set");
#endif
}

JYPPX_StatusCode jyppx_trt8_global_plugin_creator_get_name(
    int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IPluginCreator* creator = nullptr;
    status = get_trt8_global_plugin_creator(creator_index, &creator);
    if (status != JYPPX_STATUS_OK) { return status; }
    const char* value = "";
    status = jyppx_trt8_get_creator_name(creator, &value, "TensorRT 8 global plugin creator name query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(value, output_buffer, output_buffer_size, out_required_size);
#else
    (void)creator_index; (void)output_buffer; (void)output_buffer_size;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 global plugin creator name query");
#endif
}

JYPPX_StatusCode jyppx_trt8_global_plugin_creator_get_version(
    int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IPluginCreator* creator = nullptr;
    status = get_trt8_global_plugin_creator(creator_index, &creator);
    if (status != JYPPX_STATUS_OK) { return status; }
    const char* value = "";
    status = jyppx_trt8_get_creator_version(creator, &value, "TensorRT 8 global plugin creator version query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(value, output_buffer, output_buffer_size, out_required_size);
#else
    (void)creator_index; (void)output_buffer; (void)output_buffer_size;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 global plugin creator version query");
#endif
}

JYPPX_StatusCode jyppx_trt8_global_plugin_creator_get_namespace(
    int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IPluginCreator* creator = nullptr;
    status = get_trt8_global_plugin_creator(creator_index, &creator);
    if (status != JYPPX_STATUS_OK) { return status; }
    const char* value = "";
    status = jyppx_trt8_get_creator_namespace(creator, &value, "TensorRT 8 global plugin creator namespace query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(value, output_buffer, output_buffer_size, out_required_size);
#else
    (void)creator_index; (void)output_buffer; (void)output_buffer_size;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 global plugin creator namespace query");
#endif
}

JYPPX_StatusCode jyppx_trt8_global_plugin_creator_get_field_count(int32_t creator_index, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IPluginCreator* creator = nullptr;
    status = get_trt8_global_plugin_creator(creator_index, &creator);
    if (status != JYPPX_STATUS_OK) { return status; }
    Trt8GlobalCreatorFieldCountState state{creator, out_count};
    try
    {
        return run_trt8_global_registry_seh_guard(
            get_trt8_global_creator_field_count_body,
            &state,
            "TensorRT 8 global plugin creator field count query");
    }
    catch (const std::exception& exception)
    {
        *out_count = 0;
        return jyppx_trt8_report_plugin_inventory_exception("TensorRT 8 global plugin creator field count query", exception);
    }
    catch (...)
    {
        *out_count = 0;
        return jyppx_trt8_report_plugin_inventory_unknown_exception("TensorRT 8 global plugin creator field count query");
    }
#else
    (void)creator_index;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 global plugin creator field count query");
#endif
}

JYPPX_StatusCode jyppx_trt8_global_plugin_creator_get_field_name(
    int32_t creator_index,
    int32_t field_index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IPluginCreator* creator = nullptr;
    status = get_trt8_global_plugin_creator(creator_index, &creator);
    if (status != JYPPX_STATUS_OK) { return status; }
    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = jyppx_trt8_get_creator_field_collection(creator, &fields, "TensorRT 8 global plugin creator field collection query");
    if (status != JYPPX_STATUS_OK) { return status; }
    const char* field_name = "";
    status = jyppx_trt8_get_plugin_field_name(fields, field_index, &field_name, "TensorRT 8 global plugin field name query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(field_name, output_buffer, output_buffer_size, out_required_size);
#else
    (void)creator_index; (void)field_index; (void)output_buffer; (void)output_buffer_size;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 global plugin field name query");
#endif
}

JYPPX_StatusCode jyppx_trt8_global_plugin_creator_get_field_metadata(
    int32_t creator_index,
    int32_t field_index,
    int32_t* out_field_type,
    int32_t* out_length,
    JYPPX_Boolean* out_has_data)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_field_type, "out_field_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_length, "out_length");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_has_data, "out_has_data");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_field_type = 0; *out_length = 0; *out_has_data = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
    nvinfer1::IPluginCreator* creator = nullptr;
    status = get_trt8_global_plugin_creator(creator_index, &creator);
    if (status != JYPPX_STATUS_OK) { return status; }
    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = jyppx_trt8_get_creator_field_collection(creator, &fields, "TensorRT 8 global plugin creator field collection query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return jyppx_trt8_get_plugin_field_metadata(
        fields,
        field_index,
        out_field_type,
        out_length,
        out_has_data,
        "TensorRT 8 global plugin field metadata query");
#else
    (void)creator_index; (void)field_index;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 global plugin field metadata query");
#endif
}
