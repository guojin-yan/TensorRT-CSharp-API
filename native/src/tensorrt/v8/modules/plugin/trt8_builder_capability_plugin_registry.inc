#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
static JYPPX_StatusCode validate_trt8_engine_capability(const int32_t capability)
{
    if (capability < static_cast<int32_t>(nvinfer1::EngineCapability::kSTANDARD) ||
        capability > static_cast<int32_t>(nvinfer1::EngineCapability::kDLA_STANDALONE))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 engine capability value is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode get_trt8_builder_capability_registry_with_seh_guard(
    const int32_t capability,
    nvinfer1::IPluginRegistry** out_registry,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_registry = nvinfer1::getBuilderPluginRegistry(static_cast<nvinfer1::EngineCapability>(capability));
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_registry = nullptr;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_registry = nvinfer1::getBuilderPluginRegistry(static_cast<nvinfer1::EngineCapability>(capability));
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode get_trt8_builder_safe_registry_exists_with_seh_guard(
    const int32_t capability,
    JYPPX_Boolean* out_exists)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_exists = nvinfer1::getBuilderSafePluginRegistry(static_cast<nvinfer1::EngineCapability>(capability)) != nullptr
            ? JYPPX_TRUE
            : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_exists = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, "TensorRT 8 builder safe plugin registry existence query", seh_exception_code);
    }
#else
    *out_exists = nvinfer1::getBuilderSafePluginRegistry(static_cast<nvinfer1::EngineCapability>(capability)) != nullptr
        ? JYPPX_TRUE
        : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode get_trt8_builder_capability_registry(
    const int32_t capability,
    nvinfer1::IPluginRegistry** out_registry,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_registry, "out_registry");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_registry = nullptr;
    status = validate_trt8_engine_capability(capability);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        status = get_trt8_builder_capability_registry_with_seh_guard(capability, out_registry, feature_name);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }
    }
    catch (const std::exception& exception)
    {
        *out_registry = nullptr;
        return jyppx_trt8_report_plugin_inventory_exception(feature_name, exception);
    }
    catch (...)
    {
        *out_registry = nullptr;
        return jyppx_trt8_report_plugin_inventory_unknown_exception(feature_name);
    }

    if (*out_registry == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 getBuilderPluginRegistry returned null for the requested engine capability.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode get_trt8_builder_capability_creator(
    const int32_t capability,
    const int32_t creator_index,
    nvinfer1::IPluginCreator** out_creator)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_creator, "out_creator");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_creator = nullptr;
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = get_trt8_builder_capability_registry(
        capability,
        &registry,
        "TensorRT 8 builder capability plugin registry query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    int32_t creator_count = 0;
    nvinfer1::IPluginCreator* const* creators = nullptr;
    status = jyppx_trt8_get_inventory_registry_creators(
        registry,
        &creators,
        &creator_count,
        "TensorRT 8 builder capability plugin creator lookup by index");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_index(creator_index, creator_count, "builder_capability_plugin_creator");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (creators == nullptr || creators[creator_index] == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 builder capability registry returned a null creator pointer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_creator = creators[creator_index];
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode get_trt8_builder_capability_lookup_creator(
    const int32_t capability,
    const char* plugin_name,
    const char* plugin_version,
    const char* plugin_namespace,
    nvinfer1::IPluginCreator** out_creator)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_creator, "out_creator");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_creator = nullptr;
    status = jyppx_trt8_validate_plugin_inventory_lookup_string(plugin_name, "plugin_name");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt8_validate_plugin_inventory_lookup_string(plugin_version, "plugin_version");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_trt8_validate_plugin_inventory_lookup_string(plugin_namespace, "plugin_namespace");
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::IPluginRegistry* registry = nullptr;
    status = get_trt8_builder_capability_registry(
        capability,
        &registry,
        "TensorRT 8 builder capability plugin creator lookup");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return jyppx_trt8_get_inventory_creator_with_seh_guard(
            registry,
            plugin_name,
            plugin_version,
            plugin_namespace,
            out_creator,
            "TensorRT 8 builder capability plugin creator lookup");
    }
    catch (const std::exception& exception)
    {
        *out_creator = nullptr;
        return jyppx_trt8_report_plugin_inventory_exception("TensorRT 8 builder capability plugin creator lookup", exception);
    }
    catch (...)
    {
        *out_creator = nullptr;
        return jyppx_trt8_report_plugin_inventory_unknown_exception("TensorRT 8 builder capability plugin creator lookup");
    }
}

static JYPPX_StatusCode copy_trt8_builder_capability_creator_string(
    const int32_t capability,
    const int32_t creator_index,
    const int32_t selector,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::IPluginCreator* creator = nullptr;
    status = get_trt8_builder_capability_creator(capability, creator_index, &creator);
    if (status != JYPPX_STATUS_OK) { return status; }

    const char* value = "";
    switch (selector)
    {
    case 0:
        status = jyppx_trt8_get_creator_name(creator, &value, "TensorRT 8 builder capability plugin creator name query");
        break;
    case 1:
        status = jyppx_trt8_get_creator_version(creator, &value, "TensorRT 8 builder capability plugin creator version query");
        break;
    default:
        status = jyppx_trt8_get_creator_namespace(creator, &value, "TensorRT 8 builder capability plugin creator namespace query");
        break;
    }

    return status == JYPPX_STATUS_OK
        ? copy_string_to_buffer(value, output_buffer, output_buffer_size, out_required_size)
        : status;
}

static JYPPX_StatusCode get_trt8_creator_fields(
    nvinfer1::IPluginCreator* creator,
    const nvinfer1::PluginFieldCollection** out_fields,
    const char* feature_name)
{
    auto status = jyppx_trt8_get_creator_field_collection(creator, out_fields, feature_name);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int32_t field_count = *out_fields != nullptr ? (*out_fields)->nbFields : 0;
    return jyppx_trt8_validate_plugin_field_storage(*out_fields, field_count, feature_name);
}

JYPPX_StatusCode jyppx_trt8_builder_capability_plugin_registry_exists(
    const int32_t capability,
    JYPPX_Boolean* out_exists)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_exists, "out_exists");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_exists = JYPPX_FALSE;

    status = validate_trt8_engine_capability(capability);
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::IPluginRegistry* registry = nullptr;
    try
    {
        status = get_trt8_builder_capability_registry_with_seh_guard(
            capability,
            &registry,
            "TensorRT 8 builder capability plugin registry existence query");
        if (status == JYPPX_STATUS_OK)
        {
            *out_exists = registry != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        }
        return status;
    }
    catch (const std::exception& exception)
    {
        return jyppx_trt8_report_plugin_inventory_exception("TensorRT 8 builder capability plugin registry existence query", exception);
    }
    catch (...)
    {
        return jyppx_trt8_report_plugin_inventory_unknown_exception("TensorRT 8 builder capability plugin registry existence query");
    }
}

JYPPX_StatusCode jyppx_trt8_builder_safe_plugin_registry_exists(
    const int32_t capability,
    JYPPX_Boolean* out_exists)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_exists, "out_exists");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_exists = JYPPX_FALSE;
    status = validate_trt8_engine_capability(capability);
    if (status != JYPPX_STATUS_OK) { return status; }

    try
    {
        return get_trt8_builder_safe_registry_exists_with_seh_guard(capability, out_exists);
    }
    catch (const std::exception& exception)
    {
        return jyppx_trt8_report_plugin_inventory_exception("TensorRT 8 builder safe plugin registry existence query", exception);
    }
    catch (...)
    {
        return jyppx_trt8_report_plugin_inventory_unknown_exception("TensorRT 8 builder safe plugin registry existence query");
    }
}

JYPPX_StatusCode jyppx_trt8_builder_capability_plugin_registry_get_creator_count(
    const int32_t capability,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;

    nvinfer1::IPluginRegistry* registry = nullptr;
    status = get_trt8_builder_capability_registry(capability, &registry, "TensorRT 8 builder capability plugin registry creator count query");
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::IPluginCreator* const* creators = nullptr;
    status = jyppx_trt8_get_inventory_registry_creators(
        registry,
        &creators,
        out_count,
        "TensorRT 8 builder capability plugin registry creator count query");
    (void)creators;
    return status;
}

JYPPX_StatusCode jyppx_trt8_builder_capability_plugin_registry_has_error_recorder(
    const int32_t capability,
    JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_recorder = JYPPX_FALSE;

    nvinfer1::IPluginRegistry* registry = nullptr;
    status = get_trt8_builder_capability_registry(capability, &registry, "TensorRT 8 builder capability plugin registry error recorder query");
    if (status != JYPPX_STATUS_OK) { return status; }

    try
    {
        return jyppx_trt8_get_plugin_registry_has_error_recorder_with_seh_guard(
            registry,
            out_has_recorder,
            "TensorRT 8 builder capability plugin registry error recorder query");
    }
    catch (const std::exception& exception)
    {
        return jyppx_trt8_report_plugin_inventory_exception("TensorRT 8 builder capability plugin registry error recorder query", exception);
    }
    catch (...)
    {
        return jyppx_trt8_report_plugin_inventory_unknown_exception("TensorRT 8 builder capability plugin registry error recorder query");
    }
}

JYPPX_StatusCode jyppx_trt8_builder_capability_plugin_registry_is_parent_search_enabled(
    const int32_t capability,
    JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_enabled = JYPPX_FALSE;

    nvinfer1::IPluginRegistry* registry = nullptr;
    status = get_trt8_builder_capability_registry(capability, &registry, "TensorRT 8 builder capability plugin registry parent search query");
    if (status != JYPPX_STATUS_OK) { return status; }

    try
    {
        return jyppx_trt8_get_plugin_registry_parent_search_enabled_with_seh_guard(
            registry,
            out_enabled,
            "TensorRT 8 builder capability plugin registry parent search query");
    }
    catch (const std::exception& exception)
    {
        return jyppx_trt8_report_plugin_inventory_exception("TensorRT 8 builder capability plugin registry parent search query", exception);
    }
    catch (...)
    {
        return jyppx_trt8_report_plugin_inventory_unknown_exception("TensorRT 8 builder capability plugin registry parent search query");
    }
}

JYPPX_StatusCode jyppx_trt8_builder_capability_plugin_creator_get_name(
    const int32_t capability,
    const int32_t creator_index,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size)
{
    return copy_trt8_builder_capability_creator_string(capability, creator_index, 0, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt8_builder_capability_plugin_creator_get_version(
    const int32_t capability,
    const int32_t creator_index,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size)
{
    return copy_trt8_builder_capability_creator_string(capability, creator_index, 1, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt8_builder_capability_plugin_creator_get_namespace(
    const int32_t capability,
    const int32_t creator_index,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size)
{
    return copy_trt8_builder_capability_creator_string(capability, creator_index, 2, output_buffer, output_buffer_size, out_required_size);
}

JYPPX_StatusCode jyppx_trt8_builder_capability_plugin_creator_get_tensor_rt_version(
    const int32_t capability,
    const int32_t creator_index,
    int32_t* out_tensor_rt_version)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor_rt_version, "out_tensor_rt_version");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor_rt_version = 0;

    nvinfer1::IPluginCreator* creator = nullptr;
    status = get_trt8_builder_capability_creator(capability, creator_index, &creator);
    return status == JYPPX_STATUS_OK
        ? jyppx_trt8_get_creator_tensor_rt_version(creator, out_tensor_rt_version, "TensorRT 8 builder capability plugin creator TensorRT version query")
        : status;
}

JYPPX_StatusCode jyppx_trt8_builder_capability_plugin_creator_get_field_count(
    const int32_t capability,
    const int32_t creator_index,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;

    nvinfer1::IPluginCreator* creator = nullptr;
    status = get_trt8_builder_capability_creator(capability, creator_index, &creator);
    if (status != JYPPX_STATUS_OK) { return status; }

    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = jyppx_trt8_get_creator_field_collection(creator, &fields, "TensorRT 8 builder capability plugin creator field count query");
    if (status == JYPPX_STATUS_OK)
    {
        *out_count = fields != nullptr ? fields->nbFields : 0;
    }
    return status;
}

JYPPX_StatusCode jyppx_trt8_builder_capability_plugin_creator_get_field_name(
    const int32_t capability,
    const int32_t creator_index,
    const int32_t field_index,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::IPluginCreator* creator = nullptr;
    status = get_trt8_builder_capability_creator(capability, creator_index, &creator);
    if (status != JYPPX_STATUS_OK) { return status; }

    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = get_trt8_creator_fields(creator, &fields, "TensorRT 8 builder capability plugin creator field name query");
    if (status != JYPPX_STATUS_OK) { return status; }
    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "builder_capability_plugin_field");
    if (status != JYPPX_STATUS_OK) { return status; }

    const char* field_name = "";
    status = jyppx_trt8_get_plugin_field_name(fields, field_index, &field_name, "TensorRT 8 builder capability plugin creator field name query");
    return status == JYPPX_STATUS_OK
        ? copy_string_to_buffer(field_name, output_buffer, output_buffer_size, out_required_size)
        : status;
}

JYPPX_StatusCode jyppx_trt8_builder_capability_plugin_creator_get_field_metadata(
    const int32_t capability,
    const int32_t creator_index,
    const int32_t field_index,
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
    *out_field_type = 0;
    *out_length = 0;
    *out_has_data = JYPPX_FALSE;

    nvinfer1::IPluginCreator* creator = nullptr;
    status = get_trt8_builder_capability_creator(capability, creator_index, &creator);
    if (status != JYPPX_STATUS_OK) { return status; }

    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = get_trt8_creator_fields(creator, &fields, "TensorRT 8 builder capability plugin creator field metadata query");
    if (status != JYPPX_STATUS_OK) { return status; }
    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "builder_capability_plugin_field");
    return status == JYPPX_STATUS_OK
        ? jyppx_trt8_get_plugin_field_metadata(fields, field_index, out_field_type, out_length, out_has_data, "TensorRT 8 builder capability plugin creator field metadata query")
        : status;
}

JYPPX_StatusCode jyppx_trt8_builder_capability_plugin_creator_lookup(
    const int32_t capability,
    const char* plugin_name,
    const char* plugin_version,
    const char* plugin_namespace,
    JYPPX_Boolean* out_found)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_found, "out_found");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_found = JYPPX_FALSE;

    nvinfer1::IPluginCreator* creator = nullptr;
    status = get_trt8_builder_capability_lookup_creator(capability, plugin_name, plugin_version, plugin_namespace, &creator);
    if (status == JYPPX_STATUS_OK)
    {
        *out_found = creator != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    }
    return status;
}

static JYPPX_StatusCode require_trt8_builder_capability_lookup_creator(
    const int32_t capability,
    const char* plugin_name,
    const char* plugin_version,
    const char* plugin_namespace,
    nvinfer1::IPluginCreator** out_creator)
{
    const auto status = get_trt8_builder_capability_lookup_creator(
        capability,
        plugin_name,
        plugin_version,
        plugin_namespace,
        out_creator);
    if (status != JYPPX_STATUS_OK || *out_creator != nullptr)
    {
        return status;
    }

    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 builder capability plugin creator lookup did not find a matching creator.");
    return JYPPX_STATUS_NOT_FOUND;
}

JYPPX_StatusCode jyppx_trt8_builder_capability_plugin_creator_lookup_get_tensor_rt_version(
    const int32_t capability,
    const char* plugin_name,
    const char* plugin_version,
    const char* plugin_namespace,
    int32_t* out_tensor_rt_version)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor_rt_version, "out_tensor_rt_version");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tensor_rt_version = 0;

    nvinfer1::IPluginCreator* creator = nullptr;
    status = require_trt8_builder_capability_lookup_creator(capability, plugin_name, plugin_version, plugin_namespace, &creator);
    return status == JYPPX_STATUS_OK
        ? jyppx_trt8_get_creator_tensor_rt_version(creator, out_tensor_rt_version, "TensorRT 8 builder capability lookup creator TensorRT version query")
        : status;
}

JYPPX_StatusCode jyppx_trt8_builder_capability_plugin_creator_lookup_get_field_count(
    const int32_t capability,
    const char* plugin_name,
    const char* plugin_version,
    const char* plugin_namespace,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;

    nvinfer1::IPluginCreator* creator = nullptr;
    status = require_trt8_builder_capability_lookup_creator(capability, plugin_name, plugin_version, plugin_namespace, &creator);
    if (status != JYPPX_STATUS_OK) { return status; }

    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = jyppx_trt8_get_creator_field_collection(creator, &fields, "TensorRT 8 builder capability lookup creator field count query");
    if (status == JYPPX_STATUS_OK)
    {
        *out_count = fields != nullptr ? fields->nbFields : 0;
    }
    return status;
}

JYPPX_StatusCode jyppx_trt8_builder_capability_plugin_creator_lookup_get_field_name(
    const int32_t capability,
    const char* plugin_name,
    const char* plugin_version,
    const char* plugin_namespace,
    const int32_t field_index,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::IPluginCreator* creator = nullptr;
    status = require_trt8_builder_capability_lookup_creator(capability, plugin_name, plugin_version, plugin_namespace, &creator);
    if (status != JYPPX_STATUS_OK) { return status; }

    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = get_trt8_creator_fields(creator, &fields, "TensorRT 8 builder capability lookup creator field name query");
    if (status != JYPPX_STATUS_OK) { return status; }
    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "builder_capability_plugin_lookup_field");
    if (status != JYPPX_STATUS_OK) { return status; }

    const char* field_name = "";
    status = jyppx_trt8_get_plugin_field_name(fields, field_index, &field_name, "TensorRT 8 builder capability lookup creator field name query");
    return status == JYPPX_STATUS_OK
        ? copy_string_to_buffer(field_name, output_buffer, output_buffer_size, out_required_size)
        : status;
}

JYPPX_StatusCode jyppx_trt8_builder_capability_plugin_creator_lookup_get_field_metadata(
    const int32_t capability,
    const char* plugin_name,
    const char* plugin_version,
    const char* plugin_namespace,
    const int32_t field_index,
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
    *out_field_type = 0;
    *out_length = 0;
    *out_has_data = JYPPX_FALSE;

    nvinfer1::IPluginCreator* creator = nullptr;
    status = require_trt8_builder_capability_lookup_creator(capability, plugin_name, plugin_version, plugin_namespace, &creator);
    if (status != JYPPX_STATUS_OK) { return status; }

    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = get_trt8_creator_fields(creator, &fields, "TensorRT 8 builder capability lookup creator field metadata query");
    if (status != JYPPX_STATUS_OK) { return status; }
    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "builder_capability_plugin_lookup_field");
    return status == JYPPX_STATUS_OK
        ? jyppx_trt8_get_plugin_field_metadata(fields, field_index, out_field_type, out_length, out_has_data, "TensorRT 8 builder capability lookup creator field metadata query")
        : status;
}
#else
#define JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE(name, parameters, unused) \
    JYPPX_StatusCode name parameters \
    { \
        unused \
        return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 builder capability plugin registry query"); \
    }

JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE(jyppx_trt8_builder_capability_plugin_registry_exists, (int32_t capability, JYPPX_Boolean* out_exists), (void)capability; (void)out_exists;)
JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE(jyppx_trt8_builder_safe_plugin_registry_exists, (int32_t capability, JYPPX_Boolean* out_exists), (void)capability; (void)out_exists;)
JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE(jyppx_trt8_builder_capability_plugin_registry_get_creator_count, (int32_t capability, int32_t* out_count), (void)capability; (void)out_count;)
JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE(jyppx_trt8_builder_capability_plugin_registry_has_error_recorder, (int32_t capability, JYPPX_Boolean* out_has_recorder), (void)capability; (void)out_has_recorder;)
JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE(jyppx_trt8_builder_capability_plugin_registry_is_parent_search_enabled, (int32_t capability, JYPPX_Boolean* out_enabled), (void)capability; (void)out_enabled;)
JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE(jyppx_trt8_builder_capability_plugin_creator_get_name, (int32_t capability, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size), (void)capability; (void)creator_index; (void)output_buffer; (void)output_buffer_size; (void)out_required_size;)
JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE(jyppx_trt8_builder_capability_plugin_creator_get_version, (int32_t capability, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size), (void)capability; (void)creator_index; (void)output_buffer; (void)output_buffer_size; (void)out_required_size;)
JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE(jyppx_trt8_builder_capability_plugin_creator_get_namespace, (int32_t capability, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size), (void)capability; (void)creator_index; (void)output_buffer; (void)output_buffer_size; (void)out_required_size;)
JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE(jyppx_trt8_builder_capability_plugin_creator_get_tensor_rt_version, (int32_t capability, int32_t creator_index, int32_t* out_tensor_rt_version), (void)capability; (void)creator_index; (void)out_tensor_rt_version;)
JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE(jyppx_trt8_builder_capability_plugin_creator_get_field_count, (int32_t capability, int32_t creator_index, int32_t* out_count), (void)capability; (void)creator_index; (void)out_count;)
JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE(jyppx_trt8_builder_capability_plugin_creator_get_field_name, (int32_t capability, int32_t creator_index, int32_t field_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size), (void)capability; (void)creator_index; (void)field_index; (void)output_buffer; (void)output_buffer_size; (void)out_required_size;)
JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE(jyppx_trt8_builder_capability_plugin_creator_get_field_metadata, (int32_t capability, int32_t creator_index, int32_t field_index, int32_t* out_field_type, int32_t* out_length, JYPPX_Boolean* out_has_data), (void)capability; (void)creator_index; (void)field_index; (void)out_field_type; (void)out_length; (void)out_has_data;)
JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE(jyppx_trt8_builder_capability_plugin_creator_lookup, (int32_t capability, const char* plugin_name, const char* plugin_version, const char* plugin_namespace, JYPPX_Boolean* out_found), (void)capability; (void)plugin_name; (void)plugin_version; (void)plugin_namespace; (void)out_found;)
JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE(jyppx_trt8_builder_capability_plugin_creator_lookup_get_tensor_rt_version, (int32_t capability, const char* plugin_name, const char* plugin_version, const char* plugin_namespace, int32_t* out_tensor_rt_version), (void)capability; (void)plugin_name; (void)plugin_version; (void)plugin_namespace; (void)out_tensor_rt_version;)
JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE(jyppx_trt8_builder_capability_plugin_creator_lookup_get_field_count, (int32_t capability, const char* plugin_name, const char* plugin_version, const char* plugin_namespace, int32_t* out_count), (void)capability; (void)plugin_name; (void)plugin_version; (void)plugin_namespace; (void)out_count;)
JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE(jyppx_trt8_builder_capability_plugin_creator_lookup_get_field_name, (int32_t capability, const char* plugin_name, const char* plugin_version, const char* plugin_namespace, int32_t field_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size), (void)capability; (void)plugin_name; (void)plugin_version; (void)plugin_namespace; (void)field_index; (void)output_buffer; (void)output_buffer_size; (void)out_required_size;)
JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE(jyppx_trt8_builder_capability_plugin_creator_lookup_get_field_metadata, (int32_t capability, const char* plugin_name, const char* plugin_version, const char* plugin_namespace, int32_t field_index, int32_t* out_field_type, int32_t* out_length, JYPPX_Boolean* out_has_data), (void)capability; (void)plugin_name; (void)plugin_version; (void)plugin_namespace; (void)field_index; (void)out_field_type; (void)out_length; (void)out_has_data;)
#undef JYPPX_TRT8_BUILDER_CAPABILITY_UNAVAILABLE
#endif
