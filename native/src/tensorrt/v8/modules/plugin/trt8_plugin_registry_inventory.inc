#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8
#define JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY 1
#else
#define JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY 0
#endif

#if !JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
static JYPPX_StatusCode jyppx_trt8_plugin_registry_report_unavailable(const char* feature_name)
{
#if JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, feature_name);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, feature_name);
#endif
}
#endif

#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
JYPPX_StatusCode jyppx_trt8_report_plugin_inventory_exception(const char* feature_name, const std::exception& exception)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
}

JYPPX_StatusCode jyppx_trt8_report_plugin_inventory_unknown_exception(const char* feature_name)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
}

JYPPX_StatusCode jyppx_trt8_get_builder_plugin_registry_with_seh_guard(
    nvinfer1::IBuilder* builder_payload,
    nvinfer1::IPluginRegistry** out_registry,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_registry = &builder_payload->getPluginRegistry();
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_registry = nullptr;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_registry = &builder_payload->getPluginRegistry();
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode jyppx_trt8_get_builder_plugin_registry(
    JYPPX_TensorRtBuilder* builder,
    nvinfer1::IPluginRegistry** out_registry,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_registry, "out_registry");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_registry = nullptr;
    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    if (builder_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder handle does not carry a TensorRT builder payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    try
    {
        status = jyppx_trt8_get_builder_plugin_registry_with_seh_guard(builder_payload, out_registry, feature_name);
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
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 builder plugin registry is null.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt8_get_inventory_registry_creators_with_seh_guard(
    nvinfer1::IPluginRegistry* registry,
    nvinfer1::IPluginCreator* const** out_creators,
    int32_t* out_count,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_creators = registry->getPluginCreatorList(out_count);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_creators = nullptr;
        *out_count = 0;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_creators = registry->getPluginCreatorList(out_count);
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode jyppx_trt8_get_inventory_registry_creators(
    nvinfer1::IPluginRegistry* registry,
    nvinfer1::IPluginCreator* const** out_creators,
    int32_t* out_count,
    const char* feature_name)
{
    *out_creators = nullptr;
    *out_count = 0;
    if (registry == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 plugin registry is null.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    try
    {
        return jyppx_trt8_get_inventory_registry_creators_with_seh_guard(registry, out_creators, out_count, feature_name);
    }
    catch (const std::exception& exception)
    {
        *out_creators = nullptr;
        *out_count = 0;
        return jyppx_trt8_report_plugin_inventory_exception(feature_name, exception);
    }
    catch (...)
    {
        *out_creators = nullptr;
        *out_count = 0;
        return jyppx_trt8_report_plugin_inventory_unknown_exception(feature_name);
    }
}

JYPPX_StatusCode jyppx_trt8_get_plugin_registry_parent_search_enabled_with_seh_guard(
    nvinfer1::IPluginRegistry* registry,
    JYPPX_Boolean* out_enabled,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_enabled = registry->isParentSearchEnabled() ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_enabled = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_enabled = registry->isParentSearchEnabled() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode jyppx_trt8_get_plugin_registry_has_error_recorder_with_seh_guard(
    nvinfer1::IPluginRegistry* registry,
    JYPPX_Boolean* out_has_recorder,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_recorder = registry->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_recorder = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_recorder = registry->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode jyppx_trt8_get_builder_plugin_creator(
    JYPPX_TensorRtBuilder* builder,
    int32_t creator_index,
    nvinfer1::IPluginCreator** out_creator)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_creator, "out_creator");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_creator = nullptr;
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt8_get_builder_plugin_registry(builder, &registry, "TensorRT 8 builder plugin registry query");
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
        "TensorRT 8 plugin creator lookup by index");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_index(creator_index, creator_count, "plugin_creator");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (creators == nullptr || creators[creator_index] == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 plugin registry returned a null creator pointer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_creator = creators[creator_index];
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt8_validate_plugin_inventory_lookup_string(const char* value, const char* name)
{
    if (value == nullptr)
    {
        std::ostringstream builder;
        builder << name << " must not be null.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (std::strlen(value) > 1023)
    {
        std::ostringstream builder;
        builder << name << " must be 1024 bytes or less including the null terminator.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt8_get_inventory_creator_with_seh_guard(
    nvinfer1::IPluginRegistry* registry,
    const char* plugin_name,
    const char* plugin_version,
    const char* plugin_namespace,
    nvinfer1::IPluginCreator** out_creator,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_creator = registry->getPluginCreator(plugin_name, plugin_version, plugin_namespace);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_creator = nullptr;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_creator = registry->getPluginCreator(plugin_name, plugin_version, plugin_namespace);
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode jyppx_trt8_get_builder_lookup_creator(
    JYPPX_TensorRtBuilder* builder,
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
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx_trt8_validate_plugin_inventory_lookup_string(plugin_version, "plugin_version");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx_trt8_validate_plugin_inventory_lookup_string(plugin_namespace, "plugin_namespace");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt8_get_builder_plugin_registry(builder, &registry, "TensorRT 8 builder plugin creator lookup");
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
            "TensorRT 8 builder plugin creator lookup");
    }
    catch (const std::exception& exception)
    {
        *out_creator = nullptr;
        return jyppx_trt8_report_plugin_inventory_exception("TensorRT 8 builder plugin creator lookup", exception);
    }
    catch (...)
    {
        *out_creator = nullptr;
        return jyppx_trt8_report_plugin_inventory_unknown_exception("TensorRT 8 builder plugin creator lookup");
    }
}

JYPPX_StatusCode jyppx_trt8_get_creator_name_with_seh_guard(
    nvinfer1::IPluginCreator* creator,
    const char** out_value,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_value = creator != nullptr ? creator->getPluginName() : "";
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_value = "";
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_value = creator != nullptr ? creator->getPluginName() : "";
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode jyppx_trt8_get_creator_version_with_seh_guard(
    nvinfer1::IPluginCreator* creator,
    const char** out_value,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_value = creator != nullptr ? creator->getPluginVersion() : "";
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_value = "";
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_value = creator != nullptr ? creator->getPluginVersion() : "";
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode jyppx_trt8_get_creator_namespace_with_seh_guard(
    nvinfer1::IPluginCreator* creator,
    const char** out_value,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_value = creator != nullptr ? creator->getPluginNamespace() : "";
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_value = "";
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_value = creator != nullptr ? creator->getPluginNamespace() : "";
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode jyppx_trt8_get_creator_field_collection_with_seh_guard(
    nvinfer1::IPluginCreator* creator,
    const nvinfer1::PluginFieldCollection** out_fields,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_fields = creator != nullptr ? creator->getFieldNames() : nullptr;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_fields = nullptr;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_fields = creator != nullptr ? creator->getFieldNames() : nullptr;
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode jyppx_trt8_get_creator_name(nvinfer1::IPluginCreator* creator, const char** out_value, const char* feature_name)
{
    try
    {
        return jyppx_trt8_get_creator_name_with_seh_guard(creator, out_value, feature_name);
    }
    catch (const std::exception& exception)
    {
        *out_value = "";
        return jyppx_trt8_report_plugin_inventory_exception(feature_name, exception);
    }
    catch (...)
    {
        *out_value = "";
        return jyppx_trt8_report_plugin_inventory_unknown_exception(feature_name);
    }
}

JYPPX_StatusCode jyppx_trt8_get_creator_version(nvinfer1::IPluginCreator* creator, const char** out_value, const char* feature_name)
{
    try
    {
        return jyppx_trt8_get_creator_version_with_seh_guard(creator, out_value, feature_name);
    }
    catch (const std::exception& exception)
    {
        *out_value = "";
        return jyppx_trt8_report_plugin_inventory_exception(feature_name, exception);
    }
    catch (...)
    {
        *out_value = "";
        return jyppx_trt8_report_plugin_inventory_unknown_exception(feature_name);
    }
}

JYPPX_StatusCode jyppx_trt8_get_creator_namespace(nvinfer1::IPluginCreator* creator, const char** out_value, const char* feature_name)
{
    try
    {
        return jyppx_trt8_get_creator_namespace_with_seh_guard(creator, out_value, feature_name);
    }
    catch (const std::exception& exception)
    {
        *out_value = "";
        return jyppx_trt8_report_plugin_inventory_exception(feature_name, exception);
    }
    catch (...)
    {
        *out_value = "";
        return jyppx_trt8_report_plugin_inventory_unknown_exception(feature_name);
    }
}

JYPPX_StatusCode jyppx_trt8_get_creator_field_collection(
    nvinfer1::IPluginCreator* creator,
    const nvinfer1::PluginFieldCollection** out_fields,
    const char* feature_name)
{
    try
    {
        return jyppx_trt8_get_creator_field_collection_with_seh_guard(creator, out_fields, feature_name);
    }
    catch (const std::exception& exception)
    {
        *out_fields = nullptr;
        return jyppx_trt8_report_plugin_inventory_exception(feature_name, exception);
    }
    catch (...)
    {
        *out_fields = nullptr;
        return jyppx_trt8_report_plugin_inventory_unknown_exception(feature_name);
    }
}

JYPPX_StatusCode jyppx_trt8_validate_plugin_field_storage(
    const nvinfer1::PluginFieldCollection* fields,
    int32_t field_count,
    const char* feature_name)
{
    if (field_count <= 0)
    {
        return JYPPX_STATUS_OK;
    }

    if (fields == nullptr || fields->fields == nullptr)
    {
        std::ostringstream builder;
        builder << feature_name << " returned a non-empty PluginFieldCollection without field storage.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt8_get_plugin_field_name_with_seh_guard(
    const nvinfer1::PluginFieldCollection* fields,
    int32_t field_index,
    const char** out_name,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_name = fields->fields[field_index].name;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_name = "";
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_name = fields->fields[field_index].name;
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode jyppx_trt8_get_plugin_field_metadata_with_seh_guard(
    const nvinfer1::PluginFieldCollection* fields,
    int32_t field_index,
    int32_t* out_field_type,
    int32_t* out_length,
    JYPPX_Boolean* out_has_data,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        const auto& field = fields->fields[field_index];
        *out_field_type = static_cast<int32_t>(field.type);
        *out_length = field.length;
        *out_has_data = field.data != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_field_type = 0;
        *out_length = 0;
        *out_has_data = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    const auto& field = fields->fields[field_index];
    *out_field_type = static_cast<int32_t>(field.type);
    *out_length = field.length;
    *out_has_data = field.data != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode jyppx_trt8_get_plugin_field_name(
    const nvinfer1::PluginFieldCollection* fields,
    int32_t field_index,
    const char** out_name,
    const char* feature_name)
{
    try
    {
        return jyppx_trt8_get_plugin_field_name_with_seh_guard(fields, field_index, out_name, feature_name);
    }
    catch (const std::exception& exception)
    {
        *out_name = "";
        return jyppx_trt8_report_plugin_inventory_exception(feature_name, exception);
    }
    catch (...)
    {
        *out_name = "";
        return jyppx_trt8_report_plugin_inventory_unknown_exception(feature_name);
    }
}

JYPPX_StatusCode jyppx_trt8_get_plugin_field_metadata(
    const nvinfer1::PluginFieldCollection* fields,
    int32_t field_index,
    int32_t* out_field_type,
    int32_t* out_length,
    JYPPX_Boolean* out_has_data,
    const char* feature_name)
{
    try
    {
        return jyppx_trt8_get_plugin_field_metadata_with_seh_guard(
            fields,
            field_index,
            out_field_type,
            out_length,
            out_has_data,
            feature_name);
    }
    catch (const std::exception& exception)
    {
        *out_field_type = 0;
        *out_length = 0;
        *out_has_data = JYPPX_FALSE;
        return jyppx_trt8_report_plugin_inventory_exception(feature_name, exception);
    }
    catch (...)
    {
        *out_field_type = 0;
        *out_length = 0;
        *out_has_data = JYPPX_FALSE;
        return jyppx_trt8_report_plugin_inventory_unknown_exception(feature_name);
    }
}

JYPPX_StatusCode jyppx_trt8_get_runtime_plugin_registry_with_seh_guard(
    nvinfer1::IRuntime* runtime_payload,
    nvinfer1::IPluginRegistry** out_registry,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_registry = &runtime_payload->getPluginRegistry();
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_registry = nullptr;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_registry = &runtime_payload->getPluginRegistry();
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode jyppx_trt8_get_runtime_plugin_registry(
    JYPPX_TensorRtRuntime* runtime,
    nvinfer1::IPluginRegistry** out_registry,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_registry, "out_registry");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_registry = nullptr;
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Runtime handle does not carry a TensorRT runtime payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    try
    {
        status = jyppx_trt8_get_runtime_plugin_registry_with_seh_guard(runtime_payload, out_registry, feature_name);
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
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 runtime plugin registry is null.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt8_get_runtime_plugin_creator(
    JYPPX_TensorRtRuntime* runtime,
    int32_t creator_index,
    nvinfer1::IPluginCreator** out_creator)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_creator, "out_creator");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_creator = nullptr;
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt8_get_runtime_plugin_registry(runtime, &registry, "TensorRT 8 runtime plugin registry query");
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
        "TensorRT 8 runtime plugin creator lookup by index");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_index(creator_index, creator_count, "runtime_plugin_creator");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (creators == nullptr || creators[creator_index] == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 runtime plugin registry returned a null creator pointer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_creator = creators[creator_index];
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt8_get_runtime_lookup_creator(
    JYPPX_TensorRtRuntime* runtime,
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
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx_trt8_validate_plugin_inventory_lookup_string(plugin_version, "plugin_version");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx_trt8_validate_plugin_inventory_lookup_string(plugin_namespace, "plugin_namespace");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt8_get_runtime_plugin_registry(runtime, &registry, "TensorRT 8 runtime plugin creator lookup");
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
            "TensorRT 8 runtime plugin creator lookup");
    }
    catch (const std::exception& exception)
    {
        *out_creator = nullptr;
        return jyppx_trt8_report_plugin_inventory_exception("TensorRT 8 runtime plugin creator lookup", exception);
    }
    catch (...)
    {
        *out_creator = nullptr;
        return jyppx_trt8_report_plugin_inventory_unknown_exception("TensorRT 8 runtime plugin creator lookup");
    }
}
#endif

JYPPX_StatusCode jyppx_trt8_builder_plugin_registry_exists(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_exists)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_exists, "out_exists");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_exists = JYPPX_FALSE;
#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    if (builder_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder handle does not carry a TensorRT builder payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IPluginRegistry* registry = nullptr;
    try
    {
        status = jyppx_trt8_get_builder_plugin_registry_with_seh_guard(
            builder_payload,
            &registry,
            "TensorRT 8 builder plugin registry existence query");
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }
    }
    catch (const std::exception& exception)
    {
        return jyppx_trt8_report_plugin_inventory_exception("TensorRT 8 builder plugin registry existence query", exception);
    }
    catch (...)
    {
        return jyppx_trt8_report_plugin_inventory_unknown_exception("TensorRT 8 builder plugin registry existence query");
    }

    *out_exists = registry != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 builder plugin registry existence query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_plugin_registry_get_creator_count(JYPPX_TensorRtBuilder* builder, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt8_get_builder_plugin_registry(builder, &registry, "TensorRT 8 plugin registry creator count query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    nvinfer1::IPluginCreator* const* creators = nullptr;
    return jyppx_trt8_get_inventory_registry_creators(
        registry,
        &creators,
        out_count,
        "TensorRT 8 plugin registry creator count query");
#else
    (void)builder;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 plugin registry creator count query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_plugin_registry_has_error_recorder(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt8_get_builder_plugin_registry(builder, &registry, "TensorRT 8 plugin registry error recorder query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return jyppx_trt8_get_plugin_registry_has_error_recorder_with_seh_guard(
            registry,
            out_has_recorder,
            "TensorRT 8 plugin registry error recorder query");
    }
    catch (const std::exception& exception)
    {
        *out_has_recorder = JYPPX_FALSE;
        return jyppx_trt8_report_plugin_inventory_exception("TensorRT 8 plugin registry error recorder query", exception);
    }
    catch (...)
    {
        *out_has_recorder = JYPPX_FALSE;
        return jyppx_trt8_report_plugin_inventory_unknown_exception("TensorRT 8 plugin registry error recorder query");
    }
#else
    (void)builder;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 plugin registry error recorder query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_plugin_registry_is_parent_search_enabled(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_enabled = JYPPX_FALSE;
#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt8_get_builder_plugin_registry(builder, &registry, "TensorRT 8 plugin registry parent search query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return jyppx_trt8_get_plugin_registry_parent_search_enabled_with_seh_guard(
            registry,
            out_enabled,
            "TensorRT 8 plugin registry parent search query");
    }
    catch (const std::exception& exception)
    {
        *out_enabled = JYPPX_FALSE;
        return jyppx_trt8_report_plugin_inventory_exception("TensorRT 8 plugin registry parent search query", exception);
    }
    catch (...)
    {
        *out_enabled = JYPPX_FALSE;
        return jyppx_trt8_report_plugin_inventory_unknown_exception("TensorRT 8 plugin registry parent search query");
    }
#else
    (void)builder;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 plugin registry parent search query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_plugin_creator_get_name(
    JYPPX_TensorRtBuilder* builder,
    int32_t creator_index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_builder_plugin_creator(builder, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const char* name = "";
    status = jyppx_trt8_get_creator_name(creator, &name, "TensorRT 8 plugin creator name query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(name, output_buffer, output_buffer_size, out_required_size);
#else
    (void)builder;
    (void)creator_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 plugin creator name query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_plugin_creator_get_version(
    JYPPX_TensorRtBuilder* builder,
    int32_t creator_index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_builder_plugin_creator(builder, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const char* version = "";
    status = jyppx_trt8_get_creator_version(creator, &version, "TensorRT 8 plugin creator version query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(version, output_buffer, output_buffer_size, out_required_size);
#else
    (void)builder;
    (void)creator_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 plugin creator version query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_plugin_creator_get_namespace(
    JYPPX_TensorRtBuilder* builder,
    int32_t creator_index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_builder_plugin_creator(builder, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const char* plugin_namespace = "";
    status = jyppx_trt8_get_creator_namespace(creator, &plugin_namespace, "TensorRT 8 plugin creator namespace query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(plugin_namespace, output_buffer, output_buffer_size, out_required_size);
#else
    (void)builder;
    (void)creator_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 plugin creator namespace query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_plugin_creator_get_field_count(
    JYPPX_TensorRtBuilder* builder,
    int32_t creator_index,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_builder_plugin_creator(builder, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = jyppx_trt8_get_creator_field_collection(creator, &fields, "TensorRT 8 plugin creator field count query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = fields != nullptr ? fields->nbFields : 0;
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    (void)creator_index;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 plugin creator field count query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_plugin_creator_get_field_name(
    JYPPX_TensorRtBuilder* builder,
    int32_t creator_index,
    int32_t field_index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_builder_plugin_creator(builder, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = jyppx_trt8_get_creator_field_collection(creator, &fields, "TensorRT 8 plugin creator field name query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "plugin_field");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx_trt8_validate_plugin_field_storage(fields, field_count, "TensorRT 8 plugin creator field name query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const char* field_name = "";
    status = jyppx_trt8_get_plugin_field_name(fields, field_index, &field_name, "TensorRT 8 plugin creator field name query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(field_name, output_buffer, output_buffer_size, out_required_size);
#else
    (void)builder;
    (void)creator_index;
    (void)field_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 plugin creator field name query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_plugin_creator_get_field_metadata(
    JYPPX_TensorRtBuilder* builder,
    int32_t creator_index,
    int32_t field_index,
    int32_t* out_field_type,
    int32_t* out_length,
    JYPPX_Boolean* out_has_data)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_field_type, "out_field_type");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_length, "out_length");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_has_data, "out_has_data");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_field_type = 0;
    *out_length = 0;
    *out_has_data = JYPPX_FALSE;
#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_builder_plugin_creator(builder, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = jyppx_trt8_get_creator_field_collection(creator, &fields, "TensorRT 8 plugin creator field metadata query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "plugin_field");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx_trt8_validate_plugin_field_storage(fields, field_count, "TensorRT 8 plugin creator field metadata query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return jyppx_trt8_get_plugin_field_metadata(
        fields,
        field_index,
        out_field_type,
        out_length,
        out_has_data,
        "TensorRT 8 plugin creator field metadata query");
#else
    (void)builder;
    (void)creator_index;
    (void)field_index;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 plugin creator field metadata query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_plugin_creator_lookup(
    JYPPX_TensorRtBuilder* builder,
    const char* plugin_name,
    const char* plugin_version,
    const char* plugin_namespace,
    JYPPX_Boolean* out_found)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_found, "out_found");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_found = JYPPX_FALSE;
#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_builder_lookup_creator(builder, plugin_name, plugin_version, plugin_namespace, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_found = creator != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    (void)plugin_name;
    (void)plugin_version;
    (void)plugin_namespace;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 plugin creator lookup query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_plugin_creator_lookup_get_field_count(
    JYPPX_TensorRtBuilder* builder,
    const char* plugin_name,
    const char* plugin_version,
    const char* plugin_namespace,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_builder_lookup_creator(builder, plugin_name, plugin_version, plugin_namespace, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (creator == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 builder plugin creator lookup did not find a matching creator.");
        return JYPPX_STATUS_NOT_FOUND;
    }

    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = jyppx_trt8_get_creator_field_collection(creator, &fields, "TensorRT 8 builder plugin creator lookup field count query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = fields != nullptr ? fields->nbFields : 0;
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    (void)plugin_name;
    (void)plugin_version;
    (void)plugin_namespace;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 plugin creator lookup field count query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_plugin_creator_lookup_get_field_name(
    JYPPX_TensorRtBuilder* builder,
    const char* plugin_name,
    const char* plugin_version,
    const char* plugin_namespace,
    int32_t field_index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_builder_lookup_creator(builder, plugin_name, plugin_version, plugin_namespace, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (creator == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 builder plugin creator lookup did not find a matching creator.");
        return JYPPX_STATUS_NOT_FOUND;
    }

    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = jyppx_trt8_get_creator_field_collection(creator, &fields, "TensorRT 8 builder plugin creator lookup field name query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "plugin_lookup_field");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx_trt8_validate_plugin_field_storage(fields, field_count, "TensorRT 8 builder plugin creator lookup field name query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const char* field_name = "";
    status = jyppx_trt8_get_plugin_field_name(fields, field_index, &field_name, "TensorRT 8 builder plugin creator lookup field name query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(field_name, output_buffer, output_buffer_size, out_required_size);
#else
    (void)builder;
    (void)plugin_name;
    (void)plugin_version;
    (void)plugin_namespace;
    (void)field_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 plugin creator lookup field name query");
#endif
}

JYPPX_StatusCode jyppx_trt8_builder_plugin_creator_lookup_get_field_metadata(
    JYPPX_TensorRtBuilder* builder,
    const char* plugin_name,
    const char* plugin_version,
    const char* plugin_namespace,
    int32_t field_index,
    int32_t* out_field_type,
    int32_t* out_length,
    JYPPX_Boolean* out_has_data)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_field_type, "out_field_type");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_length, "out_length");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_has_data, "out_has_data");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_field_type = 0;
    *out_length = 0;
    *out_has_data = JYPPX_FALSE;
#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_builder_lookup_creator(builder, plugin_name, plugin_version, plugin_namespace, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (creator == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 builder plugin creator lookup did not find a matching creator.");
        return JYPPX_STATUS_NOT_FOUND;
    }

    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = jyppx_trt8_get_creator_field_collection(creator, &fields, "TensorRT 8 builder plugin creator lookup field metadata query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "plugin_lookup_field");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx_trt8_validate_plugin_field_storage(fields, field_count, "TensorRT 8 builder plugin creator lookup field metadata query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return jyppx_trt8_get_plugin_field_metadata(
        fields,
        field_index,
        out_field_type,
        out_length,
        out_has_data,
        "TensorRT 8 builder plugin creator lookup field metadata query");
#else
    (void)builder;
    (void)plugin_name;
    (void)plugin_version;
    (void)plugin_namespace;
    (void)field_index;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 plugin creator lookup field metadata query");
#endif
}

JYPPX_StatusCode jyppx_trt8_runtime_plugin_registry_exists(JYPPX_TensorRtRuntime* runtime, JYPPX_Boolean* out_exists)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_exists, "out_exists");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_exists = JYPPX_FALSE;
#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Runtime handle does not carry a TensorRT runtime payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    nvinfer1::IPluginRegistry* registry = nullptr;
    try
    {
        status = jyppx_trt8_get_runtime_plugin_registry_with_seh_guard(
            runtime_payload,
            &registry,
            "TensorRT 8 runtime plugin registry existence query");
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }
    }
    catch (const std::exception& exception)
    {
        return jyppx_trt8_report_plugin_inventory_exception("TensorRT 8 runtime plugin registry existence query", exception);
    }
    catch (...)
    {
        return jyppx_trt8_report_plugin_inventory_unknown_exception("TensorRT 8 runtime plugin registry existence query");
    }

    *out_exists = registry != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 runtime plugin registry existence query");
#endif
}

JYPPX_StatusCode jyppx_trt8_runtime_plugin_registry_get_creator_count(JYPPX_TensorRtRuntime* runtime, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt8_get_runtime_plugin_registry(runtime, &registry, "TensorRT 8 runtime plugin registry creator count query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    nvinfer1::IPluginCreator* const* creators = nullptr;
    return jyppx_trt8_get_inventory_registry_creators(
        registry,
        &creators,
        out_count,
        "TensorRT 8 runtime plugin registry creator count query");
#else
    (void)runtime;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 runtime plugin registry creator count query");
#endif
}

JYPPX_StatusCode jyppx_trt8_runtime_plugin_registry_has_error_recorder(JYPPX_TensorRtRuntime* runtime, JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt8_get_runtime_plugin_registry(runtime, &registry, "TensorRT 8 runtime plugin registry error recorder query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return jyppx_trt8_get_plugin_registry_has_error_recorder_with_seh_guard(
            registry,
            out_has_recorder,
            "TensorRT 8 runtime plugin registry error recorder query");
    }
    catch (const std::exception& exception)
    {
        *out_has_recorder = JYPPX_FALSE;
        return jyppx_trt8_report_plugin_inventory_exception("TensorRT 8 runtime plugin registry error recorder query", exception);
    }
    catch (...)
    {
        *out_has_recorder = JYPPX_FALSE;
        return jyppx_trt8_report_plugin_inventory_unknown_exception("TensorRT 8 runtime plugin registry error recorder query");
    }
#else
    (void)runtime;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 runtime plugin registry error recorder query");
#endif
}

JYPPX_StatusCode jyppx_trt8_runtime_plugin_registry_is_parent_search_enabled(JYPPX_TensorRtRuntime* runtime, JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_enabled = JYPPX_FALSE;
#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt8_get_runtime_plugin_registry(runtime, &registry, "TensorRT 8 runtime plugin registry parent search query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return jyppx_trt8_get_plugin_registry_parent_search_enabled_with_seh_guard(
            registry,
            out_enabled,
            "TensorRT 8 runtime plugin registry parent search query");
    }
    catch (const std::exception& exception)
    {
        *out_enabled = JYPPX_FALSE;
        return jyppx_trt8_report_plugin_inventory_exception("TensorRT 8 runtime plugin registry parent search query", exception);
    }
    catch (...)
    {
        *out_enabled = JYPPX_FALSE;
        return jyppx_trt8_report_plugin_inventory_unknown_exception("TensorRT 8 runtime plugin registry parent search query");
    }
#else
    (void)runtime;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 runtime plugin registry parent search query");
#endif
}

JYPPX_StatusCode jyppx_trt8_runtime_plugin_creator_get_name(
    JYPPX_TensorRtRuntime* runtime,
    int32_t creator_index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_runtime_plugin_creator(runtime, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const char* name = "";
    status = jyppx_trt8_get_creator_name(creator, &name, "TensorRT 8 runtime plugin creator name query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(name, output_buffer, output_buffer_size, out_required_size);
#else
    (void)runtime;
    (void)creator_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 runtime plugin creator name query");
#endif
}

JYPPX_StatusCode jyppx_trt8_runtime_plugin_creator_get_version(
    JYPPX_TensorRtRuntime* runtime,
    int32_t creator_index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_runtime_plugin_creator(runtime, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const char* version = "";
    status = jyppx_trt8_get_creator_version(creator, &version, "TensorRT 8 runtime plugin creator version query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(version, output_buffer, output_buffer_size, out_required_size);
#else
    (void)runtime;
    (void)creator_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 runtime plugin creator version query");
#endif
}

JYPPX_StatusCode jyppx_trt8_runtime_plugin_creator_get_namespace(
    JYPPX_TensorRtRuntime* runtime,
    int32_t creator_index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_runtime_plugin_creator(runtime, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const char* plugin_namespace = "";
    status = jyppx_trt8_get_creator_namespace(creator, &plugin_namespace, "TensorRT 8 runtime plugin creator namespace query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(plugin_namespace, output_buffer, output_buffer_size, out_required_size);
#else
    (void)runtime;
    (void)creator_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 runtime plugin creator namespace query");
#endif
}

JYPPX_StatusCode jyppx_trt8_runtime_plugin_creator_get_field_count(
    JYPPX_TensorRtRuntime* runtime,
    int32_t creator_index,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_runtime_plugin_creator(runtime, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = jyppx_trt8_get_creator_field_collection(creator, &fields, "TensorRT 8 runtime plugin creator field count query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = fields != nullptr ? fields->nbFields : 0;
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    (void)creator_index;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 runtime plugin creator field count query");
#endif
}

JYPPX_StatusCode jyppx_trt8_runtime_plugin_creator_get_field_name(
    JYPPX_TensorRtRuntime* runtime,
    int32_t creator_index,
    int32_t field_index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_runtime_plugin_creator(runtime, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = jyppx_trt8_get_creator_field_collection(creator, &fields, "TensorRT 8 runtime plugin creator field name query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "runtime_plugin_field");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx_trt8_validate_plugin_field_storage(fields, field_count, "TensorRT 8 runtime plugin creator field name query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const char* field_name = "";
    status = jyppx_trt8_get_plugin_field_name(fields, field_index, &field_name, "TensorRT 8 runtime plugin creator field name query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(field_name, output_buffer, output_buffer_size, out_required_size);
#else
    (void)runtime;
    (void)creator_index;
    (void)field_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 runtime plugin creator field name query");
#endif
}

JYPPX_StatusCode jyppx_trt8_runtime_plugin_creator_get_field_metadata(
    JYPPX_TensorRtRuntime* runtime,
    int32_t creator_index,
    int32_t field_index,
    int32_t* out_field_type,
    int32_t* out_length,
    JYPPX_Boolean* out_has_data)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_field_type, "out_field_type");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_length, "out_length");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_has_data, "out_has_data");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_field_type = 0;
    *out_length = 0;
    *out_has_data = JYPPX_FALSE;
#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_runtime_plugin_creator(runtime, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = jyppx_trt8_get_creator_field_collection(creator, &fields, "TensorRT 8 runtime plugin creator field metadata query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "runtime_plugin_field");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx_trt8_validate_plugin_field_storage(fields, field_count, "TensorRT 8 runtime plugin creator field metadata query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return jyppx_trt8_get_plugin_field_metadata(
        fields,
        field_index,
        out_field_type,
        out_length,
        out_has_data,
        "TensorRT 8 runtime plugin creator field metadata query");
#else
    (void)runtime;
    (void)creator_index;
    (void)field_index;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 runtime plugin creator field metadata query");
#endif
}

JYPPX_StatusCode jyppx_trt8_runtime_plugin_creator_lookup(
    JYPPX_TensorRtRuntime* runtime,
    const char* plugin_name,
    const char* plugin_version,
    const char* plugin_namespace,
    JYPPX_Boolean* out_found)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_found, "out_found");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_found = JYPPX_FALSE;
#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_runtime_lookup_creator(runtime, plugin_name, plugin_version, plugin_namespace, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_found = creator != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    (void)plugin_name;
    (void)plugin_version;
    (void)plugin_namespace;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 runtime plugin creator lookup query");
#endif
}

JYPPX_StatusCode jyppx_trt8_runtime_plugin_creator_lookup_get_field_count(
    JYPPX_TensorRtRuntime* runtime,
    const char* plugin_name,
    const char* plugin_version,
    const char* plugin_namespace,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_runtime_lookup_creator(runtime, plugin_name, plugin_version, plugin_namespace, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (creator == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 runtime plugin creator lookup did not find a matching creator.");
        return JYPPX_STATUS_NOT_FOUND;
    }

    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = jyppx_trt8_get_creator_field_collection(creator, &fields, "TensorRT 8 runtime plugin creator lookup field count query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = fields != nullptr ? fields->nbFields : 0;
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    (void)plugin_name;
    (void)plugin_version;
    (void)plugin_namespace;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 runtime plugin creator lookup field count query");
#endif
}

JYPPX_StatusCode jyppx_trt8_runtime_plugin_creator_lookup_get_field_name(
    JYPPX_TensorRtRuntime* runtime,
    const char* plugin_name,
    const char* plugin_version,
    const char* plugin_namespace,
    int32_t field_index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_runtime_lookup_creator(runtime, plugin_name, plugin_version, plugin_namespace, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (creator == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 runtime plugin creator lookup did not find a matching creator.");
        return JYPPX_STATUS_NOT_FOUND;
    }

    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = jyppx_trt8_get_creator_field_collection(creator, &fields, "TensorRT 8 runtime plugin creator lookup field name query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "runtime_plugin_lookup_field");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx_trt8_validate_plugin_field_storage(fields, field_count, "TensorRT 8 runtime plugin creator lookup field name query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const char* field_name = "";
    status = jyppx_trt8_get_plugin_field_name(fields, field_index, &field_name, "TensorRT 8 runtime plugin creator lookup field name query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(field_name, output_buffer, output_buffer_size, out_required_size);
#else
    (void)runtime;
    (void)plugin_name;
    (void)plugin_version;
    (void)plugin_namespace;
    (void)field_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 runtime plugin creator lookup field name query");
#endif
}

JYPPX_StatusCode jyppx_trt8_runtime_plugin_creator_lookup_get_field_metadata(
    JYPPX_TensorRtRuntime* runtime,
    const char* plugin_name,
    const char* plugin_version,
    const char* plugin_namespace,
    int32_t field_index,
    int32_t* out_field_type,
    int32_t* out_length,
    JYPPX_Boolean* out_has_data)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_field_type, "out_field_type");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_length, "out_length");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_has_data, "out_has_data");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_field_type = 0;
    *out_length = 0;
    *out_has_data = JYPPX_FALSE;
#if JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreator* creator = nullptr;
    status = jyppx_trt8_get_runtime_lookup_creator(runtime, plugin_name, plugin_version, plugin_namespace, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (creator == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT 8 runtime plugin creator lookup did not find a matching creator.");
        return JYPPX_STATUS_NOT_FOUND;
    }

    const nvinfer1::PluginFieldCollection* fields = nullptr;
    status = jyppx_trt8_get_creator_field_collection(creator, &fields, "TensorRT 8 runtime plugin creator lookup field metadata query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "runtime_plugin_lookup_field");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx_trt8_validate_plugin_field_storage(fields, field_count, "TensorRT 8 runtime plugin creator lookup field metadata query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return jyppx_trt8_get_plugin_field_metadata(
        fields,
        field_index,
        out_field_type,
        out_length,
        out_has_data,
        "TensorRT 8 runtime plugin creator lookup field metadata query");
#else
    (void)runtime;
    (void)plugin_name;
    (void)plugin_version;
    (void)plugin_namespace;
    (void)field_index;
    return jyppx_trt8_plugin_registry_report_unavailable("TensorRT 8 runtime plugin creator lookup field metadata query");
#endif
}
#undef JYPPX_TRT8_PLUGIN_REGISTRY_HAS_INVENTORY
