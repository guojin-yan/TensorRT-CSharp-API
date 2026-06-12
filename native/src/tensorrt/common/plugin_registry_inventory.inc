#ifndef JYPPX_TRT_PLUGIN_PREFIX
#error "JYPPX_TRT_PLUGIN_PREFIX must be defined before including plugin_registry_inventory.inc"
#endif

#define JYPPX_TRT_PLUGIN_CONCAT2(prefix, name) prefix##name
#define JYPPX_TRT_PLUGIN_CONCAT(prefix, name) JYPPX_TRT_PLUGIN_CONCAT2(prefix, name)
#define JYPPX_TRT_PLUGIN_FN(name) JYPPX_TRT_PLUGIN_CONCAT(JYPPX_TRT_PLUGIN_PREFIX, name)

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
#define JYPPX_TRT_PLUGIN_REGISTRY_HAS_INVENTORY 1
#else
#define JYPPX_TRT_PLUGIN_REGISTRY_HAS_INVENTORY 0
#endif

#if !JYPPX_TRT_PLUGIN_REGISTRY_HAS_INVENTORY
static JYPPX_StatusCode jyppx_trt_plugin_registry_report_unavailable(const char* feature_name)
{
#if JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, feature_name);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, feature_name);
#endif
}
#endif

#if JYPPX_TRT_PLUGIN_REGISTRY_HAS_INVENTORY
JYPPX_StatusCode jyppx_trt_get_builder_plugin_registry(JYPPX_TensorRtBuilder* builder, nvinfer1::IPluginRegistry** out_registry)
{
    auto status = jyppx::tensorrt::validate_handle(builder, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER, "builder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* builder_payload = get_payload<nvinfer1::IBuilder>(builder);
    if (builder_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_registry = &builder_payload->getPluginRegistry();
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt_get_plugin_creator(
    JYPPX_TensorRtBuilder* builder,
    int32_t creator_index,
    nvinfer1::IPluginCreatorInterface** out_creator)
{
    nvinfer1::IPluginRegistry* registry = nullptr;
    auto status = jyppx_trt_get_builder_plugin_registry(builder, &registry);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    int32_t creator_count = 0;
    auto* creators = registry->getAllCreators(&creator_count);
    status = validate_index(creator_index, creator_count, "plugin_creator");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (creators == nullptr || creators[creator_index] == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT plugin registry returned a null creator pointer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_creator = creators[creator_index];
    return JYPPX_STATUS_OK;
}

const nvinfer1::PluginFieldCollection* jyppx_trt_get_creator_field_collection(nvinfer1::IPluginCreatorInterface* creator)
{
    if (creator == nullptr)
    {
        return nullptr;
    }

    if (auto* legacy_creator = dynamic_cast<nvinfer1::IPluginCreator*>(creator))
    {
        return legacy_creator->getFieldNames();
    }

    if (auto* v3_creator = dynamic_cast<nvinfer1::IPluginCreatorV3One*>(creator))
    {
        return v3_creator->getFieldNames();
    }

    return nullptr;
}

const char* jyppx_trt_get_creator_name(nvinfer1::IPluginCreatorInterface* creator)
{
    if (auto* legacy_creator = dynamic_cast<nvinfer1::IPluginCreator*>(creator))
    {
        return legacy_creator->getPluginName();
    }

    if (auto* v3_creator = dynamic_cast<nvinfer1::IPluginCreatorV3One*>(creator))
    {
        return v3_creator->getPluginName();
    }

    return "";
}

const char* jyppx_trt_get_creator_version(nvinfer1::IPluginCreatorInterface* creator)
{
    if (auto* legacy_creator = dynamic_cast<nvinfer1::IPluginCreator*>(creator))
    {
        return legacy_creator->getPluginVersion();
    }

    if (auto* v3_creator = dynamic_cast<nvinfer1::IPluginCreatorV3One*>(creator))
    {
        return v3_creator->getPluginVersion();
    }

    return "";
}

const char* jyppx_trt_get_creator_namespace(nvinfer1::IPluginCreatorInterface* creator)
{
    if (auto* legacy_creator = dynamic_cast<nvinfer1::IPluginCreator*>(creator))
    {
        return legacy_creator->getPluginNamespace();
    }

    if (auto* v3_creator = dynamic_cast<nvinfer1::IPluginCreatorV3One*>(creator))
    {
        return v3_creator->getPluginNamespace();
    }

    return "";
}
#endif

JYPPX_StatusCode JYPPX_TRT_PLUGIN_FN(builder_plugin_registry_get_creator_count)(JYPPX_TensorRtBuilder* builder, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
#if JYPPX_TRT_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt_get_builder_plugin_registry(builder, &registry);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    int32_t count = 0;
    registry->getAllCreators(&count);
    *out_count = count;
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    return jyppx_trt_plugin_registry_report_unavailable("plugin registry creator count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_FN(builder_plugin_registry_has_error_recorder)(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_TRT_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt_get_builder_plugin_registry(builder, &registry);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_recorder = registry->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    return jyppx_trt_plugin_registry_report_unavailable("plugin registry error recorder query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_FN(builder_plugin_registry_is_parent_search_enabled)(JYPPX_TensorRtBuilder* builder, JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_enabled = JYPPX_FALSE;
#if JYPPX_TRT_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt_get_builder_plugin_registry(builder, &registry);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_enabled = registry->isParentSearchEnabled() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    return jyppx_trt_plugin_registry_report_unavailable("plugin registry parent search query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_FN(builder_plugin_creator_get_name)(JYPPX_TensorRtBuilder* builder, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_plugin_creator(builder, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(jyppx_trt_get_creator_name(creator), output_buffer, output_buffer_size, out_required_size);
#else
    (void)builder;
    (void)creator_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt_plugin_registry_report_unavailable("plugin creator name query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_FN(builder_plugin_creator_get_version)(JYPPX_TensorRtBuilder* builder, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_plugin_creator(builder, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(jyppx_trt_get_creator_version(creator), output_buffer, output_buffer_size, out_required_size);
#else
    (void)builder;
    (void)creator_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt_plugin_registry_report_unavailable("plugin creator version query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_FN(builder_plugin_creator_get_namespace)(JYPPX_TensorRtBuilder* builder, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_plugin_creator(builder, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(jyppx_trt_get_creator_namespace(creator), output_buffer, output_buffer_size, out_required_size);
#else
    (void)builder;
    (void)creator_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt_plugin_registry_report_unavailable("plugin creator namespace query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_FN(builder_plugin_creator_get_interface_info)(
    JYPPX_TensorRtBuilder* builder,
    int32_t creator_index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_major, "out_major");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::tensorrt::validate_output_pointer(out_minor, "out_minor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_major = 0;
    *out_minor = 0;
#if JYPPX_TRT_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_plugin_creator(builder, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const nvinfer1::InterfaceInfo info = creator->getInterfaceInfo();
    *out_major = info.major;
    *out_minor = info.minor;
    return copy_string_to_buffer(info.kind, output_buffer, output_buffer_size, out_required_size);
#else
    (void)builder;
    (void)creator_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt_plugin_registry_report_unavailable("plugin creator interface info query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_FN(builder_plugin_creator_get_field_count)(JYPPX_TensorRtBuilder* builder, int32_t creator_index, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
#if JYPPX_TRT_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_plugin_creator(builder, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const auto* fields = jyppx_trt_get_creator_field_collection(creator);
    *out_count = fields != nullptr ? fields->nbFields : 0;
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    (void)creator_index;
    return jyppx_trt_plugin_registry_report_unavailable("plugin creator field count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_FN(builder_plugin_creator_get_field_name)(
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

#if JYPPX_TRT_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_plugin_creator(builder, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const auto* fields = jyppx_trt_get_creator_field_collection(creator);
    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "plugin_field");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(fields->fields[field_index].name, output_buffer, output_buffer_size, out_required_size);
#else
    (void)builder;
    (void)creator_index;
    (void)field_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt_plugin_registry_report_unavailable("plugin creator field name query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_FN(builder_plugin_creator_get_field_metadata)(
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
#if JYPPX_TRT_PLUGIN_REGISTRY_HAS_INVENTORY
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_plugin_creator(builder, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const auto* fields = jyppx_trt_get_creator_field_collection(creator);
    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "plugin_field");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const auto& field = fields->fields[field_index];
    *out_field_type = static_cast<int32_t>(field.type);
    *out_length = field.length;
    *out_has_data = field.data != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)builder;
    (void)creator_index;
    (void)field_index;
    return jyppx_trt_plugin_registry_report_unavailable("plugin creator field metadata query");
#endif
}

#undef JYPPX_TRT_PLUGIN_REGISTRY_HAS_INVENTORY
#undef JYPPX_TRT_PLUGIN_FN
#undef JYPPX_TRT_PLUGIN_CONCAT
#undef JYPPX_TRT_PLUGIN_CONCAT2
