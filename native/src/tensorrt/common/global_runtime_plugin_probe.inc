#ifndef JYPPX_TRT_GLOBAL_PLUGIN_PREFIX
#error "JYPPX_TRT_GLOBAL_PLUGIN_PREFIX must be defined before including global_runtime_plugin_probe.inc"
#endif

#define JYPPX_TRT_GLOBAL_PLUGIN_CONCAT2(prefix, name) prefix##name
#define JYPPX_TRT_GLOBAL_PLUGIN_CONCAT(prefix, name) JYPPX_TRT_GLOBAL_PLUGIN_CONCAT2(prefix, name)
#define JYPPX_TRT_GLOBAL_PLUGIN_FN(name) JYPPX_TRT_GLOBAL_PLUGIN_CONCAT(JYPPX_TRT_GLOBAL_PLUGIN_PREFIX, name)

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
#define JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE 1
#else
#define JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE 0
#endif

#if !JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
static JYPPX_StatusCode jyppx_trt_global_probe_report_unavailable(const char* feature_name)
{
#if JYPPX_HAS_TENSORRT
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, feature_name);
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, feature_name);
#endif
}
#endif

#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
JYPPX_StatusCode jyppx_trt_get_global_plugin_registry(nvinfer1::IPluginRegistry** out_registry)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_registry, "out_registry");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_registry = ::getPluginRegistry();
    if (*out_registry == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT global plugin registry is null.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt_get_global_plugin_creator(
    int32_t creator_index,
    nvinfer1::IPluginCreatorInterface** out_creator)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_creator, "out_creator");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_creator = nullptr;
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt_get_global_plugin_registry(&registry);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    int32_t creator_count = 0;
    auto* creators = registry->getAllCreators(&creator_count);
    status = validate_index(creator_index, creator_count, "global_plugin_creator");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (creators == nullptr || creators[creator_index] == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT global plugin registry returned a null creator pointer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_creator = creators[creator_index];
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt_validate_plugin_lookup_string(const char* value, const char* name)
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

JYPPX_StatusCode jyppx_trt_get_builder_capability_plugin_registry(int32_t capability, nvinfer1::IPluginRegistry** out_registry)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_registry, "out_registry");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_registry = nvinfer1::getBuilderPluginRegistry(static_cast<nvinfer1::EngineCapability>(capability));
    if (*out_registry == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT getBuilderPluginRegistry returned null for the requested engine capability.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt_get_builder_capability_plugin_creator(
    int32_t capability,
    int32_t creator_index,
    nvinfer1::IPluginCreatorInterface** out_creator)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_creator, "out_creator");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_creator = nullptr;
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt_get_builder_capability_plugin_registry(capability, &registry);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    int32_t creator_count = 0;
    auto* creators = registry->getAllCreators(&creator_count);
    status = validate_index(creator_index, creator_count, "builder_capability_plugin_creator");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (creators == nullptr || creators[creator_index] == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT builder capability plugin registry returned a null creator pointer.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_creator = creators[creator_index];
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_trt_get_builder_capability_lookup_creator(
    int32_t capability,
    const char* plugin_name,
    const char* plugin_version,
    const char* plugin_namespace,
    nvinfer1::IPluginCreatorInterface** out_creator)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_creator, "out_creator");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_creator = nullptr;
    status = jyppx_trt_validate_plugin_lookup_string(plugin_name, "plugin_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx_trt_validate_plugin_lookup_string(plugin_version, "plugin_version");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx_trt_validate_plugin_lookup_string(plugin_namespace, "plugin_namespace");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt_get_builder_capability_plugin_registry(capability, &registry);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_creator = registry->getCreator(plugin_name, plugin_version, plugin_namespace);
    return JYPPX_STATUS_OK;
}
#endif

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_get_infer_lib_version)(int32_t* out_version)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_version, "out_version");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_version = 0;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    *out_version = ::getInferLibVersion();
    return JYPPX_STATUS_OK;
#else
    return jyppx_trt_global_probe_report_unavailable("TensorRT infer library packed version query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_get_infer_lib_major_version)(int32_t* out_version)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_version, "out_version");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_version = 0;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    *out_version = ::getInferLibMajorVersion();
    return JYPPX_STATUS_OK;
#else
    return jyppx_trt_global_probe_report_unavailable("TensorRT infer library major version query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_get_infer_lib_minor_version)(int32_t* out_version)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_version, "out_version");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_version = 0;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    *out_version = ::getInferLibMinorVersion();
    return JYPPX_STATUS_OK;
#else
    return jyppx_trt_global_probe_report_unavailable("TensorRT infer library minor version query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_get_infer_lib_patch_version)(int32_t* out_version)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_version, "out_version");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_version = 0;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    *out_version = ::getInferLibPatchVersion();
    return JYPPX_STATUS_OK;
#else
    return jyppx_trt_global_probe_report_unavailable("TensorRT infer library patch version query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_get_infer_lib_build_version)(int32_t* out_version)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_version, "out_version");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_version = 0;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    *out_version = ::getInferLibBuildVersion();
    return JYPPX_STATUS_OK;
#else
    return jyppx_trt_global_probe_report_unavailable("TensorRT infer library build version query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_get_onnx_parser_version)(int32_t* out_version)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_version, "out_version");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_version = 0;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
#if JYPPX_HAS_TENSORRT_ONNXPARSER
    *out_version = ::getNvOnnxParserVersion();
    return JYPPX_STATUS_OK;
#else
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT ONNX parser vendor API is not available.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#endif
#else
    return jyppx_trt_global_probe_report_unavailable("TensorRT ONNX parser version query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_has_logger)(JYPPX_Boolean* out_has_logger)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_logger, "out_has_logger");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_logger = JYPPX_FALSE;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    *out_has_logger = ::getLogger() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx_trt_global_probe_report_unavailable("TensorRT global logger query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_plugin_registry_get_creator_count)(int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt_get_global_plugin_registry(&registry);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    registry->getAllCreators(out_count);
    return JYPPX_STATUS_OK;
#else
    return jyppx_trt_global_probe_report_unavailable("TensorRT global plugin registry creator count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_plugin_registry_get_recursive_creator_count)(int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt_get_global_plugin_registry(&registry);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    registry->getAllCreatorsRecursive(out_count);
    return JYPPX_STATUS_OK;
#else
    return jyppx_trt_global_probe_report_unavailable("TensorRT global plugin registry recursive creator count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_plugin_registry_has_error_recorder)(JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt_get_global_plugin_registry(&registry);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_recorder = registry->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx_trt_global_probe_report_unavailable("TensorRT global plugin registry error recorder query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_plugin_registry_is_parent_search_enabled)(JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_enabled = JYPPX_FALSE;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt_get_global_plugin_registry(&registry);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_enabled = registry->isParentSearchEnabled() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx_trt_global_probe_report_unavailable("TensorRT global plugin registry parent search query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_plugin_creator_get_name)(int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_global_plugin_creator(creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(jyppx_trt_get_creator_name(creator), output_buffer, output_buffer_size, out_required_size);
#else
    (void)creator_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt_global_probe_report_unavailable("TensorRT global plugin creator name query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_plugin_creator_get_version)(int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_global_plugin_creator(creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(jyppx_trt_get_creator_version(creator), output_buffer, output_buffer_size, out_required_size);
#else
    (void)creator_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt_global_probe_report_unavailable("TensorRT global plugin creator version query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_plugin_creator_get_namespace)(int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_global_plugin_creator(creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(jyppx_trt_get_creator_namespace(creator), output_buffer, output_buffer_size, out_required_size);
#else
    (void)creator_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt_global_probe_report_unavailable("TensorRT global plugin creator namespace query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_plugin_creator_get_interface_info)(
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
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_global_plugin_creator(creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const nvinfer1::InterfaceInfo info = creator->getInterfaceInfo();
    *out_major = info.major;
    *out_minor = info.minor;
    return copy_string_to_buffer(info.kind, output_buffer, output_buffer_size, out_required_size);
#else
    (void)creator_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt_global_probe_report_unavailable("TensorRT global plugin creator interface info query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_plugin_creator_get_field_count)(int32_t creator_index, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_global_plugin_creator(creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const auto* fields = jyppx_trt_get_creator_field_collection(creator);
    *out_count = fields != nullptr ? fields->nbFields : 0;
    return JYPPX_STATUS_OK;
#else
    (void)creator_index;
    return jyppx_trt_global_probe_report_unavailable("TensorRT global plugin creator field count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_plugin_creator_get_field_name)(
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

#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_global_plugin_creator(creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const auto* fields = jyppx_trt_get_creator_field_collection(creator);
    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "global_plugin_field");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(fields->fields[field_index].name, output_buffer, output_buffer_size, out_required_size);
#else
    (void)creator_index;
    (void)field_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt_global_probe_report_unavailable("TensorRT global plugin creator field name query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_plugin_creator_get_field_metadata)(
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
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_global_plugin_creator(creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const auto* fields = jyppx_trt_get_creator_field_collection(creator);
    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "global_plugin_field");
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
    (void)creator_index;
    (void)field_index;
    return jyppx_trt_global_probe_report_unavailable("TensorRT global plugin creator field metadata query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(global_plugin_creator_lookup)(
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
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    status = jyppx_trt_validate_plugin_lookup_string(plugin_name, "plugin_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx_trt_validate_plugin_lookup_string(plugin_version, "plugin_version");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx_trt_validate_plugin_lookup_string(plugin_namespace, "plugin_namespace");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt_get_global_plugin_registry(&registry);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_found = registry->getCreator(plugin_name, plugin_version, plugin_namespace) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)plugin_name;
    (void)plugin_version;
    (void)plugin_namespace;
    return jyppx_trt_global_probe_report_unavailable("TensorRT global plugin creator lookup");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_registry_exists)(int32_t capability, JYPPX_Boolean* out_exists)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_exists, "out_exists");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_exists = JYPPX_FALSE;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    *out_exists = nvinfer1::getBuilderPluginRegistry(static_cast<nvinfer1::EngineCapability>(capability)) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)capability;
    return jyppx_trt_global_probe_report_unavailable("TensorRT builder capability plugin registry existence query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_safe_plugin_registry_exists)(int32_t capability, JYPPX_Boolean* out_exists)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_exists, "out_exists");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_exists = JYPPX_FALSE;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    auto* registry = nvinfer1::getBuilderSafePluginRegistry(static_cast<nvinfer1::EngineCapability>(capability));
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    *out_exists = registry != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)capability;
    return jyppx_trt_global_probe_report_unavailable("TensorRT builder safe plugin registry existence query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_registry_get_creator_count)(int32_t capability, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt_get_builder_capability_plugin_registry(capability, &registry);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    registry->getAllCreators(out_count);
    return JYPPX_STATUS_OK;
#else
    (void)capability;
    return jyppx_trt_global_probe_report_unavailable("TensorRT builder capability plugin registry creator count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_registry_get_recursive_creator_count)(int32_t capability, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt_get_builder_capability_plugin_registry(capability, &registry);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    registry->getAllCreatorsRecursive(out_count);
    return JYPPX_STATUS_OK;
#else
    (void)capability;
    return jyppx_trt_global_probe_report_unavailable("TensorRT builder capability plugin registry recursive creator count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_registry_has_error_recorder)(int32_t capability, JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt_get_builder_capability_plugin_registry(capability, &registry);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_recorder = registry->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)capability;
    return jyppx_trt_global_probe_report_unavailable("TensorRT builder capability plugin registry error recorder query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_registry_is_parent_search_enabled)(int32_t capability, JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_enabled = JYPPX_FALSE;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginRegistry* registry = nullptr;
    status = jyppx_trt_get_builder_capability_plugin_registry(capability, &registry);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_enabled = registry->isParentSearchEnabled() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)capability;
    return jyppx_trt_global_probe_report_unavailable("TensorRT builder capability plugin registry parent search query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_creator_get_name)(int32_t capability, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_builder_capability_plugin_creator(capability, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(jyppx_trt_get_creator_name(creator), output_buffer, output_buffer_size, out_required_size);
#else
    (void)capability;
    (void)creator_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt_global_probe_report_unavailable("TensorRT builder capability plugin creator name query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_creator_get_version)(int32_t capability, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_builder_capability_plugin_creator(capability, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(jyppx_trt_get_creator_version(creator), output_buffer, output_buffer_size, out_required_size);
#else
    (void)capability;
    (void)creator_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt_global_probe_report_unavailable("TensorRT builder capability plugin creator version query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_creator_get_namespace)(int32_t capability, int32_t creator_index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_builder_capability_plugin_creator(capability, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(jyppx_trt_get_creator_namespace(creator), output_buffer, output_buffer_size, out_required_size);
#else
    (void)capability;
    (void)creator_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt_global_probe_report_unavailable("TensorRT builder capability plugin creator namespace query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_creator_get_interface_info)(
    int32_t capability,
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
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_builder_capability_plugin_creator(capability, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const nvinfer1::InterfaceInfo info = creator->getInterfaceInfo();
    *out_major = info.major;
    *out_minor = info.minor;
    return copy_string_to_buffer(info.kind, output_buffer, output_buffer_size, out_required_size);
#else
    (void)capability;
    (void)creator_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt_global_probe_report_unavailable("TensorRT builder capability plugin creator interface info query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_creator_get_field_count)(int32_t capability, int32_t creator_index, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_builder_capability_plugin_creator(capability, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const auto* fields = jyppx_trt_get_creator_field_collection(creator);
    *out_count = fields != nullptr ? fields->nbFields : 0;
    return JYPPX_STATUS_OK;
#else
    (void)capability;
    (void)creator_index;
    return jyppx_trt_global_probe_report_unavailable("TensorRT builder capability plugin creator field count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_creator_get_field_name)(
    int32_t capability,
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

#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_builder_capability_plugin_creator(capability, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const auto* fields = jyppx_trt_get_creator_field_collection(creator);
    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "builder_capability_plugin_field");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(fields->fields[field_index].name, output_buffer, output_buffer_size, out_required_size);
#else
    (void)capability;
    (void)creator_index;
    (void)field_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt_global_probe_report_unavailable("TensorRT builder capability plugin creator field name query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_creator_get_field_metadata)(
    int32_t capability,
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
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_builder_capability_plugin_creator(capability, creator_index, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const auto* fields = jyppx_trt_get_creator_field_collection(creator);
    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "builder_capability_plugin_field");
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
    (void)capability;
    (void)creator_index;
    (void)field_index;
    return jyppx_trt_global_probe_report_unavailable("TensorRT builder capability plugin creator field metadata query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_creator_lookup)(
    int32_t capability,
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
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_builder_capability_lookup_creator(capability, plugin_name, plugin_version, plugin_namespace, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_found = creator != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)capability;
    (void)plugin_name;
    (void)plugin_version;
    (void)plugin_namespace;
    return jyppx_trt_global_probe_report_unavailable("TensorRT builder capability plugin creator lookup");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_creator_lookup_get_interface_info)(
    int32_t capability,
    const char* plugin_name,
    const char* plugin_version,
    const char* plugin_namespace,
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
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_builder_capability_lookup_creator(capability, plugin_name, plugin_version, plugin_namespace, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (creator == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT builder capability plugin creator lookup did not find a matching creator.");
        return JYPPX_STATUS_NOT_FOUND;
    }

    const nvinfer1::InterfaceInfo info = creator->getInterfaceInfo();
    *out_major = info.major;
    *out_minor = info.minor;
    return copy_string_to_buffer(info.kind, output_buffer, output_buffer_size, out_required_size);
#else
    (void)capability;
    (void)plugin_name;
    (void)plugin_version;
    (void)plugin_namespace;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt_global_probe_report_unavailable("TensorRT builder capability plugin creator lookup interface info query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_creator_lookup_get_field_count)(
    int32_t capability,
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
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_builder_capability_lookup_creator(capability, plugin_name, plugin_version, plugin_namespace, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (creator == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    const auto* fields = jyppx_trt_get_creator_field_collection(creator);
    *out_count = fields != nullptr ? fields->nbFields : 0;
    return JYPPX_STATUS_OK;
#else
    (void)capability;
    (void)plugin_name;
    (void)plugin_version;
    (void)plugin_namespace;
    return jyppx_trt_global_probe_report_unavailable("TensorRT builder capability plugin creator lookup field count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_creator_lookup_get_field_name)(
    int32_t capability,
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

#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_builder_capability_lookup_creator(capability, plugin_name, plugin_version, plugin_namespace, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (creator == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT builder capability plugin creator lookup did not find a matching creator.");
        return JYPPX_STATUS_NOT_FOUND;
    }

    const auto* fields = jyppx_trt_get_creator_field_collection(creator);
    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "builder_capability_plugin_lookup_field");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(fields->fields[field_index].name, output_buffer, output_buffer_size, out_required_size);
#else
    (void)capability;
    (void)plugin_name;
    (void)plugin_version;
    (void)plugin_namespace;
    (void)field_index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx_trt_global_probe_report_unavailable("TensorRT builder capability plugin creator lookup field name query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_GLOBAL_PLUGIN_FN(builder_capability_plugin_creator_lookup_get_field_metadata)(
    int32_t capability,
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
#if JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
    nvinfer1::IPluginCreatorInterface* creator = nullptr;
    status = jyppx_trt_get_builder_capability_lookup_creator(capability, plugin_name, plugin_version, plugin_namespace, &creator);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (creator == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT builder capability plugin creator lookup did not find a matching creator.");
        return JYPPX_STATUS_NOT_FOUND;
    }

    const auto* fields = jyppx_trt_get_creator_field_collection(creator);
    const int32_t field_count = fields != nullptr ? fields->nbFields : 0;
    status = validate_index(field_index, field_count, "builder_capability_plugin_lookup_field");
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
    (void)capability;
    (void)plugin_name;
    (void)plugin_version;
    (void)plugin_namespace;
    (void)field_index;
    return jyppx_trt_global_probe_report_unavailable("TensorRT builder capability plugin creator lookup field metadata query");
#endif
}

#undef JYPPX_TRT_GLOBAL_HAS_READONLY_PROBE
#undef JYPPX_TRT_GLOBAL_PLUGIN_FN
#undef JYPPX_TRT_GLOBAL_PLUGIN_CONCAT
#undef JYPPX_TRT_GLOBAL_PLUGIN_CONCAT2
