#ifndef JYPPX_TRT_PLUGIN_V3_LAYER_PREFIX
#error "JYPPX_TRT_PLUGIN_V3_LAYER_PREFIX must be defined before including plugin_v3_layer_metadata_snapshot.inc"
#endif

#define JYPPX_TRT_PLUGIN_V3_LAYER_CONCAT2(prefix, name) prefix##name
#define JYPPX_TRT_PLUGIN_V3_LAYER_CONCAT(prefix, name) JYPPX_TRT_PLUGIN_V3_LAYER_CONCAT2(prefix, name)
#define JYPPX_TRT_PLUGIN_V3_LAYER_FN(name) JYPPX_TRT_PLUGIN_V3_LAYER_CONCAT(JYPPX_TRT_PLUGIN_V3_LAYER_PREFIX, name)

namespace
{
enum class PluginV3LayerTarget
{
    Plugin,
    Core,
    Build,
    Runtime
};

enum class PluginV3CoreStringQuery
{
    Name,
    Version,
    Namespace
};

enum class PluginV3BuildStringQuery
{
    TimingCacheId,
    Metadata
};

enum class PluginV3BuildScalarQuery
{
    OutputCount,
    TacticCount,
    FormatCombinationLimit
};

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
JYPPX_StatusCode get_plugin_v3_layer(
    JYPPX_TensorRtLayer* layer,
    nvinfer1::IPluginV3Layer** out_plugin_layer,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(
        layer,
        kLine,
        JYPPX_TENSORRT_OBJECT_KIND_LAYER,
        "layer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* layer_payload = get_layer_payload(layer);
    if (layer_payload == nullptr)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "Layer handle does not carry the expected TensorRT layer payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    if (layer_payload->getType() != nvinfer1::LayerType::kPLUGIN_V3)
    {
        std::ostringstream builder;
        builder << feature_name << " requires a TensorRT PluginV3 layer.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_plugin_layer = static_cast<nvinfer1::IPluginV3Layer*>(layer_payload);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode resolve_plugin_v3_target_with_seh_guard(
    nvinfer1::IPluginV3Layer* plugin_layer,
    const PluginV3LayerTarget target,
    nvinfer1::IVersionedInterface** out_target,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        nvinfer1::IPluginV3& plugin = plugin_layer->getPlugin();
        if (target == PluginV3LayerTarget::Plugin)
        {
            *out_target = &plugin;
            return JYPPX_STATUS_OK;
        }

        nvinfer1::PluginCapabilityType capability_type = nvinfer1::PluginCapabilityType::kCORE;
        switch (target)
        {
        case PluginV3LayerTarget::Core:
            capability_type = nvinfer1::PluginCapabilityType::kCORE;
            break;
        case PluginV3LayerTarget::Build:
            capability_type = nvinfer1::PluginCapabilityType::kBUILD;
            break;
        case PluginV3LayerTarget::Runtime:
            capability_type = nvinfer1::PluginCapabilityType::kRUNTIME;
            break;
        case PluginV3LayerTarget::Plugin:
            break;
        }

        *out_target = plugin.getCapabilityInterface(capability_type);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_target = nullptr;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    nvinfer1::IPluginV3& plugin = plugin_layer->getPlugin();
    if (target == PluginV3LayerTarget::Plugin)
    {
        *out_target = &plugin;
        return JYPPX_STATUS_OK;
    }

    nvinfer1::PluginCapabilityType capability_type = nvinfer1::PluginCapabilityType::kCORE;
    switch (target)
    {
    case PluginV3LayerTarget::Core:
        capability_type = nvinfer1::PluginCapabilityType::kCORE;
        break;
    case PluginV3LayerTarget::Build:
        capability_type = nvinfer1::PluginCapabilityType::kBUILD;
        break;
    case PluginV3LayerTarget::Runtime:
        capability_type = nvinfer1::PluginCapabilityType::kRUNTIME;
        break;
    case PluginV3LayerTarget::Plugin:
        break;
    }

    *out_target = plugin.getCapabilityInterface(capability_type);
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode resolve_plugin_v3_target(
    nvinfer1::IPluginV3Layer* plugin_layer,
    const PluginV3LayerTarget target,
    nvinfer1::IVersionedInterface** out_target,
    const char* feature_name)
{
    try
    {
        auto status = resolve_plugin_v3_target_with_seh_guard(plugin_layer, target, out_target, feature_name);
        if (status != JYPPX_STATUS_OK || *out_target != nullptr)
        {
            return status;
        }

        std::ostringstream builder;
        builder << feature_name << " is unavailable because the requested PluginV3 capability is not exposed.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_NOT_FOUND;
    }
    catch (const std::exception& exception)
    {
        *out_target = nullptr;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        *out_target = nullptr;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}

JYPPX_StatusCode query_plugin_v3_interface_info_with_seh_guard(
    const nvinfer1::IVersionedInterface* target,
    nvinfer1::InterfaceInfo* out_info,
    int32_t* out_api_language,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_info = target->getInterfaceInfo();
        *out_api_language = static_cast<int32_t>(target->getAPILanguage());
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_info = nvinfer1::InterfaceInfo{"", 0, 0};
        *out_api_language = -1;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_info = target->getInterfaceInfo();
    *out_api_language = static_cast<int32_t>(target->getAPILanguage());
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode query_plugin_v3_interface_info(
    const nvinfer1::IVersionedInterface* target,
    nvinfer1::InterfaceInfo* out_info,
    int32_t* out_api_language,
    const char* feature_name)
{
    try
    {
        return query_plugin_v3_interface_info_with_seh_guard(target, out_info, out_api_language, feature_name);
    }
    catch (const std::exception& exception)
    {
        *out_info = nvinfer1::InterfaceInfo{"", 0, 0};
        *out_api_language = -1;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        *out_info = nvinfer1::InterfaceInfo{"", 0, 0};
        *out_api_language = -1;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}

JYPPX_StatusCode query_plugin_v3_core_string_with_seh_guard(
    nvinfer1::IPluginV3OneCore* core,
    const PluginV3CoreStringQuery query,
    const char** out_value,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        switch (query)
        {
        case PluginV3CoreStringQuery::Name:
            *out_value = core->getPluginName();
            break;
        case PluginV3CoreStringQuery::Version:
            *out_value = core->getPluginVersion();
            break;
        case PluginV3CoreStringQuery::Namespace:
            *out_value = core->getPluginNamespace();
            break;
        }
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_value = "";
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    switch (query)
    {
    case PluginV3CoreStringQuery::Name:
        *out_value = core->getPluginName();
        break;
    case PluginV3CoreStringQuery::Version:
        *out_value = core->getPluginVersion();
        break;
    case PluginV3CoreStringQuery::Namespace:
        *out_value = core->getPluginNamespace();
        break;
    }
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode query_plugin_v3_core_string(
    nvinfer1::IPluginV3OneCore* core,
    const PluginV3CoreStringQuery query,
    const char** out_value,
    const char* feature_name)
{
    try
    {
        return query_plugin_v3_core_string_with_seh_guard(core, query, out_value, feature_name);
    }
    catch (const std::exception& exception)
    {
        *out_value = "";
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        *out_value = "";
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}

JYPPX_StatusCode query_plugin_v3_build_string_with_seh_guard(
    nvinfer1::IPluginV3OneBuild* build,
    const PluginV3BuildStringQuery query,
    const char** out_value,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_value = query == PluginV3BuildStringQuery::TimingCacheId
            ? build->getTimingCacheID()
            : build->getMetadataString();
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_value = "";
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_value = query == PluginV3BuildStringQuery::TimingCacheId
        ? build->getTimingCacheID()
        : build->getMetadataString();
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode query_plugin_v3_build_string(
    nvinfer1::IPluginV3OneBuild* build,
    const PluginV3BuildStringQuery query,
    const char** out_value,
    const char* feature_name)
{
    try
    {
        return query_plugin_v3_build_string_with_seh_guard(build, query, out_value, feature_name);
    }
    catch (const std::exception& exception)
    {
        *out_value = "";
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        *out_value = "";
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}

JYPPX_StatusCode query_plugin_v3_build_scalar_with_seh_guard(
    nvinfer1::IPluginV3OneBuild* build,
    const PluginV3BuildScalarQuery query,
    int32_t* out_value,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        switch (query)
        {
        case PluginV3BuildScalarQuery::OutputCount:
            *out_value = build->getNbOutputs();
            break;
        case PluginV3BuildScalarQuery::TacticCount:
            *out_value = build->getNbTactics();
            break;
        case PluginV3BuildScalarQuery::FormatCombinationLimit:
            *out_value = build->getFormatCombinationLimit();
            break;
        }
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_value = 0;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    switch (query)
    {
    case PluginV3BuildScalarQuery::OutputCount:
        *out_value = build->getNbOutputs();
        break;
    case PluginV3BuildScalarQuery::TacticCount:
        *out_value = build->getNbTactics();
        break;
    case PluginV3BuildScalarQuery::FormatCombinationLimit:
        *out_value = build->getFormatCombinationLimit();
        break;
    }
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode query_plugin_v3_build_scalar(
    nvinfer1::IPluginV3OneBuild* build,
    const PluginV3BuildScalarQuery query,
    int32_t* out_value,
    const char* feature_name)
{
    try
    {
        return query_plugin_v3_build_scalar_with_seh_guard(build, query, out_value, feature_name);
    }
    catch (const std::exception& exception)
    {
        *out_value = 0;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        *out_value = 0;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}

JYPPX_StatusCode query_plugin_v3_capability_presence_with_seh_guard(
    nvinfer1::IPluginV3Layer* plugin_layer,
    JYPPX_Boolean* out_has_core,
    JYPPX_Boolean* out_has_build,
    JYPPX_Boolean* out_has_runtime,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        nvinfer1::IPluginV3& plugin = plugin_layer->getPlugin();
        *out_has_core = plugin.getCapabilityInterface(nvinfer1::PluginCapabilityType::kCORE) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        *out_has_build = plugin.getCapabilityInterface(nvinfer1::PluginCapabilityType::kBUILD) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        *out_has_runtime = plugin.getCapabilityInterface(nvinfer1::PluginCapabilityType::kRUNTIME) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_core = JYPPX_FALSE;
        *out_has_build = JYPPX_FALSE;
        *out_has_runtime = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    nvinfer1::IPluginV3& plugin = plugin_layer->getPlugin();
    *out_has_core = plugin.getCapabilityInterface(nvinfer1::PluginCapabilityType::kCORE) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    *out_has_build = plugin.getCapabilityInterface(nvinfer1::PluginCapabilityType::kBUILD) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    *out_has_runtime = plugin.getCapabilityInterface(nvinfer1::PluginCapabilityType::kRUNTIME) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode query_plugin_v3_capability_presence(
    nvinfer1::IPluginV3Layer* plugin_layer,
    JYPPX_Boolean* out_has_core,
    JYPPX_Boolean* out_has_build,
    JYPPX_Boolean* out_has_runtime,
    const char* feature_name)
{
    try
    {
        return query_plugin_v3_capability_presence_with_seh_guard(
            plugin_layer,
            out_has_core,
            out_has_build,
            out_has_runtime,
            feature_name);
    }
    catch (const std::exception& exception)
    {
        *out_has_core = JYPPX_FALSE;
        *out_has_build = JYPPX_FALSE;
        *out_has_runtime = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        *out_has_core = JYPPX_FALSE;
        *out_has_build = JYPPX_FALSE;
        *out_has_runtime = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}
#endif

JYPPX_StatusCode plugin_v3_layer_copy_interface_info(
    JYPPX_TensorRtLayer* layer,
    const PluginV3LayerTarget target,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor,
    int32_t* out_api_language,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_major, "out_major");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_minor, "out_minor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_api_language, "out_api_language");
    if (status != JYPPX_STATUS_OK) { return status; }

    *out_required_size = 0;
    *out_major = 0;
    *out_minor = 0;
    *out_api_language = -1;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::IPluginV3Layer* plugin_layer = nullptr;
    status = get_plugin_v3_layer(layer, &plugin_layer, feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::IVersionedInterface* target_interface = nullptr;
    status = resolve_plugin_v3_target(plugin_layer, target, &target_interface, feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::InterfaceInfo info{"", 0, 0};
    status = query_plugin_v3_interface_info(target_interface, &info, out_api_language, feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }

    return copy_interface_info_to_buffer(
        info,
        output_buffer,
        output_buffer_size,
        out_required_size,
        out_major,
        out_minor);
#else
    (void)layer;
    (void)target;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, feature_name);
#endif
}

JYPPX_StatusCode plugin_v3_layer_copy_core_string(
    JYPPX_TensorRtLayer* layer,
    const PluginV3CoreStringQuery query,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::IPluginV3Layer* plugin_layer = nullptr;
    status = get_plugin_v3_layer(layer, &plugin_layer, feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::IVersionedInterface* target = nullptr;
    status = resolve_plugin_v3_target(plugin_layer, PluginV3LayerTarget::Core, &target, feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }

    const char* value = "";
    status = query_plugin_v3_core_string(
        static_cast<nvinfer1::IPluginV3OneCore*>(target),
        query,
        &value,
        feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(value == nullptr ? "" : value, output_buffer, output_buffer_size, out_required_size);
#else
    (void)layer;
    (void)query;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, feature_name);
#endif
}

JYPPX_StatusCode plugin_v3_layer_copy_build_string(
    JYPPX_TensorRtLayer* layer,
    const PluginV3BuildStringQuery query,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::IPluginV3Layer* plugin_layer = nullptr;
    status = get_plugin_v3_layer(layer, &plugin_layer, feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::IVersionedInterface* target = nullptr;
    status = resolve_plugin_v3_target(plugin_layer, PluginV3LayerTarget::Build, &target, feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }

    const char* value = "";
    status = query_plugin_v3_build_string(
        static_cast<nvinfer1::IPluginV3OneBuild*>(target),
        query,
        &value,
        feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(value == nullptr ? "" : value, output_buffer, output_buffer_size, out_required_size);
#else
    (void)layer;
    (void)query;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, feature_name);
#endif
}

JYPPX_StatusCode plugin_v3_layer_copy_build_scalar(
    JYPPX_TensorRtLayer* layer,
    const PluginV3BuildScalarQuery query,
    int32_t* out_value,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_value, "out_value");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_value = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::IPluginV3Layer* plugin_layer = nullptr;
    status = get_plugin_v3_layer(layer, &plugin_layer, feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::IVersionedInterface* target = nullptr;
    status = resolve_plugin_v3_target(plugin_layer, PluginV3LayerTarget::Build, &target, feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }

    return query_plugin_v3_build_scalar(
        static_cast<nvinfer1::IPluginV3OneBuild*>(target),
        query,
        out_value,
        feature_name);
#else
    (void)layer;
    (void)query;
    return jyppx::tensorrt::report_vendor_missing(kLine, feature_name);
#endif
}
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V3_LAYER_FN(plugin_v3_layer_get_plugin_interface_info)(
    JYPPX_TensorRtLayer* layer,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor,
    int32_t* out_api_language)
{
    return plugin_v3_layer_copy_interface_info(
        layer, PluginV3LayerTarget::Plugin, output_buffer, output_buffer_size, out_required_size,
        out_major, out_minor, out_api_language, "PluginV3 layer plugin interface info query");
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V3_LAYER_FN(plugin_v3_layer_get_capability_presence)(
    JYPPX_TensorRtLayer* layer,
    JYPPX_Boolean* out_has_core,
    JYPPX_Boolean* out_has_build,
    JYPPX_Boolean* out_has_runtime)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_core, "out_has_core");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_has_build, "out_has_build");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_has_runtime, "out_has_runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_core = JYPPX_FALSE;
    *out_has_build = JYPPX_FALSE;
    *out_has_runtime = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::IPluginV3Layer* plugin_layer = nullptr;
    status = get_plugin_v3_layer(layer, &plugin_layer, "PluginV3 layer capability presence query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return query_plugin_v3_capability_presence(
        plugin_layer,
        out_has_core,
        out_has_build,
        out_has_runtime,
        "PluginV3 layer capability presence query");
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV3 layer capability presence query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V3_LAYER_FN(plugin_v3_layer_get_core_plugin_name)(
    JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return plugin_v3_layer_copy_core_string(
        layer, PluginV3CoreStringQuery::Name, output_buffer, output_buffer_size, out_required_size,
        "PluginV3 layer core plugin name query");
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V3_LAYER_FN(plugin_v3_layer_get_core_plugin_version)(
    JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return plugin_v3_layer_copy_core_string(
        layer, PluginV3CoreStringQuery::Version, output_buffer, output_buffer_size, out_required_size,
        "PluginV3 layer core plugin version query");
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V3_LAYER_FN(plugin_v3_layer_get_core_plugin_namespace)(
    JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return plugin_v3_layer_copy_core_string(
        layer, PluginV3CoreStringQuery::Namespace, output_buffer, output_buffer_size, out_required_size,
        "PluginV3 layer core plugin namespace query");
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V3_LAYER_FN(plugin_v3_layer_get_core_interface_info)(
    JYPPX_TensorRtLayer* layer,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor,
    int32_t* out_api_language)
{
    return plugin_v3_layer_copy_interface_info(
        layer, PluginV3LayerTarget::Core, output_buffer, output_buffer_size, out_required_size,
        out_major, out_minor, out_api_language, "PluginV3 layer core interface info query");
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V3_LAYER_FN(plugin_v3_layer_get_build_interface_info)(
    JYPPX_TensorRtLayer* layer,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor,
    int32_t* out_api_language)
{
    return plugin_v3_layer_copy_interface_info(
        layer, PluginV3LayerTarget::Build, output_buffer, output_buffer_size, out_required_size,
        out_major, out_minor, out_api_language, "PluginV3 layer build interface info query");
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V3_LAYER_FN(plugin_v3_layer_get_build_nb_outputs)(
    JYPPX_TensorRtLayer* layer, int32_t* out_count)
{
    return plugin_v3_layer_copy_build_scalar(
        layer, PluginV3BuildScalarQuery::OutputCount, out_count, "PluginV3 layer build output count query");
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V3_LAYER_FN(plugin_v3_layer_get_build_nb_tactics)(
    JYPPX_TensorRtLayer* layer, int32_t* out_count)
{
    return plugin_v3_layer_copy_build_scalar(
        layer, PluginV3BuildScalarQuery::TacticCount, out_count, "PluginV3 layer build tactic count query");
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V3_LAYER_FN(plugin_v3_layer_get_build_format_combination_limit)(
    JYPPX_TensorRtLayer* layer, int32_t* out_limit)
{
    return plugin_v3_layer_copy_build_scalar(
        layer, PluginV3BuildScalarQuery::FormatCombinationLimit, out_limit,
        "PluginV3 layer build format-combination limit query");
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V3_LAYER_FN(plugin_v3_layer_get_build_timing_cache_id)(
    JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return plugin_v3_layer_copy_build_string(
        layer, PluginV3BuildStringQuery::TimingCacheId, output_buffer, output_buffer_size, out_required_size,
        "PluginV3 layer build timing-cache ID query");
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V3_LAYER_FN(plugin_v3_layer_get_build_metadata_string)(
    JYPPX_TensorRtLayer* layer, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    return plugin_v3_layer_copy_build_string(
        layer, PluginV3BuildStringQuery::Metadata, output_buffer, output_buffer_size, out_required_size,
        "PluginV3 layer build metadata string query");
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V3_LAYER_FN(plugin_v3_layer_get_runtime_interface_info)(
    JYPPX_TensorRtLayer* layer,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor,
    int32_t* out_api_language)
{
    return plugin_v3_layer_copy_interface_info(
        layer, PluginV3LayerTarget::Runtime, output_buffer, output_buffer_size, out_required_size,
        out_major, out_minor, out_api_language, "PluginV3 layer runtime interface info query");
}

#undef JYPPX_TRT_PLUGIN_V3_LAYER_FN
#undef JYPPX_TRT_PLUGIN_V3_LAYER_CONCAT
#undef JYPPX_TRT_PLUGIN_V3_LAYER_CONCAT2
