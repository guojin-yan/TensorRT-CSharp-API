#ifndef JYPPX_TRT_PLUGIN_V2_LAYER_PREFIX
#error "JYPPX_TRT_PLUGIN_V2_LAYER_PREFIX must be defined before including plugin_v2_layer_metadata_snapshot.inc"
#endif

#define JYPPX_TRT_PLUGIN_V2_LAYER_CONCAT2(prefix, name) prefix##name
#define JYPPX_TRT_PLUGIN_V2_LAYER_CONCAT(prefix, name) JYPPX_TRT_PLUGIN_V2_LAYER_CONCAT2(prefix, name)
#define JYPPX_TRT_PLUGIN_V2_LAYER_FN(name) JYPPX_TRT_PLUGIN_V2_LAYER_CONCAT(JYPPX_TRT_PLUGIN_V2_LAYER_PREFIX, name)

namespace
{
enum class PluginV2LayerStringQuery
{
    Type,
    Version,
    Namespace
};

enum class PluginV2LayerScalarQuery
{
    SerializationSize,
    TensorRtVersion
};

#if JYPPX_HAS_TENSORRT
JYPPX_StatusCode get_plugin_v2_layer(
    JYPPX_TensorRtLayer* layer,
    nvinfer1::IPluginV2Layer** out_plugin_layer,
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

    if (layer_payload->getType() != nvinfer1::LayerType::kPLUGIN_V2)
    {
        std::ostringstream builder;
        builder << feature_name << " requires a TensorRT PluginV2 layer.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, builder.str().c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_plugin_layer = static_cast<nvinfer1::IPluginV2Layer*>(layer_payload);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode query_plugin_v2_layer_string_with_seh_guard(
    nvinfer1::IPluginV2Layer* plugin_layer,
    const PluginV2LayerStringQuery query,
    const char** out_value,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        nvinfer1::IPluginV2& plugin = plugin_layer->getPlugin();
        switch (query)
        {
        case PluginV2LayerStringQuery::Type:
            *out_value = plugin.getPluginType();
            break;
        case PluginV2LayerStringQuery::Version:
            *out_value = plugin.getPluginVersion();
            break;
        case PluginV2LayerStringQuery::Namespace:
            *out_value = plugin.getPluginNamespace();
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
    nvinfer1::IPluginV2& plugin = plugin_layer->getPlugin();
    switch (query)
    {
    case PluginV2LayerStringQuery::Type:
        *out_value = plugin.getPluginType();
        break;
    case PluginV2LayerStringQuery::Version:
        *out_value = plugin.getPluginVersion();
        break;
    case PluginV2LayerStringQuery::Namespace:
        *out_value = plugin.getPluginNamespace();
        break;
    }

    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode query_plugin_v2_layer_string(
    nvinfer1::IPluginV2Layer* plugin_layer,
    const PluginV2LayerStringQuery query,
    const char** out_value,
    const char* feature_name)
{
    try
    {
        return query_plugin_v2_layer_string_with_seh_guard(plugin_layer, query, out_value, feature_name);
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

JYPPX_StatusCode query_plugin_v2_layer_scalar_with_seh_guard(
    nvinfer1::IPluginV2Layer* plugin_layer,
    const PluginV2LayerScalarQuery query,
    size_t* out_serialization_size,
    int32_t* out_tensor_rt_version,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        nvinfer1::IPluginV2& plugin = plugin_layer->getPlugin();
        if (query == PluginV2LayerScalarQuery::SerializationSize)
        {
            *out_serialization_size = plugin.getSerializationSize();
        }
        else
        {
            *out_tensor_rt_version = plugin.getTensorRTVersion();
        }

        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_serialization_size = 0;
        *out_tensor_rt_version = 0;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    nvinfer1::IPluginV2& plugin = plugin_layer->getPlugin();
    if (query == PluginV2LayerScalarQuery::SerializationSize)
    {
        *out_serialization_size = plugin.getSerializationSize();
    }
    else
    {
        *out_tensor_rt_version = plugin.getTensorRTVersion();
    }

    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode query_plugin_v2_layer_scalar(
    nvinfer1::IPluginV2Layer* plugin_layer,
    const PluginV2LayerScalarQuery query,
    size_t* out_serialization_size,
    int32_t* out_tensor_rt_version,
    const char* feature_name)
{
    try
    {
        return query_plugin_v2_layer_scalar_with_seh_guard(
            plugin_layer,
            query,
            out_serialization_size,
            out_tensor_rt_version,
            feature_name);
    }
    catch (const std::exception& exception)
    {
        *out_serialization_size = 0;
        *out_tensor_rt_version = 0;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        *out_serialization_size = 0;
        *out_tensor_rt_version = 0;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}
#endif

JYPPX_StatusCode plugin_v2_layer_get_string(
    JYPPX_TensorRtLayer* layer,
    const PluginV2LayerStringQuery query,
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
#if JYPPX_HAS_TENSORRT
    nvinfer1::IPluginV2Layer* plugin_layer = nullptr;
    status = get_plugin_v2_layer(layer, &plugin_layer, feature_name);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const char* value = "";
    status = query_plugin_v2_layer_string(plugin_layer, query, &value, feature_name);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(value, output_buffer, output_buffer_size, out_required_size);
#else
    (void)layer;
    (void)query;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, feature_name);
#endif
}
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V2_LAYER_FN(plugin_v2_layer_get_plugin_type)(
    JYPPX_TensorRtLayer* layer,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    return plugin_v2_layer_get_string(
        layer,
        PluginV2LayerStringQuery::Type,
        output_buffer,
        output_buffer_size,
        out_required_size,
        "PluginV2 layer plugin type query");
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V2_LAYER_FN(plugin_v2_layer_get_plugin_version)(
    JYPPX_TensorRtLayer* layer,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    return plugin_v2_layer_get_string(
        layer,
        PluginV2LayerStringQuery::Version,
        output_buffer,
        output_buffer_size,
        out_required_size,
        "PluginV2 layer plugin version query");
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V2_LAYER_FN(plugin_v2_layer_get_plugin_namespace)(
    JYPPX_TensorRtLayer* layer,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    return plugin_v2_layer_get_string(
        layer,
        PluginV2LayerStringQuery::Namespace,
        output_buffer,
        output_buffer_size,
        out_required_size,
        "PluginV2 layer plugin namespace query");
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V2_LAYER_FN(plugin_v2_layer_get_serialization_size)(
    JYPPX_TensorRtLayer* layer,
    size_t* out_serialization_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_serialization_size, "out_serialization_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_serialization_size = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IPluginV2Layer* plugin_layer = nullptr;
    status = get_plugin_v2_layer(layer, &plugin_layer, "PluginV2 layer serialization size query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    int32_t ignored_tensor_rt_version = 0;
    return query_plugin_v2_layer_scalar(
        plugin_layer,
        PluginV2LayerScalarQuery::SerializationSize,
        out_serialization_size,
        &ignored_tensor_rt_version,
        "PluginV2 layer serialization size query");
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV2 layer serialization size query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V2_LAYER_FN(plugin_v2_layer_get_tensor_rt_version)(
    JYPPX_TensorRtLayer* layer,
    int32_t* out_tensor_rt_version)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_tensor_rt_version, "out_tensor_rt_version");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_tensor_rt_version = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IPluginV2Layer* plugin_layer = nullptr;
    status = get_plugin_v2_layer(layer, &plugin_layer, "PluginV2 layer TensorRT version query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    size_t ignored_serialization_size = 0;
    return query_plugin_v2_layer_scalar(
        plugin_layer,
        PluginV2LayerScalarQuery::TensorRtVersion,
        &ignored_serialization_size,
        out_tensor_rt_version,
        "PluginV2 layer TensorRT version query");
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV2 layer TensorRT version query");
#endif
}

#undef JYPPX_TRT_PLUGIN_V2_LAYER_FN
#undef JYPPX_TRT_PLUGIN_V2_LAYER_CONCAT
#undef JYPPX_TRT_PLUGIN_V2_LAYER_CONCAT2
