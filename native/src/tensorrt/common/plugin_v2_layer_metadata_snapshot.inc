#ifndef JYPPX_TRT_PLUGIN_V2_LAYER_PREFIX
#error "JYPPX_TRT_PLUGIN_V2_LAYER_PREFIX must be defined before including plugin_v2_layer_metadata_snapshot.inc"
#endif

#define JYPPX_TRT_PLUGIN_V2_LAYER_CONCAT2(prefix, name) prefix##name
#define JYPPX_TRT_PLUGIN_V2_LAYER_CONCAT(prefix, name) JYPPX_TRT_PLUGIN_V2_LAYER_CONCAT2(prefix, name)
#define JYPPX_TRT_PLUGIN_V2_LAYER_FN(name) JYPPX_TRT_PLUGIN_V2_LAYER_CONCAT(JYPPX_TRT_PLUGIN_V2_LAYER_PREFIX, name)

#ifndef JYPPX_TRT_PLUGIN_V2_LAYER_ENABLE_BROADCAST_EXPORTS
#define JYPPX_TRT_PLUGIN_V2_LAYER_ENABLE_BROADCAST_EXPORTS 0
#endif

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

struct PluginV2LayerCapabilityInfo
{
    int32_t input_count;
    int32_t output_count;
    JYPPX_Boolean has_ext;
    JYPPX_Boolean has_io_ext;
    JYPPX_Boolean has_dynamic_ext;
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

JYPPX_StatusCode query_plugin_v2_layer_capabilities_with_seh_guard(
    nvinfer1::IPluginV2Layer* plugin_layer,
    PluginV2LayerCapabilityInfo* out_info,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        nvinfer1::IPluginV2& plugin = plugin_layer->getPlugin();
        out_info->input_count = plugin_layer->getNbInputs();
        out_info->output_count = plugin.getNbOutputs();
        out_info->has_ext = dynamic_cast<nvinfer1::IPluginV2Ext*>(&plugin) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->has_io_ext = dynamic_cast<nvinfer1::IPluginV2IOExt*>(&plugin) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->has_dynamic_ext = dynamic_cast<nvinfer1::IPluginV2DynamicExt*>(&plugin) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_info = {};
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    nvinfer1::IPluginV2& plugin = plugin_layer->getPlugin();
    out_info->input_count = plugin_layer->getNbInputs();
    out_info->output_count = plugin.getNbOutputs();
    out_info->has_ext = dynamic_cast<nvinfer1::IPluginV2Ext*>(&plugin) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->has_io_ext = dynamic_cast<nvinfer1::IPluginV2IOExt*>(&plugin) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->has_dynamic_ext = dynamic_cast<nvinfer1::IPluginV2DynamicExt*>(&plugin) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode query_plugin_v2_layer_capabilities(
    nvinfer1::IPluginV2Layer* plugin_layer,
    PluginV2LayerCapabilityInfo* out_info,
    const char* feature_name)
{
    *out_info = {};
    try
    {
        return query_plugin_v2_layer_capabilities_with_seh_guard(plugin_layer, out_info, feature_name);
    }
    catch (const std::exception& exception)
    {
        *out_info = {};
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        *out_info = {};
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}

JYPPX_StatusCode query_plugin_v2_legacy_output_dimensions_with_seh_guard(
    nvinfer1::IPluginV2Layer* plugin_layer,
    const int32_t output_index,
    nvinfer1::Dims* input_dimensions,
    const int32_t input_count,
    nvinfer1::Dims* out_dimensions,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        for (int32_t index = 0; index < input_count; ++index)
        {
            nvinfer1::ITensor* input = plugin_layer->getInput(index);
            input_dimensions[index] = input == nullptr ? nvinfer1::Dims{} : input->getDimensions();
        }

        *out_dimensions = plugin_layer->getPlugin().getOutputDimensions(
            output_index,
            input_dimensions,
            input_count);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_dimensions = {};
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    for (int32_t index = 0; index < input_count; ++index)
    {
        nvinfer1::ITensor* input = plugin_layer->getInput(index);
        input_dimensions[index] = input == nullptr ? nvinfer1::Dims{} : input->getDimensions();
    }

    *out_dimensions = plugin_layer->getPlugin().getOutputDimensions(output_index, input_dimensions, input_count);
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode query_plugin_v2_legacy_output_dimensions(
    nvinfer1::IPluginV2Layer* plugin_layer,
    const int32_t output_index,
    nvinfer1::Dims* input_dimensions,
    const int32_t input_count,
    nvinfer1::Dims* out_dimensions,
    const char* feature_name)
{
    *out_dimensions = {};
    try
    {
        return query_plugin_v2_legacy_output_dimensions_with_seh_guard(
            plugin_layer,
            output_index,
            input_dimensions,
            input_count,
            out_dimensions,
            feature_name);
    }
    catch (const std::exception& exception)
    {
        *out_dimensions = {};
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        *out_dimensions = {};
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}

JYPPX_StatusCode query_plugin_v2_output_data_type_with_seh_guard(
    nvinfer1::IPluginV2Layer* plugin_layer,
    const int32_t output_index,
    nvinfer1::DataType* input_types,
    const int32_t input_count,
    int32_t* out_data_type,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        nvinfer1::IPluginV2& plugin = plugin_layer->getPlugin();
        auto* ext = dynamic_cast<nvinfer1::IPluginV2Ext*>(&plugin);
        if (ext == nullptr)
        {
            return JYPPX_STATUS_NOT_FOUND;
        }

        for (int32_t index = 0; index < input_count; ++index)
        {
            nvinfer1::ITensor* input = plugin_layer->getInput(index);
            input_types[index] = input == nullptr ? nvinfer1::DataType::kFLOAT : input->getType();
        }

        *out_data_type = static_cast<int32_t>(ext->getOutputDataType(output_index, input_types, input_count));
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_data_type = -1;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    nvinfer1::IPluginV2& plugin = plugin_layer->getPlugin();
    auto* ext = dynamic_cast<nvinfer1::IPluginV2Ext*>(&plugin);
    if (ext == nullptr)
    {
        return JYPPX_STATUS_NOT_FOUND;
    }

    for (int32_t index = 0; index < input_count; ++index)
    {
        nvinfer1::ITensor* input = plugin_layer->getInput(index);
        input_types[index] = input == nullptr ? nvinfer1::DataType::kFLOAT : input->getType();
    }

    *out_data_type = static_cast<int32_t>(ext->getOutputDataType(output_index, input_types, input_count));
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode query_plugin_v2_output_data_type(
    nvinfer1::IPluginV2Layer* plugin_layer,
    const int32_t output_index,
    nvinfer1::DataType* input_types,
    const int32_t input_count,
    int32_t* out_data_type,
    const char* feature_name)
{
    *out_data_type = -1;
    try
    {
        auto status = query_plugin_v2_output_data_type_with_seh_guard(
            plugin_layer,
            output_index,
            input_types,
            input_count,
            out_data_type,
            feature_name);
        if (status == JYPPX_STATUS_NOT_FOUND)
        {
            jyppx::common::set_last_error(
                JYPPX_ERROR_CATEGORY_TENSORRT,
                "PluginV2 output data type requires an IPluginV2Ext-compatible plugin.");
        }
        return status;
    }
    catch (const std::exception& exception)
    {
        *out_data_type = -1;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        *out_data_type = -1;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}

JYPPX_StatusCode query_plugin_v2_legacy_scalar_with_seh_guard(
    nvinfer1::IPluginV2Layer* plugin_layer,
    const int32_t max_batch_size,
    const int32_t data_type,
    const int32_t tensor_format,
    size_t* out_workspace_size,
    JYPPX_Boolean* out_supported,
    const bool query_workspace,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        nvinfer1::IPluginV2& plugin = plugin_layer->getPlugin();
        if (query_workspace)
        {
            *out_workspace_size = plugin.getWorkspaceSize(max_batch_size);
        }
        else
        {
            *out_supported = plugin.supportsFormat(
                static_cast<nvinfer1::DataType>(data_type),
                static_cast<nvinfer1::PluginFormat>(tensor_format)) ? JYPPX_TRUE : JYPPX_FALSE;
        }
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_workspace_size = 0;
        *out_supported = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    nvinfer1::IPluginV2& plugin = plugin_layer->getPlugin();
    if (query_workspace)
    {
        *out_workspace_size = plugin.getWorkspaceSize(max_batch_size);
    }
    else
    {
        *out_supported = plugin.supportsFormat(
            static_cast<nvinfer1::DataType>(data_type),
            static_cast<nvinfer1::PluginFormat>(tensor_format)) ? JYPPX_TRUE : JYPPX_FALSE;
    }
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode query_plugin_v2_legacy_scalar(
    nvinfer1::IPluginV2Layer* plugin_layer,
    const int32_t max_batch_size,
    const int32_t data_type,
    const int32_t tensor_format,
    size_t* out_workspace_size,
    JYPPX_Boolean* out_supported,
    const bool query_workspace,
    const char* feature_name)
{
    try
    {
        return query_plugin_v2_legacy_scalar_with_seh_guard(
            plugin_layer,
            max_batch_size,
            data_type,
            tensor_format,
            out_workspace_size,
            out_supported,
            query_workspace,
            feature_name);
    }
    catch (const std::exception& exception)
    {
        *out_workspace_size = 0;
        *out_supported = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        *out_workspace_size = 0;
        *out_supported = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}

#if JYPPX_TENSORRT_VERSION_MAJOR_NUM < 11
JYPPX_StatusCode query_plugin_v2_broadcast_with_seh_guard(
    nvinfer1::IPluginV2Layer* plugin_layer,
    const int32_t slot_index,
    const bool* input_is_broadcasted,
    const int32_t input_count,
    JYPPX_Boolean* out_value,
    const bool query_output,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        nvinfer1::IPluginV2& plugin = plugin_layer->getPlugin();
        auto* ext = dynamic_cast<nvinfer1::IPluginV2Ext*>(&plugin);
        if (ext == nullptr)
        {
            return JYPPX_STATUS_NOT_FOUND;
        }

        const bool value = query_output
            ? ext->isOutputBroadcastAcrossBatch(slot_index, input_is_broadcasted, input_count)
            : ext->canBroadcastInputAcrossBatch(slot_index);
        *out_value = value ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_value = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    nvinfer1::IPluginV2& plugin = plugin_layer->getPlugin();
    auto* ext = dynamic_cast<nvinfer1::IPluginV2Ext*>(&plugin);
    if (ext == nullptr)
    {
        return JYPPX_STATUS_NOT_FOUND;
    }

    const bool value = query_output
        ? ext->isOutputBroadcastAcrossBatch(slot_index, input_is_broadcasted, input_count)
        : ext->canBroadcastInputAcrossBatch(slot_index);
    *out_value = value ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode query_plugin_v2_broadcast(
    nvinfer1::IPluginV2Layer* plugin_layer,
    const int32_t slot_index,
    const bool* input_is_broadcasted,
    const int32_t input_count,
    JYPPX_Boolean* out_value,
    const bool query_output,
    const char* feature_name)
{
    *out_value = JYPPX_FALSE;
    try
    {
        auto status = query_plugin_v2_broadcast_with_seh_guard(
            plugin_layer,
            slot_index,
            input_is_broadcasted,
            input_count,
            out_value,
            query_output,
            feature_name);
        if (status == JYPPX_STATUS_NOT_FOUND)
        {
            jyppx::common::set_last_error(
                JYPPX_ERROR_CATEGORY_TENSORRT,
                "PluginV2 broadcast queries require an IPluginV2Ext-compatible plugin.");
        }
        return status;
    }
    catch (const std::exception& exception)
    {
        *out_value = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        *out_value = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}
#endif
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

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V2_LAYER_FN(plugin_v2_layer_get_plugin_output_count)(
    JYPPX_TensorRtLayer* layer,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IPluginV2Layer* plugin_layer = nullptr;
    status = get_plugin_v2_layer(layer, &plugin_layer, "PluginV2 output count query");
    if (status != JYPPX_STATUS_OK) { return status; }

    PluginV2LayerCapabilityInfo info{};
    status = query_plugin_v2_layer_capabilities(plugin_layer, &info, "PluginV2 output count query");
    if (status == JYPPX_STATUS_OK) { *out_count = info.output_count; }
    return status;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV2 output count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V2_LAYER_FN(plugin_v2_layer_get_capability_presence)(
    JYPPX_TensorRtLayer* layer,
    JYPPX_Boolean* out_has_ext,
    JYPPX_Boolean* out_has_io_ext,
    JYPPX_Boolean* out_has_dynamic_ext)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_ext, "out_has_ext");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_has_io_ext, "out_has_io_ext");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_has_dynamic_ext, "out_has_dynamic_ext");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_ext = JYPPX_FALSE;
    *out_has_io_ext = JYPPX_FALSE;
    *out_has_dynamic_ext = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IPluginV2Layer* plugin_layer = nullptr;
    status = get_plugin_v2_layer(layer, &plugin_layer, "PluginV2 capability presence query");
    if (status != JYPPX_STATUS_OK) { return status; }

    PluginV2LayerCapabilityInfo info{};
    status = query_plugin_v2_layer_capabilities(plugin_layer, &info, "PluginV2 capability presence query");
    if (status == JYPPX_STATUS_OK)
    {
        *out_has_ext = info.has_ext;
        *out_has_io_ext = info.has_io_ext;
        *out_has_dynamic_ext = info.has_dynamic_ext;
    }
    return status;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV2 capability presence query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V2_LAYER_FN(plugin_v2_layer_get_legacy_output_dimensions)(
    JYPPX_TensorRtLayer* layer,
    const int32_t output_index,
    JYPPX_TensorRtDims* out_dimensions)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_dimensions, "out_dimensions");
    if (status != JYPPX_STATUS_OK) { return status; }
    std::memset(out_dimensions, 0, sizeof(*out_dimensions));
#if JYPPX_HAS_TENSORRT
    nvinfer1::IPluginV2Layer* plugin_layer = nullptr;
    status = get_plugin_v2_layer(layer, &plugin_layer, "PluginV2 legacy output dimensions query");
    if (status != JYPPX_STATUS_OK) { return status; }

    PluginV2LayerCapabilityInfo info{};
    status = query_plugin_v2_layer_capabilities(plugin_layer, &info, "PluginV2 legacy output dimensions query");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_index(output_index, info.output_count, "plugin output");
    if (status != JYPPX_STATUS_OK) { return status; }

    std::unique_ptr<nvinfer1::Dims[]> input_dimensions;
    if (info.input_count > 0)
    {
        input_dimensions.reset(new (std::nothrow) nvinfer1::Dims[static_cast<size_t>(info.input_count)]);
        if (!input_dimensions)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Unable to allocate PluginV2 input dimensions scratch storage.");
            return JYPPX_STATUS_OUT_OF_MEMORY;
        }
    }

    nvinfer1::Dims dimensions{};
    status = query_plugin_v2_legacy_output_dimensions(
        plugin_layer,
        output_index,
        input_dimensions.get(),
        info.input_count,
        &dimensions,
        "PluginV2 legacy output dimensions query");
    if (status == JYPPX_STATUS_OK) { copy_dims(dimensions, out_dimensions); }
    return status;
#else
    (void)layer;
    (void)output_index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV2 legacy output dimensions query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V2_LAYER_FN(plugin_v2_layer_get_legacy_workspace_size)(
    JYPPX_TensorRtLayer* layer,
    const int32_t max_batch_size,
    size_t* out_workspace_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_workspace_size, "out_workspace_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_workspace_size = 0;
    if (max_batch_size <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "max_batch_size must be positive.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IPluginV2Layer* plugin_layer = nullptr;
    status = get_plugin_v2_layer(layer, &plugin_layer, "PluginV2 legacy workspace size query");
    if (status != JYPPX_STATUS_OK) { return status; }
    JYPPX_Boolean ignored_supported = JYPPX_FALSE;
    return query_plugin_v2_legacy_scalar(
        plugin_layer,
        max_batch_size,
        0,
        0,
        out_workspace_size,
        &ignored_supported,
        true,
        "PluginV2 legacy workspace size query");
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV2 legacy workspace size query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V2_LAYER_FN(plugin_v2_layer_supports_legacy_format)(
    JYPPX_TensorRtLayer* layer,
    const int32_t data_type,
    const int32_t tensor_format,
    JYPPX_Boolean* out_supported)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_supported, "out_supported");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_supported = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IPluginV2Layer* plugin_layer = nullptr;
    status = get_plugin_v2_layer(layer, &plugin_layer, "PluginV2 legacy format support query");
    if (status != JYPPX_STATUS_OK) { return status; }
    size_t ignored_workspace_size = 0;
    return query_plugin_v2_legacy_scalar(
        plugin_layer,
        1,
        data_type,
        tensor_format,
        &ignored_workspace_size,
        out_supported,
        false,
        "PluginV2 legacy format support query");
#else
    (void)layer;
    (void)data_type;
    (void)tensor_format;
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV2 legacy format support query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V2_LAYER_FN(plugin_v2_layer_get_output_data_type)(
    JYPPX_TensorRtLayer* layer,
    const int32_t output_index,
    int32_t* out_data_type)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_data_type, "out_data_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_data_type = -1;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IPluginV2Layer* plugin_layer = nullptr;
    status = get_plugin_v2_layer(layer, &plugin_layer, "PluginV2 output data type query");
    if (status != JYPPX_STATUS_OK) { return status; }

    PluginV2LayerCapabilityInfo info{};
    status = query_plugin_v2_layer_capabilities(plugin_layer, &info, "PluginV2 output data type query");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_index(output_index, info.output_count, "plugin output");
    if (status != JYPPX_STATUS_OK) { return status; }

    std::unique_ptr<nvinfer1::DataType[]> input_types;
    if (info.input_count > 0)
    {
        input_types.reset(new (std::nothrow) nvinfer1::DataType[static_cast<size_t>(info.input_count)]);
        if (!input_types)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Unable to allocate PluginV2 input data type scratch storage.");
            return JYPPX_STATUS_OUT_OF_MEMORY;
        }
    }

    return query_plugin_v2_output_data_type(
        plugin_layer,
        output_index,
        input_types.get(),
        info.input_count,
        out_data_type,
        "PluginV2 output data type query");
#else
    (void)layer;
    (void)output_index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV2 output data type query");
#endif
}

#if JYPPX_TRT_PLUGIN_V2_LAYER_ENABLE_BROADCAST_EXPORTS
JYPPX_StatusCode JYPPX_TRT_PLUGIN_V2_LAYER_FN(plugin_v2_layer_can_broadcast_input_across_batch)(
    JYPPX_TensorRtLayer* layer,
    const int32_t input_index,
    JYPPX_Boolean* out_can_broadcast)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_can_broadcast, "out_can_broadcast");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_can_broadcast = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM < 11
    nvinfer1::IPluginV2Layer* plugin_layer = nullptr;
    status = get_plugin_v2_layer(layer, &plugin_layer, "PluginV2 input broadcast query");
    if (status != JYPPX_STATUS_OK) { return status; }
    PluginV2LayerCapabilityInfo info{};
    status = query_plugin_v2_layer_capabilities(plugin_layer, &info, "PluginV2 input broadcast query");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_index(input_index, info.input_count, "plugin input");
    if (status != JYPPX_STATUS_OK) { return status; }
    return query_plugin_v2_broadcast(
        plugin_layer,
        input_index,
        nullptr,
        info.input_count,
        out_can_broadcast,
        false,
        "PluginV2 input broadcast query");
#else
    (void)layer;
    (void)input_index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV2 input broadcast query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_V2_LAYER_FN(plugin_v2_layer_is_output_broadcast_across_batch)(
    JYPPX_TensorRtLayer* layer,
    const int32_t output_index,
    const uint8_t* input_is_broadcasted,
    const size_t input_count,
    JYPPX_Boolean* out_is_broadcast)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_broadcast, "out_is_broadcast");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_broadcast = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM < 11
    nvinfer1::IPluginV2Layer* plugin_layer = nullptr;
    status = get_plugin_v2_layer(layer, &plugin_layer, "PluginV2 output broadcast query");
    if (status != JYPPX_STATUS_OK) { return status; }
    PluginV2LayerCapabilityInfo info{};
    status = query_plugin_v2_layer_capabilities(plugin_layer, &info, "PluginV2 output broadcast query");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_index(output_index, info.output_count, "plugin output");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (input_count != static_cast<size_t>(info.input_count))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "input_is_broadcasted count must match the PluginV2 layer input count.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (input_count > 0 && input_is_broadcasted == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "input_is_broadcasted cannot be null when input_count is nonzero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    std::unique_ptr<bool[]> broadcast_flags;
    if (input_count > 0)
    {
        broadcast_flags.reset(new (std::nothrow) bool[input_count]);
        if (!broadcast_flags)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Unable to allocate PluginV2 broadcast flag scratch storage.");
            return JYPPX_STATUS_OUT_OF_MEMORY;
        }
        for (size_t index = 0; index < input_count; ++index)
        {
            if (input_is_broadcasted[index] > 1)
            {
                jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "input_is_broadcasted values must be 0 or 1.");
                return JYPPX_STATUS_INVALID_ARGUMENT;
            }
            broadcast_flags[index] = input_is_broadcasted[index] != 0;
        }
    }

    return query_plugin_v2_broadcast(
        plugin_layer,
        output_index,
        broadcast_flags.get(),
        info.input_count,
        out_is_broadcast,
        true,
        "PluginV2 output broadcast query");
#else
    (void)layer;
    (void)output_index;
    (void)input_is_broadcasted;
    (void)input_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV2 output broadcast query");
#endif
}
#endif

#undef JYPPX_TRT_PLUGIN_V2_LAYER_FN
#undef JYPPX_TRT_PLUGIN_V2_LAYER_CONCAT
#undef JYPPX_TRT_PLUGIN_V2_LAYER_CONCAT2
#undef JYPPX_TRT_PLUGIN_V2_LAYER_ENABLE_BROADCAST_EXPORTS
