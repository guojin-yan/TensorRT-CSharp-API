#ifndef JYPPX_TRT_PLUGIN_OWNER_QUERY_PREFIX
#error "JYPPX_TRT_PLUGIN_OWNER_QUERY_PREFIX must be defined before including plugin_layer_owner_scoped_query_snapshots.inc"
#endif

#ifndef JYPPX_TRT_PLUGIN_OWNER_QUERY_ENABLE_V3
#error "JYPPX_TRT_PLUGIN_OWNER_QUERY_ENABLE_V3 must be defined before including plugin_layer_owner_scoped_query_snapshots.inc"
#endif

#define JYPPX_TRT_PLUGIN_OWNER_QUERY_CONCAT2(prefix, name) prefix##name
#define JYPPX_TRT_PLUGIN_OWNER_QUERY_CONCAT(prefix, name) JYPPX_TRT_PLUGIN_OWNER_QUERY_CONCAT2(prefix, name)
#define JYPPX_TRT_PLUGIN_OWNER_QUERY_FN(name) JYPPX_TRT_PLUGIN_OWNER_QUERY_CONCAT(JYPPX_TRT_PLUGIN_OWNER_QUERY_PREFIX, name)

namespace
{
using PluginOwnerQueryBody = JYPPX_StatusCode (*)(void* state);

JYPPX_StatusCode run_plugin_owner_query_with_seh_guard(
    PluginOwnerQueryBody body,
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

JYPPX_StatusCode run_plugin_owner_query(
    PluginOwnerQueryBody body,
    void* state,
    const char* feature_name)
{
    try
    {
        return run_plugin_owner_query_with_seh_guard(body, state, feature_name);
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

#if JYPPX_HAS_TENSORRT
enum class PluginV2FormatCapability
{
    IoExt,
    DynamicExt
};

nvinfer1::TensorFormat select_plugin_tensor_format(nvinfer1::ITensor* tensor)
{
    const uint32_t allowed_formats = static_cast<uint32_t>(tensor->getAllowedFormats());
    if (allowed_formats == 0)
    {
        return nvinfer1::TensorFormat::kLINEAR;
    }

    for (uint32_t format = 0; format < 32; ++format)
    {
        if ((allowed_formats & (uint32_t{1} << format)) != 0)
        {
            return static_cast<nvinfer1::TensorFormat>(format);
        }
    }

    return nvinfer1::TensorFormat::kLINEAR;
}

void fill_plugin_tensor_desc(nvinfer1::ITensor* tensor, nvinfer1::PluginTensorDesc* out_desc)
{
    out_desc->dims = tensor->getDimensions();
    out_desc->type = tensor->getType();
    out_desc->format = select_plugin_tensor_format(tensor);
    out_desc->scale = -1.0F;
}

struct PluginV2FormatSupportState
{
    nvinfer1::IPluginV2Layer* layer;
    PluginV2FormatCapability capability;
    nvinfer1::PluginTensorDesc* descriptors;
    int32_t input_count;
    int32_t output_count;
    uint8_t* output_support;
};

JYPPX_StatusCode plugin_v2_format_support_body(void* raw_state)
{
    auto* state = static_cast<PluginV2FormatSupportState*>(raw_state);
    nvinfer1::IPluginV2& plugin = state->layer->getPlugin();
    auto* io_ext = dynamic_cast<nvinfer1::IPluginV2IOExt*>(&plugin);
    auto* dynamic_ext = dynamic_cast<nvinfer1::IPluginV2DynamicExt*>(&plugin);
    if ((state->capability == PluginV2FormatCapability::IoExt && io_ext == nullptr)
        || (state->capability == PluginV2FormatCapability::DynamicExt && dynamic_ext == nullptr))
    {
        return JYPPX_STATUS_NOT_FOUND;
    }

    for (int32_t index = 0; index < state->input_count; ++index)
    {
        nvinfer1::ITensor* tensor = state->layer->getInput(index);
        if (tensor == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
        fill_plugin_tensor_desc(tensor, &state->descriptors[index]);
    }
    for (int32_t index = 0; index < state->output_count; ++index)
    {
        nvinfer1::ITensor* tensor = state->layer->getOutput(index);
        if (tensor == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
        fill_plugin_tensor_desc(tensor, &state->descriptors[state->input_count + index]);
    }

    bool prefix_supported = true;
    const int32_t descriptor_count = state->input_count + state->output_count;
    for (int32_t position = 0; position < descriptor_count; ++position)
    {
        bool supported = false;
        if (prefix_supported)
        {
            supported = state->capability == PluginV2FormatCapability::IoExt
                ? io_ext->supportsFormatCombination(position, state->descriptors, state->input_count, state->output_count)
                : dynamic_ext->supportsFormatCombination(position, state->descriptors, state->input_count, state->output_count);
        }
        state->output_support[position] = supported ? uint8_t{1} : uint8_t{0};
        prefix_supported = prefix_supported && supported;
    }
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode copy_plugin_v2_current_format_support(
    JYPPX_TensorRtLayer* layer,
    const PluginV2FormatCapability capability,
    uint8_t* output_support,
    const size_t output_capacity,
    size_t* out_required_count,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_count, "out_required_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_count = 0;

    nvinfer1::IPluginV2Layer* plugin_layer = nullptr;
    status = get_plugin_v2_layer(layer, &plugin_layer, feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }

    PluginV2LayerCapabilityInfo info{};
    status = query_plugin_v2_layer_capabilities(plugin_layer, &info, feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    if (info.input_count < 0 || info.output_count <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "PluginV2 layer reported invalid input or output counts.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const size_t required_count = static_cast<size_t>(info.input_count) + static_cast<size_t>(info.output_count);
    *out_required_count = required_count;
    if (output_capacity < required_count || (required_count > 0 && output_support == nullptr))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "PluginV2 format-support output buffer is too small.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }

    std::unique_ptr<nvinfer1::PluginTensorDesc[]> descriptors(
        new (std::nothrow) nvinfer1::PluginTensorDesc[required_count]);
    if (!descriptors)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Unable to allocate PluginV2 format descriptor scratch storage.");
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }
    std::memset(output_support, 0, required_count);

    PluginV2FormatSupportState state{
        plugin_layer,
        capability,
        descriptors.get(),
        info.input_count,
        info.output_count,
        output_support};
    status = run_plugin_owner_query(plugin_v2_format_support_body, &state, feature_name);
    if (status == JYPPX_STATUS_NOT_FOUND)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            capability == PluginV2FormatCapability::IoExt
                ? "PluginV2 layer does not expose IPluginV2IOExt."
                : "PluginV2 layer does not expose IPluginV2DynamicExt.");
    }
    else if (status == JYPPX_STATUS_INVALID_STATE)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "PluginV2 layer contains a missing input or output tensor.");
    }
    return status;
}
#endif

#if JYPPX_TRT_PLUGIN_OWNER_QUERY_ENABLE_V3 && JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
struct PluginV3IoCountState
{
    nvinfer1::IPluginV3Layer* layer;
    nvinfer1::IPluginV3OneBuild* build;
    int32_t* out_input_count;
    int32_t* out_output_count;
};

JYPPX_StatusCode plugin_v3_io_count_body(void* raw_state)
{
    auto* state = static_cast<PluginV3IoCountState*>(raw_state);
    *state->out_input_count = state->layer->getNbInputs();
    *state->out_output_count = state->build->getNbOutputs();
    return (*state->out_input_count >= 0 && *state->out_output_count > 0)
        ? JYPPX_STATUS_OK
        : JYPPX_STATUS_INVALID_STATE;
}

JYPPX_StatusCode get_plugin_v3_build_target(
    JYPPX_TensorRtLayer* layer,
    nvinfer1::IPluginV3Layer** out_plugin_layer,
    nvinfer1::IPluginV3OneBuild** out_build,
    const char* feature_name)
{
    auto status = get_plugin_v3_layer(layer, out_plugin_layer, feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::IVersionedInterface* target = nullptr;
    status = resolve_plugin_v3_target(*out_plugin_layer, PluginV3LayerTarget::Build, &target, feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_build = static_cast<nvinfer1::IPluginV3OneBuild*>(target);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode get_plugin_v3_runtime_target(
    JYPPX_TensorRtLayer* layer,
    nvinfer1::IPluginV3OneRuntime** out_runtime,
    const char* feature_name)
{
    nvinfer1::IPluginV3Layer* plugin_layer = nullptr;
    auto status = get_plugin_v3_layer(layer, &plugin_layer, feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }

    nvinfer1::IVersionedInterface* target = nullptr;
    status = resolve_plugin_v3_target(plugin_layer, PluginV3LayerTarget::Runtime, &target, feature_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_runtime = static_cast<nvinfer1::IPluginV3OneRuntime*>(target);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode query_plugin_v3_io_counts(
    nvinfer1::IPluginV3Layer* plugin_layer,
    nvinfer1::IPluginV3OneBuild* build,
    int32_t* out_input_count,
    int32_t* out_output_count,
    const char* feature_name)
{
    *out_input_count = 0;
    *out_output_count = 0;
    PluginV3IoCountState state{plugin_layer, build, out_input_count, out_output_count};
    auto status = run_plugin_owner_query(plugin_v3_io_count_body, &state, feature_name);
    if (status == JYPPX_STATUS_INVALID_STATE)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "PluginV3 build capability reported invalid input or output counts.");
    }
    return status;
}

struct PluginV3OutputTypesState
{
    nvinfer1::IPluginV3Layer* layer;
    nvinfer1::IPluginV3OneBuild* build;
    nvinfer1::DataType* input_types;
    int32_t input_count;
    nvinfer1::DataType* output_types;
    int32_t output_count;
    int32_t callback_status;
};

JYPPX_StatusCode plugin_v3_output_types_body(void* raw_state)
{
    auto* state = static_cast<PluginV3OutputTypesState*>(raw_state);
    nvinfer1::DataType no_input_type = nvinfer1::DataType::kFLOAT;
    for (int32_t index = 0; index < state->input_count; ++index)
    {
        nvinfer1::ITensor* tensor = state->layer->getInput(index);
        if (tensor == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
        state->input_types[index] = tensor->getType();
    }
    state->callback_status = state->build->getOutputDataTypes(
        state->output_types,
        state->output_count,
        state->input_count == 0 ? &no_input_type : state->input_types,
        state->input_count);
    return state->callback_status == 0 ? JYPPX_STATUS_OK : JYPPX_STATUS_RUNTIME_ERROR;
}

struct PluginV3AliasedInputsState
{
    nvinfer1::IPluginV3OneBuild* build;
    int32_t input_count;
    int32_t output_count;
    int32_t* output_aliases;
};

JYPPX_StatusCode plugin_v3_aliased_inputs_body(void* raw_state)
{
    auto* state = static_cast<PluginV3AliasedInputsState*>(raw_state);
    auto* build_v2 = dynamic_cast<nvinfer1::IPluginV3OneBuildV2*>(state->build);
    if (build_v2 == nullptr) { return JYPPX_STATUS_NOT_FOUND; }
    for (int32_t output_index = 0; output_index < state->output_count; ++output_index)
    {
        const int32_t input_index = build_v2->getAliasedInput(output_index);
        if (input_index < -1 || input_index >= state->input_count) { return JYPPX_STATUS_INVALID_STATE; }
        state->output_aliases[output_index] = input_index;
    }
    return JYPPX_STATUS_OK;
}

struct PluginV3FormatSupportState
{
    nvinfer1::IPluginV3Layer* layer;
    nvinfer1::IPluginV3OneBuild* build;
    nvinfer1::DynamicPluginTensorDesc* descriptors;
    int32_t input_count;
    int32_t output_count;
    uint8_t* output_support;
};

void fill_dynamic_plugin_tensor_desc(nvinfer1::ITensor* tensor, nvinfer1::DynamicPluginTensorDesc* out_desc)
{
    fill_plugin_tensor_desc(tensor, &out_desc->desc);
    out_desc->min = out_desc->desc.dims;
    out_desc->opt = out_desc->desc.dims;
    out_desc->max = out_desc->desc.dims;
}

JYPPX_StatusCode plugin_v3_format_support_body(void* raw_state)
{
    auto* state = static_cast<PluginV3FormatSupportState*>(raw_state);
    for (int32_t index = 0; index < state->input_count; ++index)
    {
        nvinfer1::ITensor* tensor = state->layer->getInput(index);
        if (tensor == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
        fill_dynamic_plugin_tensor_desc(tensor, &state->descriptors[index]);
    }
    for (int32_t index = 0; index < state->output_count; ++index)
    {
        nvinfer1::ITensor* tensor = state->layer->getOutput(index);
        if (tensor == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
        fill_dynamic_plugin_tensor_desc(tensor, &state->descriptors[state->input_count + index]);
    }

    bool prefix_supported = true;
    const int32_t descriptor_count = state->input_count + state->output_count;
    for (int32_t position = 0; position < descriptor_count; ++position)
    {
        const bool supported = prefix_supported
            && state->build->supportsFormatCombination(position, state->descriptors, state->input_count, state->output_count);
        state->output_support[position] = supported ? uint8_t{1} : uint8_t{0};
        prefix_supported = prefix_supported && supported;
    }
    return JYPPX_STATUS_OK;
}

struct PluginV3RuntimeFieldCountState
{
    nvinfer1::IPluginV3OneRuntime* runtime;
    int32_t* out_count;
};

JYPPX_StatusCode plugin_v3_runtime_field_count_body(void* raw_state)
{
    auto* state = static_cast<PluginV3RuntimeFieldCountState*>(raw_state);
    const nvinfer1::PluginFieldCollection* fields = state->runtime->getFieldsToSerialize();
    if (fields == nullptr)
    {
        *state->out_count = 0;
        return JYPPX_STATUS_OK;
    }
    if (fields->nbFields < 0 || (fields->nbFields > 0 && fields->fields == nullptr))
    {
        return JYPPX_STATUS_INVALID_STATE;
    }
    *state->out_count = fields->nbFields;
    return JYPPX_STATUS_OK;
}

struct PluginV3RuntimeFieldNameState
{
    nvinfer1::IPluginV3OneRuntime* runtime;
    int32_t field_index;
    char* output_buffer;
    size_t output_buffer_size;
    size_t* out_required_size;
};

JYPPX_StatusCode plugin_v3_runtime_field_name_body(void* raw_state)
{
    auto* state = static_cast<PluginV3RuntimeFieldNameState*>(raw_state);
    const nvinfer1::PluginFieldCollection* fields = state->runtime->getFieldsToSerialize();
    if (fields == nullptr || fields->nbFields <= state->field_index || fields->fields == nullptr)
    {
        return JYPPX_STATUS_NOT_FOUND;
    }
    const char* name = fields->fields[state->field_index].name;
    return copy_string_to_buffer(name == nullptr ? "" : name, state->output_buffer, state->output_buffer_size, state->out_required_size);
}

struct PluginV3RuntimeFieldMetadataState
{
    nvinfer1::IPluginV3OneRuntime* runtime;
    int32_t field_index;
    int32_t* out_field_type;
    int32_t* out_length;
    JYPPX_Boolean* out_has_data;
};

JYPPX_StatusCode plugin_v3_runtime_field_metadata_body(void* raw_state)
{
    auto* state = static_cast<PluginV3RuntimeFieldMetadataState*>(raw_state);
    const nvinfer1::PluginFieldCollection* fields = state->runtime->getFieldsToSerialize();
    if (fields == nullptr || fields->nbFields <= state->field_index || fields->fields == nullptr)
    {
        return JYPPX_STATUS_NOT_FOUND;
    }
    const nvinfer1::PluginField& field = fields->fields[state->field_index];
    if (field.length < 0) { return JYPPX_STATUS_INVALID_STATE; }
    *state->out_field_type = static_cast<int32_t>(field.type);
    *state->out_length = field.length;
    *state->out_has_data = field.data == nullptr ? JYPPX_FALSE : JYPPX_TRUE;
    return JYPPX_STATUS_OK;
}
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_OWNER_QUERY_FN(plugin_v2_layer_copy_dynamic_format_support)(
    JYPPX_TensorRtLayer* layer,
    uint8_t* output_support,
    size_t output_capacity,
    size_t* out_required_count)
{
#if JYPPX_HAS_TENSORRT
    return copy_plugin_v2_current_format_support(
        layer,
        PluginV2FormatCapability::DynamicExt,
        output_support,
        output_capacity,
        out_required_count,
        "PluginV2 DynamicExt current format-combination query");
#else
    (void)layer; (void)output_support; (void)output_capacity;
    if (out_required_count != nullptr) { *out_required_count = 0; }
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV2 DynamicExt current format-combination query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_OWNER_QUERY_FN(plugin_v2_layer_copy_io_ext_format_support)(
    JYPPX_TensorRtLayer* layer,
    uint8_t* output_support,
    size_t output_capacity,
    size_t* out_required_count)
{
#if JYPPX_HAS_TENSORRT
    return copy_plugin_v2_current_format_support(
        layer,
        PluginV2FormatCapability::IoExt,
        output_support,
        output_capacity,
        out_required_count,
        "PluginV2 IOExt current format-combination query");
#else
    (void)layer; (void)output_support; (void)output_capacity;
    if (out_required_count != nullptr) { *out_required_count = 0; }
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV2 IOExt current format-combination query");
#endif
}

#if JYPPX_TRT_PLUGIN_OWNER_QUERY_ENABLE_V3
JYPPX_StatusCode JYPPX_TRT_PLUGIN_OWNER_QUERY_FN(plugin_v3_layer_get_build_io_counts)(
    JYPPX_TensorRtLayer* layer,
    int32_t* out_input_count,
    int32_t* out_output_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_input_count, "out_input_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_output_count, "out_output_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_input_count = 0;
    *out_output_count = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::IPluginV3Layer* plugin_layer = nullptr;
    nvinfer1::IPluginV3OneBuild* build = nullptr;
    status = get_plugin_v3_build_target(layer, &plugin_layer, &build, "PluginV3 build IO count query");
    if (status != JYPPX_STATUS_OK) { return status; }
    return query_plugin_v3_io_counts(plugin_layer, build, out_input_count, out_output_count, "PluginV3 build IO count query");
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV3 build IO count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_OWNER_QUERY_FN(plugin_v3_layer_copy_build_output_data_types)(
    JYPPX_TensorRtLayer* layer,
    int32_t* output_types,
    size_t output_capacity,
    size_t* out_required_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_count, "out_required_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_count = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::IPluginV3Layer* plugin_layer = nullptr;
    nvinfer1::IPluginV3OneBuild* build = nullptr;
    status = get_plugin_v3_build_target(layer, &plugin_layer, &build, "PluginV3 build output data-type query");
    if (status != JYPPX_STATUS_OK) { return status; }
    int32_t input_count = 0;
    int32_t output_count = 0;
    status = query_plugin_v3_io_counts(plugin_layer, build, &input_count, &output_count, "PluginV3 build output data-type query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_count = static_cast<size_t>(output_count);
    if (output_capacity < static_cast<size_t>(output_count) || output_types == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "PluginV3 output data-type buffer is too small.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }

    std::unique_ptr<nvinfer1::DataType[]> input_data_types;
    if (input_count > 0)
    {
        input_data_types.reset(new (std::nothrow) nvinfer1::DataType[static_cast<size_t>(input_count)]);
        if (!input_data_types) { return JYPPX_STATUS_OUT_OF_MEMORY; }
    }
    std::unique_ptr<nvinfer1::DataType[]> output_data_types(
        new (std::nothrow) nvinfer1::DataType[static_cast<size_t>(output_count)]);
    if (!output_data_types) { return JYPPX_STATUS_OUT_OF_MEMORY; }

    PluginV3OutputTypesState state{
        plugin_layer, build, input_data_types.get(), input_count, output_data_types.get(), output_count, 0};
    status = run_plugin_owner_query(plugin_v3_output_types_body, &state, "PluginV3 build output data-type query");
    if (status == JYPPX_STATUS_RUNTIME_ERROR)
    {
        std::ostringstream message;
        message << "PluginV3 getOutputDataTypes returned status " << state.callback_status << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, message.str().c_str());
        return status;
    }
    if (status == JYPPX_STATUS_INVALID_STATE)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "PluginV3 layer contains a missing input tensor.");
        return status;
    }
    if (status != JYPPX_STATUS_OK) { return status; }
    for (int32_t index = 0; index < output_count; ++index)
    {
        output_types[index] = static_cast<int32_t>(output_data_types[index]);
    }
    return JYPPX_STATUS_OK;
#else
    (void)layer; (void)output_types; (void)output_capacity;
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV3 build output data-type query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_OWNER_QUERY_FN(plugin_v3_layer_copy_build_aliased_inputs)(
    JYPPX_TensorRtLayer* layer,
    int32_t* output_aliases,
    size_t output_capacity,
    size_t* out_required_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_count, "out_required_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_count = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::IPluginV3Layer* plugin_layer = nullptr;
    nvinfer1::IPluginV3OneBuild* build = nullptr;
    status = get_plugin_v3_build_target(layer, &plugin_layer, &build, "PluginV3 build aliased-input query");
    if (status != JYPPX_STATUS_OK) { return status; }
    int32_t input_count = 0;
    int32_t output_count = 0;
    status = query_plugin_v3_io_counts(plugin_layer, build, &input_count, &output_count, "PluginV3 build aliased-input query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_count = static_cast<size_t>(output_count);
    if (output_capacity < static_cast<size_t>(output_count) || output_aliases == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "PluginV3 aliased-input buffer is too small.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }
    std::fill(output_aliases, output_aliases + output_count, -1);
    PluginV3AliasedInputsState state{build, input_count, output_count, output_aliases};
    status = run_plugin_owner_query(plugin_v3_aliased_inputs_body, &state, "PluginV3 build aliased-input query");
    if (status == JYPPX_STATUS_NOT_FOUND)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "PluginV3 build capability does not expose IPluginV3OneBuildV2 alias metadata.");
    }
    else if (status == JYPPX_STATUS_INVALID_STATE)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "PluginV3 build capability returned an invalid aliased-input index.");
    }
    return status;
#else
    (void)layer; (void)output_aliases; (void)output_capacity;
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV3 build aliased-input query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_OWNER_QUERY_FN(plugin_v3_layer_copy_build_current_format_support)(
    JYPPX_TensorRtLayer* layer,
    uint8_t* output_support,
    size_t output_capacity,
    size_t* out_required_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_count, "out_required_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_count = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::IPluginV3Layer* plugin_layer = nullptr;
    nvinfer1::IPluginV3OneBuild* build = nullptr;
    status = get_plugin_v3_build_target(layer, &plugin_layer, &build, "PluginV3 build current format-combination query");
    if (status != JYPPX_STATUS_OK) { return status; }
    int32_t input_count = 0;
    int32_t output_count = 0;
    status = query_plugin_v3_io_counts(plugin_layer, build, &input_count, &output_count, "PluginV3 build current format-combination query");
    if (status != JYPPX_STATUS_OK) { return status; }
    const size_t required_count = static_cast<size_t>(input_count) + static_cast<size_t>(output_count);
    *out_required_count = required_count;
    if (output_capacity < required_count || (required_count > 0 && output_support == nullptr))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "PluginV3 format-support output buffer is too small.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }
    std::unique_ptr<nvinfer1::DynamicPluginTensorDesc[]> descriptors(
        new (std::nothrow) nvinfer1::DynamicPluginTensorDesc[required_count]);
    if (!descriptors) { return JYPPX_STATUS_OUT_OF_MEMORY; }
    std::memset(output_support, 0, required_count);
    PluginV3FormatSupportState state{
        plugin_layer, build, descriptors.get(), input_count, output_count, output_support};
    status = run_plugin_owner_query(plugin_v3_format_support_body, &state, "PluginV3 build current format-combination query");
    if (status == JYPPX_STATUS_INVALID_STATE)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "PluginV3 layer contains a missing input or output tensor.");
    }
    return status;
#else
    (void)layer; (void)output_support; (void)output_capacity;
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV3 build current format-combination query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_OWNER_QUERY_FN(plugin_v3_layer_get_runtime_serialization_field_count)(
    JYPPX_TensorRtLayer* layer,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::IPluginV3OneRuntime* runtime = nullptr;
    status = get_plugin_v3_runtime_target(layer, &runtime, "PluginV3 runtime serialization-field count query");
    if (status != JYPPX_STATUS_OK) { return status; }
    PluginV3RuntimeFieldCountState state{runtime, out_count};
    status = run_plugin_owner_query(plugin_v3_runtime_field_count_body, &state, "PluginV3 runtime serialization-field count query");
    if (status == JYPPX_STATUS_INVALID_STATE)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "PluginV3 runtime returned an invalid serialization field collection.");
    }
    return status;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV3 runtime serialization-field count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_OWNER_QUERY_FN(plugin_v3_layer_get_runtime_serialization_field_name)(
    JYPPX_TensorRtLayer* layer,
    int32_t field_index,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
    if (field_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "field_index cannot be negative.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::IPluginV3OneRuntime* runtime = nullptr;
    status = get_plugin_v3_runtime_target(layer, &runtime, "PluginV3 runtime serialization-field name query");
    if (status != JYPPX_STATUS_OK) { return status; }
    PluginV3RuntimeFieldNameState state{runtime, field_index, output_buffer, output_buffer_size, out_required_size};
    status = run_plugin_owner_query(plugin_v3_runtime_field_name_body, &state, "PluginV3 runtime serialization-field name query");
    if (status == JYPPX_STATUS_NOT_FOUND)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "PluginV3 serialization field index is out of range.");
    }
    return status;
#else
    (void)layer; (void)output_buffer; (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV3 runtime serialization-field name query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_PLUGIN_OWNER_QUERY_FN(plugin_v3_layer_get_runtime_serialization_field_metadata)(
    JYPPX_TensorRtLayer* layer,
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
    *out_field_type = -1;
    *out_length = 0;
    *out_has_data = JYPPX_FALSE;
    if (field_index < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "field_index cannot be negative.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::IPluginV3OneRuntime* runtime = nullptr;
    status = get_plugin_v3_runtime_target(layer, &runtime, "PluginV3 runtime serialization-field metadata query");
    if (status != JYPPX_STATUS_OK) { return status; }
    PluginV3RuntimeFieldMetadataState state{runtime, field_index, out_field_type, out_length, out_has_data};
    status = run_plugin_owner_query(plugin_v3_runtime_field_metadata_body, &state, "PluginV3 runtime serialization-field metadata query");
    if (status == JYPPX_STATUS_NOT_FOUND)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "PluginV3 serialization field index is out of range.");
    }
    else if (status == JYPPX_STATUS_INVALID_STATE)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "PluginV3 serialization field contains invalid metadata.");
    }
    return status;
#else
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "PluginV3 runtime serialization-field metadata query");
#endif
}
#endif

#undef JYPPX_TRT_PLUGIN_OWNER_QUERY_FN
#undef JYPPX_TRT_PLUGIN_OWNER_QUERY_CONCAT
#undef JYPPX_TRT_PLUGIN_OWNER_QUERY_CONCAT2
