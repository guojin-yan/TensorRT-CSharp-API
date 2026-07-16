#if JYPPX_HAS_TENSORRT
static JYPPX_StatusCode jyppx_trt11_callback_allocator_control_exception(const char* feature_name, const std::exception& exception)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
}

static JYPPX_StatusCode jyppx_trt11_callback_allocator_control_unknown_exception(const char* feature_name)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
}

static JYPPX_StatusCode trt11_builder_config_set_progress_monitor_with_seh_guard(
    nvinfer1::IBuilderConfig* config_payload,
    ManagedProgressMonitor* monitor_payload,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        config_payload->setProgressMonitor(monitor_payload);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    config_payload->setProgressMonitor(monitor_payload);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode trt11_builder_config_has_progress_monitor_with_seh_guard(
    nvinfer1::IBuilderConfig* config_payload,
    JYPPX_Boolean* out_has_monitor,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_monitor = config_payload->getProgressMonitor() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_monitor = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_monitor = config_payload->getProgressMonitor() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode trt11_builder_config_clear_progress_monitor_with_seh_guard(
    nvinfer1::IBuilderConfig* config_payload,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        config_payload->setProgressMonitor(nullptr);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    config_payload->setProgressMonitor(nullptr);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode trt11_engine_inspector_has_error_recorder_with_seh_guard(
    nvinfer1::IEngineInspector* inspector_payload,
    JYPPX_Boolean* out_has_recorder,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_recorder = inspector_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_recorder = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_recorder = inspector_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode trt11_execution_context_clear_output_allocator_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    const char* tensor_name,
    JYPPX_Boolean* out_cleared,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_cleared = context_payload->setOutputAllocator(tensor_name, nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_cleared = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_cleared = context_payload->setOutputAllocator(tensor_name, nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode trt11_execution_context_clear_temporary_storage_allocator_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    JYPPX_Boolean* out_cleared,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_cleared = context_payload->setTemporaryStorageAllocator(nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_cleared = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_cleared = context_payload->setTemporaryStorageAllocator(nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode trt11_execution_context_clear_debug_listener_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    JYPPX_Boolean* out_cleared,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_cleared = context_payload->setDebugListener(nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_cleared = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_cleared = context_payload->setDebugListener(nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode trt11_execution_context_has_debug_listener_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    JYPPX_Boolean* out_has_listener,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_listener = context_payload->getDebugListener() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_listener = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_listener = context_payload->getDebugListener() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode trt11_execution_context_clear_profiler_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        context_payload->setProfiler(nullptr);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    context_payload->setProfiler(nullptr);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode trt11_execution_context_has_profiler_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    JYPPX_Boolean* out_has_profiler,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_profiler = context_payload->getProfiler() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_profiler = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_profiler = context_payload->getProfiler() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}
#endif

JYPPX_StatusCode jyppx_trt11_builder_config_reset(JYPPX_TensorRtBuilderConfig* config)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    config_payload->reset();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config reset");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_has_timing_cache(JYPPX_TensorRtBuilderConfig* config, JYPPX_Boolean* out_has_cache)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_cache, "out_has_cache");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_cache = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_has_cache = config_payload->getTimingCache() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config timing cache query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_can_run_on_dla(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtLayer* layer, JYPPX_Boolean* out_can_run)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_can_run, "out_can_run");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_can_run = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IBuilderConfig* config_payload = nullptr;
    nvinfer1::ILayer* layer_payload = nullptr;
    status = get_builder_config_and_layer_payload(config, layer, &config_payload, &layer_payload, "layer DLA capability query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_can_run = config_payload->canRunOnDLA(layer_payload) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)config;
    (void)layer;
    return jyppx::tensorrt::report_vendor_missing(kLine, "layer DLA capability query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_clear_plugins_to_serialize(JYPPX_TensorRtBuilderConfig* config)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    config_payload->setPluginsToSerialize(nullptr, 0);
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config plugin serialization clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_plugin_to_serialize_count(JYPPX_TensorRtBuilderConfig* config, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_count = config_payload->getNbPluginsToSerialize();
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config plugin serialization count query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_get_plugin_to_serialize(JYPPX_TensorRtBuilderConfig* config, int32_t index, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    status = validate_index(index, config_payload->getNbPluginsToSerialize(), "plugin_to_serialize");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_string_to_buffer(config_payload->getPluginToSerialize(index), output_buffer, output_buffer_size, out_required_size);
#else
    (void)index;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config plugin serialization path query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_set_progress_monitor(JYPPX_TensorRtBuilderConfig* config, JYPPX_TensorRtProgressMonitor* monitor)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(monitor, kLine, JYPPX_TENSORRT_OBJECT_KIND_PROGRESS_MONITOR, "monitor");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    auto* monitor_payload = get_payload<ManagedProgressMonitor>(monitor);
    if (config_payload == nullptr || monitor_payload == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Builder config or progress monitor handle does not carry the expected TensorRT payload.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    try
    {
        return trt11_builder_config_set_progress_monitor_with_seh_guard(
            config_payload,
            monitor_payload,
            "builder config progress monitor attach");
    }
    catch (const std::exception& exception)
    {
        return jyppx_trt11_callback_allocator_control_exception("builder config progress monitor attach", exception);
    }
    catch (...)
    {
        return jyppx_trt11_callback_allocator_control_unknown_exception("builder config progress monitor attach");
    }
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config progress monitor attach");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_has_progress_monitor(JYPPX_TensorRtBuilderConfig* config, JYPPX_Boolean* out_has_monitor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_monitor, "out_has_monitor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_monitor = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    try
    {
        return trt11_builder_config_has_progress_monitor_with_seh_guard(
            config_payload,
            out_has_monitor,
            "builder config progress monitor query");
    }
    catch (const std::exception& exception)
    {
        *out_has_monitor = JYPPX_FALSE;
        return jyppx_trt11_callback_allocator_control_exception("builder config progress monitor query", exception);
    }
    catch (...)
    {
        *out_has_monitor = JYPPX_FALSE;
        return jyppx_trt11_callback_allocator_control_unknown_exception("builder config progress monitor query");
    }
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config progress monitor query");
#endif
}

JYPPX_StatusCode jyppx_trt11_builder_config_clear_progress_monitor(JYPPX_TensorRtBuilderConfig* config)
{
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* config_payload = get_payload<nvinfer1::IBuilderConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    try
    {
        return trt11_builder_config_clear_progress_monitor_with_seh_guard(
            config_payload,
            "builder config progress monitor clear");
    }
    catch (const std::exception& exception)
    {
        return jyppx_trt11_callback_allocator_control_exception("builder config progress monitor clear", exception);
    }
    catch (...)
    {
        return jyppx_trt11_callback_allocator_control_unknown_exception("builder config progress monitor clear");
    }
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "builder config progress monitor clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_mark_debug(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_marked)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_marked, "out_marked");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network debug tensor mark");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(tensor, &tensor_payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = network_payload->markDebug(*tensor_payload) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)network;
    (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network debug tensor mark");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_unmark_debug(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_unmarked)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_unmarked, "out_unmarked");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_unmarked = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network debug tensor unmark");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(tensor, &tensor_payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_unmarked = network_payload->unmarkDebug(*tensor_payload) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)network;
    (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network debug tensor unmark");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_is_debug_tensor(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_is_debug)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_is_debug, "out_is_debug");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_debug = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network debug tensor query");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(tensor, &tensor_payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_debug = network_payload->isDebugTensor(*tensor_payload) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)network;
    (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network debug tensor query");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_mark_unfused_tensors_as_debug_tensors(JYPPX_TensorRtNetworkDefinition* network, JYPPX_Boolean* out_marked)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_marked, "out_marked");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network unfused debug tensor mark");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = network_payload->markUnfusedTensorsAsDebugTensors() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)network;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network unfused debug tensor mark");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_unmark_unfused_tensors_as_debug_tensors(JYPPX_TensorRtNetworkDefinition* network, JYPPX_Boolean* out_unmarked)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_unmarked, "out_unmarked");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_unmarked = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network unfused debug tensor unmark");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_unmarked = network_payload->unmarkUnfusedTensorsAsDebugTensors() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)network;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network unfused debug tensor unmark");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_mark_output_for_shapes(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_marked)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_marked, "out_marked");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network shape output mark");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(tensor, &tensor_payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_marked = network_payload->markOutputForShapes(*tensor_payload) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)network;
    (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network shape output mark");
#endif
}

JYPPX_StatusCode jyppx_trt11_network_unmark_output_for_shapes(JYPPX_TensorRtNetworkDefinition* network, JYPPX_TensorRtTensor* tensor, JYPPX_Boolean* out_unmarked)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_unmarked, "out_unmarked");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_unmarked = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    nvinfer1::ITensor* tensor_payload = nullptr;
    status = get_network_payload_for_api(network, &network_payload, "network shape output unmark");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = get_tensor_payload_for_api(tensor, &tensor_payload, "tensor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_unmarked = network_payload->unmarkOutputForShapes(*tensor_payload) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)network;
    (void)tensor;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network shape output unmark");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_inspector_get_layer_information(JYPPX_TensorRtEngineInspector* inspector, int32_t layer_index, int32_t format, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
    if (layer_index < 0 || format < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
    status = jyppx::tensorrt::validate_handle(inspector, kLine, JYPPX_TENSORRT_OBJECT_KIND_ENGINE_INSPECTOR, "inspector");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* inspector_payload = get_payload<nvinfer1::IEngineInspector>(inspector);
    if (inspector_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    return copy_string_to_buffer(inspector_payload->getLayerInformation(layer_index, static_cast<nvinfer1::LayerInformationFormat>(format)), output_buffer, output_buffer_size, out_required_size);
#else
    (void)layer_index;
    (void)format;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine inspector layer information query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_inspector_has_execution_context(JYPPX_TensorRtEngineInspector* inspector, JYPPX_Boolean* out_has_context)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_context, "out_has_context");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_context = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(inspector, kLine, JYPPX_TENSORRT_OBJECT_KIND_ENGINE_INSPECTOR, "inspector");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* inspector_payload = get_payload<nvinfer1::IEngineInspector>(inspector);
    if (inspector_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_has_context = inspector_payload->getExecutionContext() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine inspector execution context query");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_inspector_clear_execution_context(JYPPX_TensorRtEngineInspector* inspector)
{
    auto status = jyppx::tensorrt::validate_handle(inspector, kLine, JYPPX_TENSORRT_OBJECT_KIND_ENGINE_INSPECTOR, "inspector");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* inspector_payload = get_payload<nvinfer1::IEngineInspector>(inspector);
    if (inspector_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    if (!inspector_payload->setExecutionContext(nullptr))
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "IEngineInspector::setExecutionContext(nullptr) returned false.");
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
    return JYPPX_STATUS_OK;
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine inspector execution context clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_inspector_has_error_recorder(JYPPX_TensorRtEngineInspector* inspector, JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_recorder = JYPPX_FALSE;
    status = jyppx::tensorrt::validate_handle(inspector, kLine, JYPPX_TENSORRT_OBJECT_KIND_ENGINE_INSPECTOR, "inspector");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    auto* inspector_payload = get_payload<nvinfer1::IEngineInspector>(inspector);
    if (inspector_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    try
    {
        return trt11_engine_inspector_has_error_recorder_with_seh_guard(
            inspector_payload,
            out_has_recorder,
            "engine inspector error recorder query");
    }
    catch (const std::exception& exception)
    {
        *out_has_recorder = JYPPX_FALSE;
        return jyppx_trt11_callback_allocator_control_exception("engine inspector error recorder query", exception);
    }
    catch (...)
    {
        *out_has_recorder = JYPPX_FALSE;
        return jyppx_trt11_callback_allocator_control_unknown_exception("engine inspector error recorder query");
    }
#else
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine inspector error recorder query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_tensor_address_value(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, uintptr_t* out_address)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_address, "out_address");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_address = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context tensor address value query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_address = reinterpret_cast<uintptr_t>(context_payload->getTensorAddress(tensor_name));
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context tensor address value query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_output_tensor_address_value(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, uintptr_t* out_address)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_address, "out_address");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_address = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context output tensor address value query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_address = reinterpret_cast<uintptr_t>(context_payload->getOutputTensorAddress(tensor_name));
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context output tensor address value query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_clear_output_allocator(JYPPX_TensorRtExecutionContext* context, const char* tensor_name, JYPPX_Boolean* out_cleared)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_cleared, "out_cleared");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cleared = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context output allocator clear");
    if (status != JYPPX_STATUS_OK) { return status; }
    try
    {
        return trt11_execution_context_clear_output_allocator_with_seh_guard(
            context_payload,
            tensor_name,
            out_cleared,
            "execution context output allocator clear");
    }
    catch (const std::exception& exception)
    {
        *out_cleared = JYPPX_FALSE;
        return jyppx_trt11_callback_allocator_control_exception("execution context output allocator clear", exception);
    }
    catch (...)
    {
        *out_cleared = JYPPX_FALSE;
        return jyppx_trt11_callback_allocator_control_unknown_exception("execution context output allocator clear");
    }
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context output allocator clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_clear_temporary_storage_allocator(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_cleared)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_cleared, "out_cleared");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cleared = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context temporary storage allocator clear");
    if (status != JYPPX_STATUS_OK) { return status; }
    try
    {
        return trt11_execution_context_clear_temporary_storage_allocator_with_seh_guard(
            context_payload,
            out_cleared,
            "execution context temporary storage allocator clear");
    }
    catch (const std::exception& exception)
    {
        *out_cleared = JYPPX_FALSE;
        return jyppx_trt11_callback_allocator_control_exception("execution context temporary storage allocator clear", exception);
    }
    catch (...)
    {
        *out_cleared = JYPPX_FALSE;
        return jyppx_trt11_callback_allocator_control_unknown_exception("execution context temporary storage allocator clear");
    }
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context temporary storage allocator clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_clear_debug_listener(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_cleared)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_cleared, "out_cleared");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cleared = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context debug listener clear");
    if (status != JYPPX_STATUS_OK) { return status; }
    try
    {
        return trt11_execution_context_clear_debug_listener_with_seh_guard(
            context_payload,
            out_cleared,
            "execution context debug listener clear");
    }
    catch (const std::exception& exception)
    {
        *out_cleared = JYPPX_FALSE;
        return jyppx_trt11_callback_allocator_control_exception("execution context debug listener clear", exception);
    }
    catch (...)
    {
        *out_cleared = JYPPX_FALSE;
        return jyppx_trt11_callback_allocator_control_unknown_exception("execution context debug listener clear");
    }
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context debug listener clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_has_debug_listener(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_has_listener)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_listener, "out_has_listener");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_listener = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context debug listener query");
    if (status != JYPPX_STATUS_OK) { return status; }
    try
    {
        return trt11_execution_context_has_debug_listener_with_seh_guard(
            context_payload,
            out_has_listener,
            "execution context debug listener query");
    }
    catch (const std::exception& exception)
    {
        *out_has_listener = JYPPX_FALSE;
        return jyppx_trt11_callback_allocator_control_exception("execution context debug listener query", exception);
    }
    catch (...)
    {
        *out_has_listener = JYPPX_FALSE;
        return jyppx_trt11_callback_allocator_control_unknown_exception("execution context debug listener query");
    }
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context debug listener query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_clear_profiler(JYPPX_TensorRtExecutionContext* context)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    auto status = get_context_payload_ext(context, &context_payload, "execution context profiler clear");
    if (status != JYPPX_STATUS_OK) { return status; }
    try
    {
        return trt11_execution_context_clear_profiler_with_seh_guard(
            context_payload,
            "execution context profiler clear");
    }
    catch (const std::exception& exception)
    {
        return jyppx_trt11_callback_allocator_control_exception("execution context profiler clear", exception);
    }
    catch (...)
    {
        return jyppx_trt11_callback_allocator_control_unknown_exception("execution context profiler clear");
    }
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context profiler clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_has_profiler(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_has_profiler)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_profiler, "out_has_profiler");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_profiler = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context profiler query");
    if (status != JYPPX_STATUS_OK) { return status; }
    try
    {
        return trt11_execution_context_has_profiler_with_seh_guard(
            context_payload,
            out_has_profiler,
            "execution context profiler query");
    }
    catch (const std::exception& exception)
    {
        *out_has_profiler = JYPPX_FALSE;
        return jyppx_trt11_callback_allocator_control_exception("execution context profiler query", exception);
    }
    catch (...)
    {
        *out_has_profiler = JYPPX_FALSE;
        return jyppx_trt11_callback_allocator_control_unknown_exception("execution context profiler query");
    }
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context profiler query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_has_runtime_config(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_has_runtime_config)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_runtime_config, "out_has_runtime_config");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_runtime_config = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context runtime config query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_runtime_config = context_payload->getRuntimeConfig() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context runtime config query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_set_nvtx_verbosity(JYPPX_TensorRtExecutionContext* context, int32_t verbosity, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;
    if (verbosity < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context NVTX verbosity set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = context_payload->setNvtxVerbosity(static_cast<nvinfer1::ProfilingVerbosity>(verbosity)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context NVTX verbosity set");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_nvtx_verbosity(JYPPX_TensorRtExecutionContext* context, int32_t* out_verbosity)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_verbosity, "out_verbosity");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_verbosity = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context NVTX verbosity query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_verbosity = static_cast<int32_t>(context_payload->getNvtxVerbosity());
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context NVTX verbosity query");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_set_unfused_tensors_debug_state(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean debug_state, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context unfused tensors debug state set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = context_payload->setUnfusedTensorsDebugState(debug_state != JYPPX_FALSE) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    (void)debug_state;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context unfused tensors debug state set");
#endif
}

JYPPX_StatusCode jyppx_trt11_execution_context_get_unfused_tensors_debug_state(JYPPX_TensorRtExecutionContext* context, JYPPX_Boolean* out_debug_state)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_debug_state, "out_debug_state");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_debug_state = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context unfused tensors debug state query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_debug_state = context_payload->getUnfusedTensorsDebugState() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context unfused tensors debug state query");
#endif
}
