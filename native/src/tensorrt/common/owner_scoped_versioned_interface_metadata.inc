#ifndef JYPPX_TRT_OWNER_SCOPED_VERSIONED_METADATA_API
#error "JYPPX_TRT_OWNER_SCOPED_VERSIONED_METADATA_API must be defined before including owner_scoped_versioned_interface_metadata.inc"
#endif

static void owner_scoped_versioned_metadata_reset(
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor,
    int32_t* out_api_language)
{
    if (out_required_size != nullptr) { *out_required_size = 0; }
    if (out_major != nullptr) { *out_major = 0; }
    if (out_minor != nullptr) { *out_minor = 0; }
    if (out_api_language != nullptr) { *out_api_language = -1; }
}

static JYPPX_StatusCode owner_scoped_versioned_metadata_validate_outputs(
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor,
    int32_t* out_api_language)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_major, "out_major");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_minor, "out_minor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_api_language, "out_api_language");
    if (status != JYPPX_STATUS_OK) { return status; }
    owner_scoped_versioned_metadata_reset(out_required_size, out_major, out_minor, out_api_language);
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode owner_scoped_versioned_metadata_validate_api_language_output(int32_t* out_api_language)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_api_language, "out_api_language");
    if (status == JYPPX_STATUS_OK) { *out_api_language = -1; }
    return status;
}

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
template <typename TOwner, typename TInterface>
using owner_scoped_versioned_metadata_getter = TInterface* (*)(TOwner* owner);

static nvinfer1::IErrorRecorder* owner_scoped_runtime_error_recorder(nvinfer1::IRuntime* owner)
{
    return owner->getErrorRecorder();
}

static nvinfer1::IErrorRecorder* owner_scoped_refitter_error_recorder(nvinfer1::IRefitter* owner)
{
    return owner->getErrorRecorder();
}

static nvinfer1::IErrorRecorder* owner_scoped_engine_error_recorder(nvinfer1::ICudaEngine* owner)
{
    return owner->getErrorRecorder();
}

static nvinfer1::IErrorRecorder* owner_scoped_execution_context_error_recorder(nvinfer1::IExecutionContext* owner)
{
    return owner->getErrorRecorder();
}

static nvinfer1::IErrorRecorder* owner_scoped_builder_error_recorder(nvinfer1::IBuilder* owner)
{
    return owner->getErrorRecorder();
}

static nvinfer1::IErrorRecorder* owner_scoped_network_error_recorder(nvinfer1::INetworkDefinition* owner)
{
    return owner->getErrorRecorder();
}

static nvinfer1::IErrorRecorder* owner_scoped_engine_inspector_error_recorder(nvinfer1::IEngineInspector* owner)
{
    return owner->getErrorRecorder();
}

static nvinfer1::IProgressMonitor* owner_scoped_builder_config_progress_monitor(nvinfer1::IBuilderConfig* owner)
{
    return owner->getProgressMonitor();
}

static nvinfer1::IGpuAllocator* owner_scoped_execution_context_temporary_storage_allocator(
    nvinfer1::IExecutionContext* owner)
{
    return owner->getTemporaryStorageAllocator();
}

static nvinfer1::IDebugListener* owner_scoped_execution_context_debug_listener(nvinfer1::IExecutionContext* owner)
{
    return owner->getDebugListener();
}

template <typename TInterface>
static JYPPX_StatusCode owner_scoped_copy_versioned_metadata(
    TInterface* object,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor,
    int32_t* out_api_language,
    const char* missing_message)
{
    if (object == nullptr)
    {
        owner_scoped_versioned_metadata_reset(out_required_size, out_major, out_minor, out_api_language);
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, missing_message);
        return JYPPX_STATUS_NOT_FOUND;
    }

    const auto info = object->getInterfaceInfo();
    *out_api_language = static_cast<int32_t>(object->getAPILanguage());
    return copy_interface_info_to_buffer(
        info,
        output_buffer,
        output_buffer_size,
        out_required_size,
        out_major,
        out_minor);
}

template <typename TInterface>
static JYPPX_StatusCode owner_scoped_copy_api_language(
    TInterface* object,
    int32_t* out_api_language,
    const char* missing_message)
{
    if (object == nullptr)
    {
        *out_api_language = -1;
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, missing_message);
        return JYPPX_STATUS_NOT_FOUND;
    }

    *out_api_language = static_cast<int32_t>(object->getAPILanguage());
    return JYPPX_STATUS_OK;
}

template <typename TOwner, typename TInterface>
static JYPPX_StatusCode owner_scoped_query_versioned_metadata_with_seh_guard(
    TOwner* owner,
    owner_scoped_versioned_metadata_getter<TOwner, TInterface> getter,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor,
    int32_t* out_api_language,
    const char* feature_name,
    const char* missing_message)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        return owner_scoped_copy_versioned_metadata(
            getter(owner),
            output_buffer,
            output_buffer_size,
            out_required_size,
            out_major,
            out_minor,
            out_api_language,
            missing_message);
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        owner_scoped_versioned_metadata_reset(out_required_size, out_major, out_minor, out_api_language);
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    return owner_scoped_copy_versioned_metadata(
        getter(owner),
        output_buffer,
        output_buffer_size,
        out_required_size,
        out_major,
        out_minor,
        out_api_language,
        missing_message);
#endif
}

template <typename TOwner, typename TInterface>
static JYPPX_StatusCode owner_scoped_query_api_language_with_seh_guard(
    TOwner* owner,
    owner_scoped_versioned_metadata_getter<TOwner, TInterface> getter,
    int32_t* out_api_language,
    const char* feature_name,
    const char* missing_message)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        return owner_scoped_copy_api_language(getter(owner), out_api_language, missing_message);
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_api_language = -1;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    return owner_scoped_copy_api_language(getter(owner), out_api_language, missing_message);
#endif
}

template <typename TOwner, typename TInterface>
static JYPPX_StatusCode owner_scoped_query_versioned_metadata(
    JYPPX_TensorRtObjectBase* handle,
    const JYPPX_TensorRtObjectKind expected_kind,
    const char* handle_name,
    owner_scoped_versioned_metadata_getter<TOwner, TInterface> getter,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor,
    int32_t* out_api_language,
    const char* feature_name,
    const char* missing_message)
{
    auto status = jyppx::tensorrt::validate_handle(handle, kLine, expected_kind, handle_name);
    if (status != JYPPX_STATUS_OK) { return status; }

    auto* owner = get_payload<TOwner>(handle);
    if (owner == nullptr)
    {
        std::ostringstream message;
        message << handle_name << " handle does not carry the expected TensorRT payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, message.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    try
    {
        return owner_scoped_query_versioned_metadata_with_seh_guard(
            owner,
            getter,
            output_buffer,
            output_buffer_size,
            out_required_size,
            out_major,
            out_minor,
            out_api_language,
            feature_name,
            missing_message);
    }
    catch (const std::exception& exception)
    {
        owner_scoped_versioned_metadata_reset(out_required_size, out_major, out_minor, out_api_language);
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        owner_scoped_versioned_metadata_reset(out_required_size, out_major, out_minor, out_api_language);
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}

template <typename TOwner, typename TInterface>
static JYPPX_StatusCode owner_scoped_query_api_language(
    JYPPX_TensorRtObjectBase* handle,
    const JYPPX_TensorRtObjectKind expected_kind,
    const char* handle_name,
    owner_scoped_versioned_metadata_getter<TOwner, TInterface> getter,
    int32_t* out_api_language,
    const char* feature_name,
    const char* missing_message)
{
    auto status = jyppx::tensorrt::validate_handle(handle, kLine, expected_kind, handle_name);
    if (status != JYPPX_STATUS_OK) { return status; }

    auto* owner = get_payload<TOwner>(handle);
    if (owner == nullptr)
    {
        std::ostringstream message;
        message << handle_name << " handle does not carry the expected TensorRT payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, message.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    try
    {
        return owner_scoped_query_api_language_with_seh_guard(
            owner,
            getter,
            out_api_language,
            feature_name,
            missing_message);
    }
    catch (const std::exception& exception)
    {
        *out_api_language = -1;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
    }
    catch (...)
    {
        *out_api_language = -1;
        return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
    }
}

static JYPPX_StatusCode owner_scoped_output_allocator_api_language_with_seh_guard(
    nvinfer1::IExecutionContext* context,
    const char* tensor_name,
    int32_t* out_api_language,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        return owner_scoped_copy_api_language(
            context->getOutputAllocator(tensor_name),
            out_api_language,
            "Execution context output allocator is not set for the requested tensor.");
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_api_language = -1;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    return owner_scoped_copy_api_language(
        context->getOutputAllocator(tensor_name),
        out_api_language,
        "Execution context output allocator is not set for the requested tensor.");
#endif
}
#endif

#define JYPPX_OWNER_SCOPED_METADATA_EXPORT(function_name, handle_type, handle_name, object_kind, owner_type, interface_type, getter, feature_name, missing_message) \
JYPPX_StatusCode JYPPX_TRT_OWNER_SCOPED_VERSIONED_METADATA_API(function_name)( \
    handle_type* handle, \
    char* output_buffer, \
    size_t output_buffer_size, \
    size_t* out_required_size, \
    int32_t* out_major, \
    int32_t* out_minor, \
    int32_t* out_api_language) \
{ \
    auto status = owner_scoped_versioned_metadata_validate_outputs(out_required_size, out_major, out_minor, out_api_language); \
    if (status != JYPPX_STATUS_OK) { return status; } \
    /* The borrowed interface is queried and copied only while the owning handle is validated and alive. */ \
    /* No native interface pointer is returned or retained. */ \
    /* NOLINTNEXTLINE(bugprone-macro-parentheses) */ \
    JYPPX_OWNER_SCOPED_METADATA_VENDOR_BODY(handle, handle_name, object_kind, owner_type, interface_type, getter, output_buffer, output_buffer_size, out_required_size, out_major, out_minor, out_api_language, feature_name, missing_message) \
}

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
#define JYPPX_OWNER_SCOPED_METADATA_VENDOR_BODY(handle, handle_name, object_kind, owner_type, interface_type, getter, output_buffer, output_buffer_size, out_required_size, out_major, out_minor, out_api_language, feature_name, missing_message) \
    return owner_scoped_query_versioned_metadata<owner_type, interface_type>(handle, object_kind, handle_name, getter, output_buffer, output_buffer_size, out_required_size, out_major, out_minor, out_api_language, feature_name, missing_message);
#elif JYPPX_HAS_TENSORRT
#define JYPPX_OWNER_SCOPED_METADATA_VENDOR_BODY(handle, handle_name, object_kind, owner_type, interface_type, getter, output_buffer, output_buffer_size, out_required_size, out_major, out_minor, out_api_language, feature_name, missing_message) \
    (void)handle; (void)output_buffer; (void)output_buffer_size; return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, feature_name);
#else
#define JYPPX_OWNER_SCOPED_METADATA_VENDOR_BODY(handle, handle_name, object_kind, owner_type, interface_type, getter, output_buffer, output_buffer_size, out_required_size, out_major, out_minor, out_api_language, feature_name, missing_message) \
    (void)handle; (void)output_buffer; (void)output_buffer_size; return jyppx::tensorrt::report_vendor_missing(kLine, feature_name);
#endif

JYPPX_OWNER_SCOPED_METADATA_EXPORT(
    runtime_get_error_recorder_versioned_metadata,
    JYPPX_TensorRtRuntime,
    "runtime",
    JYPPX_TENSORRT_OBJECT_KIND_RUNTIME,
    nvinfer1::IRuntime,
    nvinfer1::IErrorRecorder,
    owner_scoped_runtime_error_recorder,
    "runtime error recorder versioned metadata query",
    "Runtime error recorder is not set.")

JYPPX_OWNER_SCOPED_METADATA_EXPORT(
    refitter_get_error_recorder_versioned_metadata,
    JYPPX_TensorRtRefitter,
    "refitter",
    JYPPX_TENSORRT_OBJECT_KIND_REFITTER,
    nvinfer1::IRefitter,
    nvinfer1::IErrorRecorder,
    owner_scoped_refitter_error_recorder,
    "refitter error recorder versioned metadata query",
    "Refitter error recorder is not set.")

JYPPX_OWNER_SCOPED_METADATA_EXPORT(
    engine_get_error_recorder_versioned_metadata,
    JYPPX_TensorRtCudaEngine,
    "engine",
    JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE,
    nvinfer1::ICudaEngine,
    nvinfer1::IErrorRecorder,
    owner_scoped_engine_error_recorder,
    "engine error recorder versioned metadata query",
    "Engine error recorder is not set.")

JYPPX_OWNER_SCOPED_METADATA_EXPORT(
    execution_context_get_error_recorder_versioned_metadata,
    JYPPX_TensorRtExecutionContext,
    "context",
    JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT,
    nvinfer1::IExecutionContext,
    nvinfer1::IErrorRecorder,
    owner_scoped_execution_context_error_recorder,
    "execution context error recorder versioned metadata query",
    "Execution context error recorder is not set.")

JYPPX_OWNER_SCOPED_METADATA_EXPORT(
    builder_get_error_recorder_versioned_metadata,
    JYPPX_TensorRtBuilder,
    "builder",
    JYPPX_TENSORRT_OBJECT_KIND_BUILDER,
    nvinfer1::IBuilder,
    nvinfer1::IErrorRecorder,
    owner_scoped_builder_error_recorder,
    "builder error recorder versioned metadata query",
    "Builder error recorder is not set.")

JYPPX_OWNER_SCOPED_METADATA_EXPORT(
    network_get_error_recorder_versioned_metadata,
    JYPPX_TensorRtNetworkDefinition,
    "network",
    JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION,
    nvinfer1::INetworkDefinition,
    nvinfer1::IErrorRecorder,
    owner_scoped_network_error_recorder,
    "network error recorder versioned metadata query",
    "Network error recorder is not set.")

JYPPX_OWNER_SCOPED_METADATA_EXPORT(
    engine_inspector_get_error_recorder_versioned_metadata,
    JYPPX_TensorRtEngineInspector,
    "inspector",
    JYPPX_TENSORRT_OBJECT_KIND_ENGINE_INSPECTOR,
    nvinfer1::IEngineInspector,
    nvinfer1::IErrorRecorder,
    owner_scoped_engine_inspector_error_recorder,
    "engine inspector error recorder versioned metadata query",
    "Engine inspector error recorder is not set.")

JYPPX_OWNER_SCOPED_METADATA_EXPORT(
    builder_config_get_progress_monitor_versioned_metadata,
    JYPPX_TensorRtBuilderConfig,
    "config",
    JYPPX_TENSORRT_OBJECT_KIND_BUILDER_CONFIG,
    nvinfer1::IBuilderConfig,
    nvinfer1::IProgressMonitor,
    owner_scoped_builder_config_progress_monitor,
    "builder config progress monitor versioned metadata query",
    "Builder config progress monitor is not set.")

#undef JYPPX_OWNER_SCOPED_METADATA_VENDOR_BODY
#undef JYPPX_OWNER_SCOPED_METADATA_EXPORT

JYPPX_StatusCode JYPPX_TRT_OWNER_SCOPED_VERSIONED_METADATA_API(execution_context_get_output_allocator_api_language)(
    JYPPX_TensorRtExecutionContext* context,
    const char* tensor_name,
    int32_t* out_api_language)
{
    auto status = owner_scoped_versioned_metadata_validate_api_language_output(out_api_language);
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    status = jyppx::tensorrt::validate_handle(
        context,
        kLine,
        JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT,
        "context");
    if (status != JYPPX_STATUS_OK) { return status; }

    auto* owner = get_payload<nvinfer1::IExecutionContext>(context);
    if (owner == nullptr)
    {
        jyppx::common::set_last_error(
            JYPPX_ERROR_CATEGORY_TENSORRT,
            "Execution context handle does not carry the expected TensorRT payload for output allocator API language query.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    try
    {
        return owner_scoped_output_allocator_api_language_with_seh_guard(
            owner,
            tensor_name,
            out_api_language,
            "execution context output allocator API language query");
    }
    catch (const std::exception& exception)
    {
        *out_api_language = -1;
        return jyppx::tensorrt::report_vendor_exception(
            kLine,
            "execution context output allocator API language query",
            exception.what());
    }
    catch (...)
    {
        *out_api_language = -1;
        return jyppx::tensorrt::report_vendor_exception(
            kLine,
            "execution context output allocator API language query",
            "unknown native exception");
    }
#elif JYPPX_HAS_TENSORRT
    (void)context;
    return jyppx::tensorrt::report_vendor_mismatch(
        kLine,
        JYPPX_TENSORRT_VERSION_MAJOR_NUM,
        "execution context output allocator API language query");
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context output allocator API language query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_OWNER_SCOPED_VERSIONED_METADATA_API(execution_context_get_temporary_storage_allocator_api_language)(
    JYPPX_TensorRtExecutionContext* context,
    int32_t* out_api_language)
{
    auto status = owner_scoped_versioned_metadata_validate_api_language_output(out_api_language);
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    return owner_scoped_query_api_language<nvinfer1::IExecutionContext, nvinfer1::IGpuAllocator>(
        context,
        JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT,
        "context",
        owner_scoped_execution_context_temporary_storage_allocator,
        out_api_language,
        "execution context temporary storage allocator API language query",
        "Execution context temporary storage allocator is not set.");
#elif JYPPX_HAS_TENSORRT
    (void)context;
    return jyppx::tensorrt::report_vendor_mismatch(
        kLine,
        JYPPX_TENSORRT_VERSION_MAJOR_NUM,
        "execution context temporary storage allocator API language query");
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(
        kLine,
        "execution context temporary storage allocator API language query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_OWNER_SCOPED_VERSIONED_METADATA_API(execution_context_get_debug_listener_api_language)(
    JYPPX_TensorRtExecutionContext* context,
    int32_t* out_api_language)
{
    auto status = owner_scoped_versioned_metadata_validate_api_language_output(out_api_language);
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    return owner_scoped_query_api_language<nvinfer1::IExecutionContext, nvinfer1::IDebugListener>(
        context,
        JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT,
        "context",
        owner_scoped_execution_context_debug_listener,
        out_api_language,
        "execution context debug listener API language query",
        "Execution context debug listener is not set.");
#elif JYPPX_HAS_TENSORRT
    (void)context;
    return jyppx::tensorrt::report_vendor_mismatch(
        kLine,
        JYPPX_TENSORRT_VERSION_MAJOR_NUM,
        "execution context debug listener API language query");
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context debug listener API language query");
#endif
}
