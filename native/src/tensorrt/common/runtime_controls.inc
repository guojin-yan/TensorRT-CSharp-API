#ifndef JYPPX_TRT_RUNTIME_API
#error "JYPPX_TRT_RUNTIME_API must be defined before including runtime_controls.inc"
#endif

static void reset_error_recorder_snapshot_info(JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info)
{
    if (out_info == nullptr)
    {
        return;
    }

    out_info->has_recorder = JYPPX_FALSE;
    out_info->error_count = 0;
    out_info->has_overflowed = JYPPX_FALSE;
    out_info->interface_info_available = JYPPX_FALSE;
    out_info->interface_info_major = 0;
    out_info->interface_info_minor = 0;
    out_info->interface_info_kind[0] = '\0';
}

static void reset_error_record_info(JYPPX_TensorRtErrorRecordInfo* out_error)
{
    if (out_error == nullptr)
    {
        return;
    }

    out_error->index = -1;
    out_error->code = 0;
    out_error->description[0] = '\0';
}

#if JYPPX_HAS_TENSORRT
static JYPPX_StatusCode jyppx_trt_runtime_control_exception(const char* feature_name, const std::exception& exception)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
}

static JYPPX_StatusCode jyppx_trt_runtime_control_unknown_exception(const char* feature_name)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
}

static void runtime_copy_error_recorder_interface_info(
    nvinfer1::IErrorRecorder* recorder,
    JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info)
{
    if (recorder == nullptr || out_info == nullptr)
    {
        return;
    }

#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    const auto interface_info = recorder->getInterfaceInfo();
    out_info->interface_info_available = JYPPX_TRUE;
    out_info->interface_info_major = static_cast<int32_t>(interface_info.major);
    out_info->interface_info_minor = static_cast<int32_t>(interface_info.minor);
    copy_c_string(interface_info.kind, out_info->interface_info_kind, sizeof(out_info->interface_info_kind));
#else
    (void)recorder;
#endif
}

static JYPPX_StatusCode runtime_has_error_recorder_with_seh_guard(
    nvinfer1::IRuntime* runtime_payload,
    JYPPX_Boolean* out_has_recorder,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_recorder = runtime_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_recorder = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_recorder = runtime_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode runtime_has_logger_with_seh_guard(
    nvinfer1::IRuntime* runtime_payload,
    JYPPX_Boolean* out_has_logger,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_logger = runtime_payload->getLogger() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_logger = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_logger = runtime_payload->getLogger() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode runtime_get_error_recorder_snapshot_info_with_seh_guard(
    nvinfer1::IRuntime* runtime_payload,
    JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        auto* recorder = runtime_payload->getErrorRecorder();
        if (recorder == nullptr)
        {
            return JYPPX_STATUS_OK;
        }

        out_info->has_recorder = JYPPX_TRUE;
        const int32_t error_count = recorder->getNbErrors();
        out_info->error_count = error_count > 0 ? error_count : 0;
        out_info->has_overflowed = recorder->hasOverflowed() ? JYPPX_TRUE : JYPPX_FALSE;
        runtime_copy_error_recorder_interface_info(recorder, out_info);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        reset_error_recorder_snapshot_info(out_info);
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    auto* recorder = runtime_payload->getErrorRecorder();
    if (recorder == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    out_info->has_recorder = JYPPX_TRUE;
    const int32_t error_count = recorder->getNbErrors();
    out_info->error_count = error_count > 0 ? error_count : 0;
    out_info->has_overflowed = recorder->hasOverflowed() ? JYPPX_TRUE : JYPPX_FALSE;
    runtime_copy_error_recorder_interface_info(recorder, out_info);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode runtime_get_error_recorder_error_with_seh_guard(
    nvinfer1::IRuntime* runtime_payload,
    const int32_t index,
    JYPPX_TensorRtErrorRecordInfo* out_error,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        auto* recorder = runtime_payload->getErrorRecorder();
        if (recorder == nullptr)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Runtime error recorder snapshot is unavailable because no recorder is attached.");
            return JYPPX_STATUS_INVALID_STATE;
        }

        const int32_t error_count = recorder->getNbErrors();
        if (index < 0 || index >= error_count)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Runtime error recorder error index is out of range.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }

        out_error->index = index;
        out_error->code = static_cast<int32_t>(recorder->getErrorCode(index));
        copy_c_string(recorder->getErrorDesc(index), out_error->description, sizeof(out_error->description));
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        reset_error_record_info(out_error);
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    auto* recorder = runtime_payload->getErrorRecorder();
    if (recorder == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Runtime error recorder snapshot is unavailable because no recorder is attached.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const int32_t error_count = recorder->getNbErrors();
    if (index < 0 || index >= error_count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Runtime error recorder error index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    out_error->index = index;
    out_error->code = static_cast<int32_t>(recorder->getErrorCode(index));
    copy_c_string(recorder->getErrorDesc(index), out_error->description, sizeof(out_error->description));
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode runtime_clear_error_recorder_with_seh_guard(
    nvinfer1::IRuntime* runtime_payload,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        runtime_payload->setErrorRecorder(nullptr);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    runtime_payload->setErrorRecorder(nullptr);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode runtime_clear_gpu_allocator_with_seh_guard(
    nvinfer1::IRuntime* runtime_payload,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        runtime_payload->setGpuAllocator(nullptr);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    runtime_payload->setGpuAllocator(nullptr);
    return JYPPX_STATUS_OK;
#endif
}
#endif

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_set_dla_core)(JYPPX_TensorRtRuntime* runtime, int32_t dla_core)
{
#if JYPPX_HAS_TENSORRT
    auto status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    runtime_payload->setDLACore(dla_core);
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    (void)dla_core;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime DLA core set");
#endif
}

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_get_dla_core)(JYPPX_TensorRtRuntime* runtime, int32_t* out_dla_core)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_dla_core, "out_dla_core");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_dla_core = -1;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_dla_core = runtime_payload->getDLACore();
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime DLA core query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_get_dla_core_count)(JYPPX_TensorRtRuntime* runtime, int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_count = runtime_payload->getNbDLACores();
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime DLA core count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_set_max_threads)(JYPPX_TensorRtRuntime* runtime, int32_t max_threads, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;
    if (max_threads < 1)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Runtime max_threads must be greater than or equal to 1.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_set = runtime_payload->setMaxThreads(max_threads) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime max threads set");
#endif
}

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_get_max_threads)(JYPPX_TensorRtRuntime* runtime, int32_t* out_max_threads)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_max_threads, "out_max_threads");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_max_threads = 0;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_max_threads = runtime_payload->getMaxThreads();
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime max threads query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_set_temporary_directory)(JYPPX_TensorRtRuntime* runtime, const char* directory_path)
{
    auto status = validate_c_string(directory_path, "directory_path");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    runtime_payload->setTemporaryDirectory(directory_path);
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime temporary directory set");
#endif
}

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_get_temporary_directory)(JYPPX_TensorRtRuntime* runtime, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    return copy_string_to_buffer(runtime_payload->getTemporaryDirectory(), output_buffer, output_buffer_size, out_required_size);
#else
    (void)runtime;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime temporary directory query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_clear_temporary_directory)(JYPPX_TensorRtRuntime* runtime)
{
#if JYPPX_HAS_TENSORRT
    auto status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    runtime_payload->setTemporaryDirectory(nullptr);
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime temporary directory clear");
#endif
}

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_set_tempfile_control_flags)(JYPPX_TensorRtRuntime* runtime, uint32_t flags)
{
#if JYPPX_HAS_TENSORRT
    auto status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    runtime_payload->setTempfileControlFlags(static_cast<nvinfer1::TempfileControlFlags>(flags));
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    (void)flags;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime tempfile control flags set");
#endif
}

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_get_tempfile_control_flags)(JYPPX_TensorRtRuntime* runtime, uint32_t* out_flags)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_flags, "out_flags");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_flags = 0;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_flags = static_cast<uint32_t>(runtime_payload->getTempfileControlFlags());
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime tempfile control flags query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_set_engine_host_code_allowed)(JYPPX_TensorRtRuntime* runtime, JYPPX_Boolean allowed)
{
#if JYPPX_HAS_TENSORRT
    auto status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    runtime_payload->setEngineHostCodeAllowed(allowed != JYPPX_FALSE);
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    (void)allowed;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime engine host code permission set");
#endif
}

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_get_engine_host_code_allowed)(JYPPX_TensorRtRuntime* runtime, JYPPX_Boolean* out_allowed)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_allowed, "out_allowed");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_allowed = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_allowed = runtime_payload->getEngineHostCodeAllowed() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime engine host code permission query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_has_logger)(JYPPX_TensorRtRuntime* runtime, JYPPX_Boolean* out_has_logger)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_logger, "out_has_logger");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_logger = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    try
    {
        return runtime_has_logger_with_seh_guard(runtime_payload, out_has_logger, "runtime logger presence query");
    }
    catch (const std::exception& exception)
    {
        *out_has_logger = JYPPX_FALSE;
        return jyppx_trt_runtime_control_exception("runtime logger presence query", exception);
    }
    catch (...)
    {
        *out_has_logger = JYPPX_FALSE;
        return jyppx_trt_runtime_control_unknown_exception("runtime logger presence query");
    }
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime logger presence query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_has_error_recorder)(JYPPX_TensorRtRuntime* runtime, JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    try
    {
        return runtime_has_error_recorder_with_seh_guard(runtime_payload, out_has_recorder, "runtime error recorder query");
    }
    catch (const std::exception& exception)
    {
        *out_has_recorder = JYPPX_FALSE;
        return jyppx_trt_runtime_control_exception("runtime error recorder query", exception);
    }
    catch (...)
    {
        *out_has_recorder = JYPPX_FALSE;
        return jyppx_trt_runtime_control_unknown_exception("runtime error recorder query");
    }
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime error recorder query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_get_error_recorder_snapshot_info)(
    JYPPX_TensorRtRuntime* runtime,
    JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK) { return status; }
    reset_error_recorder_snapshot_info(out_info);
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    try
    {
        return runtime_get_error_recorder_snapshot_info_with_seh_guard(runtime_payload, out_info, "runtime error recorder snapshot query");
    }
    catch (const std::exception& exception)
    {
        reset_error_recorder_snapshot_info(out_info);
        return jyppx_trt_runtime_control_exception("runtime error recorder snapshot query", exception);
    }
    catch (...)
    {
        reset_error_recorder_snapshot_info(out_info);
        return jyppx_trt_runtime_control_unknown_exception("runtime error recorder snapshot query");
    }
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime error recorder snapshot query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_get_error_recorder_error)(
    JYPPX_TensorRtRuntime* runtime,
    int32_t index,
    JYPPX_TensorRtErrorRecordInfo* out_error)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_error, "out_error");
    if (status != JYPPX_STATUS_OK) { return status; }
    reset_error_record_info(out_error);
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    try
    {
        return runtime_get_error_recorder_error_with_seh_guard(runtime_payload, index, out_error, "runtime error recorder error copy");
    }
    catch (const std::exception& exception)
    {
        reset_error_record_info(out_error);
        return jyppx_trt_runtime_control_exception("runtime error recorder error copy", exception);
    }
    catch (...)
    {
        reset_error_record_info(out_error);
        return jyppx_trt_runtime_control_unknown_exception("runtime error recorder error copy");
    }
#else
    (void)runtime;
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime error recorder error copy");
#endif
}

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_clear_error_recorder)(JYPPX_TensorRtRuntime* runtime)
{
#if JYPPX_HAS_TENSORRT
    auto status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    try
    {
        return runtime_clear_error_recorder_with_seh_guard(runtime_payload, "runtime error recorder clear");
    }
    catch (const std::exception& exception)
    {
        return jyppx_trt_runtime_control_exception("runtime error recorder clear", exception);
    }
    catch (...)
    {
        return jyppx_trt_runtime_control_unknown_exception("runtime error recorder clear");
    }
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime error recorder clear");
#endif
}

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_clear_gpu_allocator)(JYPPX_TensorRtRuntime* runtime)
{
#if JYPPX_HAS_TENSORRT
    auto status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    try
    {
        return runtime_clear_gpu_allocator_with_seh_guard(runtime_payload, "runtime GPU allocator clear");
    }
    catch (const std::exception& exception)
    {
        return jyppx_trt_runtime_control_exception("runtime GPU allocator clear", exception);
    }
    catch (...)
    {
        return jyppx_trt_runtime_control_unknown_exception("runtime GPU allocator clear");
    }
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime GPU allocator clear");
#endif
}
