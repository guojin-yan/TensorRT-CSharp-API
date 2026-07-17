static void trt11_reset_runtime_error_recorder_snapshot_info(JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info)
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

static void trt11_reset_runtime_error_record_info(JYPPX_TensorRtErrorRecordInfo* out_error)
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
static JYPPX_StatusCode trt11_runtime_control_exception(const char* feature_name, const std::exception& exception)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
}

static JYPPX_StatusCode trt11_runtime_control_unknown_exception(const char* feature_name)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
}

static void trt11_copy_runtime_error_recorder_interface_info(
    nvinfer1::IErrorRecorder* recorder,
    JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info)
{
    if (recorder == nullptr || out_info == nullptr)
    {
        return;
    }

    const auto interface_info = recorder->getInterfaceInfo();
    out_info->interface_info_available = JYPPX_TRUE;
    out_info->interface_info_major = static_cast<int32_t>(interface_info.major);
    out_info->interface_info_minor = static_cast<int32_t>(interface_info.minor);
    copy_c_string(interface_info.kind, out_info->interface_info_kind, sizeof(out_info->interface_info_kind));
}

static JYPPX_StatusCode trt11_runtime_get_error_recorder_snapshot_info_with_seh_guard(
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
        trt11_copy_runtime_error_recorder_interface_info(recorder, out_info);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        trt11_reset_runtime_error_recorder_snapshot_info(out_info);
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
    trt11_copy_runtime_error_recorder_interface_info(recorder, out_info);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode trt11_runtime_get_error_recorder_error_with_seh_guard(
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
        trt11_reset_runtime_error_record_info(out_error);
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

static JYPPX_StatusCode trt11_refitter_get_error_recorder_snapshot_info_with_seh_guard(
    nvinfer1::IRefitter* refitter_payload,
    JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        auto* recorder = refitter_payload->getErrorRecorder();
        if (recorder == nullptr)
        {
            return JYPPX_STATUS_OK;
        }

        out_info->has_recorder = JYPPX_TRUE;
        const int32_t error_count = recorder->getNbErrors();
        out_info->error_count = error_count > 0 ? error_count : 0;
        out_info->has_overflowed = recorder->hasOverflowed() ? JYPPX_TRUE : JYPPX_FALSE;
        trt11_copy_runtime_error_recorder_interface_info(recorder, out_info);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        trt11_reset_runtime_error_recorder_snapshot_info(out_info);
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    auto* recorder = refitter_payload->getErrorRecorder();
    if (recorder == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    out_info->has_recorder = JYPPX_TRUE;
    const int32_t error_count = recorder->getNbErrors();
    out_info->error_count = error_count > 0 ? error_count : 0;
    out_info->has_overflowed = recorder->hasOverflowed() ? JYPPX_TRUE : JYPPX_FALSE;
    trt11_copy_runtime_error_recorder_interface_info(recorder, out_info);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode trt11_refitter_get_error_recorder_error_with_seh_guard(
    nvinfer1::IRefitter* refitter_payload,
    const int32_t index,
    JYPPX_TensorRtErrorRecordInfo* out_error,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        auto* recorder = refitter_payload->getErrorRecorder();
        if (recorder == nullptr)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter error recorder snapshot is unavailable because no recorder is attached.");
            return JYPPX_STATUS_INVALID_STATE;
        }

        const int32_t error_count = recorder->getNbErrors();
        if (index < 0 || index >= error_count)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter error recorder error index is out of range.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }

        out_error->index = index;
        out_error->code = static_cast<int32_t>(recorder->getErrorCode(index));
        copy_c_string(recorder->getErrorDesc(index), out_error->description, sizeof(out_error->description));
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        trt11_reset_runtime_error_record_info(out_error);
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    auto* recorder = refitter_payload->getErrorRecorder();
    if (recorder == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter error recorder snapshot is unavailable because no recorder is attached.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const int32_t error_count = recorder->getNbErrors();
    if (index < 0 || index >= error_count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter error recorder error index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    out_error->index = index;
    out_error->code = static_cast<int32_t>(recorder->getErrorCode(index));
    copy_c_string(recorder->getErrorDesc(index), out_error->description, sizeof(out_error->description));
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode trt11_runtime_has_logger_with_seh_guard(
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
#endif

JYPPX_StatusCode jyppx_trt11_runtime_set_dla_core(JYPPX_TensorRtRuntime* runtime, int32_t dla_core)
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

JYPPX_StatusCode jyppx_trt11_runtime_get_dla_core(JYPPX_TensorRtRuntime* runtime, int32_t* out_dla_core)
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

JYPPX_StatusCode jyppx_trt11_runtime_get_dla_core_count(JYPPX_TensorRtRuntime* runtime, int32_t* out_count)
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

JYPPX_StatusCode jyppx_trt11_runtime_set_max_threads(JYPPX_TensorRtRuntime* runtime, int32_t max_threads, JYPPX_Boolean* out_set)
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

JYPPX_StatusCode jyppx_trt11_runtime_get_max_threads(JYPPX_TensorRtRuntime* runtime, int32_t* out_max_threads)
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

JYPPX_StatusCode jyppx_trt11_runtime_set_temporary_directory(JYPPX_TensorRtRuntime* runtime, const char* directory_path)
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

JYPPX_StatusCode jyppx_trt11_runtime_get_temporary_directory(JYPPX_TensorRtRuntime* runtime, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
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

JYPPX_StatusCode jyppx_trt11_runtime_clear_temporary_directory(JYPPX_TensorRtRuntime* runtime)
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

JYPPX_StatusCode jyppx_trt11_runtime_set_tempfile_control_flags(JYPPX_TensorRtRuntime* runtime, uint32_t flags)
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

JYPPX_StatusCode jyppx_trt11_runtime_get_tempfile_control_flags(JYPPX_TensorRtRuntime* runtime, uint32_t* out_flags)
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

JYPPX_StatusCode jyppx_trt11_runtime_set_engine_host_code_allowed(JYPPX_TensorRtRuntime* runtime, JYPPX_Boolean allowed)
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

JYPPX_StatusCode jyppx_trt11_runtime_get_engine_host_code_allowed(JYPPX_TensorRtRuntime* runtime, JYPPX_Boolean* out_allowed)
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

JYPPX_StatusCode jyppx_trt11_runtime_has_logger(JYPPX_TensorRtRuntime* runtime, JYPPX_Boolean* out_has_logger)
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
        return trt11_runtime_has_logger_with_seh_guard(runtime_payload, out_has_logger, "runtime logger presence query");
    }
    catch (const std::exception& exception)
    {
        *out_has_logger = JYPPX_FALSE;
        return trt11_runtime_control_exception("runtime logger presence query", exception);
    }
    catch (...)
    {
        *out_has_logger = JYPPX_FALSE;
        return trt11_runtime_control_unknown_exception("runtime logger presence query");
    }
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime logger presence query");
#endif
}

JYPPX_StatusCode jyppx_trt11_runtime_has_error_recorder(JYPPX_TensorRtRuntime* runtime, JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_has_recorder = runtime_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime error recorder query");
#endif
}

JYPPX_StatusCode jyppx_trt11_runtime_get_error_recorder_snapshot_info(
    JYPPX_TensorRtRuntime* runtime,
    JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK) { return status; }
    trt11_reset_runtime_error_recorder_snapshot_info(out_info);
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    try
    {
        return trt11_runtime_get_error_recorder_snapshot_info_with_seh_guard(runtime_payload, out_info, "runtime error recorder snapshot query");
    }
    catch (const std::exception& exception)
    {
        trt11_reset_runtime_error_recorder_snapshot_info(out_info);
        return trt11_runtime_control_exception("runtime error recorder snapshot query", exception);
    }
    catch (...)
    {
        trt11_reset_runtime_error_recorder_snapshot_info(out_info);
        return trt11_runtime_control_unknown_exception("runtime error recorder snapshot query");
    }
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime error recorder snapshot query");
#endif
}

JYPPX_StatusCode jyppx_trt11_runtime_get_error_recorder_error(
    JYPPX_TensorRtRuntime* runtime,
    int32_t index,
    JYPPX_TensorRtErrorRecordInfo* out_error)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_error, "out_error");
    if (status != JYPPX_STATUS_OK) { return status; }
    trt11_reset_runtime_error_record_info(out_error);
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    try
    {
        return trt11_runtime_get_error_recorder_error_with_seh_guard(runtime_payload, index, out_error, "runtime error recorder error copy");
    }
    catch (const std::exception& exception)
    {
        trt11_reset_runtime_error_record_info(out_error);
        return trt11_runtime_control_exception("runtime error recorder error copy", exception);
    }
    catch (...)
    {
        trt11_reset_runtime_error_record_info(out_error);
        return trt11_runtime_control_unknown_exception("runtime error recorder error copy");
    }
#else
    (void)runtime;
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime error recorder error copy");
#endif
}

JYPPX_StatusCode jyppx_trt11_runtime_clear_error_recorder(JYPPX_TensorRtRuntime* runtime)
{
#if JYPPX_HAS_TENSORRT
    auto status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    runtime_payload->setErrorRecorder(nullptr);
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime error recorder clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_runtime_clear_gpu_allocator(JYPPX_TensorRtRuntime* runtime)
{
#if JYPPX_HAS_TENSORRT
    auto status = jyppx::tensorrt::validate_handle(runtime, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME, "runtime");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* runtime_payload = get_payload<nvinfer1::IRuntime>(runtime);
    if (runtime_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    runtime_payload->setGpuAllocator(nullptr);
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime GPU allocator clear");
#endif
}

#if JYPPX_HAS_TENSORRT
static JYPPX_StatusCode create_host_memory_handle(nvinfer1::IHostMemory* host_memory, JYPPX_TensorRtHostMemory** out_host_memory, const char* feature_name)
{
    if (host_memory == nullptr)
    {
        return report_null_vendor_object(feature_name);
    }

    JYPPX_TensorRtObjectBase* handle = nullptr;
    const JYPPX_StatusCode status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_HOST_MEMORY, host_memory, &destroy_payload<nvinfer1::IHostMemory>);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_host_memory = reinterpret_cast<JYPPX_TensorRtHostMemory*>(handle);
    return JYPPX_STATUS_OK;
}
#endif

JYPPX_StatusCode jyppx_trt11_engine_serialize(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtHostMemory** out_host_memory)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_host_memory, "out_host_memory");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_host_memory = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine serialize");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_host_memory_handle(engine_payload->serialize(), out_host_memory, "ICudaEngine::serialize");
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine serialize");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_create_serialization_config(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtSerializationConfig** out_config)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_config, "out_config");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_config = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine serialization config creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::ISerializationConfig* config = engine_payload->createSerializationConfig();
    if (config == nullptr) { return report_null_vendor_object("ICudaEngine::createSerializationConfig"); }
    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_SERIALIZATION_CONFIG, config, &destroy_payload<nvinfer1::ISerializationConfig>);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_config = reinterpret_cast<JYPPX_TensorRtSerializationConfig*>(handle);
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine serialization config creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_serialize_with_config(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtSerializationConfig* config, JYPPX_TensorRtHostMemory** out_host_memory)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_host_memory, "out_host_memory");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_host_memory = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine serialize with config");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_SERIALIZATION_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* config_payload = get_payload<nvinfer1::ISerializationConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    return create_host_memory_handle(engine_payload->serializeWithConfig(*config_payload), out_host_memory, "ICudaEngine::serializeWithConfig");
#else
    (void)engine;
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine serialize with config");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_create_runtime_config(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtRuntimeConfig** out_config)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_config, "out_config");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_config = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "engine runtime config creation");
    if (status != JYPPX_STATUS_OK) { return status; }
    nvinfer1::IRuntimeConfig* config = engine_payload->createRuntimeConfig();
    if (config == nullptr) { return report_null_vendor_object("ICudaEngine::createRuntimeConfig"); }
    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME_CONFIG, config, &destroy_payload<nvinfer1::IRuntimeConfig>);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_config = reinterpret_cast<JYPPX_TensorRtRuntimeConfig*>(handle);
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine runtime config creation");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_create_execution_context_with_strategy(JYPPX_TensorRtCudaEngine* engine, int32_t strategy, JYPPX_TensorRtExecutionContext** out_context)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_context, "out_context");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_context = nullptr;
    if (strategy < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context allocation strategy must be non-negative.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "execution context creation with allocation strategy");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* context = engine_payload->createExecutionContext(static_cast<nvinfer1::ExecutionContextAllocationStrategy>(strategy));
    if (context == nullptr) { return report_null_vendor_object("ICudaEngine::createExecutionContext(strategy)"); }
    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, context, &destroy_payload<nvinfer1::IExecutionContext>);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_context = reinterpret_cast<JYPPX_TensorRtExecutionContext*>(handle);
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context creation with allocation strategy");
#endif
}

JYPPX_StatusCode jyppx_trt11_engine_create_execution_context_with_runtime_config(JYPPX_TensorRtCudaEngine* engine, JYPPX_TensorRtRuntimeConfig* runtime_config, JYPPX_TensorRtExecutionContext** out_context)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_context, "out_context");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_context = nullptr;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_ext(engine, &engine_payload, "execution context creation with runtime config");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_handle(runtime_config, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME_CONFIG, "runtime_config");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* config_payload = get_payload<nvinfer1::IRuntimeConfig>(runtime_config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    auto* context = engine_payload->createExecutionContext(config_payload);
    if (context == nullptr) { return report_null_vendor_object("ICudaEngine::createExecutionContext(runtimeConfig)"); }
    JYPPX_TensorRtObjectBase* handle = nullptr;
    status = create_handle_with_payload(&handle, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, context, &destroy_payload<nvinfer1::IExecutionContext>);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_context = reinterpret_cast<JYPPX_TensorRtExecutionContext*>(handle);
    return JYPPX_STATUS_OK;
#else
    (void)engine;
    (void)runtime_config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context creation with runtime config");
#endif
}

JYPPX_StatusCode jyppx_trt11_serialization_config_set_flags(JYPPX_TensorRtSerializationConfig* config, uint32_t flags, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_SERIALIZATION_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* config_payload = get_payload<nvinfer1::ISerializationConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_set = config_payload->setFlags(static_cast<nvinfer1::SerializationFlags>(flags)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)config;
    (void)flags;
    return jyppx::tensorrt::report_vendor_missing(kLine, "serialization config flags set");
#endif
}

JYPPX_StatusCode jyppx_trt11_serialization_config_get_flags(JYPPX_TensorRtSerializationConfig* config, uint32_t* out_flags)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_flags, "out_flags");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_flags = 0;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_SERIALIZATION_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* config_payload = get_payload<nvinfer1::ISerializationConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_flags = static_cast<uint32_t>(config_payload->getFlags());
    return JYPPX_STATUS_OK;
#else
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "serialization config flags query");
#endif
}

JYPPX_StatusCode jyppx_trt11_serialization_config_set_flag(JYPPX_TensorRtSerializationConfig* config, int32_t flag, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;
    if (flag < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_SERIALIZATION_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* config_payload = get_payload<nvinfer1::ISerializationConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_set = config_payload->setFlag(static_cast<nvinfer1::SerializationFlag>(flag)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "serialization config flag set");
#endif
}

JYPPX_StatusCode jyppx_trt11_serialization_config_clear_flag(JYPPX_TensorRtSerializationConfig* config, int32_t flag, JYPPX_Boolean* out_cleared)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_cleared, "out_cleared");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cleared = JYPPX_FALSE;
    if (flag < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_SERIALIZATION_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* config_payload = get_payload<nvinfer1::ISerializationConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_cleared = config_payload->clearFlag(static_cast<nvinfer1::SerializationFlag>(flag)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "serialization config flag clear");
#endif
}

JYPPX_StatusCode jyppx_trt11_serialization_config_get_flag(JYPPX_TensorRtSerializationConfig* config, int32_t flag, JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_enabled = JYPPX_FALSE;
    if (flag < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_SERIALIZATION_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* config_payload = get_payload<nvinfer1::ISerializationConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_enabled = config_payload->getFlag(static_cast<nvinfer1::SerializationFlag>(flag)) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "serialization config flag query");
#endif
}

JYPPX_StatusCode jyppx_trt11_runtime_config_set_execution_context_allocation_strategy(JYPPX_TensorRtRuntimeConfig* config, int32_t strategy)
{
    if (strategy < 0) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    auto status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* config_payload = get_payload<nvinfer1::IRuntimeConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    config_payload->setExecutionContextAllocationStrategy(static_cast<nvinfer1::ExecutionContextAllocationStrategy>(strategy));
    return JYPPX_STATUS_OK;
#else
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime config allocation strategy set");
#endif
}

JYPPX_StatusCode jyppx_trt11_runtime_config_get_execution_context_allocation_strategy(JYPPX_TensorRtRuntimeConfig* config, int32_t* out_strategy)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_strategy, "out_strategy");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_strategy = 0;
#if JYPPX_HAS_TENSORRT
    status = jyppx::tensorrt::validate_handle(config, kLine, JYPPX_TENSORRT_OBJECT_KIND_RUNTIME_CONFIG, "config");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* config_payload = get_payload<nvinfer1::IRuntimeConfig>(config);
    if (config_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_strategy = static_cast<int32_t>(config_payload->getExecutionContextAllocationStrategy());
    return JYPPX_STATUS_OK;
#else
    (void)config;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime config allocation strategy query");
#endif
}

#if JYPPX_HAS_TENSORRT
static void copy_weights_info(const nvinfer1::Weights& weights, JYPPX_TensorRtWeightsInfo* out_info)
{
    out_info->data_type = static_cast<int32_t>(weights.type);
    out_info->count = weights.count;
    out_info->has_values = weights.values != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
}
#endif

JYPPX_StatusCode jyppx_trt11_refitter_refit_cuda_engine_async(JYPPX_TensorRtRefitter* refitter, JYPPX_CudaStream* stream, JYPPX_Boolean* out_refitted)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_refitted, "out_refitted");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_refitted = JYPPX_FALSE;
    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT && JYPPX_HAS_CUDA_TOOLKIT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_ext(refitter, &payload, "refitter async refit CUDA engine");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* stream_payload = reinterpret_cast<jyppx::cuda::StreamObject*>(stream);
    if (stream_payload == nullptr) { return JYPPX_STATUS_INVALID_STATE; }
    *out_refitted = payload->refitCudaEngineAsync(stream_payload->handle) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_TENSORRT
    (void)refitter;
    return jyppx::cuda::report_cuda_dependency_missing("TensorRT refitter async refit CUDA engine");
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter async refit CUDA engine");
#endif
}

JYPPX_StatusCode jyppx_trt11_refitter_set_max_threads(JYPPX_TensorRtRefitter* refitter, int32_t max_threads, JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = JYPPX_FALSE;
    if (max_threads < 1) { return JYPPX_STATUS_INVALID_ARGUMENT; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_ext(refitter, &payload, "refitter max threads set");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_set = payload->setMaxThreads(max_threads) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter max threads set");
#endif
}

JYPPX_StatusCode jyppx_trt11_refitter_get_max_threads(JYPPX_TensorRtRefitter* refitter, int32_t* out_max_threads)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_max_threads, "out_max_threads");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_max_threads = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_ext(refitter, &payload, "refitter max threads query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_max_threads = payload->getMaxThreads();
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter max threads query");
#endif
}

JYPPX_StatusCode jyppx_trt11_refitter_set_weights_validation(JYPPX_TensorRtRefitter* refitter, JYPPX_Boolean enabled)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    auto status = get_refitter_payload_ext(refitter, &payload, "refitter weights validation set");
    if (status != JYPPX_STATUS_OK) { return status; }
    payload->setWeightsValidation(enabled != JYPPX_FALSE);
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    (void)enabled;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter weights validation set");
#endif
}

JYPPX_StatusCode jyppx_trt11_refitter_get_weights_validation(JYPPX_TensorRtRefitter* refitter, JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_enabled = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_ext(refitter, &payload, "refitter weights validation query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_enabled = payload->getWeightsValidation() ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter weights validation query");
#endif
}

JYPPX_StatusCode jyppx_trt11_refitter_unset_named_weights(JYPPX_TensorRtRefitter* refitter, const char* weights_name, JYPPX_Boolean* out_unset)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_unset, "out_unset");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_unset = JYPPX_FALSE;
    status = validate_c_string(weights_name, "weights_name");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_ext(refitter, &payload, "refitter named weights unset");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_unset = payload->unsetNamedWeights(weights_name) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter named weights unset");
#endif
}

JYPPX_StatusCode jyppx_trt11_refitter_get_weights_location(JYPPX_TensorRtRefitter* refitter, const char* weights_name, int32_t* out_location)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_location, "out_location");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_location = 0;
    status = validate_c_string(weights_name, "weights_name");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_ext(refitter, &payload, "refitter weights location query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_location = static_cast<int32_t>(payload->getWeightsLocation(weights_name));
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter weights location query");
#endif
}

JYPPX_StatusCode jyppx_trt11_refitter_get_named_weights_info(JYPPX_TensorRtRefitter* refitter, const char* weights_name, JYPPX_TensorRtWeightsInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK) { return status; }
    out_info->data_type = -1;
    out_info->count = 0;
    out_info->has_values = JYPPX_FALSE;
    status = validate_c_string(weights_name, "weights_name");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_ext(refitter, &payload, "refitter named weights info query");
    if (status != JYPPX_STATUS_OK) { return status; }
    copy_weights_info(payload->getNamedWeights(weights_name), out_info);
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter named weights info query");
#endif
}

JYPPX_StatusCode jyppx_trt11_refitter_get_weights_prototype_info(JYPPX_TensorRtRefitter* refitter, const char* weights_name, JYPPX_TensorRtWeightsInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK) { return status; }
    out_info->data_type = -1;
    out_info->count = 0;
    out_info->has_values = JYPPX_FALSE;
    status = validate_c_string(weights_name, "weights_name");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_ext(refitter, &payload, "refitter weights prototype info query");
    if (status != JYPPX_STATUS_OK) { return status; }
    copy_weights_info(payload->getWeightsPrototype(weights_name), out_info);
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter weights prototype info query");
#endif
}

JYPPX_StatusCode jyppx_trt11_refitter_has_error_recorder(JYPPX_TensorRtRefitter* refitter, JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_ext(refitter, &payload, "refitter error recorder query");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_recorder = payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter error recorder query");
#endif
}

JYPPX_StatusCode jyppx_trt11_refitter_get_error_recorder_snapshot_info(
    JYPPX_TensorRtRefitter* refitter,
    JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK) { return status; }
    trt11_reset_runtime_error_recorder_snapshot_info(out_info);
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_ext(refitter, &payload, "refitter error recorder snapshot query");
    if (status != JYPPX_STATUS_OK) { return status; }
    try
    {
        return trt11_refitter_get_error_recorder_snapshot_info_with_seh_guard(
            payload,
            out_info,
            "refitter error recorder snapshot query");
    }
    catch (const std::exception& exception)
    {
        trt11_reset_runtime_error_recorder_snapshot_info(out_info);
        return trt11_runtime_control_exception("refitter error recorder snapshot query", exception);
    }
    catch (...)
    {
        trt11_reset_runtime_error_recorder_snapshot_info(out_info);
        return trt11_runtime_control_unknown_exception("refitter error recorder snapshot query");
    }
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter error recorder snapshot query");
#endif
}

JYPPX_StatusCode jyppx_trt11_refitter_get_error_recorder_error(
    JYPPX_TensorRtRefitter* refitter,
    int32_t index,
    JYPPX_TensorRtErrorRecordInfo* out_error)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_error, "out_error");
    if (status != JYPPX_STATUS_OK) { return status; }
    trt11_reset_runtime_error_record_info(out_error);
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_ext(refitter, &payload, "refitter error recorder error copy");
    if (status != JYPPX_STATUS_OK) { return status; }
    try
    {
        return trt11_refitter_get_error_recorder_error_with_seh_guard(
            payload,
            index,
            out_error,
            "refitter error recorder error copy");
    }
    catch (const std::exception& exception)
    {
        trt11_reset_runtime_error_record_info(out_error);
        return trt11_runtime_control_exception("refitter error recorder error copy", exception);
    }
    catch (...)
    {
        trt11_reset_runtime_error_record_info(out_error);
        return trt11_runtime_control_unknown_exception("refitter error recorder error copy");
    }
#else
    (void)refitter;
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter error recorder error copy");
#endif
}

JYPPX_StatusCode jyppx_trt11_refitter_clear_error_recorder(JYPPX_TensorRtRefitter* refitter)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    auto status = get_refitter_payload_ext(refitter, &payload, "refitter error recorder clear");
    if (status != JYPPX_STATUS_OK) { return status; }
    payload->setErrorRecorder(nullptr);
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter error recorder clear");
#endif
}
