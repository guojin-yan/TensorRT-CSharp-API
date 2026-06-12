#ifndef JYPPX_TRT_RUNTIME_API
#error "JYPPX_TRT_RUNTIME_API must be defined before including runtime_controls.inc"
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
    *out_has_recorder = runtime_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)runtime;
    return jyppx::tensorrt::report_vendor_missing(kLine, "runtime error recorder query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_clear_error_recorder)(JYPPX_TensorRtRuntime* runtime)
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

JYPPX_StatusCode JYPPX_TRT_RUNTIME_API(runtime_clear_gpu_allocator)(JYPPX_TensorRtRuntime* runtime)
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
