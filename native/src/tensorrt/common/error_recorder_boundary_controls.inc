#ifndef JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API
#error "JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API must be defined before including error_recorder_boundary_controls.inc"
#endif

static void error_recorder_boundary_reset_error_recorder_snapshot_info(JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info)
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

static void error_recorder_boundary_reset_error_record_info(JYPPX_TensorRtErrorRecordInfo* out_error)
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
static JYPPX_StatusCode error_recorder_boundary_control_exception(
    const char* feature_name,
    const std::exception& exception)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
}

static JYPPX_StatusCode error_recorder_boundary_control_unknown_exception(const char* feature_name)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
}

static void error_recorder_boundary_copy_interface_info(
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

static JYPPX_StatusCode get_engine_payload_for_error_recorder_boundary(
    JYPPX_TensorRtCudaEngine* engine,
    nvinfer1::ICudaEngine** out_payload,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(engine, kLine, JYPPX_TENSORRT_OBJECT_KIND_CUDA_ENGINE, "engine");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* engine_payload = get_payload<nvinfer1::ICudaEngine>(engine);
    if (engine_payload == nullptr)
    {
        std::ostringstream message;
        message << "Engine handle does not carry the expected TensorRT payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, message.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_payload = engine_payload;
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode get_context_payload_for_error_recorder_boundary(
    JYPPX_TensorRtExecutionContext* context,
    nvinfer1::IExecutionContext** out_payload,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(context, kLine, JYPPX_TENSORRT_OBJECT_KIND_EXECUTION_CONTEXT, "context");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* context_payload = get_payload<nvinfer1::IExecutionContext>(context);
    if (context_payload == nullptr)
    {
        std::ostringstream message;
        message << "Execution context handle does not carry the expected TensorRT payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, message.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_payload = context_payload;
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode get_network_payload_for_error_recorder_boundary(
    JYPPX_TensorRtNetworkDefinition* network,
    nvinfer1::INetworkDefinition** out_payload,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(network, kLine, JYPPX_TENSORRT_OBJECT_KIND_NETWORK_DEFINITION, "network");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* network_payload = get_payload<nvinfer1::INetworkDefinition>(network);
    if (network_payload == nullptr)
    {
        std::ostringstream message;
        message << "Network definition handle does not carry the expected TensorRT payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, message.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_payload = network_payload;
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode get_engine_inspector_payload_for_error_recorder_boundary(
    JYPPX_TensorRtEngineInspector* inspector,
    nvinfer1::IEngineInspector** out_payload,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_handle(inspector, kLine, JYPPX_TENSORRT_OBJECT_KIND_ENGINE_INSPECTOR, "inspector");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* inspector_payload = get_payload<nvinfer1::IEngineInspector>(inspector);
    if (inspector_payload == nullptr)
    {
        std::ostringstream message;
        message << "Engine inspector handle does not carry the expected TensorRT payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, message.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_payload = inspector_payload;
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode engine_has_error_recorder_with_seh_guard(
    nvinfer1::ICudaEngine* engine_payload,
    JYPPX_Boolean* out_has_recorder,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_recorder = engine_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_recorder = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_recorder = engine_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode engine_clear_error_recorder_with_seh_guard(
    nvinfer1::ICudaEngine* engine_payload,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        engine_payload->setErrorRecorder(nullptr);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    engine_payload->setErrorRecorder(nullptr);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode engine_get_error_recorder_snapshot_info_with_seh_guard(
    nvinfer1::ICudaEngine* engine_payload,
    JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        auto* recorder = engine_payload->getErrorRecorder();
        if (recorder == nullptr)
        {
            return JYPPX_STATUS_OK;
        }

        out_info->has_recorder = JYPPX_TRUE;
        const int32_t error_count = recorder->getNbErrors();
        out_info->error_count = error_count > 0 ? error_count : 0;
        out_info->has_overflowed = recorder->hasOverflowed() ? JYPPX_TRUE : JYPPX_FALSE;
        error_recorder_boundary_copy_interface_info(recorder, out_info);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        error_recorder_boundary_reset_error_recorder_snapshot_info(out_info);
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    auto* recorder = engine_payload->getErrorRecorder();
    if (recorder == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    out_info->has_recorder = JYPPX_TRUE;
    const int32_t error_count = recorder->getNbErrors();
    out_info->error_count = error_count > 0 ? error_count : 0;
    out_info->has_overflowed = recorder->hasOverflowed() ? JYPPX_TRUE : JYPPX_FALSE;
    error_recorder_boundary_copy_interface_info(recorder, out_info);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode engine_get_error_recorder_error_with_seh_guard(
    nvinfer1::ICudaEngine* engine_payload,
    const int32_t index,
    JYPPX_TensorRtErrorRecordInfo* out_error,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        auto* recorder = engine_payload->getErrorRecorder();
        if (recorder == nullptr)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine error recorder snapshot is unavailable because no recorder is attached.");
            return JYPPX_STATUS_INVALID_STATE;
        }

        const int32_t error_count = recorder->getNbErrors();
        if (index < 0 || index >= error_count)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine error recorder error index is out of range.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }

        out_error->index = index;
        out_error->code = static_cast<int32_t>(recorder->getErrorCode(index));
        copy_c_string(recorder->getErrorDesc(index), out_error->description, sizeof(out_error->description));
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        error_recorder_boundary_reset_error_record_info(out_error);
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    auto* recorder = engine_payload->getErrorRecorder();
    if (recorder == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine error recorder snapshot is unavailable because no recorder is attached.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const int32_t error_count = recorder->getNbErrors();
    if (index < 0 || index >= error_count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Engine error recorder error index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    out_error->index = index;
    out_error->code = static_cast<int32_t>(recorder->getErrorCode(index));
    copy_c_string(recorder->getErrorDesc(index), out_error->description, sizeof(out_error->description));
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode execution_context_has_error_recorder_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    JYPPX_Boolean* out_has_recorder,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_recorder = context_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_recorder = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_recorder = context_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode execution_context_get_error_recorder_snapshot_info_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        auto* recorder = context_payload->getErrorRecorder();
        if (recorder == nullptr)
        {
            return JYPPX_STATUS_OK;
        }

        out_info->has_recorder = JYPPX_TRUE;
        const int32_t error_count = recorder->getNbErrors();
        out_info->error_count = error_count > 0 ? error_count : 0;
        out_info->has_overflowed = recorder->hasOverflowed() ? JYPPX_TRUE : JYPPX_FALSE;
        error_recorder_boundary_copy_interface_info(recorder, out_info);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        error_recorder_boundary_reset_error_recorder_snapshot_info(out_info);
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    auto* recorder = context_payload->getErrorRecorder();
    if (recorder == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    out_info->has_recorder = JYPPX_TRUE;
    const int32_t error_count = recorder->getNbErrors();
    out_info->error_count = error_count > 0 ? error_count : 0;
    out_info->has_overflowed = recorder->hasOverflowed() ? JYPPX_TRUE : JYPPX_FALSE;
    error_recorder_boundary_copy_interface_info(recorder, out_info);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode execution_context_get_error_recorder_error_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    const int32_t index,
    JYPPX_TensorRtErrorRecordInfo* out_error,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        auto* recorder = context_payload->getErrorRecorder();
        if (recorder == nullptr)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context error recorder snapshot is unavailable because no recorder is attached.");
            return JYPPX_STATUS_INVALID_STATE;
        }

        const int32_t error_count = recorder->getNbErrors();
        if (index < 0 || index >= error_count)
        {
            jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context error recorder error index is out of range.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }

        out_error->index = index;
        out_error->code = static_cast<int32_t>(recorder->getErrorCode(index));
        copy_c_string(recorder->getErrorDesc(index), out_error->description, sizeof(out_error->description));
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        error_recorder_boundary_reset_error_record_info(out_error);
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    auto* recorder = context_payload->getErrorRecorder();
    if (recorder == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context error recorder snapshot is unavailable because no recorder is attached.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    const int32_t error_count = recorder->getNbErrors();
    if (index < 0 || index >= error_count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Execution context error recorder error index is out of range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    out_error->index = index;
    out_error->code = static_cast<int32_t>(recorder->getErrorCode(index));
    copy_c_string(recorder->getErrorDesc(index), out_error->description, sizeof(out_error->description));
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode execution_context_clear_error_recorder_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        context_payload->setErrorRecorder(nullptr);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    context_payload->setErrorRecorder(nullptr);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode network_has_error_recorder_with_seh_guard(
    nvinfer1::INetworkDefinition* network_payload,
    JYPPX_Boolean* out_has_recorder,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_recorder = network_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_recorder = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_recorder = network_payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode network_clear_error_recorder_with_seh_guard(
    nvinfer1::INetworkDefinition* network_payload,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        network_payload->setErrorRecorder(nullptr);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    network_payload->setErrorRecorder(nullptr);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode engine_inspector_has_error_recorder_with_seh_guard(
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

static JYPPX_StatusCode engine_inspector_clear_error_recorder_with_seh_guard(
    nvinfer1::IEngineInspector* inspector_payload,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        inspector_payload->setErrorRecorder(nullptr);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    inspector_payload->setErrorRecorder(nullptr);
    return JYPPX_STATUS_OK;
#endif
}
#endif

JYPPX_StatusCode JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API(engine_has_error_recorder)(
    JYPPX_TensorRtCudaEngine* engine,
    JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_for_error_recorder_boundary(engine, &engine_payload, "engine error recorder query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return engine_has_error_recorder_with_seh_guard(
            engine_payload,
            out_has_recorder,
            "engine error recorder query");
    }
    catch (const std::exception& exception)
    {
        *out_has_recorder = JYPPX_FALSE;
        return error_recorder_boundary_control_exception("engine error recorder query", exception);
    }
    catch (...)
    {
        *out_has_recorder = JYPPX_FALSE;
        return error_recorder_boundary_control_unknown_exception("engine error recorder query");
    }
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine error recorder query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API(engine_clear_error_recorder)(JYPPX_TensorRtCudaEngine* engine)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    auto status = get_engine_payload_for_error_recorder_boundary(engine, &engine_payload, "engine error recorder clear");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return engine_clear_error_recorder_with_seh_guard(
            engine_payload,
            "engine error recorder clear");
    }
    catch (const std::exception& exception)
    {
        return error_recorder_boundary_control_exception("engine error recorder clear", exception);
    }
    catch (...)
    {
        return error_recorder_boundary_control_unknown_exception("engine error recorder clear");
    }
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine error recorder clear");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API(engine_get_error_recorder_snapshot_info)(
    JYPPX_TensorRtCudaEngine* engine,
    JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    error_recorder_boundary_reset_error_recorder_snapshot_info(out_info);
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_for_error_recorder_boundary(engine, &engine_payload, "engine error recorder snapshot query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return engine_get_error_recorder_snapshot_info_with_seh_guard(
            engine_payload,
            out_info,
            "engine error recorder snapshot query");
    }
    catch (const std::exception& exception)
    {
        error_recorder_boundary_reset_error_recorder_snapshot_info(out_info);
        return error_recorder_boundary_control_exception("engine error recorder snapshot query", exception);
    }
    catch (...)
    {
        error_recorder_boundary_reset_error_recorder_snapshot_info(out_info);
        return error_recorder_boundary_control_unknown_exception("engine error recorder snapshot query");
    }
#else
    (void)engine;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine error recorder snapshot query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API(engine_get_error_recorder_error)(
    JYPPX_TensorRtCudaEngine* engine,
    int32_t index,
    JYPPX_TensorRtErrorRecordInfo* out_error)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_error, "out_error");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    error_recorder_boundary_reset_error_record_info(out_error);
#if JYPPX_HAS_TENSORRT
    nvinfer1::ICudaEngine* engine_payload = nullptr;
    status = get_engine_payload_for_error_recorder_boundary(engine, &engine_payload, "engine error recorder error copy");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return engine_get_error_recorder_error_with_seh_guard(
            engine_payload,
            index,
            out_error,
            "engine error recorder error copy");
    }
    catch (const std::exception& exception)
    {
        error_recorder_boundary_reset_error_record_info(out_error);
        return error_recorder_boundary_control_exception("engine error recorder error copy", exception);
    }
    catch (...)
    {
        error_recorder_boundary_reset_error_record_info(out_error);
        return error_recorder_boundary_control_unknown_exception("engine error recorder error copy");
    }
#else
    (void)engine;
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine error recorder error copy");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API(execution_context_has_error_recorder)(
    JYPPX_TensorRtExecutionContext* context,
    JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_for_error_recorder_boundary(context, &context_payload, "execution context error recorder query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return execution_context_has_error_recorder_with_seh_guard(
            context_payload,
            out_has_recorder,
            "execution context error recorder query");
    }
    catch (const std::exception& exception)
    {
        *out_has_recorder = JYPPX_FALSE;
        return error_recorder_boundary_control_exception("execution context error recorder query", exception);
    }
    catch (...)
    {
        *out_has_recorder = JYPPX_FALSE;
        return error_recorder_boundary_control_unknown_exception("execution context error recorder query");
    }
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context error recorder query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API(execution_context_clear_error_recorder)(JYPPX_TensorRtExecutionContext* context)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    auto status = get_context_payload_for_error_recorder_boundary(context, &context_payload, "execution context error recorder clear");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return execution_context_clear_error_recorder_with_seh_guard(
            context_payload,
            "execution context error recorder clear");
    }
    catch (const std::exception& exception)
    {
        return error_recorder_boundary_control_exception("execution context error recorder clear", exception);
    }
    catch (...)
    {
        return error_recorder_boundary_control_unknown_exception("execution context error recorder clear");
    }
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context error recorder clear");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API(execution_context_get_error_recorder_snapshot_info)(
    JYPPX_TensorRtExecutionContext* context,
    JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    error_recorder_boundary_reset_error_recorder_snapshot_info(out_info);
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_for_error_recorder_boundary(context, &context_payload, "execution context error recorder snapshot query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return execution_context_get_error_recorder_snapshot_info_with_seh_guard(
            context_payload,
            out_info,
            "execution context error recorder snapshot query");
    }
    catch (const std::exception& exception)
    {
        error_recorder_boundary_reset_error_recorder_snapshot_info(out_info);
        return error_recorder_boundary_control_exception("execution context error recorder snapshot query", exception);
    }
    catch (...)
    {
        error_recorder_boundary_reset_error_recorder_snapshot_info(out_info);
        return error_recorder_boundary_control_unknown_exception("execution context error recorder snapshot query");
    }
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context error recorder snapshot query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API(execution_context_get_error_recorder_error)(
    JYPPX_TensorRtExecutionContext* context,
    int32_t index,
    JYPPX_TensorRtErrorRecordInfo* out_error)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_error, "out_error");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    error_recorder_boundary_reset_error_record_info(out_error);
#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_for_error_recorder_boundary(context, &context_payload, "execution context error recorder error copy");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return execution_context_get_error_recorder_error_with_seh_guard(
            context_payload,
            index,
            out_error,
            "execution context error recorder error copy");
    }
    catch (const std::exception& exception)
    {
        error_recorder_boundary_reset_error_record_info(out_error);
        return error_recorder_boundary_control_exception("execution context error recorder error copy", exception);
    }
    catch (...)
    {
        error_recorder_boundary_reset_error_record_info(out_error);
        return error_recorder_boundary_control_unknown_exception("execution context error recorder error copy");
    }
#else
    (void)context;
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context error recorder error copy");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API(network_has_error_recorder)(
    JYPPX_TensorRtNetworkDefinition* network,
    JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    status = get_network_payload_for_error_recorder_boundary(network, &network_payload, "network error recorder query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return network_has_error_recorder_with_seh_guard(
            network_payload,
            out_has_recorder,
            "network error recorder query");
    }
    catch (const std::exception& exception)
    {
        *out_has_recorder = JYPPX_FALSE;
        return error_recorder_boundary_control_exception("network error recorder query", exception);
    }
    catch (...)
    {
        *out_has_recorder = JYPPX_FALSE;
        return error_recorder_boundary_control_unknown_exception("network error recorder query");
    }
#else
    (void)network;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network error recorder query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API(network_clear_error_recorder)(JYPPX_TensorRtNetworkDefinition* network)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::INetworkDefinition* network_payload = nullptr;
    auto status = get_network_payload_for_error_recorder_boundary(network, &network_payload, "network error recorder clear");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return network_clear_error_recorder_with_seh_guard(
            network_payload,
            "network error recorder clear");
    }
    catch (const std::exception& exception)
    {
        return error_recorder_boundary_control_exception("network error recorder clear", exception);
    }
    catch (...)
    {
        return error_recorder_boundary_control_unknown_exception("network error recorder clear");
    }
#else
    (void)network;
    return jyppx::tensorrt::report_vendor_missing(kLine, "network error recorder clear");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API(engine_inspector_has_error_recorder)(
    JYPPX_TensorRtEngineInspector* inspector,
    JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IEngineInspector* inspector_payload = nullptr;
    status = get_engine_inspector_payload_for_error_recorder_boundary(inspector, &inspector_payload, "engine inspector error recorder query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return engine_inspector_has_error_recorder_with_seh_guard(
            inspector_payload,
            out_has_recorder,
            "engine inspector error recorder query");
    }
    catch (const std::exception& exception)
    {
        *out_has_recorder = JYPPX_FALSE;
        return error_recorder_boundary_control_exception("engine inspector error recorder query", exception);
    }
    catch (...)
    {
        *out_has_recorder = JYPPX_FALSE;
        return error_recorder_boundary_control_unknown_exception("engine inspector error recorder query");
    }
#else
    (void)inspector;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine inspector error recorder query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_ERROR_RECORDER_BOUNDARY_API(engine_inspector_clear_error_recorder)(JYPPX_TensorRtEngineInspector* inspector)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IEngineInspector* inspector_payload = nullptr;
    auto status = get_engine_inspector_payload_for_error_recorder_boundary(inspector, &inspector_payload, "engine inspector error recorder clear");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return engine_inspector_clear_error_recorder_with_seh_guard(
            inspector_payload,
            "engine inspector error recorder clear");
    }
    catch (const std::exception& exception)
    {
        return error_recorder_boundary_control_exception("engine inspector error recorder clear", exception);
    }
    catch (...)
    {
        return error_recorder_boundary_control_unknown_exception("engine inspector error recorder clear");
    }
#else
    (void)inspector;
    return jyppx::tensorrt::report_vendor_missing(kLine, "engine inspector error recorder clear");
#endif
}
