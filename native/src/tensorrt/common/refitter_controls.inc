#ifndef JYPPX_TRT_REFITTER_API
#error "JYPPX_TRT_REFITTER_API must be defined before including refitter_controls.inc"
#endif

static void reset_refitter_error_recorder_snapshot_info(JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info)
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

static void reset_refitter_error_record_info(JYPPX_TensorRtErrorRecordInfo* out_error)
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
static JYPPX_StatusCode get_refitter_payload_for_controls(
    JYPPX_TensorRtRefitter* refitter,
    nvinfer1::IRefitter** out_refitter,
    const char* feature_name)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_refitter, "out_refitter");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_refitter = nullptr;
    status = jyppx::tensorrt::validate_handle(refitter, kLine, JYPPX_TENSORRT_OBJECT_KIND_REFITTER, "refitter");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* payload = get_payload<nvinfer1::IRefitter>(refitter);
    if (payload == nullptr)
    {
        std::ostringstream message;
        message << "Refitter handle does not carry the expected TensorRT refitter payload for " << feature_name << ".";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, message.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_refitter = payload;
    return JYPPX_STATUS_OK;
}

static JYPPX_StatusCode jyppx_trt_refitter_control_exception(const char* feature_name, const std::exception& exception)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
}

static JYPPX_StatusCode jyppx_trt_refitter_control_unknown_exception(const char* feature_name)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
}

static void refitter_copy_error_recorder_interface_info(
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

static JYPPX_StatusCode refitter_has_logger_with_seh_guard(
    nvinfer1::IRefitter* payload,
    JYPPX_Boolean* out_has_logger,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_logger = payload->getLogger() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_logger = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_logger = payload->getLogger() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode refitter_has_error_recorder_with_seh_guard(
    nvinfer1::IRefitter* payload,
    JYPPX_Boolean* out_has_recorder,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_has_recorder = payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_has_recorder = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_has_recorder = payload->getErrorRecorder() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode refitter_get_error_recorder_snapshot_info_with_seh_guard(
    nvinfer1::IRefitter* payload,
    JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        auto* recorder = payload->getErrorRecorder();
        if (recorder == nullptr)
        {
            return JYPPX_STATUS_OK;
        }

        out_info->has_recorder = JYPPX_TRUE;
        const int32_t error_count = recorder->getNbErrors();
        out_info->error_count = error_count > 0 ? error_count : 0;
        out_info->has_overflowed = recorder->hasOverflowed() ? JYPPX_TRUE : JYPPX_FALSE;
        refitter_copy_error_recorder_interface_info(recorder, out_info);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        reset_refitter_error_recorder_snapshot_info(out_info);
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    auto* recorder = payload->getErrorRecorder();
    if (recorder == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    out_info->has_recorder = JYPPX_TRUE;
    const int32_t error_count = recorder->getNbErrors();
    out_info->error_count = error_count > 0 ? error_count : 0;
    out_info->has_overflowed = recorder->hasOverflowed() ? JYPPX_TRUE : JYPPX_FALSE;
    refitter_copy_error_recorder_interface_info(recorder, out_info);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode refitter_get_error_recorder_error_with_seh_guard(
    nvinfer1::IRefitter* payload,
    const int32_t index,
    JYPPX_TensorRtErrorRecordInfo* out_error,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        auto* recorder = payload->getErrorRecorder();
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
        reset_refitter_error_record_info(out_error);
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    auto* recorder = payload->getErrorRecorder();
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

static JYPPX_StatusCode refitter_clear_error_recorder_with_seh_guard(
    nvinfer1::IRefitter* payload,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        payload->setErrorRecorder(nullptr);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    payload->setErrorRecorder(nullptr);
    return JYPPX_STATUS_OK;
#endif
}

static void copy_refitter_names_to_entries(
    const std::vector<char const*>& names,
    JYPPX_TensorRtRefitEntryInfo* output_entries,
    int32_t copy_count)
{
    for (int32_t i = 0; i < copy_count; ++i)
    {
        copy_c_string(names[static_cast<size_t>(i)], output_entries[i].layer_name, sizeof(output_entries[i].layer_name));
        output_entries[i].role = -1;
    }
}
#endif

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_set_max_threads)(
    JYPPX_TensorRtRefitter* refitter,
    int32_t max_threads,
    JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_set = JYPPX_FALSE;
    if (max_threads < 1)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter max threads must be greater than or equal to 1.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter max threads set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_set = payload->setMaxThreads(max_threads) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    (void)max_threads;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter max threads set");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_get_max_threads)(
    JYPPX_TensorRtRefitter* refitter,
    int32_t* out_max_threads)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_max_threads, "out_max_threads");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_max_threads = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter max threads query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_max_threads = payload->getMaxThreads();
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter max threads query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_has_logger)(
    JYPPX_TensorRtRefitter* refitter,
    JYPPX_Boolean* out_has_logger)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_logger, "out_has_logger");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_logger = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter logger presence query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return refitter_has_logger_with_seh_guard(payload, out_has_logger, "refitter logger presence query");
    }
    catch (const std::exception& exception)
    {
        *out_has_logger = JYPPX_FALSE;
        return jyppx_trt_refitter_control_exception("refitter logger presence query", exception);
    }
    catch (...)
    {
        *out_has_logger = JYPPX_FALSE;
        return jyppx_trt_refitter_control_unknown_exception("refitter logger presence query");
    }
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter logger presence query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_has_error_recorder)(
    JYPPX_TensorRtRefitter* refitter,
    JYPPX_Boolean* out_has_recorder)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_has_recorder, "out_has_recorder");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_has_recorder = JYPPX_FALSE;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter error recorder query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return refitter_has_error_recorder_with_seh_guard(payload, out_has_recorder, "refitter error recorder query");
    }
    catch (const std::exception& exception)
    {
        *out_has_recorder = JYPPX_FALSE;
        return jyppx_trt_refitter_control_exception("refitter error recorder query", exception);
    }
    catch (...)
    {
        *out_has_recorder = JYPPX_FALSE;
        return jyppx_trt_refitter_control_unknown_exception("refitter error recorder query");
    }
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter error recorder query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_get_error_recorder_snapshot_info)(
    JYPPX_TensorRtRefitter* refitter,
    JYPPX_TensorRtErrorRecorderSnapshotInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    reset_refitter_error_recorder_snapshot_info(out_info);
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter error recorder snapshot query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return refitter_get_error_recorder_snapshot_info_with_seh_guard(payload, out_info, "refitter error recorder snapshot query");
    }
    catch (const std::exception& exception)
    {
        reset_refitter_error_recorder_snapshot_info(out_info);
        return jyppx_trt_refitter_control_exception("refitter error recorder snapshot query", exception);
    }
    catch (...)
    {
        reset_refitter_error_recorder_snapshot_info(out_info);
        return jyppx_trt_refitter_control_unknown_exception("refitter error recorder snapshot query");
    }
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter error recorder snapshot query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_get_error_recorder_error)(
    JYPPX_TensorRtRefitter* refitter,
    int32_t index,
    JYPPX_TensorRtErrorRecordInfo* out_error)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_error, "out_error");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    reset_refitter_error_record_info(out_error);
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter error recorder error copy");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return refitter_get_error_recorder_error_with_seh_guard(payload, index, out_error, "refitter error recorder error copy");
    }
    catch (const std::exception& exception)
    {
        reset_refitter_error_record_info(out_error);
        return jyppx_trt_refitter_control_exception("refitter error recorder error copy", exception);
    }
    catch (...)
    {
        reset_refitter_error_record_info(out_error);
        return jyppx_trt_refitter_control_unknown_exception("refitter error recorder error copy");
    }
#else
    (void)refitter;
    (void)index;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter error recorder error copy");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_clear_error_recorder)(JYPPX_TensorRtRefitter* refitter)
{
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    auto status = get_refitter_payload_for_controls(refitter, &payload, "refitter error recorder clear");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        return refitter_clear_error_recorder_with_seh_guard(payload, "refitter error recorder clear");
    }
    catch (const std::exception& exception)
    {
        return jyppx_trt_refitter_control_exception("refitter error recorder clear", exception);
    }
    catch (...)
    {
        return jyppx_trt_refitter_control_unknown_exception("refitter error recorder clear");
    }
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter error recorder clear");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_set_dynamic_range)(
    JYPPX_TensorRtRefitter* refitter,
    const char* tensor_name,
    float min,
    float max,
    JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_set = JYPPX_FALSE;
    status = validate_c_string(tensor_name, "tensor_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (min > max)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter dynamic range min must be less than or equal to max.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter dynamic range set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_set = payload->setDynamicRange(tensor_name, min, max) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    (void)min;
    (void)max;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter dynamic range set");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_get_dynamic_range_min)(
    JYPPX_TensorRtRefitter* refitter,
    const char* tensor_name,
    float* out_min)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_min, "out_min");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_min = 0.0F;
    status = validate_c_string(tensor_name, "tensor_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter dynamic range minimum query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_min = payload->getDynamicRangeMin(tensor_name);
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter dynamic range minimum query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_get_dynamic_range_max)(
    JYPPX_TensorRtRefitter* refitter,
    const char* tensor_name,
    float* out_max)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_max, "out_max");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_max = 0.0F;
    status = validate_c_string(tensor_name, "tensor_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter dynamic range maximum query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_max = payload->getDynamicRangeMax(tensor_name);
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter dynamic range maximum query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_get_dynamic_range_tensor_count)(
    JYPPX_TensorRtRefitter* refitter,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter dynamic range tensor count query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = payload->getTensorsWithDynamicRange(0, nullptr);
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter dynamic range tensor count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_get_dynamic_range_tensor_entries)(
    JYPPX_TensorRtRefitter* refitter,
    JYPPX_TensorRtRefitEntryInfo* output_entries,
    int32_t output_count,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
    if (output_count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter output count must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (output_count > 0 && output_entries == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter output entries buffer must not be null when output_count is non-zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter dynamic range tensor entries query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int32_t required_count = payload->getTensorsWithDynamicRange(0, nullptr);
    *out_count = required_count;
    if (output_entries == nullptr || output_count == 0 || required_count == 0)
    {
        return JYPPX_STATUS_OK;
    }

    const int32_t copy_count = output_count < required_count ? output_count : required_count;
    std::vector<char const*> names(static_cast<size_t>(copy_count));
    payload->getTensorsWithDynamicRange(copy_count, names.data());
    copy_refitter_names_to_entries(names, output_entries, copy_count);

    if (copy_count < required_count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter output entries buffer is too small.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }

    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    (void)output_entries;
    (void)output_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter dynamic range tensor entries query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_set_named_weights)(
    JYPPX_TensorRtRefitter* refitter,
    const char* weights_name,
    int32_t data_type,
    const void* values,
    int64_t value_count,
    JYPPX_Boolean* out_set)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_set, "out_set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_set = JYPPX_FALSE;
    status = validate_c_string(weights_name, "weights_name");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (values == nullptr || value_count <= 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT named refit weights values must not be null or empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (get_data_type_size(data_type) == 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "TensorRT named refit weights data type is not supported by the active TensorRT line.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter named weights set");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    nvinfer1::Weights weights{static_cast<nvinfer1::DataType>(data_type), values, value_count};
    *out_set = payload->setNamedWeights(weights_name, weights) ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    (void)data_type;
    (void)values;
    (void)value_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter named weights set");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_get_missing_weights_count)(
    JYPPX_TensorRtRefitter* refitter,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter missing named weights count query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = payload->getMissingWeights(0, nullptr);
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter missing named weights count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_get_all_weights_count)(
    JYPPX_TensorRtRefitter* refitter,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter all named weights count query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = payload->getAllWeights(0, nullptr);
    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter all named weights count query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_get_missing_weights_entries)(
    JYPPX_TensorRtRefitter* refitter,
    JYPPX_TensorRtRefitEntryInfo* output_entries,
    int32_t output_count,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
    if (output_count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter output count must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (output_count > 0 && output_entries == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter output entries buffer must not be null when output_count is non-zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter missing named weights entries query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int32_t required_count = payload->getMissingWeights(0, nullptr);
    *out_count = required_count;
    if (output_entries == nullptr || output_count == 0 || required_count == 0)
    {
        return JYPPX_STATUS_OK;
    }

    const int32_t copy_count = output_count < required_count ? output_count : required_count;
    std::vector<char const*> names(static_cast<size_t>(copy_count));
    payload->getMissingWeights(copy_count, names.data());
    copy_refitter_names_to_entries(names, output_entries, copy_count);

    if (copy_count < required_count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter output entries buffer is too small.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }

    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    (void)output_entries;
    (void)output_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter missing named weights entries query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_REFITTER_API(refitter_get_all_weights_entries)(
    JYPPX_TensorRtRefitter* refitter,
    JYPPX_TensorRtRefitEntryInfo* output_entries,
    int32_t output_count,
    int32_t* out_count)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
    if (output_count < 0)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter output count must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (output_count > 0 && output_entries == nullptr)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter output entries buffer must not be null when output_count is non-zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::IRefitter* payload = nullptr;
    status = get_refitter_payload_for_controls(refitter, &payload, "refitter all named weights entries query");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const int32_t required_count = payload->getAllWeights(0, nullptr);
    *out_count = required_count;
    if (output_entries == nullptr || output_count == 0 || required_count == 0)
    {
        return JYPPX_STATUS_OK;
    }

    const int32_t copy_count = output_count < required_count ? output_count : required_count;
    std::vector<char const*> names(static_cast<size_t>(copy_count));
    payload->getAllWeights(copy_count, names.data());
    copy_refitter_names_to_entries(names, output_entries, copy_count);

    if (copy_count < required_count)
    {
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, "Refitter output entries buffer is too small.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }

    return JYPPX_STATUS_OK;
#else
    (void)refitter;
    (void)output_entries;
    (void)output_count;
    return jyppx::tensorrt::report_vendor_missing(kLine, "refitter all named weights entries query");
#endif
}

#undef JYPPX_TRT_REFITTER_API
