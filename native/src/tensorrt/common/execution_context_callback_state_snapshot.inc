#ifndef JYPPX_TRT_EXECUTION_CONTEXT_CALLBACK_STATE_API
#error "JYPPX_TRT_EXECUTION_CONTEXT_CALLBACK_STATE_API must be defined before including execution_context_callback_state_snapshot.inc"
#endif

namespace
{
constexpr const char* kExecutionContextCallbackStateFeature = "execution context callback state snapshot";

static void copy_callback_state_text(char* destination, const size_t destination_size, const char* value) noexcept
{
    if (destination == nullptr || destination_size == 0U)
    {
        return;
    }

    const char* safe_value = value != nullptr ? value : "";
    const size_t length = std::strlen(safe_value);
    const size_t copy_length = length < (destination_size - 1U)
        ? length
        : (destination_size - 1U);
    if (copy_length != 0U)
    {
        std::memcpy(destination, safe_value, copy_length);
    }

    destination[copy_length] = '\0';
}

static void reset_execution_context_callback_state_info(
    JYPPX_TensorRtExecutionContextCallbackStateInfo* out_info) noexcept
{
    if (out_info == nullptr)
    {
        return;
    }

    out_info->line = static_cast<uint32_t>(kLine);
    out_info->has_output_allocator = JYPPX_FALSE;
    out_info->has_temporary_storage_allocator = JYPPX_FALSE;
    out_info->has_debug_listener = JYPPX_FALSE;
    out_info->output_allocator_interface_info_available = JYPPX_FALSE;
    out_info->temporary_storage_allocator_interface_info_available = JYPPX_FALSE;
    out_info->debug_listener_interface_info_available = JYPPX_FALSE;
    out_info->output_allocator_clear_supported = JYPPX_TRUE;
    out_info->temporary_storage_allocator_clear_supported = JYPPX_TRUE;
    out_info->debug_listener_clear_supported = JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10 ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->output_allocator_cleared = JYPPX_FALSE;
    out_info->temporary_storage_allocator_cleared = JYPPX_FALSE;
    out_info->debug_listener_cleared = JYPPX_FALSE;
    out_info->output_allocator_interface_major = 0;
    out_info->output_allocator_interface_minor = 0;
    out_info->temporary_storage_allocator_interface_major = 0;
    out_info->temporary_storage_allocator_interface_minor = 0;
    out_info->debug_listener_interface_major = 0;
    out_info->debug_listener_interface_minor = 0;
    out_info->last_status = static_cast<int32_t>(JYPPX_STATUS_OK);
    out_info->output_allocator_interface_kind[0] = '\0';
    out_info->temporary_storage_allocator_interface_kind[0] = '\0';
    out_info->debug_listener_interface_kind[0] = '\0';
    copy_callback_state_text(out_info->last_operation, sizeof(out_info->last_operation), "reset");
    copy_callback_state_text(out_info->last_diagnostic, sizeof(out_info->last_diagnostic), "execution context callback state snapshot reset.");
}

static void set_callback_state_status(
    JYPPX_TensorRtExecutionContextCallbackStateInfo* out_info,
    const JYPPX_StatusCode status,
    const char* operation,
    const char* diagnostic) noexcept
{
    out_info->last_status = static_cast<int32_t>(status);
    copy_callback_state_text(out_info->last_operation, sizeof(out_info->last_operation), operation);
    copy_callback_state_text(out_info->last_diagnostic, sizeof(out_info->last_diagnostic), diagnostic);
}

#if JYPPX_HAS_TENSORRT
static JYPPX_StatusCode callback_state_exception(const std::exception& exception)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, kExecutionContextCallbackStateFeature, exception.what());
}

static JYPPX_StatusCode callback_state_unknown_exception()
{
    return jyppx::tensorrt::report_vendor_exception(kLine, kExecutionContextCallbackStateFeature, "unknown native exception");
}

static JYPPX_StatusCode snapshot_output_allocator_presence_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    const char* tensor_name,
    JYPPX_TensorRtExecutionContextCallbackStateInfo* out_info)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        out_info->has_output_allocator = context_payload->getOutputAllocator(tensor_name) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        out_info->has_output_allocator = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, kExecutionContextCallbackStateFeature, seh_exception_code);
    }
#else
    out_info->has_output_allocator = context_payload->getOutputAllocator(tensor_name) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode snapshot_temporary_storage_allocator_presence_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    JYPPX_TensorRtExecutionContextCallbackStateInfo* out_info)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        out_info->has_temporary_storage_allocator = context_payload->getTemporaryStorageAllocator() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        out_info->has_temporary_storage_allocator = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, kExecutionContextCallbackStateFeature, seh_exception_code);
    }
#else
    out_info->has_temporary_storage_allocator = context_payload->getTemporaryStorageAllocator() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode clear_output_allocator_with_state_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    const char* tensor_name,
    JYPPX_TensorRtExecutionContextCallbackStateInfo* out_info)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        out_info->output_allocator_cleared = context_payload->setOutputAllocator(tensor_name, nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->has_output_allocator = context_payload->getOutputAllocator(tensor_name) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        out_info->output_allocator_cleared = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, kExecutionContextCallbackStateFeature, seh_exception_code);
    }
#else
    out_info->output_allocator_cleared = context_payload->setOutputAllocator(tensor_name, nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->has_output_allocator = context_payload->getOutputAllocator(tensor_name) != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode clear_temporary_storage_allocator_with_state_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    JYPPX_TensorRtExecutionContextCallbackStateInfo* out_info)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        out_info->temporary_storage_allocator_cleared = context_payload->setTemporaryStorageAllocator(nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->has_temporary_storage_allocator = context_payload->getTemporaryStorageAllocator() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        out_info->temporary_storage_allocator_cleared = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, kExecutionContextCallbackStateFeature, seh_exception_code);
    }
#else
    out_info->temporary_storage_allocator_cleared = context_payload->setTemporaryStorageAllocator(nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->has_temporary_storage_allocator = context_payload->getTemporaryStorageAllocator() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
template <typename TObject>
static JYPPX_StatusCode copy_callback_interface_info_to_state_with_seh_guard(
    const TObject* object,
    JYPPX_Boolean* out_available,
    char* kind_buffer,
    const size_t kind_buffer_size,
    int32_t* out_major,
    int32_t* out_minor)
{
    if (object == nullptr)
    {
        *out_available = JYPPX_FALSE;
        *out_major = 0;
        *out_minor = 0;
        if (kind_buffer_size != 0U)
        {
            kind_buffer[0] = '\0';
        }
        return JYPPX_STATUS_OK;
    }

#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        nvinfer1::InterfaceInfo info = object->getInterfaceInfo();
        *out_available = JYPPX_TRUE;
        *out_major = info.major;
        *out_minor = info.minor;
        copy_callback_state_text(kind_buffer, kind_buffer_size, info.kind);
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_available = JYPPX_FALSE;
        *out_major = 0;
        *out_minor = 0;
        if (kind_buffer_size != 0U)
        {
            kind_buffer[0] = '\0';
        }
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, kExecutionContextCallbackStateFeature, seh_exception_code);
    }
#else
    nvinfer1::InterfaceInfo info = object->getInterfaceInfo();
    *out_available = JYPPX_TRUE;
    *out_major = info.major;
    *out_minor = info.minor;
    copy_callback_state_text(kind_buffer, kind_buffer_size, info.kind);
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode snapshot_debug_listener_presence_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    JYPPX_TensorRtExecutionContextCallbackStateInfo* out_info)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        out_info->has_debug_listener = context_payload->getDebugListener() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        out_info->has_debug_listener = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, kExecutionContextCallbackStateFeature, seh_exception_code);
    }
#else
    out_info->has_debug_listener = context_payload->getDebugListener() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}

static JYPPX_StatusCode clear_debug_listener_with_state_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    JYPPX_TensorRtExecutionContextCallbackStateInfo* out_info)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        out_info->debug_listener_cleared = context_payload->setDebugListener(nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
        out_info->has_debug_listener = context_payload->getDebugListener() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        out_info->debug_listener_cleared = JYPPX_FALSE;
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, kExecutionContextCallbackStateFeature, seh_exception_code);
    }
#else
    out_info->debug_listener_cleared = context_payload->setDebugListener(nullptr) ? JYPPX_TRUE : JYPPX_FALSE;
    out_info->has_debug_listener = context_payload->getDebugListener() != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#endif
}
#endif

static JYPPX_StatusCode populate_callback_state_snapshot(
    nvinfer1::IExecutionContext* context_payload,
    const char* tensor_name,
    JYPPX_TensorRtExecutionContextCallbackStateInfo* out_info)
{
    auto status = snapshot_output_allocator_presence_with_seh_guard(context_payload, tensor_name, out_info);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = snapshot_temporary_storage_allocator_presence_with_seh_guard(context_payload, out_info);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    status = snapshot_debug_listener_presence_with_seh_guard(context_payload, out_info);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = copy_callback_interface_info_to_state_with_seh_guard(
        context_payload->getOutputAllocator(tensor_name),
        &out_info->output_allocator_interface_info_available,
        out_info->output_allocator_interface_kind,
        sizeof(out_info->output_allocator_interface_kind),
        &out_info->output_allocator_interface_major,
        &out_info->output_allocator_interface_minor);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = copy_callback_interface_info_to_state_with_seh_guard(
        context_payload->getTemporaryStorageAllocator(),
        &out_info->temporary_storage_allocator_interface_info_available,
        out_info->temporary_storage_allocator_interface_kind,
        sizeof(out_info->temporary_storage_allocator_interface_kind),
        &out_info->temporary_storage_allocator_interface_major,
        &out_info->temporary_storage_allocator_interface_minor);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = copy_callback_interface_info_to_state_with_seh_guard(
        context_payload->getDebugListener(),
        &out_info->debug_listener_interface_info_available,
        out_info->debug_listener_interface_kind,
        sizeof(out_info->debug_listener_interface_kind),
        &out_info->debug_listener_interface_major,
        &out_info->debug_listener_interface_minor);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
#else
    out_info->debug_listener_clear_supported = JYPPX_FALSE;
    copy_callback_state_text(out_info->last_diagnostic, sizeof(out_info->last_diagnostic), "TensorRT 8 callback state snapshot does not include debug listener or interface-info metadata.");
#endif

    return JYPPX_STATUS_OK;
}
#endif
}

JYPPX_StatusCode JYPPX_TRT_EXECUTION_CONTEXT_CALLBACK_STATE_API(execution_context_get_callback_state_snapshot)(
    JYPPX_TensorRtExecutionContext* context,
    const char* tensor_name,
    JYPPX_TensorRtExecutionContextCallbackStateInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    reset_execution_context_callback_state_info(out_info);
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, kExecutionContextCallbackStateFeature);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        status = populate_callback_state_snapshot(context_payload, tensor_name, out_info);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        set_callback_state_status(
            out_info,
            JYPPX_STATUS_OK,
            "snapshot",
            "execution-context-callback-state-snapshot copied callback boundary state without exposing borrowed pointers.");
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        set_callback_state_status(out_info, JYPPX_STATUS_RUNTIME_ERROR, "snapshot-exception", exception.what());
        return callback_state_exception(exception);
    }
    catch (...)
    {
        set_callback_state_status(out_info, JYPPX_STATUS_RUNTIME_ERROR, "snapshot-exception", "unknown native exception");
        return callback_state_unknown_exception();
    }
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, kExecutionContextCallbackStateFeature);
#endif
}

JYPPX_StatusCode JYPPX_TRT_EXECUTION_CONTEXT_CALLBACK_STATE_API(execution_context_clear_callback_state)(
    JYPPX_TensorRtExecutionContext* context,
    const char* tensor_name,
    JYPPX_TensorRtExecutionContextCallbackStateInfo* out_info)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    reset_execution_context_callback_state_info(out_info);
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_TENSORRT
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context callback state clear");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    try
    {
        status = populate_callback_state_snapshot(context_payload, tensor_name, out_info);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        status = clear_output_allocator_with_state_seh_guard(context_payload, tensor_name, out_info);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        status = clear_temporary_storage_allocator_with_state_seh_guard(context_payload, out_info);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

#if JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
        status = clear_debug_listener_with_state_seh_guard(context_payload, out_info);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }
#else
        out_info->debug_listener_clear_supported = JYPPX_FALSE;
        out_info->debug_listener_cleared = JYPPX_FALSE;
#endif

        set_callback_state_status(
            out_info,
            JYPPX_STATUS_OK,
            "clear",
            "execution-context-callback-state-snapshot cleared supported callback attachments and copied post-clear state.");
        return JYPPX_STATUS_OK;
    }
    catch (const std::exception& exception)
    {
        set_callback_state_status(out_info, JYPPX_STATUS_RUNTIME_ERROR, "clear-exception", exception.what());
        return callback_state_exception(exception);
    }
    catch (...)
    {
        set_callback_state_status(out_info, JYPPX_STATUS_RUNTIME_ERROR, "clear-exception", "unknown native exception");
        return callback_state_unknown_exception();
    }
#else
    (void)context;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context callback state clear");
#endif
}
