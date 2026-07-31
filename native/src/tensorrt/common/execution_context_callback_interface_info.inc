#ifndef JYPPX_TRT_EXECUTION_CONTEXT_CALLBACK_INTERFACE_INFO_API
#error "JYPPX_TRT_EXECUTION_CONTEXT_CALLBACK_INTERFACE_INFO_API must be defined before including execution_context_callback_interface_info.inc"
#endif

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
static JYPPX_StatusCode execution_context_callback_interface_info_exception(
    const char* feature_name,
    const std::exception& exception)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, exception.what());
}

static JYPPX_StatusCode execution_context_callback_interface_info_unknown_exception(const char* feature_name)
{
    return jyppx::tensorrt::report_vendor_exception(kLine, feature_name, "unknown native exception");
}

template <typename TObject>
static JYPPX_StatusCode execution_context_get_borrowed_interface_info_with_seh_guard(
    const TObject* object,
    nvinfer1::InterfaceInfo* out_info,
    const char* feature_name)
{
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        *out_info = object->getInterfaceInfo();
        return JYPPX_STATUS_OK;
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        *out_info = nvinfer1::InterfaceInfo{"", 0, 0};
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    *out_info = object->getInterfaceInfo();
    return JYPPX_STATUS_OK;
#endif
}

template <typename TObject>
static JYPPX_StatusCode execution_context_copy_borrowed_interface_info(
    const TObject* object,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor,
    const char* feature_name,
    const char* missing_message)
{
    if (object == nullptr)
    {
        *out_required_size = 0;
        *out_major = 0;
        *out_minor = 0;
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_TENSORRT, missing_message);
        return JYPPX_STATUS_NOT_FOUND;
    }

    nvinfer1::InterfaceInfo info{"", 0, 0};
    auto status = execution_context_get_borrowed_interface_info_with_seh_guard(object, &info, feature_name);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_interface_info_to_buffer(info, output_buffer, output_buffer_size, out_required_size, out_major, out_minor);
}

static void reset_execution_context_callback_interface_info_outputs(
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor) noexcept
{
    if (output_buffer != nullptr && output_buffer_size != 0U)
    {
        output_buffer[0] = '\0';
    }
    *out_required_size = 0;
    *out_major = 0;
    *out_minor = 0;
}

static JYPPX_StatusCode execution_context_copy_output_allocator_interface_info_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    const char* tensor_name,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor)
{
    constexpr const char* feature_name = "execution context output allocator interface info query";
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        return execution_context_copy_borrowed_interface_info(
            context_payload->getOutputAllocator(tensor_name),
            output_buffer,
            output_buffer_size,
            out_required_size,
            out_major,
            out_minor,
            feature_name,
            "Execution context output allocator is not set for the requested tensor.");
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        reset_execution_context_callback_interface_info_outputs(
            output_buffer, output_buffer_size, out_required_size, out_major, out_minor);
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    return execution_context_copy_borrowed_interface_info(
        context_payload->getOutputAllocator(tensor_name),
        output_buffer,
        output_buffer_size,
        out_required_size,
        out_major,
        out_minor,
        feature_name,
        "Execution context output allocator is not set for the requested tensor.");
#endif
}

static JYPPX_StatusCode execution_context_copy_temporary_storage_allocator_interface_info_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor)
{
    constexpr const char* feature_name = "execution context temporary storage allocator interface info query";
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        return execution_context_copy_borrowed_interface_info(
            context_payload->getTemporaryStorageAllocator(),
            output_buffer,
            output_buffer_size,
            out_required_size,
            out_major,
            out_minor,
            feature_name,
            "Execution context temporary storage allocator is not set.");
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        reset_execution_context_callback_interface_info_outputs(
            output_buffer, output_buffer_size, out_required_size, out_major, out_minor);
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    return execution_context_copy_borrowed_interface_info(
        context_payload->getTemporaryStorageAllocator(),
        output_buffer,
        output_buffer_size,
        out_required_size,
        out_major,
        out_minor,
        feature_name,
        "Execution context temporary storage allocator is not set.");
#endif
}

static JYPPX_StatusCode execution_context_copy_debug_listener_interface_info_with_seh_guard(
    nvinfer1::IExecutionContext* context_payload,
    char* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor)
{
    constexpr const char* feature_name = "execution context debug listener interface info query";
#if defined(_MSC_VER)
    uint32_t seh_exception_code = 0;
    __try
    {
        return execution_context_copy_borrowed_interface_info(
            context_payload->getDebugListener(),
            output_buffer,
            output_buffer_size,
            out_required_size,
            out_major,
            out_minor,
            feature_name,
            "Execution context debug listener is not set.");
    }
    __except (jyppx::tensorrt::capture_vendor_seh_exception_code(&seh_exception_code, GetExceptionCode()))
    {
        reset_execution_context_callback_interface_info_outputs(
            output_buffer, output_buffer_size, out_required_size, out_major, out_minor);
        return jyppx::tensorrt::report_vendor_seh_exception(kLine, feature_name, seh_exception_code);
    }
#else
    return execution_context_copy_borrowed_interface_info(
        context_payload->getDebugListener(),
        output_buffer,
        output_buffer_size,
        out_required_size,
        out_major,
        out_minor,
        feature_name,
        "Execution context debug listener is not set.");
#endif
}
#endif

JYPPX_StatusCode JYPPX_TRT_EXECUTION_CONTEXT_CALLBACK_INTERFACE_INFO_API(execution_context_get_output_allocator_interface_info)(
    JYPPX_TensorRtExecutionContext* context,
    const char* tensor_name,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_major, "out_major");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_minor, "out_minor");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_named_tensor(tensor_name);
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
    *out_major = 0;
    *out_minor = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context output allocator interface info query");
    if (status != JYPPX_STATUS_OK) { return status; }

    try
    {
        return execution_context_copy_output_allocator_interface_info_with_seh_guard(
            context_payload,
            tensor_name,
            output_buffer,
            output_buffer_size,
            out_required_size,
            out_major,
            out_minor);
    }
    catch (const std::exception& exception)
    {
        *out_required_size = 0;
        *out_major = 0;
        *out_minor = 0;
        return execution_context_callback_interface_info_exception("execution context output allocator interface info query", exception);
    }
    catch (...)
    {
        *out_required_size = 0;
        *out_major = 0;
        *out_minor = 0;
        return execution_context_callback_interface_info_unknown_exception("execution context output allocator interface info query");
    }
#elif JYPPX_HAS_TENSORRT
    (void)context;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context output allocator interface info query");
#else
    (void)context;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context output allocator interface info query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_EXECUTION_CONTEXT_CALLBACK_INTERFACE_INFO_API(execution_context_get_temporary_storage_allocator_interface_info)(
    JYPPX_TensorRtExecutionContext* context,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_major, "out_major");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_minor, "out_minor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
    *out_major = 0;
    *out_minor = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context temporary storage allocator interface info query");
    if (status != JYPPX_STATUS_OK) { return status; }

    try
    {
        return execution_context_copy_temporary_storage_allocator_interface_info_with_seh_guard(
            context_payload,
            output_buffer,
            output_buffer_size,
            out_required_size,
            out_major,
            out_minor);
    }
    catch (const std::exception& exception)
    {
        *out_required_size = 0;
        *out_major = 0;
        *out_minor = 0;
        return execution_context_callback_interface_info_exception("execution context temporary storage allocator interface info query", exception);
    }
    catch (...)
    {
        *out_required_size = 0;
        *out_major = 0;
        *out_minor = 0;
        return execution_context_callback_interface_info_unknown_exception("execution context temporary storage allocator interface info query");
    }
#elif JYPPX_HAS_TENSORRT
    (void)context;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context temporary storage allocator interface info query");
#else
    (void)context;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context temporary storage allocator interface info query");
#endif
}

JYPPX_StatusCode JYPPX_TRT_EXECUTION_CONTEXT_CALLBACK_INTERFACE_INFO_API(execution_context_get_debug_listener_interface_info)(
    JYPPX_TensorRtExecutionContext* context,
    char* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size,
    int32_t* out_major,
    int32_t* out_minor)
{
    auto status = jyppx::tensorrt::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_major, "out_major");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::tensorrt::validate_output_pointer(out_minor, "out_minor");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_required_size = 0;
    *out_major = 0;
    *out_minor = 0;

#if JYPPX_HAS_TENSORRT && JYPPX_TENSORRT_VERSION_MAJOR_NUM >= 10
    nvinfer1::IExecutionContext* context_payload = nullptr;
    status = get_context_payload_ext(context, &context_payload, "execution context debug listener interface info query");
    if (status != JYPPX_STATUS_OK) { return status; }

    try
    {
        return execution_context_copy_debug_listener_interface_info_with_seh_guard(
            context_payload,
            output_buffer,
            output_buffer_size,
            out_required_size,
            out_major,
            out_minor);
    }
    catch (const std::exception& exception)
    {
        *out_required_size = 0;
        *out_major = 0;
        *out_minor = 0;
        return execution_context_callback_interface_info_exception("execution context debug listener interface info query", exception);
    }
    catch (...)
    {
        *out_required_size = 0;
        *out_major = 0;
        *out_minor = 0;
        return execution_context_callback_interface_info_unknown_exception("execution context debug listener interface info query");
    }
#elif JYPPX_HAS_TENSORRT
    (void)context;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_mismatch(kLine, JYPPX_TENSORRT_VERSION_MAJOR_NUM, "execution context debug listener interface info query");
#else
    (void)context;
    (void)output_buffer;
    (void)output_buffer_size;
    return jyppx::tensorrt::report_vendor_missing(kLine, "execution context debug listener interface info query");
#endif
}
