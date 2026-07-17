#include <cstring>
#include <new>

namespace
{
JYPPX_StatusCode report_primary_execution_context_exception(
    const char* operation,
    const char* error_name,
    const char* message) noexcept
{
    try
    {
        jyppx::cuda::set_cuda_error(operation, 0, error_name, message);
    }
    catch (...)
    {
    }
    return std::strcmp(error_name, "native-allocation-failed") == 0
        ? JYPPX_STATUS_OUT_OF_MEMORY
        : JYPPX_STATUS_RUNTIME_ERROR;
}

JYPPX_StatusCode primary_execution_context_not_supported(const char* operation) noexcept
{
    try
    {
        jyppx::cuda::set_cuda_error(
            operation,
            0,
            "cuda-version-not-supported",
            "CUDA primary execution-context APIs require CUDA Toolkit 13.0 or later.");
    }
    catch (...)
    {
    }
    return JYPPX_STATUS_NOT_SUPPORTED;
}

template <typename TCallable>
JYPPX_StatusCode run_primary_execution_context_noexcept(const char* operation, TCallable&& callable) noexcept
{
    try
    {
        return callable();
    }
    catch (const std::bad_alloc&)
    {
        return report_primary_execution_context_exception(
            operation,
            "native-allocation-failed",
            "The CUDA primary execution-context bridge could not allocate owner storage.");
    }
    catch (...)
    {
        return report_primary_execution_context_exception(
            operation,
            "native-exception-caught",
            "A native exception was contained by the CUDA primary execution-context bridge.");
    }
}

JYPPX_StatusCode get_primary_execution_context_impl(
    const int32_t device,
    JYPPX_CudaExecutionContext** out_context) noexcept
{
    return run_primary_execution_context_noexcept("cudaDeviceGetExecutionCtx", [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_context, "out_context");
        if (status != JYPPX_STATUS_OK) return status;
        *out_context = nullptr;
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
        auto* context = new (std::nothrow) jyppx::cuda::ExecutionContextObject{};
        if (context == nullptr) return JYPPX_STATUS_OUT_OF_MEMORY;
        context->base.magic = jyppx::cuda::kObjectMagic;
        context->base.kind = jyppx::cuda::ObjectKind::ExecutionContext;
        context->device_ordinal = device;
        context->is_primary = true;
        context->handle = nullptr;
        status = jyppx::cuda::map_cuda_status(
            cudaDeviceGetExecutionCtx(&context->handle, device),
            "cudaDeviceGetExecutionCtx");
        if (status != JYPPX_STATUS_OK)
        {
            delete context;
            return status;
        }
        if (context->handle == nullptr)
        {
            delete context;
            jyppx::cuda::set_cuda_error(
                "cudaDeviceGetExecutionCtx",
                0,
                "invalid-execution-context-handle",
                "CUDA reported success but returned a null primary execution-context handle.");
            return JYPPX_STATUS_INVALID_STATE;
        }
        *out_context = reinterpret_cast<JYPPX_CudaExecutionContext*>(context);
        return JYPPX_STATUS_OK;
#elif JYPPX_HAS_CUDA_TOOLKIT
        (void)device;
        return primary_execution_context_not_supported("cudaDeviceGetExecutionCtx");
#else
        (void)device;
        return jyppx::cuda::report_cuda_dependency_missing("CUDA primary execution-context query");
#endif
    });
}

JYPPX_StatusCode release_primary_execution_context_wrapper_impl(
    JYPPX_CudaExecutionContext* context_handle) noexcept
{
    return run_primary_execution_context_noexcept("primary execution-context wrapper release", [&]() {
        if (context_handle == nullptr) return JYPPX_STATUS_OK;
        auto status = jyppx::cuda::validate_execution_context(context_handle, "context");
        if (status != JYPPX_STATUS_OK) return status;
        auto* context = reinterpret_cast<jyppx::cuda::ExecutionContextObject*>(context_handle);
        if (!context->is_primary)
        {
            jyppx::cuda::set_cuda_error(
                "primary execution-context wrapper release",
                0,
                "unexpected-context-ownership",
                "The primary execution-context bridge wrapper cannot release an owning context.");
            return JYPPX_STATUS_INVALID_STATE;
        }

        // cudaDeviceGetExecutionCtx returns the device primary context. Calling
        // cudaExecutionCtxDestroy for that handle is explicitly undefined behavior.
        context->base.magic = 0;
        context->handle = nullptr;
        delete context;
        return JYPPX_STATUS_OK;
    });
}

JYPPX_StatusCode get_execution_context_device_impl(
    JYPPX_CudaExecutionContext* context_handle,
    int32_t* out_device) noexcept
{
    return run_primary_execution_context_noexcept("cudaExecutionCtxGetDevice", [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_device, "out_device");
        if (status != JYPPX_STATUS_OK) return status;
        *out_device = -1;
        status = jyppx::cuda::validate_execution_context(context_handle, "context");
        if (status != JYPPX_STATUS_OK) return status;
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
        auto* context = reinterpret_cast<jyppx::cuda::ExecutionContextObject*>(context_handle);
        return jyppx::cuda::map_cuda_status(
            cudaExecutionCtxGetDevice(out_device, context->handle),
            "cudaExecutionCtxGetDevice");
#elif JYPPX_HAS_CUDA_TOOLKIT
        return primary_execution_context_not_supported("cudaExecutionCtxGetDevice");
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA execution-context device query");
#endif
    });
}

JYPPX_StatusCode get_execution_context_id_impl(
    JYPPX_CudaExecutionContext* context_handle,
    uint64_t* out_context_id) noexcept
{
    return run_primary_execution_context_noexcept("cudaExecutionCtxGetId", [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_context_id, "out_context_id");
        if (status != JYPPX_STATUS_OK) return status;
        *out_context_id = 0;
        status = jyppx::cuda::validate_execution_context(context_handle, "context");
        if (status != JYPPX_STATUS_OK) return status;
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
        auto* context = reinterpret_cast<jyppx::cuda::ExecutionContextObject*>(context_handle);
        unsigned long long context_id = 0;
        status = jyppx::cuda::map_cuda_status(
            cudaExecutionCtxGetId(context->handle, &context_id),
            "cudaExecutionCtxGetId");
        if (status == JYPPX_STATUS_OK) *out_context_id = static_cast<uint64_t>(context_id);
        return status;
#elif JYPPX_HAS_CUDA_TOOLKIT
        return primary_execution_context_not_supported("cudaExecutionCtxGetId");
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA execution-context id query");
#endif
    });
}

JYPPX_StatusCode synchronize_execution_context_impl(
    JYPPX_CudaExecutionContext* context_handle) noexcept
{
    return run_primary_execution_context_noexcept("cudaExecutionCtxSynchronize", [&]() {
        auto status = jyppx::cuda::validate_execution_context(context_handle, "context");
        if (status != JYPPX_STATUS_OK) return status;
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
        auto* context = reinterpret_cast<jyppx::cuda::ExecutionContextObject*>(context_handle);
        return jyppx::cuda::map_cuda_status(
            cudaExecutionCtxSynchronize(context->handle),
            "cudaExecutionCtxSynchronize");
#elif JYPPX_HAS_CUDA_TOOLKIT
        return primary_execution_context_not_supported("cudaExecutionCtxSynchronize");
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA execution-context synchronization");
#endif
    });
}

JYPPX_StatusCode create_execution_context_stream_impl(
    JYPPX_CudaExecutionContext* context_handle,
    const uint32_t flags,
    const int32_t priority,
    JYPPX_CudaStream** out_stream) noexcept
{
    return run_primary_execution_context_noexcept("cudaExecutionCtxStreamCreate", [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_stream, "out_stream");
        if (status != JYPPX_STATUS_OK) return status;
        *out_stream = nullptr;
        status = jyppx::cuda::validate_execution_context(context_handle, "context");
        if (status != JYPPX_STATUS_OK) return status;
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
        auto* stream = new (std::nothrow) jyppx::cuda::StreamObject{};
        if (stream == nullptr) return JYPPX_STATUS_OUT_OF_MEMORY;
        stream->base.magic = jyppx::cuda::kObjectMagic;
        stream->base.kind = jyppx::cuda::ObjectKind::Stream;
        stream->flags = flags;
        stream->handle = nullptr;
        auto* context = reinterpret_cast<jyppx::cuda::ExecutionContextObject*>(context_handle);
        status = jyppx::cuda::map_cuda_status(
            cudaExecutionCtxStreamCreate(&stream->handle, context->handle, flags, priority),
            "cudaExecutionCtxStreamCreate");
        if (status != JYPPX_STATUS_OK)
        {
            delete stream;
            return status;
        }
        *out_stream = reinterpret_cast<JYPPX_CudaStream*>(stream);
        return JYPPX_STATUS_OK;
#elif JYPPX_HAS_CUDA_TOOLKIT
        (void)flags;
        (void)priority;
        return primary_execution_context_not_supported("cudaExecutionCtxStreamCreate");
#else
        (void)flags;
        (void)priority;
        return jyppx::cuda::report_cuda_dependency_missing("CUDA execution-context stream creation");
#endif
    });
}

JYPPX_StatusCode record_execution_context_event_impl(
    JYPPX_CudaExecutionContext* context_handle,
    JYPPX_CudaEvent* event_handle) noexcept
{
    return run_primary_execution_context_noexcept("cudaExecutionCtxRecordEvent", [&]() {
        auto status = jyppx::cuda::validate_execution_context(context_handle, "context");
        if (status != JYPPX_STATUS_OK) return status;
        status = jyppx::cuda::validate_event(event_handle, "event");
        if (status != JYPPX_STATUS_OK) return status;
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
        auto* context = reinterpret_cast<jyppx::cuda::ExecutionContextObject*>(context_handle);
        auto* event_object = reinterpret_cast<jyppx::cuda::EventObject*>(event_handle);
        return jyppx::cuda::map_cuda_status(
            cudaExecutionCtxRecordEvent(context->handle, event_object->handle),
            "cudaExecutionCtxRecordEvent");
#elif JYPPX_HAS_CUDA_TOOLKIT
        return primary_execution_context_not_supported("cudaExecutionCtxRecordEvent");
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA execution-context event record");
#endif
    });
}

JYPPX_StatusCode wait_execution_context_event_impl(
    JYPPX_CudaExecutionContext* context_handle,
    JYPPX_CudaEvent* event_handle) noexcept
{
    return run_primary_execution_context_noexcept("cudaExecutionCtxWaitEvent", [&]() {
        auto status = jyppx::cuda::validate_execution_context(context_handle, "context");
        if (status != JYPPX_STATUS_OK) return status;
        status = jyppx::cuda::validate_event(event_handle, "event");
        if (status != JYPPX_STATUS_OK) return status;
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
        auto* context = reinterpret_cast<jyppx::cuda::ExecutionContextObject*>(context_handle);
        auto* event_object = reinterpret_cast<jyppx::cuda::EventObject*>(event_handle);
        return jyppx::cuda::map_cuda_status(
            cudaExecutionCtxWaitEvent(context->handle, event_object->handle),
            "cudaExecutionCtxWaitEvent");
#elif JYPPX_HAS_CUDA_TOOLKIT
        return primary_execution_context_not_supported("cudaExecutionCtxWaitEvent");
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA execution-context event wait");
#endif
    });
}
}

#if defined(_MSC_VER) && defined(_WIN32)
#define JYPPX_CUDA_PRIMARY_CONTEXT_GUARD(operation, call) \
    __try { return call; } \
    __except (1) { return report_primary_execution_context_exception(operation, "windows-seh-caught", "A Windows structured exception was contained by the CUDA primary execution-context bridge."); }
#else
#define JYPPX_CUDA_PRIMARY_CONTEXT_GUARD(operation, call) return call
#endif

JYPPX_StatusCode jyppx_cuda_primary_execution_context_get_safe(int32_t device, JYPPX_CudaExecutionContext** out_context)
{
    JYPPX_CUDA_PRIMARY_CONTEXT_GUARD("cudaDeviceGetExecutionCtx", get_primary_execution_context_impl(device, out_context));
}

JYPPX_StatusCode jyppx_cuda_primary_execution_context_release_wrapper_safe(JYPPX_CudaExecutionContext* context)
{
    JYPPX_CUDA_PRIMARY_CONTEXT_GUARD("primary execution-context wrapper release", release_primary_execution_context_wrapper_impl(context));
}

JYPPX_StatusCode jyppx_cuda_execution_context_get_device_safe(JYPPX_CudaExecutionContext* context, int32_t* out_device)
{
    JYPPX_CUDA_PRIMARY_CONTEXT_GUARD("cudaExecutionCtxGetDevice", get_execution_context_device_impl(context, out_device));
}

JYPPX_StatusCode jyppx_cuda_execution_context_get_id_safe(JYPPX_CudaExecutionContext* context, uint64_t* out_context_id)
{
    JYPPX_CUDA_PRIMARY_CONTEXT_GUARD("cudaExecutionCtxGetId", get_execution_context_id_impl(context, out_context_id));
}

JYPPX_StatusCode jyppx_cuda_execution_context_synchronize_safe(JYPPX_CudaExecutionContext* context)
{
    JYPPX_CUDA_PRIMARY_CONTEXT_GUARD("cudaExecutionCtxSynchronize", synchronize_execution_context_impl(context));
}

JYPPX_StatusCode jyppx_cuda_execution_context_create_stream_safe(JYPPX_CudaExecutionContext* context, uint32_t flags, int32_t priority, JYPPX_CudaStream** out_stream)
{
    JYPPX_CUDA_PRIMARY_CONTEXT_GUARD("cudaExecutionCtxStreamCreate", create_execution_context_stream_impl(context, flags, priority, out_stream));
}

JYPPX_StatusCode jyppx_cuda_execution_context_record_event_safe(JYPPX_CudaExecutionContext* context, JYPPX_CudaEvent* event_handle)
{
    JYPPX_CUDA_PRIMARY_CONTEXT_GUARD("cudaExecutionCtxRecordEvent", record_execution_context_event_impl(context, event_handle));
}

JYPPX_StatusCode jyppx_cuda_execution_context_wait_event_safe(JYPPX_CudaExecutionContext* context, JYPPX_CudaEvent* event_handle)
{
    JYPPX_CUDA_PRIMARY_CONTEXT_GUARD("cudaExecutionCtxWaitEvent", wait_execution_context_event_impl(context, event_handle));
}

#undef JYPPX_CUDA_PRIMARY_CONTEXT_GUARD
