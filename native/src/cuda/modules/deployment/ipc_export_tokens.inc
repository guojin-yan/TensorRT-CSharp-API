namespace
{
constexpr size_t kCudaIpcExportTokenSize = 64;

JYPPX_StatusCode validate_ipc_output_buffer(
    const char* operation,
    uint8_t* output_buffer,
    const size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::cuda::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = kCudaIpcExportTokenSize;
    if (output_buffer_size == 0)
    {
        return JYPPX_STATUS_OK;
    }

    if (output_buffer == nullptr || output_buffer_size < kCudaIpcExportTokenSize)
    {
        jyppx::cuda::set_cuda_error(
            operation,
            0,
            "ipc-export-buffer-too-small",
            "CUDA IPC export output buffer must contain at least 64 bytes.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}
}

JYPPX_StatusCode jyppx_cuda_ipc_export_event_token_safe(
    JYPPX_CudaEvent* event_handle,
    uint8_t* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::cuda::validate_event(event_handle, "event_handle");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_ipc_output_buffer("cudaIpcGetEventHandle", output_buffer, output_buffer_size, out_required_size);
    if (status != JYPPX_STATUS_OK || output_buffer_size == 0)
    {
        return status;
    }

    auto* event_object = reinterpret_cast<EventObject*>(event_handle);
    constexpr uint32_t required_flags = 2U | 4U;
    if ((event_object->flags & required_flags) != required_flags)
    {
        jyppx::cuda::set_cuda_error(
            "cudaIpcGetEventHandle",
            0,
            "ipc-event-flags-required",
            "CUDA IPC event export requires Interprocess and DisableTiming event flags.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    static_assert(sizeof(cudaIpcEventHandle_t) == kCudaIpcExportTokenSize, "Unexpected CUDA IPC event handle size.");
    cudaIpcEventHandle_t token{};
    status = jyppx::cuda::map_cuda_status(cudaIpcGetEventHandle(&token, event_object->handle), "cudaIpcGetEventHandle");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memcpy(output_buffer, &token, sizeof(token));
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA IPC event handle export");
#endif
}

JYPPX_StatusCode jyppx_cuda_ipc_export_memory_token_safe(
    JYPPX_CudaMemory* memory,
    uint8_t* output_buffer,
    size_t output_buffer_size,
    size_t* out_required_size)
{
    auto status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_ipc_output_buffer("cudaIpcGetMemHandle", output_buffer, output_buffer_size, out_required_size);
    if (status != JYPPX_STATUS_OK || output_buffer_size == 0)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(memory);
    if (!memory_object->is_ipc_exportable)
    {
        jyppx::cuda::set_cuda_error(
            "cudaIpcGetMemHandle",
            0,
            "ipc-memory-allocation-required",
            "CUDA IPC memory export requires the base address of a synchronous cudaMalloc allocation.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    static_assert(sizeof(cudaIpcMemHandle_t) == kCudaIpcExportTokenSize, "Unexpected CUDA IPC memory handle size.");
    cudaIpcMemHandle_t token{};
    status = jyppx::cuda::map_cuda_status(cudaIpcGetMemHandle(&token, memory_object->pointer), "cudaIpcGetMemHandle");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memcpy(output_buffer, &token, sizeof(token));
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA IPC memory handle export");
#endif
}
