namespace
{
JYPPX_StatusCode validate_ipc_import_token(
    const char* operation,
    const uint8_t* token,
    const size_t token_size)
{
    if (token == nullptr)
    {
        jyppx::cuda::set_cuda_error(
            operation,
            0,
            "ipc-import-token-required",
            "CUDA IPC import token must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (token_size != kCudaIpcExportTokenSize)
    {
        jyppx::cuda::set_cuda_error(
            operation,
            0,
            "ipc-import-token-size-invalid",
            "CUDA IPC import token must contain exactly 64 bytes.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}
}

JYPPX_StatusCode jyppx_cuda_ipc_import_event_safe(
    const uint8_t* token,
    const size_t token_size,
    JYPPX_CudaEvent** out_event)
{
    auto status = jyppx::cuda::validate_output_pointer(out_event, "out_event");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_event = nullptr;
    status = validate_ipc_import_token("cudaIpcOpenEventHandle", token, token_size);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    static_assert(sizeof(cudaIpcEventHandle_t) == kCudaIpcExportTokenSize, "Unexpected CUDA IPC event handle size.");
    cudaIpcEventHandle_t ipc_handle{};
    std::memcpy(&ipc_handle, token, sizeof(ipc_handle));

    auto* event_object = new (std::nothrow) EventObject{};
    if (event_object == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    event_object->base.magic = jyppx::cuda::kObjectMagic;
    event_object->base.kind = ObjectKind::Event;
    event_object->flags = 2U;
    event_object->handle = nullptr;

    status = jyppx::cuda::map_cuda_status(
        cudaIpcOpenEventHandle(&event_object->handle, ipc_handle),
        "cudaIpcOpenEventHandle");
    if (status != JYPPX_STATUS_OK)
    {
        delete event_object;
        return status;
    }

    *out_event = reinterpret_cast<JYPPX_CudaEvent*>(event_object);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA IPC event import");
#endif
}

JYPPX_StatusCode jyppx_cuda_ipc_import_memory_safe(
    const uint8_t* token,
    const size_t token_size,
    const size_t allocation_size,
    JYPPX_CudaMemory** out_memory)
{
    auto status = jyppx::cuda::validate_output_pointer(out_memory, "out_memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_memory = nullptr;
    status = validate_ipc_import_token("cudaIpcOpenMemHandle", token, token_size);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (allocation_size == 0)
    {
        jyppx::cuda::set_cuda_error(
            "cudaIpcOpenMemHandle",
            0,
            "ipc-import-size-required",
            "CUDA IPC memory import requires the exported allocation size.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    static_assert(sizeof(cudaIpcMemHandle_t) == kCudaIpcExportTokenSize, "Unexpected CUDA IPC memory handle size.");
    cudaIpcMemHandle_t ipc_handle{};
    std::memcpy(&ipc_handle, token, sizeof(ipc_handle));

    auto* memory = new (std::nothrow) MemoryObject{};
    if (memory == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    memory->base.magic = jyppx::cuda::kObjectMagic;
    memory->base.kind = ObjectKind::Memory;
    memory->pointer = nullptr;
    memory->size = allocation_size;
    memory->is_managed = false;
    memory->is_ipc_exportable = false;
    memory->release_mode = MemoryReleaseMode::IpcClose;

    status = jyppx::cuda::map_cuda_status(
        cudaIpcOpenMemHandle(&memory->pointer, ipc_handle, cudaIpcMemLazyEnablePeerAccess),
        "cudaIpcOpenMemHandle");
    if (status != JYPPX_STATUS_OK)
    {
        delete memory;
        return status;
    }

    *out_memory = reinterpret_cast<JYPPX_CudaMemory*>(memory);
    return JYPPX_STATUS_OK;
#else
    (void)allocation_size;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA IPC memory import");
#endif
}

JYPPX_StatusCode jyppx_cuda_ipc_close_imported_memory_safe(JYPPX_CudaMemory* memory)
{
    if (memory == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    auto status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(memory);
    if (memory_object->release_mode != MemoryReleaseMode::IpcClose)
    {
        jyppx::cuda::set_cuda_error(
            "cudaIpcCloseMemHandle",
            0,
            "ipc-imported-memory-required",
            "cudaIpcCloseMemHandle requires memory created by the IPC import path.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    status = jyppx::cuda::map_cuda_status(
        cudaIpcCloseMemHandle(memory_object->pointer),
        "cudaIpcCloseMemHandle");
#else
    status = JYPPX_STATUS_OK;
#endif
    delete memory_object;
    return status;
}
