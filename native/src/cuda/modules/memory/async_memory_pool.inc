JYPPX_StatusCode jyppx_cuda_memory_alloc_async(size_t size, JYPPX_CudaStream* stream, JYPPX_CudaMemory** out_memory)
{
    auto status = jyppx::cuda::validate_output_pointer(out_memory, "out_memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_memory = nullptr;

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* memory = new (std::nothrow) MemoryObject{};
    if (memory == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    memory->base.magic = jyppx::cuda::kObjectMagic;
    memory->base.kind = ObjectKind::Memory;
    memory->pointer = nullptr;
    memory->size = size;
    memory->is_managed = false;

    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    status = jyppx::cuda::map_cuda_status(cudaMallocAsync(&memory->pointer, size, stream_object->handle), "cudaMallocAsync");
    if (status != JYPPX_STATUS_OK)
    {
        delete memory;
        return status;
    }

    *out_memory = reinterpret_cast<JYPPX_CudaMemory*>(memory);
    return JYPPX_STATUS_OK;
#else
    (void)size;
    (void)stream;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async memory allocation");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_free_async(JYPPX_CudaMemory* memory, JYPPX_CudaStream* stream)
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

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(memory);
#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    status = jyppx::cuda::map_cuda_status(cudaFreeAsync(memory_object->pointer, stream_object->handle), "cudaFreeAsync");
#else
    status = JYPPX_STATUS_OK;
#endif
    delete memory_object;
    return status;
}

JYPPX_StatusCode jyppx_cuda_memory_alloc_from_pool_async(size_t size, uint64_t pool_handle, JYPPX_CudaStream* stream, JYPPX_CudaMemory** out_memory)
{
    auto status = jyppx::cuda::validate_output_pointer(out_memory, "out_memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_memory = nullptr;
    if (pool_handle == 0)
    {
        jyppx::cuda::set_cuda_error("cudaMallocFromPoolAsync", 0, "invalid-pool", "CUDA memory pool handle must not be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* memory = new (std::nothrow) MemoryObject{};
    if (memory == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    memory->base.magic = jyppx::cuda::kObjectMagic;
    memory->base.kind = ObjectKind::Memory;
    memory->pointer = nullptr;
    memory->size = size;
    memory->is_managed = false;

    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    cudaMemPool_t pool = reinterpret_cast<cudaMemPool_t>(static_cast<uintptr_t>(pool_handle));
    status = jyppx::cuda::map_cuda_status(cudaMallocFromPoolAsync(&memory->pointer, size, pool, stream_object->handle), "cudaMallocFromPoolAsync");
    if (status != JYPPX_STATUS_OK)
    {
        delete memory;
        return status;
    }

    *out_memory = reinterpret_cast<JYPPX_CudaMemory*>(memory);
    return JYPPX_STATUS_OK;
#else
    (void)size;
    (void)pool_handle;
    (void)stream;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async memory allocation from memory pool");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_pool_create(int32_t device, uint64_t* out_pool_handle)
{
    auto status = jyppx::cuda::validate_output_pointer(out_pool_handle, "out_pool_handle");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_pool_handle = 0;
    if (device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemPoolCreate", 0, "invalid-device", "Device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemPoolProps properties{};
    properties.allocType = cudaMemAllocationTypePinned;
    properties.handleTypes = cudaMemHandleTypeNone;
    properties.location.type = cudaMemLocationTypeDevice;
    properties.location.id = device;

    cudaMemPool_t pool = nullptr;
    status = jyppx::cuda::map_cuda_status(cudaMemPoolCreate(&pool, &properties), "cudaMemPoolCreate");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_pool_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pool));
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memory pool creation");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_pool_destroy(uint64_t pool_handle)
{
    if (pool_handle == 0)
    {
        return JYPPX_STATUS_OK;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemPool_t pool = reinterpret_cast<cudaMemPool_t>(static_cast<uintptr_t>(pool_handle));
    return jyppx::cuda::map_cuda_status(cudaMemPoolDestroy(pool), "cudaMemPoolDestroy");
#else
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode jyppx_cuda_device_get_default_memory_pool(int32_t device, uint64_t* out_pool_handle)
{
    auto status = jyppx::cuda::validate_output_pointer(out_pool_handle, "out_pool_handle");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_pool_handle = 0;
    if (device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaDeviceGetDefaultMemPool", 0, "invalid-device", "Device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemPool_t pool = nullptr;
    status = jyppx::cuda::map_cuda_status(cudaDeviceGetDefaultMemPool(&pool, device), "cudaDeviceGetDefaultMemPool");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_pool_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pool));
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA default memory pool query");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_get_current_memory_pool(int32_t device, uint64_t* out_pool_handle)
{
    auto status = jyppx::cuda::validate_output_pointer(out_pool_handle, "out_pool_handle");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_pool_handle = 0;
    if (device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaDeviceGetMemPool", 0, "invalid-device", "Device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemPool_t pool = nullptr;
    status = jyppx::cuda::map_cuda_status(cudaDeviceGetMemPool(&pool, device), "cudaDeviceGetMemPool");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_pool_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pool));
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA current memory pool query");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_set_current_memory_pool(int32_t device, uint64_t pool_handle)
{
    if (device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaDeviceSetMemPool", 0, "invalid-device", "Device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (pool_handle == 0)
    {
        jyppx::cuda::set_cuda_error("cudaDeviceSetMemPool", 0, "invalid-pool", "CUDA memory pool handle must not be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemPool_t pool = reinterpret_cast<cudaMemPool_t>(static_cast<uintptr_t>(pool_handle));
    return jyppx::cuda::map_cuda_status(cudaDeviceSetMemPool(device, pool), "cudaDeviceSetMemPool");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA current memory pool set");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_pool_get_attribute(uint64_t pool_handle, int32_t attribute, int64_t* out_value)
{
    auto status = jyppx::cuda::validate_output_pointer(out_value, "out_value");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_value = 0;
    if (pool_handle == 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemPoolGetAttribute", 0, "invalid-pool", "CUDA memory pool handle must not be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemPool_t pool = reinterpret_cast<cudaMemPool_t>(static_cast<uintptr_t>(pool_handle));
    return jyppx::cuda::map_cuda_status(cudaMemPoolGetAttribute(pool, static_cast<cudaMemPoolAttr>(attribute), out_value), "cudaMemPoolGetAttribute");
#else
    (void)attribute;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memory pool attribute query");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_pool_set_attribute(uint64_t pool_handle, int32_t attribute, int64_t value)
{
    if (pool_handle == 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemPoolSetAttribute", 0, "invalid-pool", "CUDA memory pool handle must not be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemPool_t pool = reinterpret_cast<cudaMemPool_t>(static_cast<uintptr_t>(pool_handle));
    return jyppx::cuda::map_cuda_status(cudaMemPoolSetAttribute(pool, static_cast<cudaMemPoolAttr>(attribute), &value), "cudaMemPoolSetAttribute");
#else
    (void)attribute;
    (void)value;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memory pool attribute set");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_pool_trim_to(uint64_t pool_handle, size_t min_bytes_to_keep)
{
    if (pool_handle == 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemPoolTrimTo", 0, "invalid-pool", "CUDA memory pool handle must not be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemPool_t pool = reinterpret_cast<cudaMemPool_t>(static_cast<uintptr_t>(pool_handle));
    return jyppx::cuda::map_cuda_status(cudaMemPoolTrimTo(pool, min_bytes_to_keep), "cudaMemPoolTrimTo");
#else
    (void)min_bytes_to_keep;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memory pool trim");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_pool_set_access(uint64_t pool_handle, int32_t device, uint32_t flags)
{
    if (pool_handle == 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemPoolSetAccess", 0, "invalid-pool", "CUDA memory pool handle must not be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemPoolSetAccess", 0, "invalid-device", "Device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemAccessDesc access{};
    access.location.type = cudaMemLocationTypeDevice;
    access.location.id = device;
    access.flags = static_cast<cudaMemAccessFlags>(flags);

    cudaMemPool_t pool = reinterpret_cast<cudaMemPool_t>(static_cast<uintptr_t>(pool_handle));
    return jyppx::cuda::map_cuda_status(cudaMemPoolSetAccess(pool, &access, 1), "cudaMemPoolSetAccess");
#else
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memory pool access set");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_pool_get_access(uint64_t pool_handle, int32_t device, uint32_t* out_flags)
{
    auto status = jyppx::cuda::validate_output_pointer(out_flags, "out_flags");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_flags = 0;
    if (pool_handle == 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemPoolGetAccess", 0, "invalid-pool", "CUDA memory pool handle must not be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemPoolGetAccess", 0, "invalid-device", "Device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemLocation location{};
    location.type = cudaMemLocationTypeDevice;
    location.id = device;

    cudaMemAccessFlags access_flags = cudaMemAccessFlagsProtNone;
    cudaMemPool_t pool = reinterpret_cast<cudaMemPool_t>(static_cast<uintptr_t>(pool_handle));
    status = jyppx::cuda::map_cuda_status(cudaMemPoolGetAccess(&access_flags, pool, &location), "cudaMemPoolGetAccess");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_flags = static_cast<uint32_t>(access_flags);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memory pool access query");
#endif
}

JYPPX_StatusCode jyppx_cuda_mem_get_default_mem_pool(const JYPPX_CudaMemLocation* location, int32_t allocation_type, uint64_t* out_pool_handle)
{
    auto status = jyppx::cuda::validate_output_pointer(out_pool_handle, "out_pool_handle");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_pool_handle = 0;
    if (location == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemGetDefaultMemPool", 0, "invalid-location", "CUDA memory location must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    cudaMemLocation cuda_location = make_bridge_mem_location(*location);
    cudaMemPool_t pool = nullptr;
    status = jyppx::cuda::map_cuda_status(
        cudaMemGetDefaultMemPool(&pool, &cuda_location, static_cast<cudaMemAllocationType>(allocation_type)),
        "cudaMemGetDefaultMemPool");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_pool_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pool));
    return JYPPX_STATUS_OK;
#else
    (void)allocation_type;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA location default memory pool query");
#endif
}

JYPPX_StatusCode jyppx_cuda_mem_get_mem_pool(const JYPPX_CudaMemLocation* location, int32_t allocation_type, uint64_t* out_pool_handle)
{
    auto status = jyppx::cuda::validate_output_pointer(out_pool_handle, "out_pool_handle");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_pool_handle = 0;
    if (location == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemGetMemPool", 0, "invalid-location", "CUDA memory location must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    cudaMemLocation cuda_location = make_bridge_mem_location(*location);
    cudaMemPool_t pool = nullptr;
    status = jyppx::cuda::map_cuda_status(
        cudaMemGetMemPool(&pool, &cuda_location, static_cast<cudaMemAllocationType>(allocation_type)),
        "cudaMemGetMemPool");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_pool_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pool));
    return JYPPX_STATUS_OK;
#else
    (void)allocation_type;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA location current memory pool query");
#endif
}

JYPPX_StatusCode jyppx_cuda_mem_set_mem_pool(const JYPPX_CudaMemLocation* location, int32_t allocation_type, uint64_t pool_handle)
{
    if (location == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemSetMemPool", 0, "invalid-location", "CUDA memory location must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (pool_handle == 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemSetMemPool", 0, "invalid-pool", "CUDA memory pool handle must not be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    cudaMemLocation cuda_location = make_bridge_mem_location(*location);
    cudaMemPool_t pool = reinterpret_cast<cudaMemPool_t>(static_cast<uintptr_t>(pool_handle));
    return jyppx::cuda::map_cuda_status(
        cudaMemSetMemPool(&cuda_location, static_cast<cudaMemAllocationType>(allocation_type), pool),
        "cudaMemSetMemPool");
#else
    (void)allocation_type;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA location current memory pool set");
#endif
}

JYPPX_StatusCode jyppx_cuda_mem_pool_export_to_shareable_handle(uint64_t pool_handle, int32_t handle_type, uint32_t flags, void* shareable_handle)
{
    if (pool_handle == 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemPoolExportToShareableHandle", 0, "invalid-pool", "CUDA memory pool handle must not be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (shareable_handle == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemPoolExportToShareableHandle", 0, "invalid-shareable-handle", "Shareable handle output storage must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemPool_t pool = reinterpret_cast<cudaMemPool_t>(static_cast<uintptr_t>(pool_handle));
    return jyppx::cuda::map_cuda_status(
        cudaMemPoolExportToShareableHandle(shareable_handle, pool, static_cast<cudaMemAllocationHandleType>(handle_type), flags),
        "cudaMemPoolExportToShareableHandle");
#else
    (void)handle_type;
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memory pool shareable export");
#endif
}

JYPPX_StatusCode jyppx_cuda_mem_pool_import_from_shareable_handle(void* shareable_handle, int32_t handle_type, uint32_t flags, uint64_t* out_pool_handle)
{
    auto status = jyppx::cuda::validate_output_pointer(out_pool_handle, "out_pool_handle");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_pool_handle = 0;
    if (shareable_handle == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemPoolImportFromShareableHandle", 0, "invalid-shareable-handle", "Shareable handle storage must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemPool_t pool = nullptr;
    status = jyppx::cuda::map_cuda_status(
        cudaMemPoolImportFromShareableHandle(&pool, shareable_handle, static_cast<cudaMemAllocationHandleType>(handle_type), flags),
        "cudaMemPoolImportFromShareableHandle");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_pool_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pool));
    return JYPPX_STATUS_OK;
#else
    (void)handle_type;
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memory pool shareable import");
#endif
}

JYPPX_StatusCode jyppx_cuda_mem_pool_export_pointer(void* pointer, JYPPX_CudaMemPoolPtrExportData* out_export_data)
{
    auto status = jyppx::cuda::validate_output_pointer(out_export_data, "out_export_data");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_export_data->reserved, 0, sizeof(out_export_data->reserved));
    if (pointer == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemPoolExportPointer", 0, "invalid-pointer", "CUDA memory pool pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemPoolPtrExportData export_data{};
    status = jyppx::cuda::map_cuda_status(cudaMemPoolExportPointer(&export_data, pointer), "cudaMemPoolExportPointer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memcpy(out_export_data->reserved, export_data.reserved, sizeof(out_export_data->reserved));
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memory pool pointer export");
#endif
}

JYPPX_StatusCode jyppx_cuda_mem_pool_import_pointer(uint64_t pool_handle, const JYPPX_CudaMemPoolPtrExportData* export_data, void** out_pointer)
{
    auto status = jyppx::cuda::validate_output_pointer(out_pointer, "out_pointer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_pointer = nullptr;
    if (pool_handle == 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemPoolImportPointer", 0, "invalid-pool", "CUDA memory pool handle must not be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (export_data == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemPoolImportPointer", 0, "invalid-export-data", "CUDA memory pool pointer export data must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemPoolPtrExportData cuda_export_data{};
    std::memcpy(cuda_export_data.reserved, export_data->reserved, sizeof(cuda_export_data.reserved));
    cudaMemPool_t pool = reinterpret_cast<cudaMemPool_t>(static_cast<uintptr_t>(pool_handle));
    return jyppx::cuda::map_cuda_status(cudaMemPoolImportPointer(out_pointer, pool, &cuda_export_data), "cudaMemPoolImportPointer");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memory pool pointer import");
#endif
}
