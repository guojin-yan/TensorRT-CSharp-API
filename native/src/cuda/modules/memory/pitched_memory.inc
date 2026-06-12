JYPPX_StatusCode jyppx_cuda_pitched_memory_alloc(size_t width_bytes, size_t height, JYPPX_CudaPitchedMemory** out_memory)
{
    auto status = jyppx::cuda::validate_output_pointer(out_memory, "out_memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_memory = nullptr;
    status = validate_pitched_dimensions(width_bytes, height, "cudaMallocPitch");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* memory = new (std::nothrow) PitchedMemoryObject{};
    if (memory == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    memory->base.magic = jyppx::cuda::kObjectMagic;
    memory->base.kind = ObjectKind::PitchedMemory;
    memory->pointer = nullptr;
    memory->pitch_bytes = 0;
    memory->width_bytes = width_bytes;
    memory->height = height;
    memory->depth = 1;

    status = jyppx::cuda::map_cuda_status(cudaMallocPitch(&memory->pointer, &memory->pitch_bytes, width_bytes, height), "cudaMallocPitch");
    if (status != JYPPX_STATUS_OK)
    {
        delete memory;
        return status;
    }

    *out_memory = reinterpret_cast<JYPPX_CudaPitchedMemory*>(memory);
    return JYPPX_STATUS_OK;
#else
    (void)width_bytes;
    (void)height;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA pitched memory allocation");
#endif
}

JYPPX_StatusCode jyppx_cuda_pitched_memory_get_info(JYPPX_CudaPitchedMemory* memory, JYPPX_CudaPitchedMemoryInfo* out_info)
{
    auto status = jyppx::cuda::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_pitched_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<PitchedMemoryObject*>(memory);
    out_info->pitch_bytes = static_cast<uint64_t>(memory_object->pitch_bytes);
    out_info->width_bytes = static_cast<uint64_t>(memory_object->width_bytes);
    out_info->height = static_cast<uint64_t>(memory_object->height);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_cuda_pitched_memory_copy_from_host_2d(JYPPX_CudaPitchedMemory* memory, const void* source, size_t source_pitch, size_t width_bytes, size_t height)
{
    auto status = jyppx::cuda::validate_pitched_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (source == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy2D", 0, "invalid-source", "Source pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* memory_object = reinterpret_cast<PitchedMemoryObject*>(memory);
    status = validate_pitched_copy_region(memory_object, width_bytes, height, "cudaMemcpy2D(HtoD)", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_host_pitch(source_pitch, width_bytes, "cudaMemcpy2D(HtoD)", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaMemcpy2D(memory_object->pointer, memory_object->pitch_bytes, source, source_pitch, width_bytes, height, cudaMemcpyHostToDevice), "cudaMemcpy2D(HtoD)");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 2D host-to-device copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_pitched_memory_copy_to_host_2d(JYPPX_CudaPitchedMemory* memory, void* destination, size_t destination_pitch, size_t width_bytes, size_t height)
{
    auto status = jyppx::cuda::validate_pitched_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (destination == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy2D", 0, "invalid-destination", "Destination pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* memory_object = reinterpret_cast<PitchedMemoryObject*>(memory);
    status = validate_pitched_copy_region(memory_object, width_bytes, height, "cudaMemcpy2D(DtoH)", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_host_pitch(destination_pitch, width_bytes, "cudaMemcpy2D(DtoH)", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaMemcpy2D(destination, destination_pitch, memory_object->pointer, memory_object->pitch_bytes, width_bytes, height, cudaMemcpyDeviceToHost), "cudaMemcpy2D(DtoH)");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 2D device-to-host copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_pitched_memory_copy_device_to_device_2d(JYPPX_CudaPitchedMemory* destination, JYPPX_CudaPitchedMemory* source, size_t width_bytes, size_t height)
{
    auto status = jyppx::cuda::validate_pitched_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_pitched_memory(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* destination_object = reinterpret_cast<PitchedMemoryObject*>(destination);
    auto* source_object = reinterpret_cast<PitchedMemoryObject*>(source);
    status = validate_pitched_copy_region(destination_object, width_bytes, height, "cudaMemcpy2D(DtoD)", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_pitched_copy_region(source_object, width_bytes, height, "cudaMemcpy2D(DtoD)", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaMemcpy2D(destination_object->pointer, destination_object->pitch_bytes, source_object->pointer, source_object->pitch_bytes, width_bytes, height, cudaMemcpyDeviceToDevice), "cudaMemcpy2D(DtoD)");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 2D device-to-device copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_pitched_memory_copy_from_host_2d_async(JYPPX_CudaPitchedMemory* memory, const void* source, size_t source_pitch, size_t width_bytes, size_t height, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_pitched_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (source == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy2DAsync", 0, "invalid-source", "Source pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* memory_object = reinterpret_cast<PitchedMemoryObject*>(memory);
    status = validate_pitched_copy_region(memory_object, width_bytes, height, "cudaMemcpy2DAsync(HtoD)", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_host_pitch(source_pitch, width_bytes, "cudaMemcpy2DAsync(HtoD)", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaMemcpy2DAsync(memory_object->pointer, memory_object->pitch_bytes, source, source_pitch, width_bytes, height, cudaMemcpyHostToDevice, stream_object->handle), "cudaMemcpy2DAsync(HtoD)");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async 2D host-to-device copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_pitched_memory_copy_to_host_2d_async(JYPPX_CudaPitchedMemory* memory, void* destination, size_t destination_pitch, size_t width_bytes, size_t height, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_pitched_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (destination == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy2DAsync", 0, "invalid-destination", "Destination pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* memory_object = reinterpret_cast<PitchedMemoryObject*>(memory);
    status = validate_pitched_copy_region(memory_object, width_bytes, height, "cudaMemcpy2DAsync(DtoH)", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_host_pitch(destination_pitch, width_bytes, "cudaMemcpy2DAsync(DtoH)", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaMemcpy2DAsync(destination, destination_pitch, memory_object->pointer, memory_object->pitch_bytes, width_bytes, height, cudaMemcpyDeviceToHost, stream_object->handle), "cudaMemcpy2DAsync(DtoH)");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async 2D device-to-host copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_pitched_memory_copy_device_to_device_2d_async(JYPPX_CudaPitchedMemory* destination, JYPPX_CudaPitchedMemory* source, size_t width_bytes, size_t height, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_pitched_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_pitched_memory(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* destination_object = reinterpret_cast<PitchedMemoryObject*>(destination);
    auto* source_object = reinterpret_cast<PitchedMemoryObject*>(source);
    status = validate_pitched_copy_region(destination_object, width_bytes, height, "cudaMemcpy2DAsync(DtoD)", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_pitched_copy_region(source_object, width_bytes, height, "cudaMemcpy2DAsync(DtoD)", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaMemcpy2DAsync(destination_object->pointer, destination_object->pitch_bytes, source_object->pointer, source_object->pitch_bytes, width_bytes, height, cudaMemcpyDeviceToDevice, stream_object->handle), "cudaMemcpy2DAsync(DtoD)");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async 2D device-to-device copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_pitched_memory_copy_from_host_3d(JYPPX_CudaPitchedMemory* memory, const void* source, size_t source_pitch, size_t width_bytes, size_t height, size_t depth)
{
    auto status = jyppx::cuda::validate_pitched_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (source == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy3D", 0, "invalid-source", "Source pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* memory_object = reinterpret_cast<PitchedMemoryObject*>(memory);
    status = validate_pitched_copy_region_3d(memory_object, width_bytes, height, depth, "cudaMemcpy3D(HtoD)", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_host_pitch(source_pitch, width_bytes, "cudaMemcpy3D(HtoD)", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemcpy3DParms parameters{};
    parameters.srcPtr = make_pitched_ptr(const_cast<void*>(source), source_pitch, width_bytes, height);
    parameters.dstPtr = make_pitched_ptr(memory_object->pointer, memory_object->pitch_bytes, width_bytes, height);
    parameters.extent = make_extent(width_bytes, height, depth);
    parameters.kind = cudaMemcpyHostToDevice;
    return jyppx::cuda::map_cuda_status(cudaMemcpy3D(&parameters), "cudaMemcpy3D(HtoD)");
#else
    (void)height;
    (void)depth;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 3D host-to-device copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_pitched_memory_copy_to_host_3d(JYPPX_CudaPitchedMemory* memory, void* destination, size_t destination_pitch, size_t width_bytes, size_t height, size_t depth)
{
    auto status = jyppx::cuda::validate_pitched_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (destination == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy3D", 0, "invalid-destination", "Destination pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* memory_object = reinterpret_cast<PitchedMemoryObject*>(memory);
    status = validate_pitched_copy_region_3d(memory_object, width_bytes, height, depth, "cudaMemcpy3D(DtoH)", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_host_pitch(destination_pitch, width_bytes, "cudaMemcpy3D(DtoH)", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemcpy3DParms parameters{};
    parameters.srcPtr = make_pitched_ptr(memory_object->pointer, memory_object->pitch_bytes, width_bytes, height);
    parameters.dstPtr = make_pitched_ptr(destination, destination_pitch, width_bytes, height);
    parameters.extent = make_extent(width_bytes, height, depth);
    parameters.kind = cudaMemcpyDeviceToHost;
    return jyppx::cuda::map_cuda_status(cudaMemcpy3D(&parameters), "cudaMemcpy3D(DtoH)");
#else
    (void)height;
    (void)depth;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 3D device-to-host copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_pitched_memory_copy_device_to_device_3d(JYPPX_CudaPitchedMemory* destination, JYPPX_CudaPitchedMemory* source, size_t width_bytes, size_t height, size_t depth)
{
    auto status = jyppx::cuda::validate_pitched_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_pitched_memory(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* destination_object = reinterpret_cast<PitchedMemoryObject*>(destination);
    auto* source_object = reinterpret_cast<PitchedMemoryObject*>(source);
    status = validate_pitched_copy_region_3d(destination_object, width_bytes, height, depth, "cudaMemcpy3D(DtoD)", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_pitched_copy_region_3d(source_object, width_bytes, height, depth, "cudaMemcpy3D(DtoD)", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemcpy3DParms parameters{};
    parameters.srcPtr = make_pitched_ptr(source_object->pointer, source_object->pitch_bytes, width_bytes, height);
    parameters.dstPtr = make_pitched_ptr(destination_object->pointer, destination_object->pitch_bytes, width_bytes, height);
    parameters.extent = make_extent(width_bytes, height, depth);
    parameters.kind = cudaMemcpyDeviceToDevice;
    return jyppx::cuda::map_cuda_status(cudaMemcpy3D(&parameters), "cudaMemcpy3D(DtoD)");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 3D device-to-device copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_pitched_memory_copy_from_host_3d_async(JYPPX_CudaPitchedMemory* memory, const void* source, size_t source_pitch, size_t width_bytes, size_t height, size_t depth, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_pitched_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (source == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy3DAsync", 0, "invalid-source", "Source pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* memory_object = reinterpret_cast<PitchedMemoryObject*>(memory);
    status = validate_pitched_copy_region_3d(memory_object, width_bytes, height, depth, "cudaMemcpy3DAsync(HtoD)", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_host_pitch(source_pitch, width_bytes, "cudaMemcpy3DAsync(HtoD)", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    cudaMemcpy3DParms parameters{};
    parameters.srcPtr = make_pitched_ptr(const_cast<void*>(source), source_pitch, width_bytes, height);
    parameters.dstPtr = make_pitched_ptr(memory_object->pointer, memory_object->pitch_bytes, width_bytes, height);
    parameters.extent = make_extent(width_bytes, height, depth);
    parameters.kind = cudaMemcpyHostToDevice;
    return jyppx::cuda::map_cuda_status(cudaMemcpy3DAsync(&parameters, stream_object->handle), "cudaMemcpy3DAsync(HtoD)");
#else
    (void)height;
    (void)depth;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async 3D host-to-device copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_pitched_memory_copy_to_host_3d_async(JYPPX_CudaPitchedMemory* memory, void* destination, size_t destination_pitch, size_t width_bytes, size_t height, size_t depth, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_pitched_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (destination == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy3DAsync", 0, "invalid-destination", "Destination pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* memory_object = reinterpret_cast<PitchedMemoryObject*>(memory);
    status = validate_pitched_copy_region_3d(memory_object, width_bytes, height, depth, "cudaMemcpy3DAsync(DtoH)", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_host_pitch(destination_pitch, width_bytes, "cudaMemcpy3DAsync(DtoH)", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    cudaMemcpy3DParms parameters{};
    parameters.srcPtr = make_pitched_ptr(memory_object->pointer, memory_object->pitch_bytes, width_bytes, height);
    parameters.dstPtr = make_pitched_ptr(destination, destination_pitch, width_bytes, height);
    parameters.extent = make_extent(width_bytes, height, depth);
    parameters.kind = cudaMemcpyDeviceToHost;
    return jyppx::cuda::map_cuda_status(cudaMemcpy3DAsync(&parameters, stream_object->handle), "cudaMemcpy3DAsync(DtoH)");
#else
    (void)height;
    (void)depth;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async 3D device-to-host copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_pitched_memory_copy_device_to_device_3d_async(JYPPX_CudaPitchedMemory* destination, JYPPX_CudaPitchedMemory* source, size_t width_bytes, size_t height, size_t depth, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_pitched_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_pitched_memory(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* destination_object = reinterpret_cast<PitchedMemoryObject*>(destination);
    auto* source_object = reinterpret_cast<PitchedMemoryObject*>(source);
    status = validate_pitched_copy_region_3d(destination_object, width_bytes, height, depth, "cudaMemcpy3DAsync(DtoD)", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_pitched_copy_region_3d(source_object, width_bytes, height, depth, "cudaMemcpy3DAsync(DtoD)", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    cudaMemcpy3DParms parameters{};
    parameters.srcPtr = make_pitched_ptr(source_object->pointer, source_object->pitch_bytes, width_bytes, height);
    parameters.dstPtr = make_pitched_ptr(destination_object->pointer, destination_object->pitch_bytes, width_bytes, height);
    parameters.extent = make_extent(width_bytes, height, depth);
    parameters.kind = cudaMemcpyDeviceToDevice;
    return jyppx::cuda::map_cuda_status(cudaMemcpy3DAsync(&parameters, stream_object->handle), "cudaMemcpy3DAsync(DtoD)");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async 3D device-to-device copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_pitched_memory_memset_2d(JYPPX_CudaPitchedMemory* memory, int32_t value, size_t width_bytes, size_t height)
{
    auto status = jyppx::cuda::validate_pitched_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<PitchedMemoryObject*>(memory);
    status = validate_pitched_copy_region(memory_object, width_bytes, height, "cudaMemset2D", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaMemset2D(memory_object->pointer, memory_object->pitch_bytes, value, width_bytes, height), "cudaMemset2D");
#else
    (void)value;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 2D memory memset");
#endif
}

JYPPX_StatusCode jyppx_cuda_pitched_memory_memset_2d_async(JYPPX_CudaPitchedMemory* memory, int32_t value, size_t width_bytes, size_t height, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_pitched_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<PitchedMemoryObject*>(memory);
    status = validate_pitched_copy_region(memory_object, width_bytes, height, "cudaMemset2DAsync", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaMemset2DAsync(memory_object->pointer, memory_object->pitch_bytes, value, width_bytes, height, stream_object->handle), "cudaMemset2DAsync");
#else
    (void)value;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async 2D memory memset");
#endif
}

JYPPX_StatusCode jyppx_cuda_pitched_memory_memset_3d(JYPPX_CudaPitchedMemory* memory, int32_t value, size_t width_bytes, size_t height, size_t depth)
{
    auto status = jyppx::cuda::validate_pitched_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<PitchedMemoryObject*>(memory);
    status = validate_pitched_copy_region_3d(memory_object, width_bytes, height, depth, "cudaMemset3D", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    const cudaPitchedPtr target = make_pitched_ptr(memory_object->pointer, memory_object->pitch_bytes, width_bytes, height);
    const cudaExtent extent = make_extent(width_bytes, height, depth);
    return jyppx::cuda::map_cuda_status(cudaMemset3D(target, value, extent), "cudaMemset3D");
#else
    (void)value;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 3D memory memset");
#endif
}

JYPPX_StatusCode jyppx_cuda_pitched_memory_memset_3d_async(JYPPX_CudaPitchedMemory* memory, int32_t value, size_t width_bytes, size_t height, size_t depth, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_pitched_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<PitchedMemoryObject*>(memory);
    status = validate_pitched_copy_region_3d(memory_object, width_bytes, height, depth, "cudaMemset3DAsync", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    const cudaPitchedPtr target = make_pitched_ptr(memory_object->pointer, memory_object->pitch_bytes, width_bytes, height);
    const cudaExtent extent = make_extent(width_bytes, height, depth);
    return jyppx::cuda::map_cuda_status(cudaMemset3DAsync(target, value, extent, stream_object->handle), "cudaMemset3DAsync");
#else
    (void)value;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async 3D memory memset");
#endif
}

JYPPX_StatusCode jyppx_cuda_pitched_memory_free(JYPPX_CudaPitchedMemory* memory)
{
    if (memory == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    auto status = jyppx::cuda::validate_pitched_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<PitchedMemoryObject*>(memory);
#if JYPPX_HAS_CUDA_TOOLKIT
    status = jyppx::cuda::map_cuda_status(cudaFree(memory_object->pointer), "cudaFree(pitched)");
#else
    status = JYPPX_STATUS_OK;
#endif
    delete memory_object;
    return status;
}
