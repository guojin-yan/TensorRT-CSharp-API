namespace
{
#if JYPPX_HAS_CUDA_TOOLKIT
cudaChannelFormatDesc make_channel_format_desc(const JYPPX_CudaChannelFormatDesc& descriptor)
{
    cudaChannelFormatDesc value{};
    value.x = descriptor.x;
    value.y = descriptor.y;
    value.z = descriptor.z;
    value.w = descriptor.w;
    value.f = static_cast<cudaChannelFormatKind>(descriptor.format_kind);
    return value;
}

cudaPos make_cuda_pos(const size_t x, const size_t y, const size_t z)
{
    cudaPos value{};
    value.x = x;
    value.y = y;
    value.z = z;
    return value;
}

void copy_channel_format_desc(const cudaChannelFormatDesc& source, JYPPX_CudaChannelFormatDesc* destination)
{
    destination->x = source.x;
    destination->y = source.y;
    destination->z = source.z;
    destination->w = source.w;
    destination->format_kind = static_cast<int32_t>(source.f);
}

void copy_array_extent(const cudaExtent& source, JYPPX_CudaArrayExtent* destination)
{
    destination->width = static_cast<uint64_t>(source.width);
    destination->height = static_cast<uint64_t>(source.height);
    destination->depth = static_cast<uint64_t>(source.depth);
}

void copy_memory_requirements(const cudaArrayMemoryRequirements& source, JYPPX_CudaArrayMemoryRequirements* destination)
{
    destination->size = static_cast<uint64_t>(source.size);
    destination->alignment = static_cast<uint64_t>(source.alignment);
}

void copy_sparse_properties(const cudaArraySparseProperties& source, JYPPX_CudaArraySparseProperties* destination)
{
    destination->tile_width = source.tileExtent.width;
    destination->tile_height = source.tileExtent.height;
    destination->tile_depth = source.tileExtent.depth;
    destination->mip_tail_first_level = source.miptailFirstLevel;
    destination->mip_tail_size = static_cast<uint64_t>(source.miptailSize);
    destination->flags = source.flags;
}
#endif

void clear_channel_format_desc(JYPPX_CudaChannelFormatDesc* descriptor)
{
    descriptor->x = 0;
    descriptor->y = 0;
    descriptor->z = 0;
    descriptor->w = 0;
    descriptor->format_kind = 0;
}

void clear_array_extent(JYPPX_CudaArrayExtent* extent)
{
    extent->width = 0;
    extent->height = 0;
    extent->depth = 0;
}

void clear_array_info(JYPPX_CudaArrayInfo* info)
{
    clear_channel_format_desc(&info->channel);
    clear_array_extent(&info->extent);
    info->flags = 0;
}

void clear_memory_requirements(JYPPX_CudaArrayMemoryRequirements* requirements)
{
    requirements->size = 0;
    requirements->alignment = 0;
}

void clear_sparse_properties(JYPPX_CudaArraySparseProperties* properties)
{
    properties->tile_width = 0;
    properties->tile_height = 0;
    properties->tile_depth = 0;
    properties->mip_tail_first_level = 0;
    properties->mip_tail_size = 0;
    properties->flags = 0;
}

JYPPX_StatusCode validate_channel_format_descriptor(const JYPPX_CudaChannelFormatDesc* descriptor, const char* operation)
{
    if (descriptor == nullptr)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-channel-descriptor", "CUDA channel format descriptor must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (descriptor->x < 0 || descriptor->y < 0 || descriptor->z < 0 || descriptor->w < 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-channel-bits", "CUDA channel bit counts must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (descriptor->x == 0 && descriptor->y == 0 && descriptor->z == 0 && descriptor->w == 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "empty-channel-descriptor", "At least one CUDA channel must have a positive bit count.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (descriptor->format_kind < 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-channel-format-kind", "CUDA channel format kind must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode validate_array_extent_values(const size_t width, const size_t height, const size_t depth, const char* operation)
{
    if (width == 0 || height == 0 || depth == 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-array-extent", "CUDA array width, height, and depth must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode validate_array_extent(const JYPPX_CudaArrayExtent* extent, const char* operation)
{
    if (extent == nullptr)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-array-extent", "CUDA array extent must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return validate_array_extent_values(
        static_cast<size_t>(extent->width),
        static_cast<size_t>(extent->height),
        static_cast<size_t>(extent->depth),
        operation);
}

JYPPX_StatusCode validate_copy_bytes(const size_t byte_count, const char* operation)
{
    if (byte_count == 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-copy-size", "CUDA array copy byte count must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode validate_copy_2d_dimensions(const size_t pitch, const size_t width_bytes, const size_t height, const char* operation, const char* role)
{
    if (width_bytes == 0 || height == 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-2d-copy-size", "CUDA array 2D copy width and height must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (pitch < width_bytes)
    {
        std::string message = std::string(role) + " pitch must be greater than or equal to copy row width.";
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-2d-copy-pitch", message.c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode validate_copy_3d_dimensions(const size_t pitch, const size_t pitched_width_bytes, const size_t pitched_height, const size_t width_bytes, const size_t height, const size_t depth, const char* operation, const char* role)
{
    auto status = validate_copy_2d_dimensions(pitch, width_bytes, height, operation, role);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (depth == 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-3d-copy-depth", "CUDA array 3D copy depth must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (pitched_width_bytes < width_bytes || pitched_height < height)
    {
        std::string message = std::string(role) + " pitched extent must cover the requested CUDA array 3D copy extent.";
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-3d-copy-pitched-extent", message.c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

#if JYPPX_HAS_CUDA_TOOLKIT
JYPPX_StatusCode make_array_object(cudaArray_t handle, JYPPX_CudaArray** out_array, bool owns_handle = true)
{
    auto* array_object = new (std::nothrow) jyppx::cuda::ArrayObject{};
    if (array_object == nullptr)
    {
        if (owns_handle)
        {
            cudaFreeArray(handle);
        }

        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    array_object->base.magic = jyppx::cuda::kObjectMagic;
    array_object->base.kind = jyppx::cuda::ObjectKind::Array;
    array_object->owns_handle = owns_handle;
    array_object->handle = handle;
    *out_array = reinterpret_cast<JYPPX_CudaArray*>(array_object);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode fill_array_info(cudaArray_t handle, JYPPX_CudaArrayInfo* out_info)
{
    cudaChannelFormatDesc descriptor{};
    cudaExtent extent{};
    unsigned int flags = 0;
    auto status = jyppx::cuda::map_cuda_status(cudaArrayGetInfo(&descriptor, &extent, &flags, handle), "cudaArrayGetInfo");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    copy_channel_format_desc(descriptor, &out_info->channel);
    copy_array_extent(extent, &out_info->extent);
    out_info->flags = flags;
    return JYPPX_STATUS_OK;
}
#endif
}

JYPPX_StatusCode jyppx_cuda_malloc_array(
    const JYPPX_CudaChannelFormatDesc* descriptor,
    size_t width,
    size_t height,
    uint32_t flags,
    JYPPX_CudaArray** out_array)
{
    auto status = jyppx::cuda::validate_output_pointer(out_array, "out_array");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_array = nullptr;
    status = validate_channel_format_descriptor(descriptor, "cudaMallocArray");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (width == 0)
    {
        jyppx::cuda::set_cuda_error("cudaMallocArray", 0, "invalid-array-width", "CUDA array width must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    const auto cuda_descriptor = make_channel_format_desc(*descriptor);
    cudaArray_t handle = nullptr;
    status = jyppx::cuda::map_cuda_status(cudaMallocArray(&handle, &cuda_descriptor, width, height, flags), "cudaMallocArray");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return make_array_object(handle, out_array);
#else
    (void)height;
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA array allocation");
#endif
}

JYPPX_StatusCode jyppx_cuda_malloc_3d_array(
    const JYPPX_CudaChannelFormatDesc* descriptor,
    const JYPPX_CudaArrayExtent* extent,
    uint32_t flags,
    JYPPX_CudaArray** out_array)
{
    auto status = jyppx::cuda::validate_output_pointer(out_array, "out_array");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_array = nullptr;
    status = validate_channel_format_descriptor(descriptor, "cudaMalloc3DArray");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_array_extent(extent, "cudaMalloc3DArray");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    const auto cuda_descriptor = make_channel_format_desc(*descriptor);
    const auto cuda_extent = make_extent(static_cast<size_t>(extent->width), static_cast<size_t>(extent->height), static_cast<size_t>(extent->depth));
    cudaArray_t handle = nullptr;
    status = jyppx::cuda::map_cuda_status(cudaMalloc3DArray(&handle, &cuda_descriptor, cuda_extent, flags), "cudaMalloc3DArray");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return make_array_object(handle, out_array);
#else
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 3D array allocation");
#endif
}

JYPPX_StatusCode jyppx_cuda_free_array(JYPPX_CudaArray* array)
{
    if (array == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    auto status = jyppx::cuda::validate_array(array, "array");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* array_object = reinterpret_cast<jyppx::cuda::ArrayObject*>(array);
#if JYPPX_HAS_CUDA_TOOLKIT
    if (array_object->owns_handle)
    {
        status = jyppx::cuda::map_cuda_status(cudaFreeArray(array_object->handle), "cudaFreeArray");
    }
    else
    {
        status = JYPPX_STATUS_OK;
    }
#else
    status = JYPPX_STATUS_OK;
#endif
    delete array_object;
    return status;
}

JYPPX_StatusCode jyppx_cuda_array_get_plane(JYPPX_CudaArray* array, uint32_t plane_index, JYPPX_CudaArray** out_plane)
{
    auto status = jyppx::cuda::validate_output_pointer(out_plane, "out_plane");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_plane = nullptr;
    status = jyppx::cuda::validate_array(array, "array");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaArray_t plane = nullptr;
    auto* array_object = reinterpret_cast<jyppx::cuda::ArrayObject*>(array);
    status = jyppx::cuda::map_cuda_status(cudaArrayGetPlane(&plane, array_object->handle, plane_index), "cudaArrayGetPlane");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return make_array_object(plane, out_plane, false);
#else
    (void)plane_index;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA array plane query");
#endif
}

JYPPX_StatusCode jyppx_cuda_array_get_info(JYPPX_CudaArray* array, JYPPX_CudaArrayInfo* out_info)
{
    auto status = jyppx::cuda::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    clear_array_info(out_info);
    status = jyppx::cuda::validate_array(array, "array");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return fill_array_info(reinterpret_cast<jyppx::cuda::ArrayObject*>(array)->handle, out_info);
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA array info query");
#endif
}

JYPPX_StatusCode jyppx_cuda_array_get_channel_desc(JYPPX_CudaArray* array, JYPPX_CudaChannelFormatDesc* out_descriptor)
{
    auto status = jyppx::cuda::validate_output_pointer(out_descriptor, "out_descriptor");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    clear_channel_format_desc(out_descriptor);
    status = jyppx::cuda::validate_array(array, "array");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaChannelFormatDesc descriptor{};
    status = jyppx::cuda::map_cuda_status(cudaGetChannelDesc(&descriptor, reinterpret_cast<jyppx::cuda::ArrayObject*>(array)->handle), "cudaGetChannelDesc");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    copy_channel_format_desc(descriptor, out_descriptor);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA array channel descriptor query");
#endif
}

JYPPX_StatusCode jyppx_cuda_array_get_memory_requirements(JYPPX_CudaArray* array, int32_t device, JYPPX_CudaArrayMemoryRequirements* out_requirements)
{
    auto status = jyppx::cuda::validate_output_pointer(out_requirements, "out_requirements");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    clear_memory_requirements(out_requirements);
    status = jyppx::cuda::validate_array(array, "array");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaArrayGetMemoryRequirements", 0, "invalid-device", "Device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaArrayMemoryRequirements requirements{};
    status = jyppx::cuda::map_cuda_status(cudaArrayGetMemoryRequirements(&requirements, reinterpret_cast<jyppx::cuda::ArrayObject*>(array)->handle, device), "cudaArrayGetMemoryRequirements");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    copy_memory_requirements(requirements, out_requirements);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA array memory requirements query");
#endif
}

JYPPX_StatusCode jyppx_cuda_array_get_sparse_properties(JYPPX_CudaArray* array, JYPPX_CudaArraySparseProperties* out_properties)
{
    auto status = jyppx::cuda::validate_output_pointer(out_properties, "out_properties");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    clear_sparse_properties(out_properties);
    status = jyppx::cuda::validate_array(array, "array");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaArraySparseProperties properties{};
    status = jyppx::cuda::map_cuda_status(cudaArrayGetSparseProperties(&properties, reinterpret_cast<jyppx::cuda::ArrayObject*>(array)->handle), "cudaArrayGetSparseProperties");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    copy_sparse_properties(properties, out_properties);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA array sparse properties query");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_get_texture_1d_linear_max_width(const JYPPX_CudaChannelFormatDesc* descriptor, int32_t device, size_t* out_max_width)
{
    auto status = jyppx::cuda::validate_output_pointer(out_max_width, "out_max_width");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_max_width = 0;
    status = validate_channel_format_descriptor(descriptor, "cudaDeviceGetTexture1DLinearMaxWidth");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaDeviceGetTexture1DLinearMaxWidth", 0, "invalid-device", "Device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    const auto cuda_descriptor = make_channel_format_desc(*descriptor);
    return jyppx::cuda::map_cuda_status(cudaDeviceGetTexture1DLinearMaxWidth(out_max_width, &cuda_descriptor, device), "cudaDeviceGetTexture1DLinearMaxWidth");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA texture 1D linear max width query");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_to_array(JYPPX_CudaArray* destination, size_t destination_x_bytes, size_t destination_y, const void* source, size_t byte_count)
{
    auto status = jyppx::cuda::validate_array(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_output_pointer(const_cast<void*>(source), "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_copy_bytes(byte_count, "cudaMemcpyToArray");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaMemcpyToArray(reinterpret_cast<jyppx::cuda::ArrayObject*>(destination)->handle, destination_x_bytes, destination_y, source, byte_count, cudaMemcpyHostToDevice), "cudaMemcpyToArray");
#else
    (void)destination_x_bytes;
    (void)destination_y;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA host-to-array copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_from_array(void* destination, JYPPX_CudaArray* source, size_t source_x_bytes, size_t source_y, size_t byte_count)
{
    auto status = jyppx::cuda::validate_output_pointer(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_array(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_copy_bytes(byte_count, "cudaMemcpyFromArray");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaMemcpyFromArray(destination, reinterpret_cast<jyppx::cuda::ArrayObject*>(source)->handle, source_x_bytes, source_y, byte_count, cudaMemcpyDeviceToHost), "cudaMemcpyFromArray");
#else
    (void)source_x_bytes;
    (void)source_y;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA array-to-host copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_array_to_array(
    JYPPX_CudaArray* destination,
    size_t destination_x_bytes,
    size_t destination_y,
    JYPPX_CudaArray* source,
    size_t source_x_bytes,
    size_t source_y,
    size_t byte_count)
{
    auto status = jyppx::cuda::validate_array(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_array(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_copy_bytes(byte_count, "cudaMemcpyArrayToArray");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(
        cudaMemcpyArrayToArray(
            reinterpret_cast<jyppx::cuda::ArrayObject*>(destination)->handle,
            destination_x_bytes,
            destination_y,
            reinterpret_cast<jyppx::cuda::ArrayObject*>(source)->handle,
            source_x_bytes,
            source_y,
            byte_count,
            cudaMemcpyDeviceToDevice),
        "cudaMemcpyArrayToArray");
#else
    (void)destination_x_bytes;
    (void)destination_y;
    (void)source_x_bytes;
    (void)source_y;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA array-to-array copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_to_array_async(JYPPX_CudaArray* destination, size_t destination_x_bytes, size_t destination_y, const void* source, size_t byte_count, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_array(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_output_pointer(const_cast<void*>(source), "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_copy_bytes(byte_count, "cudaMemcpyToArrayAsync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* destination_object = reinterpret_cast<jyppx::cuda::ArrayObject*>(destination);
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaMemcpyToArrayAsync(destination_object->handle, destination_x_bytes, destination_y, source, byte_count, cudaMemcpyHostToDevice, stream_object->handle), "cudaMemcpyToArrayAsync");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async host-to-array copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_from_array_async(void* destination, JYPPX_CudaArray* source, size_t source_x_bytes, size_t source_y, size_t byte_count, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_output_pointer(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_array(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_copy_bytes(byte_count, "cudaMemcpyFromArrayAsync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* source_object = reinterpret_cast<jyppx::cuda::ArrayObject*>(source);
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaMemcpyFromArrayAsync(destination, source_object->handle, source_x_bytes, source_y, byte_count, cudaMemcpyDeviceToHost, stream_object->handle), "cudaMemcpyFromArrayAsync");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async array-to-host copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_2d_to_array(
    JYPPX_CudaArray* destination,
    size_t destination_x_bytes,
    size_t destination_y,
    const void* source,
    size_t source_pitch,
    size_t width_bytes,
    size_t height)
{
    auto status = jyppx::cuda::validate_array(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_output_pointer(const_cast<void*>(source), "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_copy_2d_dimensions(source_pitch, width_bytes, height, "cudaMemcpy2DToArray", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaMemcpy2DToArray(reinterpret_cast<jyppx::cuda::ArrayObject*>(destination)->handle, destination_x_bytes, destination_y, source, source_pitch, width_bytes, height, cudaMemcpyHostToDevice), "cudaMemcpy2DToArray");
#else
    (void)destination_x_bytes;
    (void)destination_y;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 2D host-to-array copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_2d_from_array(
    void* destination,
    size_t destination_pitch,
    JYPPX_CudaArray* source,
    size_t source_x_bytes,
    size_t source_y,
    size_t width_bytes,
    size_t height)
{
    auto status = jyppx::cuda::validate_output_pointer(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_array(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_copy_2d_dimensions(destination_pitch, width_bytes, height, "cudaMemcpy2DFromArray", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaMemcpy2DFromArray(destination, destination_pitch, reinterpret_cast<jyppx::cuda::ArrayObject*>(source)->handle, source_x_bytes, source_y, width_bytes, height, cudaMemcpyDeviceToHost), "cudaMemcpy2DFromArray");
#else
    (void)source_x_bytes;
    (void)source_y;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 2D array-to-host copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_2d_array_to_array(
    JYPPX_CudaArray* destination,
    size_t destination_x_bytes,
    size_t destination_y,
    JYPPX_CudaArray* source,
    size_t source_x_bytes,
    size_t source_y,
    size_t width_bytes,
    size_t height)
{
    auto status = jyppx::cuda::validate_array(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_array(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (width_bytes == 0 || height == 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy2DArrayToArray", 0, "invalid-2d-copy-size", "CUDA array 2D copy width and height must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(
        cudaMemcpy2DArrayToArray(
            reinterpret_cast<jyppx::cuda::ArrayObject*>(destination)->handle,
            destination_x_bytes,
            destination_y,
            reinterpret_cast<jyppx::cuda::ArrayObject*>(source)->handle,
            source_x_bytes,
            source_y,
            width_bytes,
            height,
            cudaMemcpyDeviceToDevice),
        "cudaMemcpy2DArrayToArray");
#else
    (void)destination_x_bytes;
    (void)destination_y;
    (void)source_x_bytes;
    (void)source_y;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 2D array-to-array copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_2d_to_array_async(
    JYPPX_CudaArray* destination,
    size_t destination_x_bytes,
    size_t destination_y,
    const void* source,
    size_t source_pitch,
    size_t width_bytes,
    size_t height,
    JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_array(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_output_pointer(const_cast<void*>(source), "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_copy_2d_dimensions(source_pitch, width_bytes, height, "cudaMemcpy2DToArrayAsync", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(
        cudaMemcpy2DToArrayAsync(
            reinterpret_cast<jyppx::cuda::ArrayObject*>(destination)->handle,
            destination_x_bytes,
            destination_y,
            source,
            source_pitch,
            width_bytes,
            height,
            cudaMemcpyHostToDevice,
            reinterpret_cast<StreamObject*>(stream)->handle),
        "cudaMemcpy2DToArrayAsync");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async 2D host-to-array copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_2d_from_array_async(
    void* destination,
    size_t destination_pitch,
    JYPPX_CudaArray* source,
    size_t source_x_bytes,
    size_t source_y,
    size_t width_bytes,
    size_t height,
    JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_output_pointer(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_array(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_copy_2d_dimensions(destination_pitch, width_bytes, height, "cudaMemcpy2DFromArrayAsync", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(
        cudaMemcpy2DFromArrayAsync(
            destination,
            destination_pitch,
            reinterpret_cast<jyppx::cuda::ArrayObject*>(source)->handle,
            source_x_bytes,
            source_y,
            width_bytes,
            height,
            cudaMemcpyDeviceToHost,
            reinterpret_cast<StreamObject*>(stream)->handle),
        "cudaMemcpy2DFromArrayAsync");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async 2D array-to-host copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_3d_to_array(
    JYPPX_CudaArray* destination,
    size_t destination_x_bytes,
    size_t destination_y,
    size_t destination_z,
    const void* source,
    size_t source_pitch,
    size_t source_width_bytes,
    size_t source_height,
    size_t width_bytes,
    size_t height,
    size_t depth)
{
    auto status = jyppx::cuda::validate_array(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_output_pointer(const_cast<void*>(source), "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_copy_3d_dimensions(source_pitch, source_width_bytes, source_height, width_bytes, height, depth, "cudaMemcpy3D", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemcpy3DParms parms{};
    parms.srcPtr = make_pitched_ptr(const_cast<void*>(source), source_pitch, source_width_bytes, source_height);
    parms.dstArray = reinterpret_cast<jyppx::cuda::ArrayObject*>(destination)->handle;
    parms.dstPos = make_cuda_pos(destination_x_bytes, destination_y, destination_z);
    parms.extent = make_extent(width_bytes, height, depth);
    parms.kind = cudaMemcpyHostToDevice;
    return jyppx::cuda::map_cuda_status(cudaMemcpy3D(&parms), "cudaMemcpy3D");
#else
    (void)destination_x_bytes;
    (void)destination_y;
    (void)destination_z;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 3D host-to-array copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_3d_from_array(
    void* destination,
    size_t destination_pitch,
    size_t destination_width_bytes,
    size_t destination_height,
    JYPPX_CudaArray* source,
    size_t source_x_bytes,
    size_t source_y,
    size_t source_z,
    size_t width_bytes,
    size_t height,
    size_t depth)
{
    auto status = jyppx::cuda::validate_output_pointer(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_array(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_copy_3d_dimensions(destination_pitch, destination_width_bytes, destination_height, width_bytes, height, depth, "cudaMemcpy3D", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemcpy3DParms parms{};
    parms.srcArray = reinterpret_cast<jyppx::cuda::ArrayObject*>(source)->handle;
    parms.srcPos = make_cuda_pos(source_x_bytes, source_y, source_z);
    parms.dstPtr = make_pitched_ptr(destination, destination_pitch, destination_width_bytes, destination_height);
    parms.extent = make_extent(width_bytes, height, depth);
    parms.kind = cudaMemcpyDeviceToHost;
    return jyppx::cuda::map_cuda_status(cudaMemcpy3D(&parms), "cudaMemcpy3D");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 3D array-to-host copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_3d_array_to_array(
    JYPPX_CudaArray* destination,
    size_t destination_x_bytes,
    size_t destination_y,
    size_t destination_z,
    JYPPX_CudaArray* source,
    size_t source_x_bytes,
    size_t source_y,
    size_t source_z,
    size_t width_bytes,
    size_t height,
    size_t depth)
{
    auto status = jyppx::cuda::validate_array(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_array(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_array_extent_values(width_bytes, height, depth, "cudaMemcpy3D");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemcpy3DParms parms{};
    parms.srcArray = reinterpret_cast<jyppx::cuda::ArrayObject*>(source)->handle;
    parms.srcPos = make_cuda_pos(source_x_bytes, source_y, source_z);
    parms.dstArray = reinterpret_cast<jyppx::cuda::ArrayObject*>(destination)->handle;
    parms.dstPos = make_cuda_pos(destination_x_bytes, destination_y, destination_z);
    parms.extent = make_extent(width_bytes, height, depth);
    parms.kind = cudaMemcpyDeviceToDevice;
    return jyppx::cuda::map_cuda_status(cudaMemcpy3D(&parms), "cudaMemcpy3D");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 3D array-to-array copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_3d_to_array_async(
    JYPPX_CudaArray* destination,
    size_t destination_x_bytes,
    size_t destination_y,
    size_t destination_z,
    const void* source,
    size_t source_pitch,
    size_t source_width_bytes,
    size_t source_height,
    size_t width_bytes,
    size_t height,
    size_t depth,
    JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_array(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_output_pointer(const_cast<void*>(source), "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_copy_3d_dimensions(source_pitch, source_width_bytes, source_height, width_bytes, height, depth, "cudaMemcpy3DAsync", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemcpy3DParms parms{};
    parms.srcPtr = make_pitched_ptr(const_cast<void*>(source), source_pitch, source_width_bytes, source_height);
    parms.dstArray = reinterpret_cast<jyppx::cuda::ArrayObject*>(destination)->handle;
    parms.dstPos = make_cuda_pos(destination_x_bytes, destination_y, destination_z);
    parms.extent = make_extent(width_bytes, height, depth);
    parms.kind = cudaMemcpyHostToDevice;
    return jyppx::cuda::map_cuda_status(cudaMemcpy3DAsync(&parms, reinterpret_cast<StreamObject*>(stream)->handle), "cudaMemcpy3DAsync");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async 3D host-to-array copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_3d_from_array_async(
    void* destination,
    size_t destination_pitch,
    size_t destination_width_bytes,
    size_t destination_height,
    JYPPX_CudaArray* source,
    size_t source_x_bytes,
    size_t source_y,
    size_t source_z,
    size_t width_bytes,
    size_t height,
    size_t depth,
    JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_output_pointer(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_array(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_copy_3d_dimensions(destination_pitch, destination_width_bytes, destination_height, width_bytes, height, depth, "cudaMemcpy3DAsync", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemcpy3DParms parms{};
    parms.srcArray = reinterpret_cast<jyppx::cuda::ArrayObject*>(source)->handle;
    parms.srcPos = make_cuda_pos(source_x_bytes, source_y, source_z);
    parms.dstPtr = make_pitched_ptr(destination, destination_pitch, destination_width_bytes, destination_height);
    parms.extent = make_extent(width_bytes, height, depth);
    parms.kind = cudaMemcpyDeviceToHost;
    return jyppx::cuda::map_cuda_status(cudaMemcpy3DAsync(&parms, reinterpret_cast<StreamObject*>(stream)->handle), "cudaMemcpy3DAsync");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async 3D array-to-host copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_3d_array_to_array_async(
    JYPPX_CudaArray* destination,
    size_t destination_x_bytes,
    size_t destination_y,
    size_t destination_z,
    JYPPX_CudaArray* source,
    size_t source_x_bytes,
    size_t source_y,
    size_t source_z,
    size_t width_bytes,
    size_t height,
    size_t depth,
    JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_array(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_array(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_array_extent_values(width_bytes, height, depth, "cudaMemcpy3DAsync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemcpy3DParms parms{};
    parms.srcArray = reinterpret_cast<jyppx::cuda::ArrayObject*>(source)->handle;
    parms.srcPos = make_cuda_pos(source_x_bytes, source_y, source_z);
    parms.dstArray = reinterpret_cast<jyppx::cuda::ArrayObject*>(destination)->handle;
    parms.dstPos = make_cuda_pos(destination_x_bytes, destination_y, destination_z);
    parms.extent = make_extent(width_bytes, height, depth);
    parms.kind = cudaMemcpyDeviceToDevice;
    return jyppx::cuda::map_cuda_status(cudaMemcpy3DAsync(&parms, reinterpret_cast<StreamObject*>(stream)->handle), "cudaMemcpy3DAsync");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async 3D array-to-array copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_malloc_mipmapped_array(
    const JYPPX_CudaChannelFormatDesc* descriptor,
    const JYPPX_CudaArrayExtent* extent,
    uint32_t level_count,
    uint32_t flags,
    JYPPX_CudaMipmappedArray** out_array)
{
    auto status = jyppx::cuda::validate_output_pointer(out_array, "out_array");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_array = nullptr;
    status = validate_channel_format_descriptor(descriptor, "cudaMallocMipmappedArray");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_array_extent(extent, "cudaMallocMipmappedArray");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (level_count == 0)
    {
        jyppx::cuda::set_cuda_error("cudaMallocMipmappedArray", 0, "invalid-mipmap-level-count", "CUDA mipmapped array level count must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    const auto cuda_descriptor = make_channel_format_desc(*descriptor);
    const auto cuda_extent = make_extent(static_cast<size_t>(extent->width), static_cast<size_t>(extent->height), static_cast<size_t>(extent->depth));
    cudaMipmappedArray_t handle = nullptr;
    status = jyppx::cuda::map_cuda_status(cudaMallocMipmappedArray(&handle, &cuda_descriptor, cuda_extent, level_count, flags), "cudaMallocMipmappedArray");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* array_object = new (std::nothrow) jyppx::cuda::MipmappedArrayObject{};
    if (array_object == nullptr)
    {
        cudaFreeMipmappedArray(handle);
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    array_object->base.magic = jyppx::cuda::kObjectMagic;
    array_object->base.kind = jyppx::cuda::ObjectKind::MipmappedArray;
    array_object->levels = level_count;
    array_object->handle = handle;
    *out_array = reinterpret_cast<JYPPX_CudaMipmappedArray*>(array_object);
    return JYPPX_STATUS_OK;
#else
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA mipmapped array allocation");
#endif
}

JYPPX_StatusCode jyppx_cuda_free_mipmapped_array(JYPPX_CudaMipmappedArray* array)
{
    if (array == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    auto status = jyppx::cuda::validate_mipmapped_array(array, "array");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* array_object = reinterpret_cast<jyppx::cuda::MipmappedArrayObject*>(array);
#if JYPPX_HAS_CUDA_TOOLKIT
    status = jyppx::cuda::map_cuda_status(cudaFreeMipmappedArray(array_object->handle), "cudaFreeMipmappedArray");
#else
    status = JYPPX_STATUS_OK;
#endif
    delete array_object;
    return status;
}

JYPPX_StatusCode jyppx_cuda_get_mipmapped_array_level_info(JYPPX_CudaMipmappedArray* array, uint32_t level, JYPPX_CudaArrayInfo* out_info)
{
    auto status = jyppx::cuda::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    clear_array_info(out_info);
    status = jyppx::cuda::validate_mipmapped_array(array, "array");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* array_object = reinterpret_cast<jyppx::cuda::MipmappedArrayObject*>(array);
    if (level >= array_object->levels)
    {
        jyppx::cuda::set_cuda_error("cudaGetMipmappedArrayLevel", 0, "mipmap-level-out-of-range", "Requested mip level is outside the allocated CUDA mipmapped array level range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaArray_t level_array = nullptr;
    status = jyppx::cuda::map_cuda_status(cudaGetMipmappedArrayLevel(&level_array, array_object->handle, level), "cudaGetMipmappedArrayLevel");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return fill_array_info(level_array, out_info);
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA mipmapped array level info query");
#endif
}

JYPPX_StatusCode jyppx_cuda_mipmapped_array_get_memory_requirements(JYPPX_CudaMipmappedArray* array, int32_t device, JYPPX_CudaArrayMemoryRequirements* out_requirements)
{
    auto status = jyppx::cuda::validate_output_pointer(out_requirements, "out_requirements");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    clear_memory_requirements(out_requirements);
    status = jyppx::cuda::validate_mipmapped_array(array, "array");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaMipmappedArrayGetMemoryRequirements", 0, "invalid-device", "Device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaArrayMemoryRequirements requirements{};
    status = jyppx::cuda::map_cuda_status(cudaMipmappedArrayGetMemoryRequirements(&requirements, reinterpret_cast<jyppx::cuda::MipmappedArrayObject*>(array)->handle, device), "cudaMipmappedArrayGetMemoryRequirements");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    copy_memory_requirements(requirements, out_requirements);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA mipmapped array memory requirements query");
#endif
}

JYPPX_StatusCode jyppx_cuda_mipmapped_array_get_sparse_properties(JYPPX_CudaMipmappedArray* array, JYPPX_CudaArraySparseProperties* out_properties)
{
    auto status = jyppx::cuda::validate_output_pointer(out_properties, "out_properties");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    clear_sparse_properties(out_properties);
    status = jyppx::cuda::validate_mipmapped_array(array, "array");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaArraySparseProperties properties{};
    status = jyppx::cuda::map_cuda_status(cudaMipmappedArrayGetSparseProperties(&properties, reinterpret_cast<jyppx::cuda::MipmappedArrayObject*>(array)->handle), "cudaMipmappedArrayGetSparseProperties");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    copy_sparse_properties(properties, out_properties);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA mipmapped array sparse properties query");
#endif
}

JYPPX_StatusCode jyppx_cuda_ctx_reset_persisting_l2_cache(void)
{
#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaCtxResetPersistingL2Cache(), "cudaCtxResetPersistingL2Cache");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA persisting L2 cache reset");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_flush_gpu_direct_rdma_writes(int32_t target, int32_t scope)
{
    if (target < 0 || scope < 0)
    {
        jyppx::cuda::set_cuda_error("cudaDeviceFlushGPUDirectRDMAWrites", 0, "invalid-rdma-flush-argument", "CUDA GPUDirect RDMA flush target and scope must be non-negative enum values.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaDeviceFlushGPUDirectRDMAWrites(static_cast<cudaFlushGPUDirectRDMAWritesTarget>(target), static_cast<cudaFlushGPUDirectRDMAWritesScope>(scope)), "cudaDeviceFlushGPUDirectRDMAWrites");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA GPUDirect RDMA writes flush");
#endif
}
