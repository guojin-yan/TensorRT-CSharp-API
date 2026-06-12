namespace
{
JYPPX_StatusCode validate_memcpy_kind(const int32_t kind, const char* operation)
{
    if (kind < 0 || kind > 4)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-copy-kind", "CUDA memcpy kind must be a valid cudaMemcpyKind value.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode validate_pointer_argument(const void* pointer, const char* name, const char* operation)
{
    if (pointer == nullptr)
    {
        const std::string message = std::string(name) + " pointer must not be null.";
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-pointer", message.c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

#if JYPPX_HAS_CUDA_TOOLKIT
cudaPos make_bridge_cuda_pos(const JYPPX_CudaPos& position)
{
    cudaPos value{};
    value.x = position.x;
    value.y = position.y;
    value.z = position.z;
    return value;
}

cudaPitchedPtr make_bridge_cuda_pitched_ptr(const JYPPX_CudaPitchedPtr& pointer)
{
    return make_pitched_ptr(pointer.pointer, pointer.pitch, pointer.x_size, pointer.y_size);
}

JYPPX_StatusCode get_optional_array_handle(JYPPX_CudaArray* array, cudaArray_t* out_handle, const char* operation, const char* role)
{
    *out_handle = nullptr;
    if (array == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    auto status = jyppx::cuda::validate_array(array, role);
    if (status != JYPPX_STATUS_OK)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-array", "CUDA array operand is not a valid bridge array.");
        return status;
    }

    *out_handle = reinterpret_cast<jyppx::cuda::ArrayObject*>(array)->handle;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode make_cuda_memcpy_3d_peer_params(const JYPPX_CudaMemcpy3DPeerParams& source, cudaMemcpy3DPeerParms* destination, const char* operation)
{
    if (source.src_device < 0 || source.dst_device < 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-device", "Source and destination device ordinals must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = validate_array_extent_values(
        static_cast<size_t>(source.extent.width),
        static_cast<size_t>(source.extent.height),
        static_cast<size_t>(source.extent.depth),
        operation);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    cudaArray_t src_array = nullptr;
    status = get_optional_array_handle(source.src_array, &src_array, operation, "source array");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    cudaArray_t dst_array = nullptr;
    status = get_optional_array_handle(source.dst_array, &dst_array, operation, "destination array");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (src_array == nullptr && source.src_ptr.pointer == nullptr)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-source", "Either a source array or source pitched pointer must be provided.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (dst_array == nullptr && source.dst_ptr.pointer == nullptr)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-destination", "Either a destination array or destination pitched pointer must be provided.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    destination->srcArray = src_array;
    destination->srcPos = make_bridge_cuda_pos(source.src_pos);
    destination->srcPtr = make_bridge_cuda_pitched_ptr(source.src_ptr);
    destination->srcDevice = source.src_device;
    destination->dstArray = dst_array;
    destination->dstPos = make_bridge_cuda_pos(source.dst_pos);
    destination->dstPtr = make_bridge_cuda_pitched_ptr(source.dst_ptr);
    destination->dstDevice = source.dst_device;
    destination->extent = make_extent(
        static_cast<size_t>(source.extent.width),
        static_cast<size_t>(source.extent.height),
        static_cast<size_t>(source.extent.depth));
    return JYPPX_STATUS_OK;
}

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12090
cudaMemcpyAttributes make_cuda_memcpy_attributes(const JYPPX_CudaMemcpyAttributes& source)
{
    cudaMemcpyAttributes destination{};
    destination.srcAccessOrder = static_cast<cudaMemcpySrcAccessOrder>(source.src_access_order);
    destination.srcLocHint = make_bridge_mem_location(source.src_location_hint);
    destination.dstLocHint = make_bridge_mem_location(source.dst_location_hint);
    destination.flags = source.flags;
    return destination;
}

JYPPX_StatusCode make_cuda_memcpy_3d_operand(const JYPPX_CudaMemcpy3DOperand& source, cudaMemcpy3DOperand* destination, const char* operation, const char* role)
{
    destination->type = static_cast<cudaMemcpy3DOperandType>(source.type);
    switch (destination->type)
    {
    case cudaMemcpyOperandTypePointer:
        if (source.pointer.pointer == nullptr)
        {
            const std::string message = std::string(role) + " pointer operand must not be null.";
            jyppx::cuda::set_cuda_error(operation, 0, "invalid-pointer-operand", message.c_str());
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }

        destination->op.ptr.ptr = source.pointer.pointer;
        destination->op.ptr.rowLength = source.pointer.row_length;
        destination->op.ptr.layerHeight = source.pointer.layer_height;
        destination->op.ptr.locHint = make_bridge_mem_location(source.pointer.location_hint);
        return JYPPX_STATUS_OK;

    case cudaMemcpyOperandTypeArray:
    {
        if (source.array.array == nullptr)
        {
            const std::string message = std::string(role) + " array operand must not be null.";
            jyppx::cuda::set_cuda_error(operation, 0, "invalid-array-operand", message.c_str());
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }

        auto status = jyppx::cuda::validate_array(source.array.array, role);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        destination->op.array.array = reinterpret_cast<jyppx::cuda::ArrayObject*>(source.array.array)->handle;
        destination->op.array.offset.x = source.array.offset.x;
        destination->op.array.offset.y = source.array.offset.y;
        destination->op.array.offset.z = source.array.offset.z;
        return JYPPX_STATUS_OK;
    }

    default:
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-operand-type", "CUDA 3D batch copy operand type must be pointer or array.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
}

JYPPX_StatusCode make_cuda_memcpy_3d_batch_op(const JYPPX_CudaMemcpy3DBatchOp& source, cudaMemcpy3DBatchOp* destination, const char* operation)
{
    auto status = make_cuda_memcpy_3d_operand(source.source, &destination->src, operation, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = make_cuda_memcpy_3d_operand(source.destination, &destination->dst, operation, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_array_extent_values(
        static_cast<size_t>(source.extent.width),
        static_cast<size_t>(source.extent.height),
        static_cast<size_t>(source.extent.depth),
        operation);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    destination->extent = make_extent(
        static_cast<size_t>(source.extent.width),
        static_cast<size_t>(source.extent.height),
        static_cast<size_t>(source.extent.depth));
    destination->srcAccessOrder = static_cast<cudaMemcpySrcAccessOrder>(source.source_access_order);
    destination->flags = source.flags;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode convert_memcpy_attributes(
    const JYPPX_CudaMemcpyAttributes* source,
    size_t count,
    cudaMemcpyAttributes** out_attributes,
    const char* operation)
{
    *out_attributes = nullptr;
    if (count == 0)
    {
        return JYPPX_STATUS_OK;
    }

    if (source == nullptr)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-attributes", "CUDA memcpy attributes must not be null when attribute count is non-zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* attributes = new (std::nothrow) cudaMemcpyAttributes[count];
    if (attributes == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    for (size_t index = 0; index < count; ++index)
    {
        attributes[index] = make_cuda_memcpy_attributes(source[index]);
    }

    *out_attributes = attributes;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode convert_memcpy_3d_batch_ops(
    const JYPPX_CudaMemcpy3DBatchOp* source,
    size_t count,
    cudaMemcpy3DBatchOp** out_operations,
    const char* operation)
{
    *out_operations = nullptr;
    if (count == 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-operation-count", "CUDA 3D batch copy operation count must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (source == nullptr)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-operations", "CUDA 3D batch copy operations must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* operations = new (std::nothrow) cudaMemcpy3DBatchOp[count];
    if (operations == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    for (size_t index = 0; index < count; ++index)
    {
        auto status = make_cuda_memcpy_3d_batch_op(source[index], &operations[index], operation);
        if (status != JYPPX_STATUS_OK)
        {
            delete[] operations;
            return status;
        }
    }

    *out_operations = operations;
    return JYPPX_STATUS_OK;
}
#endif
#endif
}

JYPPX_StatusCode jyppx_cuda_malloc_host(size_t size, JYPPX_CudaPinnedMemory** out_memory)
{
    auto status = jyppx::cuda::validate_output_pointer(out_memory, "out_memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_memory = nullptr;
    if (size == 0)
    {
        jyppx::cuda::set_cuda_error("cudaMallocHost", 0, "invalid-size", "Pinned host allocation size must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* memory = new (std::nothrow) PinnedMemoryObject{};
    if (memory == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    memory->base.magic = jyppx::cuda::kObjectMagic;
    memory->base.kind = ObjectKind::PinnedMemory;
    memory->pointer = nullptr;
    memory->size = size;
    memory->flags = 0;
    memory->is_registered = false;

    status = jyppx::cuda::map_cuda_status(cudaMallocHost(&memory->pointer, size), "cudaMallocHost");
    if (status != JYPPX_STATUS_OK)
    {
        delete memory;
        return status;
    }

    *out_memory = reinterpret_cast<JYPPX_CudaPinnedMemory*>(memory);
    return JYPPX_STATUS_OK;
#else
    (void)size;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA pinned host allocation");
#endif
}

JYPPX_StatusCode jyppx_cuda_host_unregister(JYPPX_CudaPinnedMemory* memory)
{
    if (memory == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    auto status = jyppx::cuda::validate_pinned_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<PinnedMemoryObject*>(memory);
    if (!memory_object->is_registered)
    {
        jyppx::cuda::set_cuda_error("cudaHostUnregister", 0, "not-registered", "Only registered host memory can be unregistered.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    status = jyppx::cuda::map_cuda_status(cudaHostUnregister(memory_object->pointer), "cudaHostUnregister");
#else
    status = JYPPX_STATUS_OK;
#endif
    delete memory_object;
    return status;
}

JYPPX_StatusCode jyppx_cuda_malloc_3d(size_t width_bytes, size_t height, size_t depth, JYPPX_CudaPitchedMemory** out_memory)
{
    auto status = jyppx::cuda::validate_output_pointer(out_memory, "out_memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_memory = nullptr;
    status = validate_pitched_dimensions(width_bytes, height, "cudaMalloc3D");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (depth == 0)
    {
        jyppx::cuda::set_cuda_error("cudaMalloc3D", 0, "invalid-3d-depth", "CUDA 3D allocation depth must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
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
    memory->depth = depth;

    cudaPitchedPtr pitched{};
    const cudaExtent extent = make_extent(width_bytes, height, depth);
    status = jyppx::cuda::map_cuda_status(cudaMalloc3D(&pitched, extent), "cudaMalloc3D");
    if (status != JYPPX_STATUS_OK)
    {
        delete memory;
        return status;
    }

    memory->pointer = pitched.ptr;
    memory->pitch_bytes = pitched.pitch;
    *out_memory = reinterpret_cast<JYPPX_CudaPitchedMemory*>(memory);
    return JYPPX_STATUS_OK;
#else
    (void)depth;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 3D pitched memory allocation");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_async(void* destination, const void* source, size_t size, int32_t kind, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_output_pointer(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (source == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyAsync", 0, "invalid-source", "Source pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = validate_copy_bytes(size, "cudaMemcpyAsync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_memcpy_kind(kind, "cudaMemcpyAsync");
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
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(
        cudaMemcpyAsync(destination, source, size, static_cast<cudaMemcpyKind>(kind), stream_object->handle),
        "cudaMemcpyAsync");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA asynchronous memcpy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_2d_async(
    void* destination,
    size_t destination_pitch,
    const void* source,
    size_t source_pitch,
    size_t width_bytes,
    size_t height,
    int32_t kind,
    JYPPX_CudaStream* stream)
{
    if (destination == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy2DAsync", 0, "invalid-destination", "Destination pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (source == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy2DAsync", 0, "invalid-source", "Source pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = validate_pitched_dimensions(width_bytes, height, "cudaMemcpy2DAsync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_host_pitch(destination_pitch, width_bytes, "cudaMemcpy2DAsync", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_host_pitch(source_pitch, width_bytes, "cudaMemcpy2DAsync", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_memcpy_kind(kind, "cudaMemcpy2DAsync");
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
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(
        cudaMemcpy2DAsync(destination, destination_pitch, source, source_pitch, width_bytes, height, static_cast<cudaMemcpyKind>(kind), stream_object->handle),
        "cudaMemcpy2DAsync");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA asynchronous 2D memcpy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_3d(
    void* destination,
    size_t destination_pitch,
    const void* source,
    size_t source_pitch,
    size_t width_bytes,
    size_t height,
    size_t depth,
    int32_t kind)
{
    if (destination == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy3D", 0, "invalid-destination", "Destination pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (source == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy3D", 0, "invalid-source", "Source pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = validate_pitched_dimensions(width_bytes, height, "cudaMemcpy3D");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (depth == 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy3D", 0, "invalid-3d-depth", "CUDA 3D copy depth must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = validate_host_pitch(destination_pitch, width_bytes, "cudaMemcpy3D", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_host_pitch(source_pitch, width_bytes, "cudaMemcpy3D", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_memcpy_kind(kind, "cudaMemcpy3D");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemcpy3DParms parameters{};
    parameters.srcPtr = make_pitched_ptr(const_cast<void*>(source), source_pitch, width_bytes, height);
    parameters.dstPtr = make_pitched_ptr(destination, destination_pitch, width_bytes, height);
    parameters.extent = make_extent(width_bytes, height, depth);
    parameters.kind = static_cast<cudaMemcpyKind>(kind);
    return jyppx::cuda::map_cuda_status(cudaMemcpy3D(&parameters), "cudaMemcpy3D");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 3D memcpy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_3d_async(
    void* destination,
    size_t destination_pitch,
    const void* source,
    size_t source_pitch,
    size_t width_bytes,
    size_t height,
    size_t depth,
    int32_t kind,
    JYPPX_CudaStream* stream)
{
    if (destination == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy3DAsync", 0, "invalid-destination", "Destination pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (source == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy3DAsync", 0, "invalid-source", "Source pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = validate_pitched_dimensions(width_bytes, height, "cudaMemcpy3DAsync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (depth == 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy3DAsync", 0, "invalid-3d-depth", "CUDA 3D copy depth must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = validate_host_pitch(destination_pitch, width_bytes, "cudaMemcpy3DAsync", "Destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_host_pitch(source_pitch, width_bytes, "cudaMemcpy3DAsync", "Source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_memcpy_kind(kind, "cudaMemcpy3DAsync");
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
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    cudaMemcpy3DParms parameters{};
    parameters.srcPtr = make_pitched_ptr(const_cast<void*>(source), source_pitch, width_bytes, height);
    parameters.dstPtr = make_pitched_ptr(destination, destination_pitch, width_bytes, height);
    parameters.extent = make_extent(width_bytes, height, depth);
    parameters.kind = static_cast<cudaMemcpyKind>(kind);
    return jyppx::cuda::map_cuda_status(cudaMemcpy3DAsync(&parameters, stream_object->handle), "cudaMemcpy3DAsync");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async 3D memcpy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_to_symbol(const void* symbol, const void* source, size_t count, size_t offset, int32_t kind)
{
    if (symbol == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyToSymbol", 0, "invalid-symbol", "CUDA symbol pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (source == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyToSymbol", 0, "invalid-source", "Source pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = validate_copy_bytes(count, "cudaMemcpyToSymbol");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_memcpy_kind(kind, "cudaMemcpyToSymbol");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(
        cudaMemcpyToSymbol(symbol, source, count, offset, static_cast<cudaMemcpyKind>(kind)),
        "cudaMemcpyToSymbol");
#else
    (void)offset;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA symbol copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_from_symbol(void* destination, const void* symbol, size_t count, size_t offset, int32_t kind)
{
    if (destination == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyFromSymbol", 0, "invalid-destination", "Destination pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (symbol == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyFromSymbol", 0, "invalid-symbol", "CUDA symbol pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = validate_copy_bytes(count, "cudaMemcpyFromSymbol");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_memcpy_kind(kind, "cudaMemcpyFromSymbol");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(
        cudaMemcpyFromSymbol(destination, symbol, count, offset, static_cast<cudaMemcpyKind>(kind)),
        "cudaMemcpyFromSymbol");
#else
    (void)offset;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA symbol copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_to_symbol_async(const void* symbol, const void* source, size_t count, size_t offset, int32_t kind, JYPPX_CudaStream* stream)
{
    if (symbol == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyToSymbolAsync", 0, "invalid-symbol", "CUDA symbol pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (source == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyToSymbolAsync", 0, "invalid-source", "Source pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = validate_copy_bytes(count, "cudaMemcpyToSymbolAsync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_memcpy_kind(kind, "cudaMemcpyToSymbolAsync");
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
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(
        cudaMemcpyToSymbolAsync(symbol, source, count, offset, static_cast<cudaMemcpyKind>(kind), stream_object->handle),
        "cudaMemcpyToSymbolAsync");
#else
    (void)offset;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA asynchronous symbol copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_from_symbol_async(void* destination, const void* symbol, size_t count, size_t offset, int32_t kind, JYPPX_CudaStream* stream)
{
    if (destination == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyFromSymbolAsync", 0, "invalid-destination", "Destination pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (symbol == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyFromSymbolAsync", 0, "invalid-symbol", "CUDA symbol pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = validate_copy_bytes(count, "cudaMemcpyFromSymbolAsync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_memcpy_kind(kind, "cudaMemcpyFromSymbolAsync");
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
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(
        cudaMemcpyFromSymbolAsync(destination, symbol, count, offset, static_cast<cudaMemcpyKind>(kind), stream_object->handle),
        "cudaMemcpyFromSymbolAsync");
#else
    (void)offset;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA asynchronous symbol copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_3d_peer(const JYPPX_CudaMemcpy3DPeerParams* parameters)
{
    if (parameters == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy3DPeer", 0, "invalid-parameters", "CUDA 3D peer copy parameters must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemcpy3DPeerParms cuda_parameters{};
    auto status = make_cuda_memcpy_3d_peer_params(*parameters, &cuda_parameters, "cudaMemcpy3DPeer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return jyppx::cuda::map_cuda_status(cudaMemcpy3DPeer(&cuda_parameters), "cudaMemcpy3DPeer");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 3D peer memcpy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_3d_peer_async(const JYPPX_CudaMemcpy3DPeerParams* parameters, JYPPX_CudaStream* stream)
{
    if (parameters == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy3DPeerAsync", 0, "invalid-parameters", "CUDA 3D peer copy parameters must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemcpy3DPeerParms cuda_parameters{};
    status = make_cuda_memcpy_3d_peer_params(*parameters, &cuda_parameters, "cudaMemcpy3DPeerAsync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaMemcpy3DPeerAsync(&cuda_parameters, stream_object->handle), "cudaMemcpy3DPeerAsync");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async 3D peer memcpy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_batch_async(
    void** destinations,
    const void** sources,
    const size_t* sizes,
    size_t count,
    const JYPPX_CudaMemcpyAttributes* attributes,
    const size_t* attribute_indices,
    size_t attribute_count,
    size_t* out_fail_index,
    JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_output_pointer(out_fail_index, "out_fail_index");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_fail_index = std::numeric_limits<size_t>::max();
    status = validate_pointer_argument(destinations, "destinations", "cudaMemcpyBatchAsync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_pointer_argument(sources, "sources", "cudaMemcpyBatchAsync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_pointer_argument(sizes, "sizes", "cudaMemcpyBatchAsync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (count == 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyBatchAsync", 0, "invalid-count", "CUDA batch copy count must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (attribute_count > 0 && attribute_indices == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyBatchAsync", 0, "invalid-attribute-indices", "CUDA batch copy attribute indices must not be null when attributes are supplied.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12090
    cudaMemcpyAttributes* cuda_attributes = nullptr;
    status = convert_memcpy_attributes(attributes, attribute_count, &cuda_attributes, "cudaMemcpyBatchAsync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
#if CUDART_VERSION >= 13000
    status = jyppx::cuda::map_cuda_status(
        cudaMemcpyBatchAsync(
            reinterpret_cast<void* const*>(destinations),
            reinterpret_cast<const void* const*>(sources),
            sizes,
            count,
            cuda_attributes,
            const_cast<size_t*>(attribute_indices),
            attribute_count,
            stream_object->handle),
        "cudaMemcpyBatchAsync");
#else
    status = jyppx::cuda::map_cuda_status(
        cudaMemcpyBatchAsync(
            destinations,
            const_cast<void**>(sources),
            const_cast<size_t*>(sizes),
            count,
            cuda_attributes,
            const_cast<size_t*>(attribute_indices),
            attribute_count,
            out_fail_index,
            stream_object->handle),
        "cudaMemcpyBatchAsync");
#endif
    delete[] cuda_attributes;
    return status;
#else
    (void)attributes;
    (void)attribute_indices;
    (void)attribute_count;
    (void)stream;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA batch memcpy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_3d_batch_async(
    size_t operation_count,
    const JYPPX_CudaMemcpy3DBatchOp* operations,
    size_t* out_fail_index,
    uint64_t flags,
    JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_output_pointer(out_fail_index, "out_fail_index");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_fail_index = std::numeric_limits<size_t>::max();
    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12090
    cudaMemcpy3DBatchOp* cuda_operations = nullptr;
    status = convert_memcpy_3d_batch_ops(operations, operation_count, &cuda_operations, "cudaMemcpy3DBatchAsync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
#if CUDART_VERSION >= 13000
    status = jyppx::cuda::map_cuda_status(
        cudaMemcpy3DBatchAsync(operation_count, cuda_operations, static_cast<unsigned long long>(flags), stream_object->handle),
        "cudaMemcpy3DBatchAsync");
#else
    status = jyppx::cuda::map_cuda_status(
        cudaMemcpy3DBatchAsync(operation_count, cuda_operations, out_fail_index, static_cast<unsigned long long>(flags), stream_object->handle),
        "cudaMemcpy3DBatchAsync");
#endif
    delete[] cuda_operations;
    return status;
#else
    (void)operations;
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 3D batch memcpy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_with_attributes_async(
    void* destination,
    const void* source,
    size_t size,
    const JYPPX_CudaMemcpyAttributes* attributes,
    JYPPX_CudaStream* stream)
{
    auto status = validate_pointer_argument(destination, "destination", "cudaMemcpyWithAttributesAsync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_pointer_argument(source, "source", "cudaMemcpyWithAttributesAsync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_copy_bytes(size, "cudaMemcpyWithAttributesAsync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (attributes == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyWithAttributesAsync", 0, "invalid-attributes", "CUDA memcpy attributes must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    cudaMemcpyAttributes cuda_attributes = make_cuda_memcpy_attributes(*attributes);
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(
        cudaMemcpyWithAttributesAsync(destination, source, size, &cuda_attributes, stream_object->handle),
        "cudaMemcpyWithAttributesAsync");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memcpy with attributes");
#endif
}

JYPPX_StatusCode jyppx_cuda_memcpy_3d_with_attributes_async(
    const JYPPX_CudaMemcpy3DBatchOp* operation,
    uint64_t flags,
    JYPPX_CudaStream* stream)
{
    if (operation == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy3DWithAttributesAsync", 0, "invalid-operation", "CUDA 3D copy operation must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    cudaMemcpy3DBatchOp cuda_operation{};
    status = make_cuda_memcpy_3d_batch_op(*operation, &cuda_operation, "cudaMemcpy3DWithAttributesAsync");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(
        cudaMemcpy3DWithAttributesAsync(&cuda_operation, static_cast<unsigned long long>(flags), stream_object->handle),
        "cudaMemcpy3DWithAttributesAsync");
#else
    (void)operation;
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA 3D memcpy with attributes");
#endif
}
