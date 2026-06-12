namespace
{
JYPPX_StatusCode validate_cuda_atomic_capability_arguments(
    uint32_t* capabilities,
    const int32_t* operations,
    uint32_t count,
    const char* operation)
{
    if (capabilities == nullptr)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-capabilities", "CUDA atomic capability output array must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (operations == nullptr)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-operations", "CUDA atomic operation array must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (count == 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-count", "CUDA atomic operation count must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
JYPPX_StatusCode convert_cuda_atomic_operations(const int32_t* source, uint32_t count, cudaAtomicOperation** out_operations)
{
    auto* operations = new (std::nothrow) cudaAtomicOperation[count];
    if (operations == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    for (uint32_t index = 0; index < count; ++index)
    {
        operations[index] = static_cast<cudaAtomicOperation>(source[index]);
    }

    *out_operations = operations;
    return JYPPX_STATUS_OK;
}
#endif
}

JYPPX_StatusCode jyppx_cuda_device_get_host_atomic_capabilities(uint32_t* capabilities, const int32_t* operations, uint32_t count, int32_t device)
{
    auto status = validate_cuda_atomic_capability_arguments(capabilities, operations, count, "cudaDeviceGetHostAtomicCapabilities");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_cuda_device_ordinal(device, "cudaDeviceGetHostAtomicCapabilities");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    cudaAtomicOperation* cuda_operations = nullptr;
    status = convert_cuda_atomic_operations(operations, count, &cuda_operations);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::map_cuda_status(
        cudaDeviceGetHostAtomicCapabilities(reinterpret_cast<unsigned int*>(capabilities), cuda_operations, count, device),
        "cudaDeviceGetHostAtomicCapabilities");
    delete[] cuda_operations;
    return status;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA host atomic capability query");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_get_p2p_atomic_capabilities(
    uint32_t* capabilities,
    const int32_t* operations,
    uint32_t count,
    int32_t source_device,
    int32_t destination_device)
{
    auto status = validate_cuda_atomic_capability_arguments(capabilities, operations, count, "cudaDeviceGetP2PAtomicCapabilities");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_cuda_device_ordinal(source_device, "cudaDeviceGetP2PAtomicCapabilities");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_cuda_device_ordinal(destination_device, "cudaDeviceGetP2PAtomicCapabilities");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    cudaAtomicOperation* cuda_operations = nullptr;
    status = convert_cuda_atomic_operations(operations, count, &cuda_operations);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::map_cuda_status(
        cudaDeviceGetP2PAtomicCapabilities(reinterpret_cast<unsigned int*>(capabilities), cuda_operations, count, source_device, destination_device),
        "cudaDeviceGetP2PAtomicCapabilities");
    delete[] cuda_operations;
    return status;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA peer atomic capability query");
#endif
}
