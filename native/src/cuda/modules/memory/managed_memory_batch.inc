#include <new>

namespace
{
enum class ManagedMemoryBatchOperation
{
    Prefetch,
    Discard,
    DiscardAndPrefetch
};

const char* get_managed_memory_batch_operation_name(const ManagedMemoryBatchOperation operation) noexcept
{
    switch (operation)
    {
    case ManagedMemoryBatchOperation::Prefetch:
        return "cudaMemPrefetchBatchAsync";
    case ManagedMemoryBatchOperation::Discard:
        return "cudaMemDiscardBatchAsync";
    default:
        return "cudaMemDiscardAndPrefetchBatchAsync";
    }
}

JYPPX_StatusCode report_managed_memory_batch_exception(
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

JYPPX_StatusCode managed_memory_batch_not_supported(const char* operation) noexcept
{
    try
    {
        jyppx::cuda::set_cuda_error(
            operation,
            0,
            "cuda-version-not-supported",
            "CUDA managed-memory batch operations require CUDA Toolkit 13.0 or later.");
    }
    catch (...)
    {
    }
    return JYPPX_STATUS_NOT_SUPPORTED;
}

template <typename TCallable>
JYPPX_StatusCode run_managed_memory_batch_noexcept(const char* operation, TCallable&& callable) noexcept
{
    try
    {
        return callable();
    }
    catch (const std::bad_alloc&)
    {
        return report_managed_memory_batch_exception(
            operation,
            "native-allocation-failed",
            "The CUDA managed-memory batch bridge could not allocate temporary array storage.");
    }
    catch (...)
    {
        return report_managed_memory_batch_exception(
            operation,
            "native-exception-caught",
            "A native exception was contained by the CUDA managed-memory batch bridge.");
    }
}

JYPPX_StatusCode validate_managed_memory_batch_ranges(
    const JYPPX_CudaManagedMemoryBatchRange* ranges,
    const size_t range_count,
    const bool requires_destination,
    const char* operation,
    std::vector<void*>& pointers,
    std::vector<size_t>& sizes,
    std::vector<int32_t>& destinations)
{
    if (ranges == nullptr)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "missing-ranges", "The CUDA managed-memory batch range array must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (range_count == 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "empty-ranges", "At least one CUDA managed-memory batch range is required.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    pointers.resize(range_count);
    sizes.resize(range_count);
    destinations.resize(range_count);
    for (size_t index = 0; index < range_count; ++index)
    {
        auto status = jyppx::cuda::validate_memory(ranges[index].memory, "range.memory");
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        auto* memory = reinterpret_cast<jyppx::cuda::MemoryObject*>(ranges[index].memory);
        if (!memory->is_managed)
        {
            jyppx::cuda::set_cuda_error(operation, 0, "memory-not-managed", "CUDA batch discard/prefetch ranges must use cudaMallocManaged bridge owners.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }
        if (requires_destination && ranges[index].destination_device < 0)
        {
            jyppx::cuda::set_cuda_error(operation, 0, "invalid-device", "CUDA managed-memory batch destination devices must be greater than or equal to zero.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }

        const void* range_pointer = nullptr;
        status = validate_memory_range_query(
            memory,
            ranges[index].offset,
            ranges[index].size,
            operation,
            &range_pointer);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        pointers[index] = const_cast<void*>(range_pointer);
        sizes[index] = ranges[index].size;
        destinations[index] = ranges[index].destination_device;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode managed_memory_batch_impl(
    const JYPPX_CudaManagedMemoryBatchRange* ranges,
    const size_t range_count,
    JYPPX_CudaStream* stream_handle,
    const ManagedMemoryBatchOperation operation) noexcept
{
    const char* operation_name = get_managed_memory_batch_operation_name(operation);
    return run_managed_memory_batch_noexcept(operation_name, [&]() {
        auto status = jyppx::cuda::validate_stream(stream_handle, "stream");
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        std::vector<void*> pointers;
        std::vector<size_t> sizes;
        std::vector<int32_t> destinations;
        status = validate_managed_memory_batch_ranges(
            ranges,
            range_count,
            operation != ManagedMemoryBatchOperation::Discard,
            operation_name,
            pointers,
            sizes,
            destinations);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
        auto* stream = reinterpret_cast<jyppx::cuda::StreamObject*>(stream_handle);
        if (operation == ManagedMemoryBatchOperation::Discard)
        {
            return jyppx::cuda::map_cuda_status(
                cudaMemDiscardBatchAsync(pointers.data(), sizes.data(), range_count, 0ULL, stream->handle),
                operation_name);
        }

        std::vector<cudaMemLocation> locations(range_count);
        std::vector<size_t> location_indices(range_count);
        for (size_t index = 0; index < range_count; ++index)
        {
            locations[index] = make_device_mem_location(destinations[index]);
            location_indices[index] = index;
        }

        const auto result = operation == ManagedMemoryBatchOperation::Prefetch
            ? cudaMemPrefetchBatchAsync(
                pointers.data(),
                sizes.data(),
                range_count,
                locations.data(),
                location_indices.data(),
                range_count,
                0ULL,
                stream->handle)
            : cudaMemDiscardAndPrefetchBatchAsync(
                pointers.data(),
                sizes.data(),
                range_count,
                locations.data(),
                location_indices.data(),
                range_count,
                0ULL,
                stream->handle);
        return jyppx::cuda::map_cuda_status(result, operation_name);
#elif JYPPX_HAS_CUDA_TOOLKIT
        return managed_memory_batch_not_supported(operation_name);
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA managed-memory batch operation");
#endif
    });
}
}

#if defined(_MSC_VER) && defined(_WIN32)
#define JYPPX_CUDA_MANAGED_MEMORY_BATCH_GUARD(operation, call) \
    __try { return call; } \
    __except (1) { return report_managed_memory_batch_exception(operation, "windows-seh-caught", "A Windows structured exception was contained by the CUDA managed-memory batch bridge."); }
#else
#define JYPPX_CUDA_MANAGED_MEMORY_BATCH_GUARD(operation, call) return call
#endif

JYPPX_StatusCode jyppx_cuda_managed_memory_prefetch_batch_async_safe(
    const JYPPX_CudaManagedMemoryBatchRange* ranges,
    size_t range_count,
    JYPPX_CudaStream* stream)
{
    JYPPX_CUDA_MANAGED_MEMORY_BATCH_GUARD(
        "cudaMemPrefetchBatchAsync",
        managed_memory_batch_impl(ranges, range_count, stream, ManagedMemoryBatchOperation::Prefetch));
}

JYPPX_StatusCode jyppx_cuda_managed_memory_discard_batch_async_safe(
    const JYPPX_CudaManagedMemoryBatchRange* ranges,
    size_t range_count,
    JYPPX_CudaStream* stream)
{
    JYPPX_CUDA_MANAGED_MEMORY_BATCH_GUARD(
        "cudaMemDiscardBatchAsync",
        managed_memory_batch_impl(ranges, range_count, stream, ManagedMemoryBatchOperation::Discard));
}

JYPPX_StatusCode jyppx_cuda_managed_memory_discard_and_prefetch_batch_async_safe(
    const JYPPX_CudaManagedMemoryBatchRange* ranges,
    size_t range_count,
    JYPPX_CudaStream* stream)
{
    JYPPX_CUDA_MANAGED_MEMORY_BATCH_GUARD(
        "cudaMemDiscardAndPrefetchBatchAsync",
        managed_memory_batch_impl(ranges, range_count, stream, ManagedMemoryBatchOperation::DiscardAndPrefetch));
}

#undef JYPPX_CUDA_MANAGED_MEMORY_BATCH_GUARD
