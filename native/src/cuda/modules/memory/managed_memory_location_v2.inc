#include <new>

namespace
{
constexpr int32_t kCudaMemLocationTypeDevice = 1;
constexpr int32_t kCudaMemLocationTypeHost = 2;
constexpr int32_t kCudaMemLocationTypeHostNuma = 3;
constexpr int32_t kCudaMemLocationTypeHostNumaCurrent = 4;

JYPPX_StatusCode report_managed_memory_location_error(
    const char* operation,
    JYPPX_StatusCode status,
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
    return status;
}

JYPPX_StatusCode validate_managed_memory_owner(
    JYPPX_CudaMemory* memory,
    const char* operation,
    MemoryObject** out_memory)
{
    auto status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(memory);
    if (!memory_object->is_managed)
    {
        return report_managed_memory_location_error(
            operation,
            JYPPX_STATUS_INVALID_ARGUMENT,
            "managed-memory-owner-required",
            "The location-based CUDA memory operation requires a cudaMallocManaged allocation owner.");
    }

    *out_memory = memory_object;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode validate_managed_memory_location(
    int32_t location_type,
    int32_t location_id,
    const char* operation) noexcept
{
    switch (location_type)
    {
    case kCudaMemLocationTypeDevice:
    case kCudaMemLocationTypeHostNuma:
        if (location_id >= 0)
        {
            return JYPPX_STATUS_OK;
        }
        break;
    case kCudaMemLocationTypeHost:
    case kCudaMemLocationTypeHostNumaCurrent:
        if (location_id == 0)
        {
            return JYPPX_STATUS_OK;
        }
        break;
    default:
        break;
    }

    return report_managed_memory_location_error(
        operation,
        JYPPX_STATUS_INVALID_ARGUMENT,
        "invalid-memory-location",
        "The CUDA memory location type and identifier combination is invalid or non-canonical.");
}

JYPPX_StatusCode validate_managed_memory_location_advice(
    int32_t advice,
    int32_t location_type,
    const char* operation) noexcept
{
    if (advice < 1 || advice > 6)
    {
        return report_managed_memory_location_error(
            operation,
            JYPPX_STATUS_INVALID_ARGUMENT,
            "invalid-memory-advice",
            "The CUDA memory advice value is outside the supported managed-memory advice range.");
    }

    if ((advice == 5 || advice == 6) &&
        location_type != kCudaMemLocationTypeDevice &&
        location_type != kCudaMemLocationTypeHost)
    {
        return report_managed_memory_location_error(
            operation,
            JYPPX_STATUS_INVALID_ARGUMENT,
            "invalid-advice-location",
            "Accessed-by advice accepts only CUDA device or host memory locations.");
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode managed_memory_location_not_supported(const char* operation) noexcept
{
    return report_managed_memory_location_error(
        operation,
        JYPPX_STATUS_NOT_SUPPORTED,
        "cuda-version-not-supported",
        "Location-based CUDA managed-memory operations require CUDA Toolkit 12.3 or later.");
}

template <typename TCallable>
JYPPX_StatusCode run_managed_memory_location_noexcept(
    const char* operation,
    TCallable&& callable) noexcept
{
    try
    {
        return callable();
    }
    catch (const std::bad_alloc&)
    {
        return report_managed_memory_location_error(
            operation,
            JYPPX_STATUS_OUT_OF_MEMORY,
            "native-allocation-failed",
            "The CUDA managed-memory location bridge could not allocate temporary storage.");
    }
    catch (...)
    {
        return report_managed_memory_location_error(
            operation,
            JYPPX_STATUS_RUNTIME_ERROR,
            "native-exception-caught",
            "A native exception was contained by the CUDA managed-memory location bridge.");
    }
}

JYPPX_StatusCode prefetch_managed_memory_location_impl(
    JYPPX_CudaMemory* memory,
    size_t offset,
    size_t count,
    int32_t location_type,
    int32_t location_id,
    JYPPX_CudaStream* stream) noexcept
{
    constexpr const char* operation = "cudaMemPrefetchAsync_v2";
    return run_managed_memory_location_noexcept(operation, [&]() -> JYPPX_StatusCode {
        MemoryObject* memory_object = nullptr;
        auto status = validate_managed_memory_owner(memory, operation, &memory_object);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        status = jyppx::cuda::validate_stream(stream, "stream");
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        status = validate_managed_memory_location(location_type, location_id, operation);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        const void* range_pointer = nullptr;
        status = validate_memory_range_query(memory_object, offset, count, operation, &range_pointer);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12030
        cudaMemLocation location{};
        location.type = static_cast<cudaMemLocationType>(location_type);
        location.id = location_id;
        auto* stream_object = reinterpret_cast<StreamObject*>(stream);
#if CUDART_VERSION >= 13000
        return jyppx::cuda::map_cuda_status(
            cudaMemPrefetchAsync(range_pointer, count, location, 0U, stream_object->handle),
            "cudaMemPrefetchAsync");
#else
        return jyppx::cuda::map_cuda_status(
            cudaMemPrefetchAsync_v2(range_pointer, count, location, 0U, stream_object->handle),
            operation);
#endif
#elif JYPPX_HAS_CUDA_TOOLKIT
        return managed_memory_location_not_supported(operation);
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA location-based managed-memory prefetch");
#endif
    });
}

JYPPX_StatusCode advise_managed_memory_location_impl(
    JYPPX_CudaMemory* memory,
    size_t offset,
    size_t count,
    int32_t advice,
    int32_t location_type,
    int32_t location_id) noexcept
{
    constexpr const char* operation = "cudaMemAdvise_v2";
    return run_managed_memory_location_noexcept(operation, [&]() -> JYPPX_StatusCode {
        MemoryObject* memory_object = nullptr;
        auto status = validate_managed_memory_owner(memory, operation, &memory_object);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        status = validate_managed_memory_location(location_type, location_id, operation);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        status = validate_managed_memory_location_advice(advice, location_type, operation);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

        const void* range_pointer = nullptr;
        status = validate_memory_range_query(memory_object, offset, count, operation, &range_pointer);
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12030
        cudaMemLocation location{};
        location.type = static_cast<cudaMemLocationType>(location_type);
        location.id = location_id;
#if CUDART_VERSION >= 13000
        return jyppx::cuda::map_cuda_status(
            cudaMemAdvise(range_pointer, count, static_cast<cudaMemoryAdvise>(advice), location),
            "cudaMemAdvise");
#else
        return jyppx::cuda::map_cuda_status(
            cudaMemAdvise_v2(range_pointer, count, static_cast<cudaMemoryAdvise>(advice), location),
            operation);
#endif
#elif JYPPX_HAS_CUDA_TOOLKIT
        return managed_memory_location_not_supported(operation);
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA location-based managed-memory advice");
#endif
    });
}
}

#if defined(_MSC_VER) && defined(_WIN32)
#define JYPPX_CUDA_MANAGED_MEMORY_LOCATION_GUARD(operation, call) \
    __try { return call; } \
    __except (1) { return report_managed_memory_location_error(operation, JYPPX_STATUS_RUNTIME_ERROR, "windows-seh-caught", "A Windows structured exception was contained by the CUDA managed-memory location bridge."); }
#else
#define JYPPX_CUDA_MANAGED_MEMORY_LOCATION_GUARD(operation, call) return call
#endif

JYPPX_StatusCode jyppx_cuda_managed_memory_prefetch_location_range_async_safe(
    JYPPX_CudaMemory* memory,
    size_t offset,
    size_t count,
    int32_t location_type,
    int32_t location_id,
    JYPPX_CudaStream* stream)
{
    JYPPX_CUDA_MANAGED_MEMORY_LOCATION_GUARD(
        "cudaMemPrefetchAsync_v2",
        prefetch_managed_memory_location_impl(memory, offset, count, location_type, location_id, stream));
}

JYPPX_StatusCode jyppx_cuda_managed_memory_advise_location_range_safe(
    JYPPX_CudaMemory* memory,
    size_t offset,
    size_t count,
    int32_t advice,
    int32_t location_type,
    int32_t location_id)
{
    JYPPX_CUDA_MANAGED_MEMORY_LOCATION_GUARD(
        "cudaMemAdvise_v2",
        advise_managed_memory_location_impl(memory, offset, count, advice, location_type, location_id));
}

#undef JYPPX_CUDA_MANAGED_MEMORY_LOCATION_GUARD
