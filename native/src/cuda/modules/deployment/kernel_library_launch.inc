#include <memory>

#if defined(_WIN32) && defined(_MSC_VER)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace
{
constexpr size_t kMaximumKernelArgumentCount = 256;
constexpr size_t kMaximumKernelScalarDataSize = 64 * 1024;

JYPPX_StatusCode invalid_kernel_launch_argument(const char* error_name, const char* message) noexcept
{
    try
    {
        jyppx::cuda::set_cuda_error("cudaLaunchKernel", 0, error_name, message);
    }
    catch (...)
    {
    }
    return JYPPX_STATUS_INVALID_ARGUMENT;
}

JYPPX_StatusCode validate_kernel_launch_arguments(
    const JYPPX_CudaKernelArgumentDescriptor* arguments,
    const size_t argument_count,
    const uint8_t* scalar_data,
    const size_t scalar_data_size,
    std::vector<void*>& memory_values,
    std::vector<void*>& argument_pointers) noexcept
{
    if (argument_count > kMaximumKernelArgumentCount)
    {
        return invalid_kernel_launch_argument("too-many-kernel-arguments", "CUDA typed kernel launch accepts at most 256 arguments.");
    }
    if (argument_count != 0 && arguments == nullptr)
    {
        return invalid_kernel_launch_argument("missing-kernel-arguments", "CUDA typed kernel arguments must not be null when argument_count is non-zero.");
    }
    if (scalar_data_size > kMaximumKernelScalarDataSize || (scalar_data_size != 0 && scalar_data == nullptr))
    {
        return invalid_kernel_launch_argument("invalid-scalar-payload", "CUDA typed kernel scalar payload must contain at most 64 KiB and must match its pointer/count.");
    }

    memory_values.assign(argument_count, nullptr);
    argument_pointers.assign(argument_count, nullptr);
    for (size_t index = 0; index < argument_count; ++index)
    {
        const auto& descriptor = arguments[index];
        if (descriptor.reserved != 0)
        {
            return invalid_kernel_launch_argument("invalid-kernel-argument-reserved", "CUDA typed kernel argument reserved fields must be zero.");
        }

        if (descriptor.kind == JYPPX_CUDA_KERNEL_ARGUMENT_SCALAR)
        {
            if (descriptor.memory != nullptr || descriptor.memory_offset != 0)
            {
                return invalid_kernel_launch_argument("invalid-scalar-memory", "CUDA scalar arguments must not carry a device-memory owner or offset.");
            }
            if (descriptor.scalar_size != 1 && descriptor.scalar_size != 2 && descriptor.scalar_size != 4 && descriptor.scalar_size != 8)
            {
                return invalid_kernel_launch_argument("invalid-scalar-size", "CUDA scalar arguments must be 1, 2, 4, or 8 bytes.");
            }
            if (descriptor.scalar_offset > scalar_data_size || descriptor.scalar_size > scalar_data_size - descriptor.scalar_offset)
            {
                return invalid_kernel_launch_argument("scalar-range-out-of-bounds", "CUDA scalar argument offset/size exceeds the copied scalar payload.");
            }
            argument_pointers[index] = const_cast<uint8_t*>(scalar_data + descriptor.scalar_offset);
            continue;
        }

        if (descriptor.kind == JYPPX_CUDA_KERNEL_ARGUMENT_DEVICE_MEMORY)
        {
            if (descriptor.scalar_offset != 0 || descriptor.scalar_size != 0)
            {
                return invalid_kernel_launch_argument("invalid-device-memory-scalar", "CUDA device-memory arguments must not carry scalar payload ranges.");
            }
            auto status = jyppx::cuda::validate_memory(descriptor.memory, "kernel argument memory");
            if (status != JYPPX_STATUS_OK)
            {
                return status;
            }
            auto* memory = reinterpret_cast<jyppx::cuda::MemoryObject*>(descriptor.memory);
            if (descriptor.memory_offset >= memory->size || memory->pointer == nullptr)
            {
                return invalid_kernel_launch_argument("device-memory-offset-out-of-bounds", "CUDA device-memory argument offset must identify a byte inside a valid allocation.");
            }
            memory_values[index] = static_cast<uint8_t*>(memory->pointer) + descriptor.memory_offset;
            argument_pointers[index] = &memory_values[index];
            continue;
        }

        return invalid_kernel_launch_argument("invalid-kernel-argument-kind", "CUDA typed kernel argument kind is not supported.");
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode create_typed_kernel_launch_impl(
    JYPPX_CudaKernelLibrary* library_handle,
    const char* kernel_name,
    const JYPPX_CudaDim3 grid_dim,
    const JYPPX_CudaDim3 block_dim,
    const JYPPX_CudaKernelArgumentDescriptor* arguments,
    const size_t argument_count,
    const uint8_t* scalar_data,
    const size_t scalar_data_size,
    const size_t dynamic_shared_memory_bytes,
    JYPPX_CudaStream* stream_handle,
    JYPPX_CudaKernelLaunch** out_launch) noexcept
{
    return run_kernel_library_noexcept("cudaLaunchKernel", [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_launch, "out_launch");
        if (status != JYPPX_STATUS_OK) return status;
        *out_launch = nullptr;
        status = jyppx::cuda::validate_kernel_library(library_handle, "library");
        if (status != JYPPX_STATUS_OK) return status;
        status = jyppx::cuda::validate_stream(stream_handle, "stream");
        if (status != JYPPX_STATUS_OK) return status;
        if (kernel_name == nullptr || kernel_name[0] == '\0')
        {
            return invalid_kernel_launch_argument("invalid-kernel-name", "CUDA kernel name must not be null or empty.");
        }
        status = validate_cuda_dim3(grid_dim, "cudaLaunchKernel", "Grid");
        if (status != JYPPX_STATUS_OK) return status;
        status = validate_cuda_dim3(block_dim, "cudaLaunchKernel", "Block");
        if (status != JYPPX_STATUS_OK) return status;

        std::vector<void*> memory_values;
        std::vector<void*> argument_pointers;
        status = validate_kernel_launch_arguments(
            arguments,
            argument_count,
            scalar_data,
            scalar_data_size,
            memory_values,
            argument_pointers);
        if (status != JYPPX_STATUS_OK) return status;

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12090
        auto* library = reinterpret_cast<jyppx::cuda::KernelLibraryObject*>(library_handle);
        auto* stream = reinterpret_cast<jyppx::cuda::StreamObject*>(stream_handle);
        cudaKernel_t kernel = nullptr;
        status = jyppx::cuda::map_cuda_status(cudaLibraryGetKernel(&kernel, library->handle, kernel_name), "cudaLibraryGetKernel");
        if (status != JYPPX_STATUS_OK) return status;
        if (kernel == nullptr)
        {
            jyppx::cuda::set_cuda_error("cudaLibraryGetKernel", 0, "invalid-kernel-handle", "CUDA reported success but returned a null borrowed kernel handle.");
            return JYPPX_STATUS_INVALID_STATE;
        }

        auto launch = std::make_unique<jyppx::cuda::KernelLaunchObject>();
        launch->base.magic = jyppx::cuda::kObjectMagic;
        launch->base.kind = jyppx::cuda::ObjectKind::KernelLaunch;
        launch->completed = false;
        launch->completion_event = nullptr;
        status = jyppx::cuda::map_cuda_status(
            cudaEventCreateWithFlags(&launch->completion_event, cudaEventDisableTiming),
            "cudaEventCreateWithFlags");
        if (status != JYPPX_STATUS_OK) return status;

        const cudaError_t launch_error = cudaLaunchKernel(
            reinterpret_cast<const void*>(kernel),
            make_cuda_dim3(grid_dim),
            make_cuda_dim3(block_dim),
            argument_count == 0 ? nullptr : argument_pointers.data(),
            dynamic_shared_memory_bytes,
            stream->handle);
        if (launch_error != cudaSuccess)
        {
            (void)cudaEventDestroy(launch->completion_event);
            launch->completion_event = nullptr;
            return jyppx::cuda::map_cuda_status(launch_error, "cudaLaunchKernel");
        }

        const cudaError_t record_error = cudaEventRecord(launch->completion_event, stream->handle);
        if (record_error != cudaSuccess)
        {
            (void)cudaStreamSynchronize(stream->handle);
            (void)cudaEventDestroy(launch->completion_event);
            launch->completion_event = nullptr;
            return jyppx::cuda::map_cuda_status(record_error, "cudaEventRecord");
        }

        *out_launch = reinterpret_cast<JYPPX_CudaKernelLaunch*>(launch.release());
        return JYPPX_STATUS_OK;
#elif JYPPX_HAS_CUDA_TOOLKIT
        (void)dynamic_shared_memory_bytes;
        return kernel_library_not_supported("cudaLaunchKernel");
#else
        (void)dynamic_shared_memory_bytes;
        return jyppx::cuda::report_cuda_dependency_missing("CUDA owner-bound typed kernel launch");
#endif
    });
}

JYPPX_StatusCode query_kernel_launch_impl(JYPPX_CudaKernelLaunch* launch_handle, JYPPX_Boolean* out_completed) noexcept
{
    return run_kernel_library_noexcept("cudaEventQuery", [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_completed, "out_completed");
        if (status != JYPPX_STATUS_OK) return status;
        *out_completed = JYPPX_FALSE;
        status = jyppx::cuda::validate_kernel_launch(launch_handle, "launch");
        if (status != JYPPX_STATUS_OK) return status;
        auto* launch = reinterpret_cast<jyppx::cuda::KernelLaunchObject*>(launch_handle);
        if (launch->completed)
        {
            *out_completed = JYPPX_TRUE;
            return JYPPX_STATUS_OK;
        }
#if JYPPX_HAS_CUDA_TOOLKIT
        const cudaError_t error = cudaEventQuery(launch->completion_event);
        if (error == cudaErrorNotReady)
        {
            (void)cudaGetLastError();
            return JYPPX_STATUS_OK;
        }
        status = jyppx::cuda::map_cuda_status(error, "cudaEventQuery");
        if (status == JYPPX_STATUS_OK)
        {
            launch->completed = true;
            *out_completed = JYPPX_TRUE;
        }
        return status;
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA typed kernel launch completion query");
#endif
    });
}

JYPPX_StatusCode synchronize_kernel_launch_impl(JYPPX_CudaKernelLaunch* launch_handle) noexcept
{
    return run_kernel_library_noexcept("cudaEventSynchronize", [&]() {
        auto status = jyppx::cuda::validate_kernel_launch(launch_handle, "launch");
        if (status != JYPPX_STATUS_OK) return status;
        auto* launch = reinterpret_cast<jyppx::cuda::KernelLaunchObject*>(launch_handle);
        if (launch->completed) return JYPPX_STATUS_OK;
#if JYPPX_HAS_CUDA_TOOLKIT
        status = jyppx::cuda::map_cuda_status(cudaEventSynchronize(launch->completion_event), "cudaEventSynchronize");
        if (status == JYPPX_STATUS_OK) launch->completed = true;
        return status;
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA typed kernel launch synchronization");
#endif
    });
}

JYPPX_StatusCode destroy_kernel_launch_impl(JYPPX_CudaKernelLaunch* launch_handle) noexcept
{
    return run_kernel_library_noexcept("cudaEventDestroy", [&]() {
        if (launch_handle == nullptr) return JYPPX_STATUS_OK;
        auto status = jyppx::cuda::validate_kernel_launch(launch_handle, "launch");
        if (status != JYPPX_STATUS_OK) return status;
        auto* launch = reinterpret_cast<jyppx::cuda::KernelLaunchObject*>(launch_handle);
        JYPPX_StatusCode synchronize_status = JYPPX_STATUS_OK;
        JYPPX_StatusCode destroy_status = JYPPX_STATUS_OK;
#if JYPPX_HAS_CUDA_TOOLKIT
        if (!launch->completed)
        {
            synchronize_status = jyppx::cuda::map_cuda_status(cudaEventSynchronize(launch->completion_event), "cudaEventSynchronize");
        }
        destroy_status = jyppx::cuda::map_cuda_status(cudaEventDestroy(launch->completion_event), "cudaEventDestroy");
#endif
        launch->base.magic = 0;
        launch->completion_event = nullptr;
        delete launch;
        return synchronize_status != JYPPX_STATUS_OK ? synchronize_status : destroy_status;
    });
}

JYPPX_StatusCode report_kernel_launch_seh(const char* operation) noexcept
{
    try
    {
        jyppx::cuda::set_cuda_error(operation, 0, "windows-seh-contained", "A Windows structured exception was contained by the CUDA typed kernel launch bridge.");
    }
    catch (...)
    {
    }
    return JYPPX_STATUS_RUNTIME_ERROR;
}

#if defined(_WIN32) && defined(_MSC_VER)
#define JYPPX_CUDA_KERNEL_LAUNCH_GUARD(operation, expression) \
    __try { return (expression); } \
    __except (EXCEPTION_EXECUTE_HANDLER) { return report_kernel_launch_seh(operation); }
#else
#define JYPPX_CUDA_KERNEL_LAUNCH_GUARD(operation, expression) return (expression)
#endif
}

JYPPX_StatusCode jyppx_cuda_kernel_library_launch_typed_safe(
    JYPPX_CudaKernelLibrary* library,
    const char* kernel_name,
    JYPPX_CudaDim3 grid_dim,
    JYPPX_CudaDim3 block_dim,
    const JYPPX_CudaKernelArgumentDescriptor* arguments,
    size_t argument_count,
    const uint8_t* scalar_data,
    size_t scalar_data_size,
    size_t dynamic_shared_memory_bytes,
    JYPPX_CudaStream* stream,
    JYPPX_CudaKernelLaunch** out_launch)
{
    JYPPX_CUDA_KERNEL_LAUNCH_GUARD("cudaLaunchKernel", create_typed_kernel_launch_impl(library, kernel_name, grid_dim, block_dim, arguments, argument_count, scalar_data, scalar_data_size, dynamic_shared_memory_bytes, stream, out_launch));
}

JYPPX_StatusCode jyppx_cuda_kernel_launch_query_safe(JYPPX_CudaKernelLaunch* launch, JYPPX_Boolean* out_completed)
{
    JYPPX_CUDA_KERNEL_LAUNCH_GUARD("cudaEventQuery", query_kernel_launch_impl(launch, out_completed));
}

JYPPX_StatusCode jyppx_cuda_kernel_launch_synchronize_safe(JYPPX_CudaKernelLaunch* launch)
{
    JYPPX_CUDA_KERNEL_LAUNCH_GUARD("cudaEventSynchronize", synchronize_kernel_launch_impl(launch));
}

JYPPX_StatusCode jyppx_cuda_kernel_launch_destroy_safe(JYPPX_CudaKernelLaunch* launch)
{
    JYPPX_CUDA_KERNEL_LAUNCH_GUARD("cudaEventDestroy", destroy_kernel_launch_impl(launch));
}
