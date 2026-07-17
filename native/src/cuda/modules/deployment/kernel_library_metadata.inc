#include <algorithm>
#include <cstring>
#include <new>

namespace
{
constexpr size_t kMaximumRetainedKernelLibraryBytes = static_cast<size_t>(1024) * 1024 * 1024;

JYPPX_StatusCode report_kernel_library_exception(const char* operation, const char* error_name, const char* message) noexcept
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

JYPPX_StatusCode kernel_library_not_supported(const char* operation) noexcept
{
    try
    {
        jyppx::cuda::set_cuda_error(operation, 0, "cuda-version-not-supported", "CUDA runtime kernel libraries require CUDA Toolkit 12.9 or later.");
    }
    catch (...)
    {
    }
    return JYPPX_STATUS_NOT_SUPPORTED;
}

template <typename TCallable>
JYPPX_StatusCode run_kernel_library_noexcept(const char* operation, TCallable&& callable) noexcept
{
    try
    {
        return callable();
    }
    catch (const std::bad_alloc&)
    {
        return report_kernel_library_exception(operation, "native-allocation-failed", "The CUDA kernel library bridge could not allocate retained code or inventory storage.");
    }
    catch (...)
    {
        return report_kernel_library_exception(operation, "native-exception-caught", "A native exception was contained by the CUDA kernel library bridge.");
    }
}

void initialize_kernel_library_object(jyppx::cuda::KernelLibraryObject* library)
{
    library->base.magic = jyppx::cuda::kObjectMagic;
    library->base.kind = jyppx::cuda::ObjectKind::KernelLibrary;
    library->handle = nullptr;
}

JYPPX_StatusCode load_kernel_library_file_impl(const char* path, JYPPX_CudaKernelLibrary** out_library) noexcept
{
    return run_kernel_library_noexcept("cudaLibraryLoadFromFile", [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_library, "out_library");
        if (status != JYPPX_STATUS_OK) return status;
        *out_library = nullptr;
        if (path == nullptr || path[0] == '\0')
        {
            jyppx::cuda::set_cuda_error("cudaLibraryLoadFromFile", 0, "invalid-library-path", "The CUDA kernel library path must not be null or empty.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12090
        auto* library = new jyppx::cuda::KernelLibraryObject{};
        initialize_kernel_library_object(library);
        status = jyppx::cuda::map_cuda_status(
            cudaLibraryLoadFromFile(&library->handle, path, nullptr, nullptr, 0U, nullptr, nullptr, 0U),
            "cudaLibraryLoadFromFile");
        if (status != JYPPX_STATUS_OK)
        {
            delete library;
            return status;
        }
        *out_library = reinterpret_cast<JYPPX_CudaKernelLibrary*>(library);
        return JYPPX_STATUS_OK;
#elif JYPPX_HAS_CUDA_TOOLKIT
        return kernel_library_not_supported("cudaLibraryLoadFromFile");
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA kernel library file load");
#endif
    });
}

JYPPX_StatusCode load_kernel_library_data_impl(
    const uint8_t* code,
    const size_t code_size,
    JYPPX_CudaKernelLibrary** out_library) noexcept
{
    return run_kernel_library_noexcept("cudaLibraryLoadData", [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_library, "out_library");
        if (status != JYPPX_STATUS_OK) return status;
        *out_library = nullptr;
        if (code == nullptr || code_size == 0 || code_size > kMaximumRetainedKernelLibraryBytes)
        {
            jyppx::cuda::set_cuda_error("cudaLibraryLoadData", 0, "invalid-library-code", "CUDA kernel library code must contain between 1 byte and 1 GiB.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12090
        auto* library = new jyppx::cuda::KernelLibraryObject{};
        initialize_kernel_library_object(library);
        library->retained_code.assign(code, code + code_size);
        library->retained_code.push_back(0U);
        status = jyppx::cuda::map_cuda_status(
            cudaLibraryLoadData(&library->handle, library->retained_code.data(), nullptr, nullptr, 0U, nullptr, nullptr, 0U),
            "cudaLibraryLoadData");
        if (status != JYPPX_STATUS_OK)
        {
            delete library;
            return status;
        }
        *out_library = reinterpret_cast<JYPPX_CudaKernelLibrary*>(library);
        return JYPPX_STATUS_OK;
#elif JYPPX_HAS_CUDA_TOOLKIT
        return kernel_library_not_supported("cudaLibraryLoadData");
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA kernel library copied-data load");
#endif
    });
}

JYPPX_StatusCode destroy_kernel_library_impl(JYPPX_CudaKernelLibrary* library_handle) noexcept
{
    return run_kernel_library_noexcept("cudaLibraryUnload", [&]() {
        auto status = jyppx::cuda::validate_kernel_library(library_handle, "library");
        if (status != JYPPX_STATUS_OK) return status;
        auto* library = reinterpret_cast<jyppx::cuda::KernelLibraryObject*>(library_handle);
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12090
        status = jyppx::cuda::map_cuda_status(cudaLibraryUnload(library->handle), "cudaLibraryUnload");
#elif JYPPX_HAS_CUDA_TOOLKIT
        status = kernel_library_not_supported("cudaLibraryUnload");
#else
        status = jyppx::cuda::report_cuda_dependency_missing("CUDA kernel library unload");
#endif
        library->base.magic = 0;
        library->handle = nullptr;
        delete library;
        return status;
    });
}

JYPPX_StatusCode get_kernel_library_count_impl(JYPPX_CudaKernelLibrary* library_handle, uint32_t* out_count) noexcept
{
    return run_kernel_library_noexcept("cudaLibraryGetKernelCount", [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_count, "out_count");
        if (status != JYPPX_STATUS_OK) return status;
        *out_count = 0;
        status = jyppx::cuda::validate_kernel_library(library_handle, "library");
        if (status != JYPPX_STATUS_OK) return status;
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12090
        auto* library = reinterpret_cast<jyppx::cuda::KernelLibraryObject*>(library_handle);
        unsigned int count = 0;
        status = jyppx::cuda::map_cuda_status(cudaLibraryGetKernelCount(&count, library->handle), "cudaLibraryGetKernelCount");
        if (status == JYPPX_STATUS_OK) *out_count = count;
        return status;
#elif JYPPX_HAS_CUDA_TOOLKIT
        return kernel_library_not_supported("cudaLibraryGetKernelCount");
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA kernel library count query");
#endif
    });
}

JYPPX_StatusCode get_kernel_library_inventory_impl(
    JYPPX_CudaKernelLibrary* library_handle,
    JYPPX_CudaKernelLibraryInventory* out_inventory) noexcept
{
    return run_kernel_library_noexcept("cudaLibraryEnumerateKernels", [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_inventory, "out_inventory");
        if (status != JYPPX_STATUS_OK) return status;
        std::memset(out_inventory, 0, sizeof(*out_inventory));
        status = jyppx::cuda::validate_kernel_library(library_handle, "library");
        if (status != JYPPX_STATUS_OK) return status;
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12090
        auto* library = reinterpret_cast<jyppx::cuda::KernelLibraryObject*>(library_handle);
        unsigned int count = 0;
        status = jyppx::cuda::map_cuda_status(cudaLibraryGetKernelCount(&count, library->handle), "cudaLibraryGetKernelCount");
        if (status != JYPPX_STATUS_OK) return status;
        out_inventory->reported_kernel_count = count;
        if (count == 0U)
        {
            out_inventory->is_complete = JYPPX_TRUE;
            return JYPPX_STATUS_OK;
        }

        std::vector<cudaKernel_t> kernels(count, nullptr);
        status = jyppx::cuda::map_cuda_status(cudaLibraryEnumerateKernels(kernels.data(), count, library->handle), "cudaLibraryEnumerateKernels");
        if (status != JYPPX_STATUS_OK) return status;
        const auto non_null = static_cast<uint32_t>(std::count_if(kernels.begin(), kernels.end(), [](cudaKernel_t kernel) { return kernel != nullptr; }));
        out_inventory->enumerated_kernel_count = non_null;
        out_inventory->null_kernel_count = count - non_null;
        out_inventory->is_complete = non_null == count ? JYPPX_TRUE : JYPPX_FALSE;
        return JYPPX_STATUS_OK;
#elif JYPPX_HAS_CUDA_TOOLKIT
        return kernel_library_not_supported("cudaLibraryEnumerateKernels");
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA kernel library copied inventory");
#endif
    });
}

JYPPX_StatusCode contains_kernel_library_kernel_impl(
    JYPPX_CudaKernelLibrary* library_handle,
    const char* name,
    JYPPX_Boolean* out_exists) noexcept
{
    return run_kernel_library_noexcept("cudaLibraryGetKernel", [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_exists, "out_exists");
        if (status != JYPPX_STATUS_OK) return status;
        *out_exists = JYPPX_FALSE;
        status = jyppx::cuda::validate_kernel_library(library_handle, "library");
        if (status != JYPPX_STATUS_OK) return status;
        if (name == nullptr || name[0] == '\0')
        {
            jyppx::cuda::set_cuda_error("cudaLibraryGetKernel", 0, "invalid-kernel-name", "The CUDA kernel name must not be null or empty.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12090
        auto* library = reinterpret_cast<jyppx::cuda::KernelLibraryObject*>(library_handle);
        cudaKernel_t kernel = nullptr;
        const cudaError_t error = cudaLibraryGetKernel(&kernel, library->handle, name);
        if (error == cudaErrorSymbolNotFound)
        {
            (void)cudaGetLastError();
            return JYPPX_STATUS_OK;
        }
        status = jyppx::cuda::map_cuda_status(error, "cudaLibraryGetKernel");
        if (status != JYPPX_STATUS_OK) return status;
        if (kernel == nullptr)
        {
            jyppx::cuda::set_cuda_error("cudaLibraryGetKernel", 0, "invalid-kernel-handle", "CUDA reported success but returned a null borrowed kernel handle.");
            return JYPPX_STATUS_INVALID_STATE;
        }
        *out_exists = JYPPX_TRUE;
        return JYPPX_STATUS_OK;
#elif JYPPX_HAS_CUDA_TOOLKIT
        return kernel_library_not_supported("cudaLibraryGetKernel");
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA kernel library named-kernel query");
#endif
    });
}
}

#if defined(_MSC_VER) && defined(_WIN32)
#define JYPPX_CUDA_KERNEL_LIBRARY_GUARD(operation, call) \
    __try { return call; } \
    __except (1) { return report_kernel_library_exception(operation, "windows-seh-caught", "A Windows structured exception was contained by the CUDA kernel library bridge."); }
#else
#define JYPPX_CUDA_KERNEL_LIBRARY_GUARD(operation, call) return call
#endif

JYPPX_StatusCode jyppx_cuda_kernel_library_load_data_copy_safe(const uint8_t* code, size_t code_size, JYPPX_CudaKernelLibrary** out_library)
{
    JYPPX_CUDA_KERNEL_LIBRARY_GUARD("cudaLibraryLoadData", load_kernel_library_data_impl(code, code_size, out_library));
}

JYPPX_StatusCode jyppx_cuda_kernel_library_load_file_safe(const char* path, JYPPX_CudaKernelLibrary** out_library)
{
    JYPPX_CUDA_KERNEL_LIBRARY_GUARD("cudaLibraryLoadFromFile", load_kernel_library_file_impl(path, out_library));
}

JYPPX_StatusCode jyppx_cuda_kernel_library_destroy_safe(JYPPX_CudaKernelLibrary* library)
{
    JYPPX_CUDA_KERNEL_LIBRARY_GUARD("cudaLibraryUnload", destroy_kernel_library_impl(library));
}

JYPPX_StatusCode jyppx_cuda_kernel_library_get_count_safe(JYPPX_CudaKernelLibrary* library, uint32_t* out_count)
{
    JYPPX_CUDA_KERNEL_LIBRARY_GUARD("cudaLibraryGetKernelCount", get_kernel_library_count_impl(library, out_count));
}

JYPPX_StatusCode jyppx_cuda_kernel_library_get_inventory_safe(JYPPX_CudaKernelLibrary* library, JYPPX_CudaKernelLibraryInventory* out_inventory)
{
    JYPPX_CUDA_KERNEL_LIBRARY_GUARD("cudaLibraryEnumerateKernels", get_kernel_library_inventory_impl(library, out_inventory));
}

JYPPX_StatusCode jyppx_cuda_kernel_library_contains_kernel_safe(JYPPX_CudaKernelLibrary* library, const char* name, JYPPX_Boolean* out_exists)
{
    JYPPX_CUDA_KERNEL_LIBRARY_GUARD("cudaLibraryGetKernel", contains_kernel_library_kernel_impl(library, name, out_exists));
}

#undef JYPPX_CUDA_KERNEL_LIBRARY_GUARD
