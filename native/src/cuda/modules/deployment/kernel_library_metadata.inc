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
        return report_kernel_library_exception(operation, "native-allocation-failed", "The CUDA kernel library bridge could not allocate retained code, inventory, or launch argument storage.");
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

enum class KernelLibrarySymbolKind
{
    Global,
    Managed
};

JYPPX_StatusCode try_get_kernel_library_symbol_size_impl(
    JYPPX_CudaKernelLibrary* library_handle,
    const char* name,
    const KernelLibrarySymbolKind kind,
    JYPPX_Boolean* out_exists,
    uint64_t* out_size) noexcept
{
    const char* operation = kind == KernelLibrarySymbolKind::Global
        ? "cudaLibraryGetGlobal"
        : "cudaLibraryGetManaged";
    return run_kernel_library_noexcept(operation, [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_exists, "out_exists");
        if (status != JYPPX_STATUS_OK) return status;
        status = jyppx::cuda::validate_output_pointer(out_size, "out_size");
        if (status != JYPPX_STATUS_OK) return status;
        *out_exists = JYPPX_FALSE;
        *out_size = 0;
        status = jyppx::cuda::validate_kernel_library(library_handle, "library");
        if (status != JYPPX_STATUS_OK) return status;
        if (name == nullptr || name[0] == '\0')
        {
            jyppx::cuda::set_cuda_error(operation, 0, "invalid-symbol-name", "The CUDA kernel-library symbol name must not be null or empty.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12090
        auto* library = reinterpret_cast<jyppx::cuda::KernelLibraryObject*>(library_handle);
        size_t symbol_size = 0;
        const cudaError_t error = kind == KernelLibrarySymbolKind::Global
            ? cudaLibraryGetGlobal(nullptr, &symbol_size, library->handle, name)
            : cudaLibraryGetManaged(nullptr, &symbol_size, library->handle, name);
        if (error == cudaErrorSymbolNotFound)
        {
            (void)cudaGetLastError();
            return JYPPX_STATUS_OK;
        }
        status = jyppx::cuda::map_cuda_status(error, operation);
        if (status != JYPPX_STATUS_OK) return status;
        *out_exists = JYPPX_TRUE;
        *out_size = static_cast<uint64_t>(symbol_size);
        return JYPPX_STATUS_OK;
#elif JYPPX_HAS_CUDA_TOOLKIT
        return kernel_library_not_supported(operation);
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA kernel library copied symbol-size query");
#endif
    });
}

JYPPX_StatusCode contains_kernel_library_unified_function_impl(
    JYPPX_CudaKernelLibrary* library_handle,
    const char* name,
    JYPPX_Boolean* out_exists) noexcept
{
    return run_kernel_library_noexcept("cudaLibraryGetUnifiedFunction", [&]() {
        auto status = jyppx::cuda::validate_output_pointer(out_exists, "out_exists");
        if (status != JYPPX_STATUS_OK) return status;
        *out_exists = JYPPX_FALSE;
        status = jyppx::cuda::validate_kernel_library(library_handle, "library");
        if (status != JYPPX_STATUS_OK) return status;
        if (name == nullptr || name[0] == '\0')
        {
            jyppx::cuda::set_cuda_error("cudaLibraryGetUnifiedFunction", 0, "invalid-unified-function-name", "The CUDA unified-function name must not be null or empty.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12090
        auto* library = reinterpret_cast<jyppx::cuda::KernelLibraryObject*>(library_handle);
        void* function_pointer = nullptr;
        const cudaError_t error = cudaLibraryGetUnifiedFunction(&function_pointer, library->handle, name);
        if (error == cudaErrorSymbolNotFound)
        {
            (void)cudaGetLastError();
            return JYPPX_STATUS_OK;
        }
        status = jyppx::cuda::map_cuda_status(error, "cudaLibraryGetUnifiedFunction");
        if (status != JYPPX_STATUS_OK) return status;
        if (function_pointer == nullptr)
        {
            jyppx::cuda::set_cuda_error("cudaLibraryGetUnifiedFunction", 0, "invalid-unified-function-pointer", "CUDA reported success but returned a null unified-function pointer.");
            return JYPPX_STATUS_INVALID_STATE;
        }
        *out_exists = JYPPX_TRUE;
        return JYPPX_STATUS_OK;
#elif JYPPX_HAS_CUDA_TOOLKIT
        return kernel_library_not_supported("cudaLibraryGetUnifiedFunction");
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA kernel library unified-function existence query");
#endif
    });
}

bool is_mutable_kernel_library_attribute(const int32_t attribute) noexcept
{
    switch (attribute)
    {
    case 8:  // cudaFuncAttributeMaxDynamicSharedMemorySize
    case 9:  // cudaFuncAttributePreferredSharedMemoryCarveout
    case 11: // cudaFuncAttributeRequiredClusterWidth
    case 12: // cudaFuncAttributeRequiredClusterHeight
    case 13: // cudaFuncAttributeRequiredClusterDepth
    case 14: // cudaFuncAttributeNonPortableClusterSizeAllowed
    case 15: // cudaFuncAttributeClusterSchedulingPolicyPreference
        return true;
    default:
        return false;
    }
}

JYPPX_StatusCode set_kernel_library_kernel_attribute_impl(
    JYPPX_CudaKernelLibrary* library_handle,
    const char* kernel_name,
    const int32_t attribute,
    const int32_t value,
    const int32_t device) noexcept
{
    return run_kernel_library_noexcept("cudaKernelSetAttributeForDevice", [&]() {
        auto status = jyppx::cuda::validate_kernel_library(library_handle, "library");
        if (status != JYPPX_STATUS_OK) return status;
        if (kernel_name == nullptr || kernel_name[0] == '\0')
        {
            jyppx::cuda::set_cuda_error("cudaKernelSetAttributeForDevice", 0, "invalid-kernel-name", "The CUDA kernel name must not be null or empty.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }
        if (!is_mutable_kernel_library_attribute(attribute))
        {
            jyppx::cuda::set_cuda_error("cudaKernelSetAttributeForDevice", 0, "invalid-kernel-attribute", "Only CUDA attributes documented as mutable by cudaKernelSetAttributeForDevice are accepted.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }
        if (device < 0)
        {
            jyppx::cuda::set_cuda_error("cudaKernelSetAttributeForDevice", 0, "invalid-device", "The CUDA device ordinal must be greater than or equal to zero.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12090
        auto* library = reinterpret_cast<jyppx::cuda::KernelLibraryObject*>(library_handle);
        cudaKernel_t kernel = nullptr;
        status = jyppx::cuda::map_cuda_status(
            cudaLibraryGetKernel(&kernel, library->handle, kernel_name),
            "cudaLibraryGetKernel");
        if (status != JYPPX_STATUS_OK) return status;
        if (kernel == nullptr)
        {
            jyppx::cuda::set_cuda_error("cudaKernelSetAttributeForDevice", 0, "invalid-kernel-handle", "CUDA reported success but returned a null borrowed kernel handle.");
            return JYPPX_STATUS_INVALID_STATE;
        }
        return jyppx::cuda::map_cuda_status(
            cudaKernelSetAttributeForDevice(kernel, static_cast<cudaFuncAttribute>(attribute), value, device),
            "cudaKernelSetAttributeForDevice");
#elif JYPPX_HAS_CUDA_TOOLKIT
        return kernel_library_not_supported("cudaKernelSetAttributeForDevice");
#else
        return jyppx::cuda::report_cuda_dependency_missing("CUDA owner-bound kernel attribute set");
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

JYPPX_StatusCode jyppx_cuda_kernel_library_try_get_global_size_safe(JYPPX_CudaKernelLibrary* library, const char* name, JYPPX_Boolean* out_exists, uint64_t* out_size)
{
    JYPPX_CUDA_KERNEL_LIBRARY_GUARD("cudaLibraryGetGlobal", try_get_kernel_library_symbol_size_impl(library, name, KernelLibrarySymbolKind::Global, out_exists, out_size));
}

JYPPX_StatusCode jyppx_cuda_kernel_library_try_get_managed_size_safe(JYPPX_CudaKernelLibrary* library, const char* name, JYPPX_Boolean* out_exists, uint64_t* out_size)
{
    JYPPX_CUDA_KERNEL_LIBRARY_GUARD("cudaLibraryGetManaged", try_get_kernel_library_symbol_size_impl(library, name, KernelLibrarySymbolKind::Managed, out_exists, out_size));
}

JYPPX_StatusCode jyppx_cuda_kernel_library_contains_unified_function_safe(JYPPX_CudaKernelLibrary* library, const char* name, JYPPX_Boolean* out_exists)
{
    JYPPX_CUDA_KERNEL_LIBRARY_GUARD("cudaLibraryGetUnifiedFunction", contains_kernel_library_unified_function_impl(library, name, out_exists));
}

JYPPX_StatusCode jyppx_cuda_kernel_library_set_kernel_attribute_for_device_safe(JYPPX_CudaKernelLibrary* library, const char* kernel_name, int32_t attribute, int32_t value, int32_t device)
{
    JYPPX_CUDA_KERNEL_LIBRARY_GUARD("cudaKernelSetAttributeForDevice", set_kernel_library_kernel_attribute_impl(library, kernel_name, attribute, value, device));
}

#undef JYPPX_CUDA_KERNEL_LIBRARY_GUARD
