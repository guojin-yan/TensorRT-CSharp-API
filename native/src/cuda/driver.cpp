#include "jyppx/cuda/runtime.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "object.hpp"
#include "../common/error_state.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if JYPPX_HAS_CUDA_TOOLKIT
#include <cuda.h>
#endif

namespace
{
constexpr size_t kMaximumDriverModuleCodeSize = 512U * 1024U * 1024U;
constexpr size_t kMaximumKernelArgumentCount = 256U;
constexpr size_t kMaximumKernelScalarDataSize = 64U * 1024U;

template <typename TCallable>
JYPPX_StatusCode run_driver_noexcept(const char* operation, TCallable&& callable) noexcept
{
    try
    {
        return callable();
    }
    catch (const std::bad_alloc&)
    {
        try
        {
            jyppx::cuda::set_cuda_error(operation, 0, "native-allocation-failed", "The CUDA Driver owner bridge could not allocate retained module or launch state.");
        }
        catch (...)
        {
        }
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }
    catch (...)
    {
        try
        {
            jyppx::cuda::set_cuda_error(operation, 0, "native-exception-caught", "A native exception was contained by the CUDA Driver owner bridge.");
        }
        catch (...)
        {
        }
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
}

JYPPX_StatusCode report_driver_seh(const char* operation) noexcept
{
    try
    {
        jyppx::cuda::set_cuda_error(operation, 0, "windows-seh-contained", "A Windows structured exception was contained by the CUDA Driver owner bridge.");
    }
    catch (...)
    {
    }
    return JYPPX_STATUS_RUNTIME_ERROR;
}

#if defined(_WIN32) && defined(_MSC_VER)
#define JYPPX_CUDA_DRIVER_GUARD(operation, expression) \
    __try { return (expression); } \
    __except (EXCEPTION_EXECUTE_HANDLER) { return report_driver_seh(operation); }
#else
#define JYPPX_CUDA_DRIVER_GUARD(operation, expression) return (expression)
#endif

#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(_WIN32)
#define JYPPX_DRIVER_API_CALL CUDAAPI
#else
#define JYPPX_DRIVER_API_CALL
#endif

using DynamicLibrary =
#if defined(_WIN32)
    HMODULE;
#else
    void*;
#endif

template <typename TFunction>
TFunction function_pointer_from_address(void* address) noexcept
{
    static_assert(sizeof(TFunction) == sizeof(address), "Dynamic function pointer size mismatch.");
    TFunction function{};
    std::memcpy(&function, &address, sizeof(function));
    return function;
}

void* load_symbol(const DynamicLibrary library, const char* name) noexcept
{
#if defined(_WIN32)
    FARPROC address = GetProcAddress(library, name);
    void* result = nullptr;
    static_assert(sizeof(address) == sizeof(result), "Windows function pointer size mismatch.");
    std::memcpy(&result, &address, sizeof(result));
    return result;
#else
    return dlsym(library, name);
#endif
}

DynamicLibrary load_library(const char* name) noexcept
{
#if defined(_WIN32)
    return LoadLibraryA(name);
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

template <typename TFunction>
TFunction resolve(const DynamicLibrary library, const char* name) noexcept
{
    return function_pointer_from_address<TFunction>(load_symbol(library, name));
}

using CuInit = CUresult (JYPPX_DRIVER_API_CALL *)(unsigned int);
using CuDriverGetVersion = CUresult (JYPPX_DRIVER_API_CALL *)(int*);
using CuGetErrorName = CUresult (JYPPX_DRIVER_API_CALL *)(CUresult, const char**);
using CuGetErrorString = CUresult (JYPPX_DRIVER_API_CALL *)(CUresult, const char**);
using CuDeviceGet = CUresult (JYPPX_DRIVER_API_CALL *)(CUdevice*, int);
using CuDevicePrimaryCtxRetain = CUresult (JYPPX_DRIVER_API_CALL *)(CUcontext*, CUdevice);
using CuDevicePrimaryCtxRelease = CUresult (JYPPX_DRIVER_API_CALL *)(CUdevice);
using CuCtxPushCurrent = CUresult (JYPPX_DRIVER_API_CALL *)(CUcontext);
using CuCtxPopCurrent = CUresult (JYPPX_DRIVER_API_CALL *)(CUcontext*);
using CuModuleLoadDataEx = CUresult (JYPPX_DRIVER_API_CALL *)(CUmodule*, const void*, unsigned int, CUjit_option*, void**);
using CuModuleUnload = CUresult (JYPPX_DRIVER_API_CALL *)(CUmodule);
using CuModuleGetFunction = CUresult (JYPPX_DRIVER_API_CALL *)(CUfunction*, CUmodule, const char*);
using CuLaunchKernel = CUresult (JYPPX_DRIVER_API_CALL *)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);
using CuEventCreate = CUresult (JYPPX_DRIVER_API_CALL *)(CUevent*, unsigned int);
using CuEventRecord = CUresult (JYPPX_DRIVER_API_CALL *)(CUevent, CUstream);
using CuEventQuery = CUresult (JYPPX_DRIVER_API_CALL *)(CUevent);
using CuEventSynchronize = CUresult (JYPPX_DRIVER_API_CALL *)(CUevent);
using CuEventDestroy = CUresult (JYPPX_DRIVER_API_CALL *)(CUevent);
using CuStreamSynchronize = CUresult (JYPPX_DRIVER_API_CALL *)(CUstream);

struct DriverApi final
{
    DynamicLibrary library{};
    std::string loaded_library_name;
    std::string dependency_diagnostic;
    CuInit init{};
    CuDriverGetVersion driver_get_version{};
    CuGetErrorName get_error_name{};
    CuGetErrorString get_error_string{};
    CuDeviceGet device_get{};
    CuDevicePrimaryCtxRetain primary_ctx_retain{};
    CuDevicePrimaryCtxRelease primary_ctx_release{};
    CuCtxPushCurrent ctx_push_current{};
    CuCtxPopCurrent ctx_pop_current{};
    CuModuleLoadDataEx module_load_data_ex{};
    CuModuleUnload module_unload{};
    CuModuleGetFunction module_get_function{};
    CuLaunchKernel launch_kernel{};
    CuEventCreate event_create{};
    CuEventRecord event_record{};
    CuEventQuery event_query{};
    CuEventSynchronize event_synchronize{};
    CuEventDestroy event_destroy{};
    CuStreamSynchronize stream_synchronize{};

    DriverApi()
    {
        std::vector<std::string> candidates;
#if defined(_WIN32)
        char* explicit_library = nullptr;
        size_t explicit_library_size = 0;
        if (_dupenv_s(&explicit_library, &explicit_library_size, "JYPPX_CUDA_DRIVER_LIBRARY") == 0 &&
            explicit_library != nullptr && explicit_library[0] != '\0')
        {
            candidates.emplace_back(explicit_library);
        }
        std::free(explicit_library);
        candidates.emplace_back("nvcuda.dll");
#else
        const char* explicit_library = std::getenv("JYPPX_CUDA_DRIVER_LIBRARY");
        if (explicit_library != nullptr && explicit_library[0] != '\0')
        {
            candidates.emplace_back(explicit_library);
        }
        candidates.emplace_back("libcuda.so.1");
        candidates.emplace_back("libcuda.so");
#endif

        std::vector<std::string> unique_candidates;
        for (const std::string& candidate : candidates)
        {
            if (std::find(unique_candidates.begin(), unique_candidates.end(), candidate) == unique_candidates.end())
            {
                unique_candidates.push_back(candidate);
            }
        }

        for (const std::string& candidate : unique_candidates)
        {
            DynamicLibrary candidate_library = load_library(candidate.c_str());
            if (candidate_library == nullptr)
            {
                continue;
            }

            init = resolve<CuInit>(candidate_library, "cuInit");
            driver_get_version = resolve<CuDriverGetVersion>(candidate_library, "cuDriverGetVersion");
            get_error_name = resolve<CuGetErrorName>(candidate_library, "cuGetErrorName");
            get_error_string = resolve<CuGetErrorString>(candidate_library, "cuGetErrorString");
            device_get = resolve<CuDeviceGet>(candidate_library, "cuDeviceGet");
            primary_ctx_retain = resolve<CuDevicePrimaryCtxRetain>(candidate_library, "cuDevicePrimaryCtxRetain");
            primary_ctx_release = resolve<CuDevicePrimaryCtxRelease>(candidate_library, "cuDevicePrimaryCtxRelease_v2");
            ctx_push_current = resolve<CuCtxPushCurrent>(candidate_library, "cuCtxPushCurrent_v2");
            ctx_pop_current = resolve<CuCtxPopCurrent>(candidate_library, "cuCtxPopCurrent_v2");
            module_load_data_ex = resolve<CuModuleLoadDataEx>(candidate_library, "cuModuleLoadDataEx");
            module_unload = resolve<CuModuleUnload>(candidate_library, "cuModuleUnload");
            module_get_function = resolve<CuModuleGetFunction>(candidate_library, "cuModuleGetFunction");
            launch_kernel = resolve<CuLaunchKernel>(candidate_library, "cuLaunchKernel");
            event_create = resolve<CuEventCreate>(candidate_library, "cuEventCreate");
            event_record = resolve<CuEventRecord>(candidate_library, "cuEventRecord");
            event_query = resolve<CuEventQuery>(candidate_library, "cuEventQuery");
            event_synchronize = resolve<CuEventSynchronize>(candidate_library, "cuEventSynchronize");
            event_destroy = resolve<CuEventDestroy>(candidate_library, "cuEventDestroy_v2");
            stream_synchronize = resolve<CuStreamSynchronize>(candidate_library, "cuStreamSynchronize");

            if (init != nullptr && driver_get_version != nullptr && get_error_name != nullptr &&
                get_error_string != nullptr && device_get != nullptr && primary_ctx_retain != nullptr &&
                primary_ctx_release != nullptr && ctx_push_current != nullptr && ctx_pop_current != nullptr &&
                module_load_data_ex != nullptr && module_unload != nullptr && module_get_function != nullptr &&
                launch_kernel != nullptr && event_create != nullptr && event_record != nullptr &&
                event_query != nullptr && event_synchronize != nullptr && event_destroy != nullptr &&
                stream_synchronize != nullptr)
            {
                library = candidate_library;
                loaded_library_name = candidate;
                dependency_diagnostic.clear();
                return;
            }

#if defined(_WIN32)
            FreeLibrary(candidate_library);
#else
            dlclose(candidate_library);
#endif
        }

        dependency_diagnostic = "CUDA Driver API library nvcuda.dll/libcuda.so.1 was not found or did not expose the required module, launch, context, and event symbols.";
    }

    bool IsAvailable() const noexcept
    {
        return library != nullptr && init != nullptr;
    }
};

DriverApi& driver_api()
{
    static DriverApi api;
    return api;
}

void set_driver_error(DriverApi& api, const char* operation, const CUresult result, const char* fallback_name, const char* fallback_message)
{
    const char* error_name = fallback_name;
    const char* error_message = fallback_message;
    if (api.get_error_name != nullptr)
    {
        const char* candidate = nullptr;
        if (api.get_error_name(result, &candidate) == CUDA_SUCCESS && candidate != nullptr)
        {
            error_name = candidate;
        }
    }
    if (api.get_error_string != nullptr)
    {
        const char* candidate = nullptr;
        if (api.get_error_string(result, &candidate) == CUDA_SUCCESS && candidate != nullptr)
        {
            error_message = candidate;
        }
    }
    jyppx::cuda::set_cuda_error(operation, static_cast<int32_t>(result), error_name, error_message);
}

JYPPX_StatusCode map_driver_status(DriverApi& api, const CUresult result, const char* operation)
{
    if (result == CUDA_SUCCESS)
    {
        return JYPPX_STATUS_OK;
    }

    set_driver_error(api, operation, result, "cuda-driver-error", "CUDA Driver API returned an error.");
    switch (result)
    {
    case CUDA_ERROR_INVALID_VALUE:
    case CUDA_ERROR_INVALID_DEVICE:
    case CUDA_ERROR_NOT_FOUND:
        return JYPPX_STATUS_INVALID_ARGUMENT;
    case CUDA_ERROR_OUT_OF_MEMORY:
        return JYPPX_STATUS_OUT_OF_MEMORY;
    case CUDA_ERROR_NOT_INITIALIZED:
    case CUDA_ERROR_NO_DEVICE:
    case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
        return JYPPX_STATUS_DEPENDENCY_MISSING;
    case CUDA_ERROR_NOT_READY:
        return JYPPX_STATUS_NOT_READY;
    case CUDA_ERROR_NOT_SUPPORTED:
        return JYPPX_STATUS_NOT_SUPPORTED;
    default:
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
}

JYPPX_StatusCode driver_dependency_missing(DriverApi& api, const char* operation)
{
    jyppx::cuda::set_cuda_error(operation, 0, "cuda-driver-dependency-missing", api.dependency_diagnostic.c_str());
    return JYPPX_STATUS_DEPENDENCY_MISSING;
}

JYPPX_StatusCode copy_driver_string(const std::string& value, char* output_buffer, const size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::cuda::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const size_t required_size = value.size() + 1U;
    *out_required_size = required_size;
    if (output_buffer == nullptr || output_buffer_size == 0U)
    {
        return JYPPX_STATUS_OK;
    }
    if (output_buffer_size < required_size)
    {
        output_buffer[0] = '\0';
        jyppx::cuda::set_cuda_error("cudaDriverStringCopy", 0, "buffer-too-small", "The CUDA Driver diagnostic buffer is too small.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }
    std::memcpy(output_buffer, value.c_str(), required_size);
    return JYPPX_STATUS_OK;
}

struct DriverContextScope final
{
    DriverApi& api;
    bool active{false};
    CUresult status{CUDA_SUCCESS};

    DriverContextScope(DriverApi& driver, const CUcontext context)
        : api(driver)
    {
        status = api.ctx_push_current(context);
        active = status == CUDA_SUCCESS;
    }

    ~DriverContextScope()
    {
        if (active)
        {
            CUcontext ignored = nullptr;
            (void)api.ctx_pop_current(&ignored);
        }
    }

    CUresult Close() noexcept
    {
        if (!active)
        {
            return status;
        }
        CUcontext ignored = nullptr;
        active = false;
        const CUresult pop_status = api.ctx_pop_current(&ignored);
        return status == CUDA_SUCCESS ? pop_status : status;
    }
};

bool valid_driver_dim3(const JYPPX_CudaDim3 value) noexcept
{
    return value.x != 0U && value.y != 0U && value.z != 0U;
}

JYPPX_StatusCode validate_driver_arguments(
    const JYPPX_CudaKernelArgumentDescriptor* arguments,
    const size_t argument_count,
    const uint8_t* scalar_data,
    const size_t scalar_data_size,
    std::vector<void*>& memory_values,
    std::vector<void*>& argument_pointers)
{
    if (argument_count > kMaximumKernelArgumentCount)
    {
        jyppx::cuda::set_cuda_error("cuLaunchKernel", 0, "too-many-kernel-arguments", "CUDA Driver typed launch accepts at most 256 arguments.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (argument_count != 0U && arguments == nullptr)
    {
        jyppx::cuda::set_cuda_error("cuLaunchKernel", 0, "missing-kernel-arguments", "CUDA Driver typed arguments must not be null when argument_count is non-zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (scalar_data_size > kMaximumKernelScalarDataSize || (scalar_data_size != 0U && scalar_data == nullptr))
    {
        jyppx::cuda::set_cuda_error("cuLaunchKernel", 0, "invalid-scalar-payload", "CUDA Driver scalar payload must contain at most 64 KiB and match its pointer/count.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    memory_values.assign(argument_count, nullptr);
    argument_pointers.assign(argument_count, nullptr);
    for (size_t index = 0; index < argument_count; ++index)
    {
        const auto& descriptor = arguments[index];
        if (descriptor.reserved != 0U)
        {
            jyppx::cuda::set_cuda_error("cuLaunchKernel", 0, "invalid-kernel-argument-reserved", "CUDA Driver typed argument reserved fields must be zero.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }
        if (descriptor.kind == JYPPX_CUDA_KERNEL_ARGUMENT_SCALAR)
        {
            if (descriptor.memory != nullptr || descriptor.memory_offset != 0U ||
                (descriptor.scalar_size != 1U && descriptor.scalar_size != 2U && descriptor.scalar_size != 4U && descriptor.scalar_size != 8U) ||
                descriptor.scalar_offset > scalar_data_size || descriptor.scalar_size > scalar_data_size - descriptor.scalar_offset)
            {
                jyppx::cuda::set_cuda_error("cuLaunchKernel", 0, "invalid-scalar-argument", "CUDA Driver scalar argument metadata is invalid.");
                return JYPPX_STATUS_INVALID_ARGUMENT;
            }
            argument_pointers[index] = const_cast<uint8_t*>(scalar_data + descriptor.scalar_offset);
            continue;
        }
        if (descriptor.kind == JYPPX_CUDA_KERNEL_ARGUMENT_DEVICE_MEMORY)
        {
            if (descriptor.scalar_offset != 0U || descriptor.scalar_size != 0U)
            {
                jyppx::cuda::set_cuda_error("cuLaunchKernel", 0, "invalid-device-memory-scalar", "CUDA Driver device-memory arguments must not carry scalar payload ranges.");
                return JYPPX_STATUS_INVALID_ARGUMENT;
            }
            auto status = jyppx::cuda::validate_memory(descriptor.memory, "kernel argument memory");
            if (status != JYPPX_STATUS_OK)
            {
                return status;
            }
            auto* memory = reinterpret_cast<jyppx::cuda::MemoryObject*>(descriptor.memory);
            if (descriptor.memory_offset >= memory->size || memory->pointer == nullptr)
            {
                jyppx::cuda::set_cuda_error("cuLaunchKernel", 0, "device-memory-offset-out-of-bounds", "CUDA Driver device-memory argument offset is outside the allocation.");
                return JYPPX_STATUS_INVALID_ARGUMENT;
            }
            memory_values[index] = static_cast<uint8_t*>(memory->pointer) + descriptor.memory_offset;
            argument_pointers[index] = &memory_values[index];
            continue;
        }
        jyppx::cuda::set_cuda_error("cuLaunchKernel", 0, "invalid-kernel-argument-kind", "CUDA Driver typed argument kind is not supported.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode query_driver_capability_impl(JYPPX_CudaDriverCapabilityInfo* out_info)
{
    auto status = jyppx::cuda::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
    std::memset(out_info, 0, sizeof(*out_info));
    DriverApi& api = driver_api();
    if (!api.IsAvailable())
    {
        return JYPPX_STATUS_OK;
    }

    const CUresult init_status = api.init(0U);
    if (init_status != CUDA_SUCCESS)
    {
        set_driver_error(api, "cuInit", init_status, "cuda-driver-init-failed", "CUDA Driver initialization failed.");
        return JYPPX_STATUS_OK;
    }

    int driver_version = 0;
    const CUresult version_status = api.driver_get_version(&driver_version);
    if (version_status != CUDA_SUCCESS)
    {
        set_driver_error(api, "cuDriverGetVersion", version_status, "cuda-driver-version-failed", "CUDA Driver version query failed.");
        return JYPPX_STATUS_OK;
    }
    out_info->dependency_available = JYPPX_TRUE;
    out_info->driver_version = driver_version;
    out_info->supports_module_load = JYPPX_TRUE;
    out_info->supports_function_lookup = JYPPX_TRUE;
    out_info->supports_typed_launch = JYPPX_TRUE;
    out_info->supports_context_interop = JYPPX_TRUE;
    out_info->supports_completion_events = JYPPX_TRUE;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode load_driver_module_impl(const uint8_t* code, const size_t code_size, const int32_t device_ordinal, JYPPX_CudaDriverModule** out_module)
{
    auto status = jyppx::cuda::validate_output_pointer(out_module, "out_module");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
    *out_module = nullptr;
    if (code == nullptr || code_size == 0U || code_size > kMaximumDriverModuleCodeSize || device_ordinal < 0)
    {
        jyppx::cuda::set_cuda_error("cuModuleLoadDataEx", 0, "invalid-driver-module-input", "CUDA Driver module code must be non-empty, bounded, and use a non-negative device ordinal.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    DriverApi& api = driver_api();
    if (!api.IsAvailable())
    {
        return driver_dependency_missing(api, "cuModuleLoadDataEx");
    }
    status = map_driver_status(api, api.init(0U), "cuInit");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    CUdevice device{};
    status = map_driver_status(api, api.device_get(&device, device_ordinal), "cuDeviceGet");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto owner = std::make_unique<jyppx::cuda::DriverModuleObject>();
    owner->base.magic = jyppx::cuda::kObjectMagic;
    owner->base.kind = jyppx::cuda::ObjectKind::DriverModule;
    owner->context = nullptr;
    owner->module = nullptr;
    owner->device = device_ordinal;
    owner->retained_code.assign(code, code + code_size);

    CUcontext context{};
    status = map_driver_status(api, api.primary_ctx_retain(&context, device), "cuDevicePrimaryCtxRetain");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
    owner->context = reinterpret_cast<void*>(context);

    CUmodule module{};
    CUresult operation_status = CUDA_SUCCESS;
    {
        DriverContextScope scope(api, context);
        operation_status = scope.status;
        if (operation_status == CUDA_SUCCESS)
        {
            operation_status = api.module_load_data_ex(&module, owner->retained_code.data(), 0U, nullptr, nullptr);
        }
        const CUresult pop_status = scope.Close();
        if (operation_status == CUDA_SUCCESS)
        {
            operation_status = pop_status;
        }
    }

    if (operation_status != CUDA_SUCCESS)
    {
        if (module != nullptr)
        {
            (void)api.module_unload(module);
        }
        (void)api.primary_ctx_release(device);
        return map_driver_status(api, operation_status, "cuModuleLoadDataEx");
    }

    owner->module = reinterpret_cast<void*>(module);
    *out_module = reinterpret_cast<JYPPX_CudaDriverModule*>(owner.release());
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode launch_driver_kernel_impl(
    jyppx::cuda::DriverModuleObject* module,
    const char* kernel_name,
    const JYPPX_CudaDim3 grid_dim,
    const JYPPX_CudaDim3 block_dim,
    const JYPPX_CudaKernelArgumentDescriptor* arguments,
    const size_t argument_count,
    const uint8_t* scalar_data,
    const size_t scalar_data_size,
    const size_t dynamic_shared_memory_bytes,
    jyppx::cuda::StreamObject* stream,
    JYPPX_CudaDriverKernelLaunch** out_launch)
{
    auto status = jyppx::cuda::validate_output_pointer(out_launch, "out_launch");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
    *out_launch = nullptr;
    status = jyppx::cuda::validate_driver_module(reinterpret_cast<JYPPX_CudaDriverModule*>(module), "module");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
    status = jyppx::cuda::validate_stream(reinterpret_cast<JYPPX_CudaStream*>(stream), "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
    if (kernel_name == nullptr || kernel_name[0] == '\0' || !valid_driver_dim3(grid_dim) || !valid_driver_dim3(block_dim) ||
        dynamic_shared_memory_bytes > static_cast<size_t>((std::numeric_limits<unsigned int>::max)()))
    {
        jyppx::cuda::set_cuda_error("cuLaunchKernel", 0, "invalid-driver-launch-input", "CUDA Driver kernel name and dimensions must be non-empty and non-zero, and dynamic shared memory must fit the CUDA Driver unsigned-int ABI.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    DriverApi& api = driver_api();
    if (!api.IsAvailable())
    {
        return driver_dependency_missing(api, "cuLaunchKernel");
    }
    std::vector<void*> memory_values;
    std::vector<void*> argument_pointers;
    status = validate_driver_arguments(arguments, argument_count, scalar_data, scalar_data_size, memory_values, argument_pointers);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    CUcontext context = reinterpret_cast<CUcontext>(module->context);
    DriverContextScope scope(api, context);
    if (scope.status != CUDA_SUCCESS)
    {
        return map_driver_status(api, scope.status, "cuCtxPushCurrent");
    }

    CUfunction function{};
    CUresult operation_status = api.module_get_function(&function, reinterpret_cast<CUmodule>(module->module), kernel_name);
    if (operation_status == CUDA_SUCCESS)
    {
        auto launch = std::make_unique<jyppx::cuda::DriverKernelLaunchObject>();
        launch->base.magic = jyppx::cuda::kObjectMagic;
        launch->base.kind = jyppx::cuda::ObjectKind::DriverKernelLaunch;
        launch->context = module->context;
        launch->event = nullptr;
        launch->completed = false;
        CUevent event{};
        operation_status = api.event_create(&event, CU_EVENT_DISABLE_TIMING);
        if (operation_status == CUDA_SUCCESS)
        {
            launch->event = reinterpret_cast<void*>(event);
            operation_status = api.launch_kernel(
                function,
                grid_dim.x,
                grid_dim.y,
                grid_dim.z,
                block_dim.x,
                block_dim.y,
                block_dim.z,
                static_cast<unsigned int>(dynamic_shared_memory_bytes),
                reinterpret_cast<CUstream>(stream->handle),
                argument_count == 0U ? nullptr : argument_pointers.data(),
                nullptr);
        }
        if (operation_status == CUDA_SUCCESS)
        {
            operation_status = api.event_record(event, reinterpret_cast<CUstream>(stream->handle));
        }
        if (operation_status != CUDA_SUCCESS)
        {
            (void)api.stream_synchronize(reinterpret_cast<CUstream>(stream->handle));
            if (event != nullptr)
            {
                (void)api.event_destroy(event);
            }
            launch->event = nullptr;
        }
        else
        {
            const CUresult pop_status = scope.Close();
            if (pop_status != CUDA_SUCCESS)
            {
                (void)api.event_synchronize(event);
                (void)api.event_destroy(event);
                launch->event = nullptr;
                operation_status = pop_status;
            }
            else
            {
                *out_launch = reinterpret_cast<JYPPX_CudaDriverKernelLaunch*>(launch.release());
                return JYPPX_STATUS_OK;
            }
        }
    }

    (void)scope.Close();
    return map_driver_status(api, operation_status, "cuLaunchKernel");
}

JYPPX_StatusCode query_driver_kernel_impl(jyppx::cuda::DriverKernelLaunchObject* launch, JYPPX_Boolean* out_completed)
{
    auto status = jyppx::cuda::validate_output_pointer(out_completed, "out_completed");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
    *out_completed = JYPPX_FALSE;
    status = jyppx::cuda::validate_driver_kernel_launch(reinterpret_cast<JYPPX_CudaDriverKernelLaunch*>(launch), "launch");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
    DriverApi& api = driver_api();
    if (!api.IsAvailable())
    {
        return driver_dependency_missing(api, "cuEventQuery");
    }
    DriverContextScope scope(api, reinterpret_cast<CUcontext>(launch->context));
    if (scope.status != CUDA_SUCCESS)
    {
        return map_driver_status(api, scope.status, "cuCtxPushCurrent");
    }
    if (launch->completed)
    {
        *out_completed = JYPPX_TRUE;
        const CUresult pop_status = scope.Close();
        return pop_status == CUDA_SUCCESS ? JYPPX_STATUS_OK : map_driver_status(api, pop_status, "cuCtxPopCurrent");
    }
    const CUresult query_status = api.event_query(reinterpret_cast<CUevent>(launch->event));
    const CUresult pop_status = scope.Close();
    if (query_status == CUDA_ERROR_NOT_READY)
    {
        return pop_status == CUDA_SUCCESS ? JYPPX_STATUS_OK : map_driver_status(api, pop_status, "cuCtxPopCurrent");
    }
    status = map_driver_status(api, query_status, "cuEventQuery");
    if (status == JYPPX_STATUS_OK)
    {
        launch->completed = true;
        *out_completed = JYPPX_TRUE;
    }
    if (status == JYPPX_STATUS_OK && pop_status != CUDA_SUCCESS)
    {
        return map_driver_status(api, pop_status, "cuCtxPopCurrent");
    }
    return status;
}

JYPPX_StatusCode synchronize_driver_kernel_impl(jyppx::cuda::DriverKernelLaunchObject* launch)
{
    auto status = jyppx::cuda::validate_driver_kernel_launch(reinterpret_cast<JYPPX_CudaDriverKernelLaunch*>(launch), "launch");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
    DriverApi& api = driver_api();
    if (!api.IsAvailable())
    {
        return driver_dependency_missing(api, "cuEventSynchronize");
    }
    DriverContextScope scope(api, reinterpret_cast<CUcontext>(launch->context));
    if (scope.status != CUDA_SUCCESS)
    {
        return map_driver_status(api, scope.status, "cuCtxPushCurrent");
    }
    const CUresult event_status = launch->completed ? CUDA_SUCCESS : api.event_synchronize(reinterpret_cast<CUevent>(launch->event));
    if (event_status == CUDA_SUCCESS)
    {
        launch->completed = true;
    }
    const CUresult pop_status = scope.Close();
    if (event_status != CUDA_SUCCESS)
    {
        return map_driver_status(api, event_status, "cuEventSynchronize");
    }
    return pop_status == CUDA_SUCCESS ? JYPPX_STATUS_OK : map_driver_status(api, pop_status, "cuCtxPopCurrent");
}

JYPPX_StatusCode destroy_driver_kernel_impl(jyppx::cuda::DriverKernelLaunchObject* launch)
{
    if (launch == nullptr)
    {
        return JYPPX_STATUS_OK;
    }
    auto status = jyppx::cuda::validate_driver_kernel_launch(reinterpret_cast<JYPPX_CudaDriverKernelLaunch*>(launch), "launch");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
    DriverApi& api = driver_api();
    if (!api.IsAvailable())
    {
        delete launch;
        return driver_dependency_missing(api, "cuEventDestroy");
    }
    DriverContextScope scope(api, reinterpret_cast<CUcontext>(launch->context));
    if (scope.status != CUDA_SUCCESS)
    {
        delete launch;
        return map_driver_status(api, scope.status, "cuCtxPushCurrent");
    }
    CUresult synchronize_status = CUDA_SUCCESS;
    if (!launch->completed)
    {
        synchronize_status = api.event_synchronize(reinterpret_cast<CUevent>(launch->event));
    }
    const CUresult destroy_status = api.event_destroy(reinterpret_cast<CUevent>(launch->event));
    const CUresult pop_status = scope.Close();
    launch->base.magic = 0U;
    launch->event = nullptr;
    delete launch;
    if (synchronize_status != CUDA_SUCCESS)
    {
        return map_driver_status(api, synchronize_status, "cuEventSynchronize");
    }
    if (destroy_status != CUDA_SUCCESS)
    {
        return map_driver_status(api, destroy_status, "cuEventDestroy");
    }
    return pop_status == CUDA_SUCCESS ? JYPPX_STATUS_OK : map_driver_status(api, pop_status, "cuCtxPopCurrent");
}

JYPPX_StatusCode destroy_driver_module_impl(jyppx::cuda::DriverModuleObject* module)
{
    if (module == nullptr)
    {
        return JYPPX_STATUS_OK;
    }
    auto status = jyppx::cuda::validate_driver_module(reinterpret_cast<JYPPX_CudaDriverModule*>(module), "module");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }
    DriverApi& api = driver_api();
    if (!api.IsAvailable())
    {
        delete module;
        return driver_dependency_missing(api, "cuModuleUnload");
    }
    DriverContextScope scope(api, reinterpret_cast<CUcontext>(module->context));
    const CUresult push_status = scope.status;
    CUresult unload_status = CUDA_SUCCESS;
    if (push_status == CUDA_SUCCESS)
    {
        unload_status = api.module_unload(reinterpret_cast<CUmodule>(module->module));
    }
    const CUresult pop_status = scope.Close();
    const CUresult release_status = api.primary_ctx_release(static_cast<CUdevice>(module->device));
    module->base.magic = 0U;
    module->module = nullptr;
    module->context = nullptr;
    delete module;
    if (push_status != CUDA_SUCCESS)
    {
        return map_driver_status(api, push_status, "cuCtxPushCurrent");
    }
    if (unload_status != CUDA_SUCCESS)
    {
        return map_driver_status(api, unload_status, "cuModuleUnload");
    }
    if (pop_status != CUDA_SUCCESS)
    {
        return map_driver_status(api, pop_status, "cuCtxPopCurrent");
    }
    return map_driver_status(api, release_status, "cuDevicePrimaryCtxRelease");
}

#else

struct DriverApi final
{
    std::string loaded_library_name;
    std::string dependency_diagnostic = "CUDA Driver API requires a CUDA Toolkit-aware bridge build.";
    bool IsAvailable() const noexcept { return false; }
};

DriverApi& driver_api()
{
    static DriverApi api;
    return api;
}

JYPPX_StatusCode driver_dependency_missing(DriverApi& api, const char* operation)
{
    jyppx::cuda::set_cuda_error(operation, 0, "cuda-driver-dependency-missing", api.dependency_diagnostic.c_str());
    return JYPPX_STATUS_DEPENDENCY_MISSING;
}

JYPPX_StatusCode copy_driver_string(const std::string& value, char* output_buffer, const size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::cuda::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK) return status;
    *out_required_size = value.size() + 1U;
    if (output_buffer != nullptr && output_buffer_size >= *out_required_size)
    {
        std::memcpy(output_buffer, value.c_str(), *out_required_size);
        return JYPPX_STATUS_OK;
    }
    if (output_buffer == nullptr || output_buffer_size == 0U) return JYPPX_STATUS_OK;
    output_buffer[0] = '\0';
    return JYPPX_STATUS_BUFFER_TOO_SMALL;
}

JYPPX_StatusCode query_driver_capability_impl(JYPPX_CudaDriverCapabilityInfo* out_info)
{
    auto status = jyppx::cuda::validate_output_pointer(out_info, "out_info");
    if (status == JYPPX_STATUS_OK) std::memset(out_info, 0, sizeof(*out_info));
    return status;
}

#endif

} // namespace

#if JYPPX_HAS_CUDA_TOOLKIT

JYPPX_StatusCode jyppx_cuda_driver_query_capability_safe(JYPPX_CudaDriverCapabilityInfo* out_info)
{
    JYPPX_CUDA_DRIVER_GUARD("cuInit", run_driver_noexcept("cuInit", [&]() { return query_driver_capability_impl(out_info); }));
}

JYPPX_StatusCode jyppx_cuda_driver_get_loaded_library_name_safe(char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    JYPPX_CUDA_DRIVER_GUARD("cudaDriverGetLoadedLibraryName", run_driver_noexcept("cudaDriverGetLoadedLibraryName", [&]() { return copy_driver_string(driver_api().loaded_library_name, output_buffer, output_buffer_size, out_required_size); }));
}

JYPPX_StatusCode jyppx_cuda_driver_get_dependency_diagnostic_safe(char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    JYPPX_CUDA_DRIVER_GUARD("cudaDriverGetDependencyDiagnostic", run_driver_noexcept("cudaDriverGetDependencyDiagnostic", [&]() { return copy_driver_string(driver_api().dependency_diagnostic, output_buffer, output_buffer_size, out_required_size); }));
}

JYPPX_StatusCode jyppx_cuda_driver_module_load_data_copy_safe(const uint8_t* code, size_t code_size, int32_t device, JYPPX_CudaDriverModule** out_module)
{
    JYPPX_CUDA_DRIVER_GUARD("cuModuleLoadDataEx", run_driver_noexcept("cuModuleLoadDataEx", [&]() { return load_driver_module_impl(code, code_size, device, out_module); }));
}

JYPPX_StatusCode jyppx_cuda_driver_module_launch_typed_safe(JYPPX_CudaDriverModule* module, const char* kernel_name, JYPPX_CudaDim3 grid_dim, JYPPX_CudaDim3 block_dim, const JYPPX_CudaKernelArgumentDescriptor* arguments, size_t argument_count, const uint8_t* scalar_data, size_t scalar_data_size, size_t dynamic_shared_memory_bytes, JYPPX_CudaStream* stream, JYPPX_CudaDriverKernelLaunch** out_launch)
{
    JYPPX_CUDA_DRIVER_GUARD("cuLaunchKernel", run_driver_noexcept("cuLaunchKernel", [&]() {
        return launch_driver_kernel_impl(reinterpret_cast<jyppx::cuda::DriverModuleObject*>(module), kernel_name, grid_dim, block_dim, arguments, argument_count, scalar_data, scalar_data_size, dynamic_shared_memory_bytes, reinterpret_cast<jyppx::cuda::StreamObject*>(stream), out_launch);
    }));
}

JYPPX_StatusCode jyppx_cuda_driver_kernel_launch_query_safe(JYPPX_CudaDriverKernelLaunch* launch, JYPPX_Boolean* out_completed)
{
    JYPPX_CUDA_DRIVER_GUARD("cuEventQuery", run_driver_noexcept("cuEventQuery", [&]() { return query_driver_kernel_impl(reinterpret_cast<jyppx::cuda::DriverKernelLaunchObject*>(launch), out_completed); }));
}

JYPPX_StatusCode jyppx_cuda_driver_kernel_launch_synchronize_safe(JYPPX_CudaDriverKernelLaunch* launch)
{
    JYPPX_CUDA_DRIVER_GUARD("cuEventSynchronize", run_driver_noexcept("cuEventSynchronize", [&]() { return synchronize_driver_kernel_impl(reinterpret_cast<jyppx::cuda::DriverKernelLaunchObject*>(launch)); }));
}

JYPPX_StatusCode jyppx_cuda_driver_kernel_launch_destroy_safe(JYPPX_CudaDriverKernelLaunch* launch)
{
    JYPPX_CUDA_DRIVER_GUARD("cuEventDestroy", run_driver_noexcept("cuEventDestroy", [&]() { return destroy_driver_kernel_impl(reinterpret_cast<jyppx::cuda::DriverKernelLaunchObject*>(launch)); }));
}

JYPPX_StatusCode jyppx_cuda_driver_module_destroy_safe(JYPPX_CudaDriverModule* module)
{
    JYPPX_CUDA_DRIVER_GUARD("cuModuleUnload", run_driver_noexcept("cuModuleUnload", [&]() { return destroy_driver_module_impl(reinterpret_cast<jyppx::cuda::DriverModuleObject*>(module)); }));
}

#else

JYPPX_StatusCode jyppx_cuda_driver_query_capability_safe(JYPPX_CudaDriverCapabilityInfo* out_info)
{
    JYPPX_CUDA_DRIVER_GUARD("cuInit", run_driver_noexcept("cuInit", [&]() { return query_driver_capability_impl(out_info); }));
}

JYPPX_StatusCode jyppx_cuda_driver_get_loaded_library_name_safe(char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    JYPPX_CUDA_DRIVER_GUARD("cudaDriverGetLoadedLibraryName", run_driver_noexcept("cudaDriverGetLoadedLibraryName", [&]() { return copy_driver_string(driver_api().loaded_library_name, output_buffer, output_buffer_size, out_required_size); }));
}

JYPPX_StatusCode jyppx_cuda_driver_get_dependency_diagnostic_safe(char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    JYPPX_CUDA_DRIVER_GUARD("cudaDriverGetDependencyDiagnostic", run_driver_noexcept("cudaDriverGetDependencyDiagnostic", [&]() { return copy_driver_string(driver_api().dependency_diagnostic, output_buffer, output_buffer_size, out_required_size); }));
}

JYPPX_StatusCode jyppx_cuda_driver_module_load_data_copy_safe(const uint8_t*, size_t, int32_t, JYPPX_CudaDriverModule** out_module)
{
    JYPPX_CUDA_DRIVER_GUARD("cuModuleLoadDataEx", run_driver_noexcept("cuModuleLoadDataEx", [&]() {
        if (out_module != nullptr) *out_module = nullptr;
        return driver_dependency_missing(driver_api(), "cuModuleLoadDataEx");
    }));
}

JYPPX_StatusCode jyppx_cuda_driver_module_launch_typed_safe(JYPPX_CudaDriverModule*, const char*, JYPPX_CudaDim3, JYPPX_CudaDim3, const JYPPX_CudaKernelArgumentDescriptor*, size_t, const uint8_t*, size_t, size_t, JYPPX_CudaStream*, JYPPX_CudaDriverKernelLaunch** out_launch)
{
    JYPPX_CUDA_DRIVER_GUARD("cuLaunchKernel", run_driver_noexcept("cuLaunchKernel", [&]() {
        if (out_launch != nullptr) *out_launch = nullptr;
        return driver_dependency_missing(driver_api(), "cuLaunchKernel");
    }));
}

JYPPX_StatusCode jyppx_cuda_driver_kernel_launch_query_safe(JYPPX_CudaDriverKernelLaunch*, JYPPX_Boolean* out_completed)
{
    JYPPX_CUDA_DRIVER_GUARD("cuEventQuery", run_driver_noexcept("cuEventQuery", [&]() {
        if (out_completed != nullptr) *out_completed = JYPPX_FALSE;
        return driver_dependency_missing(driver_api(), "cuEventQuery");
    }));
}

JYPPX_StatusCode jyppx_cuda_driver_kernel_launch_synchronize_safe(JYPPX_CudaDriverKernelLaunch*)
{
    JYPPX_CUDA_DRIVER_GUARD("cuEventSynchronize", run_driver_noexcept("cuEventSynchronize", [&]() { return driver_dependency_missing(driver_api(), "cuEventSynchronize"); }));
}

JYPPX_StatusCode jyppx_cuda_driver_kernel_launch_destroy_safe(JYPPX_CudaDriverKernelLaunch*)
{
    JYPPX_CUDA_DRIVER_GUARD("cuEventDestroy", run_driver_noexcept("cuEventDestroy", [&]() { return JYPPX_STATUS_OK; }));
}

JYPPX_StatusCode jyppx_cuda_driver_module_destroy_safe(JYPPX_CudaDriverModule*)
{
    JYPPX_CUDA_DRIVER_GUARD("cuModuleUnload", run_driver_noexcept("cuModuleUnload", [&]() { return JYPPX_STATUS_OK; }));
}

#endif
