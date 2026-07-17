#include "jyppx/cuda/runtime.h"

#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <new>
#include <vector>

#include "object.hpp"

#if JYPPX_HAS_CUDA_TOOLKIT
#include <cuda_runtime_api.h>
#endif

namespace
{
using jyppx::cuda::EventObject;
using jyppx::cuda::GraphExecObject;
using jyppx::cuda::GraphObject;
using jyppx::cuda::MemoryObject;
using jyppx::cuda::ObjectBase;
using jyppx::cuda::ObjectKind;
using jyppx::cuda::PitchedMemoryObject;
using jyppx::cuda::PinnedMemoryObject;
using jyppx::cuda::StreamObject;

constexpr JYPPX_Boolean to_jyppx_bool(const bool value)
{
    return value ? JYPPX_TRUE : JYPPX_FALSE;
}

#if JYPPX_HAS_CUDA_TOOLKIT
cudaPitchedPtr make_pitched_ptr(void* pointer, const size_t pitch_bytes, const size_t width_bytes, const size_t height)
{
    cudaPitchedPtr value{};
    value.ptr = pointer;
    value.pitch = pitch_bytes;
    value.xsize = width_bytes;
    value.ysize = height;
    return value;
}

cudaExtent make_extent(const size_t width_bytes, const size_t height, const size_t depth)
{
    cudaExtent value{};
    value.width = width_bytes;
    value.height = height;
    value.depth = depth;
    return value;
}

cudaMemLocation make_bridge_mem_location(const JYPPX_CudaMemLocation& location)
{
    cudaMemLocation value{};
    value.type = static_cast<cudaMemLocationType>(location.type);
    value.id = location.id;
    return value;
}

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
cudaMemLocation make_device_mem_location(const int32_t device)
{
    cudaMemLocation location{};
    location.type = cudaMemLocationTypeDevice;
    location.id = device;
    return location;
}
#endif
#endif

JYPPX_StatusCode copy_string_to_buffer(const char* value, char* output_buffer, const size_t output_buffer_size, size_t* out_required_size)
{
    auto status = jyppx::cuda::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    const char* safe_value = value != nullptr ? value : "";
    const size_t required_size = std::strlen(safe_value) + 1;
    *out_required_size = required_size;

    if (output_buffer == nullptr || output_buffer_size == 0)
    {
        return JYPPX_STATUS_OK;
    }

    if (output_buffer_size < required_size)
    {
        if (output_buffer_size > 0)
        {
            output_buffer[0] = '\0';
        }

        jyppx::cuda::set_cuda_error("copy_string_to_buffer", 0, "buffer-too-small", "Output buffer is too small for the requested CUDA string.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }

    std::memcpy(output_buffer, safe_value, required_size);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode validate_pitched_dimensions(const size_t width_bytes, const size_t height, const char* operation)
{
    if (width_bytes == 0 || height == 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-2d-size", "2D CUDA copy dimensions must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode validate_pitched_copy_region(
    const PitchedMemoryObject* memory,
    const size_t width_bytes,
    const size_t height,
    const char* operation,
    const char* role)
{
    const auto status = validate_pitched_dimensions(width_bytes, height, operation);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (width_bytes > memory->width_bytes || height > memory->height)
    {
        std::string message = std::string(role) + " 2D copy region exceeds allocated pitched memory extent.";
        jyppx::cuda::set_cuda_error(operation, 0, "2d-region-out-of-range", message.c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode validate_host_pitch(const size_t pitch, const size_t width_bytes, const char* operation, const char* role)
{
    if (pitch < width_bytes)
    {
        std::string message = std::string(role) + " pitch must be greater than or equal to row width.";
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-host-pitch", message.c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode validate_pitched_copy_region_3d(
    const PitchedMemoryObject* memory,
    const size_t width_bytes,
    const size_t height,
    const size_t depth,
    const char* operation,
    const char* role)
{
    if (depth == 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-3d-depth", "3D CUDA copy depth must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    const auto status = validate_pitched_dimensions(width_bytes, height, operation);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    bool exceedsExtent = false;
    if (memory->depth <= 1)
    {
        if (height != 0 && depth > (std::numeric_limits<size_t>::max() / height))
        {
            exceedsExtent = true;
        }
        else
        {
            const size_t flattened_height = height * depth;
            exceedsExtent = width_bytes > memory->width_bytes || flattened_height > memory->height;
        }
    }
    else
    {
        exceedsExtent = width_bytes > memory->width_bytes || height > memory->height || depth > memory->depth;
    }

    if (exceedsExtent)
    {
        std::string message = std::string(role) + " 3D copy region exceeds allocated pitched memory extent.";
        jyppx::cuda::set_cuda_error(operation, 0, "3d-region-out-of-range", message.c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode validate_memory_range_query(
    const MemoryObject* memory,
    const size_t offset,
    const size_t count,
    const char* operation,
    const void** out_pointer)
{
    if (count == 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-range", "CUDA memory range count must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (offset > memory->size || count > memory->size - offset)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "range-out-of-bounds", "CUDA memory range exceeds the managed bridge allocation.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    *out_pointer = static_cast<const unsigned char*>(memory->pointer) + offset;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode validate_scalar_mem_range_attribute(const int32_t attribute, const char* operation)
{
    switch (attribute)
    {
    case 1:
    case 2:
    case 4:
        return JYPPX_STATUS_OK;

    case 3:
        jyppx::cuda::set_cuda_error(operation, 0, "array-attribute-not-supported", "cudaMemRangeAttributeAccessedBy returns an array and is intentionally not exposed by this scalar bridge.");
        return JYPPX_STATUS_INVALID_ARGUMENT;

    case 5:
    case 6:
    case 7:
    case 8:
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12030
        return JYPPX_STATUS_OK;
#else
        jyppx::cuda::set_cuda_error(operation, 0, "attribute-not-supported", "This CUDA Toolkit version does not expose the requested memory range location attribute.");
        return JYPPX_STATUS_NOT_SUPPORTED;
#endif

    default:
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-attribute", "Unsupported CUDA memory range attribute.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
}

size_t get_scalar_mem_range_attribute_size(const int32_t attribute)
{
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12030
    if (attribute == 5 || attribute == 7)
    {
        return sizeof(cudaMemLocationType);
    }
#else
    (void)attribute;
#endif

    return sizeof(int32_t);
}

JYPPX_StatusCode query_memory_range_accessed_by_devices(
    const MemoryObject* memory,
    const size_t offset,
    const size_t count,
    int32_t** out_devices,
    size_t* out_device_count)
{
    *out_devices = nullptr;
    *out_device_count = 0;

    const void* range_pointer = nullptr;
    auto status = validate_memory_range_query(memory, offset, count, "cudaMemRangeGetAttribute(AccessedBy)", &range_pointer);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    int cuda_device_count = 0;
    status = jyppx::cuda::map_cuda_status(cudaGetDeviceCount(&cuda_device_count), "cudaGetDeviceCount");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (cuda_device_count <= 0)
    {
        return JYPPX_STATUS_OK;
    }

    const size_t candidate_count = static_cast<size_t>(cuda_device_count);
    if (candidate_count > (std::numeric_limits<size_t>::max() / sizeof(int32_t)))
    {
        jyppx::cuda::set_cuda_error("cudaMemRangeGetAttribute(AccessedBy)", 0, "device-count-overflow", "CUDA device count is too large for memory range AccessedBy query.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    int32_t* candidates = new (std::nothrow) int32_t[candidate_count];
    if (candidates == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    for (size_t index = 0; index < candidate_count; ++index)
    {
        candidates[index] = static_cast<int32_t>(cudaInvalidDeviceId);
    }

    status = jyppx::cuda::map_cuda_status(
        cudaMemRangeGetAttribute(
            candidates,
            sizeof(int32_t) * candidate_count,
            cudaMemRangeAttributeAccessedBy,
            range_pointer,
            count),
        "cudaMemRangeGetAttribute(AccessedBy)");
    if (status != JYPPX_STATUS_OK)
    {
        delete[] candidates;
        return status;
    }

    size_t valid_count = 0;
    for (size_t index = 0; index < candidate_count; ++index)
    {
        if (candidates[index] != static_cast<int32_t>(cudaInvalidDeviceId))
        {
            ++valid_count;
        }
    }

    if (valid_count == 0)
    {
        delete[] candidates;
        return JYPPX_STATUS_OK;
    }

    int32_t* devices = new (std::nothrow) int32_t[valid_count];
    if (devices == nullptr)
    {
        delete[] candidates;
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    size_t write_index = 0;
    for (size_t index = 0; index < candidate_count; ++index)
    {
        if (candidates[index] != static_cast<int32_t>(cudaInvalidDeviceId))
        {
            devices[write_index++] = candidates[index];
        }
    }

    delete[] candidates;
    *out_devices = devices;
    *out_device_count = valid_count;
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memory range AccessedBy query");
#endif
}
}

JYPPX_StatusCode jyppx_cuda_query_runtime_info(JYPPX_CudaRuntimeInfo* out_info)
{
    auto status = jyppx::cuda::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    int runtime_version = 0;
    status = jyppx::cuda::map_cuda_status(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    int driver_version = 0;
    status = jyppx::cuda::map_cuda_status(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    int device_count = 0;
    status = jyppx::cuda::map_cuda_status(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    out_info->vendor_dependency_available = JYPPX_TRUE;
    out_info->supports_streams = JYPPX_TRUE;
    out_info->supports_events = JYPPX_TRUE;
    out_info->supports_memory = JYPPX_TRUE;
    out_info->runtime_version = runtime_version;
    out_info->driver_version = driver_version;
    out_info->device_count = device_count;
    out_info->status_message = "CUDA runtime bridge is available.";
    return JYPPX_STATUS_OK;
#else
    out_info->vendor_dependency_available = JYPPX_FALSE;
    out_info->supports_streams = JYPPX_FALSE;
    out_info->supports_events = JYPPX_FALSE;
    out_info->supports_memory = JYPPX_FALSE;
    out_info->runtime_version = 0;
    out_info->driver_version = 0;
    out_info->device_count = 0;
    out_info->status_message = "CUDA Toolkit was not detected when the bridge was built.";
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode jyppx_cuda_get_runtime_version(int32_t* out_version)
{
    auto status = jyppx::cuda::validate_output_pointer(out_version, "out_version");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_version = 0;

#if JYPPX_HAS_CUDA_TOOLKIT
    int version = 0;
    status = jyppx::cuda::map_cuda_status(cudaRuntimeGetVersion(&version), "cudaRuntimeGetVersion");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_version = version;
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA runtime version query");
#endif
}

JYPPX_StatusCode jyppx_cuda_get_driver_version(int32_t* out_version)
{
    auto status = jyppx::cuda::validate_output_pointer(out_version, "out_version");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_version = 0;

#if JYPPX_HAS_CUDA_TOOLKIT
    int version = 0;
    status = jyppx::cuda::map_cuda_status(cudaDriverGetVersion(&version), "cudaDriverGetVersion");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_version = version;
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA driver version query");
#endif
}

JYPPX_StatusCode jyppx_cuda_get_device_count(int32_t* out_device_count)
{
    auto status = jyppx::cuda::validate_output_pointer(out_device_count, "out_device_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    int value = 0;
    status = jyppx::cuda::map_cuda_status(cudaGetDeviceCount(&value), "cudaGetDeviceCount");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_device_count = value;
    return JYPPX_STATUS_OK;
#else
    *out_device_count = 0;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA device count query");
#endif
}

JYPPX_StatusCode jyppx_cuda_get_device_info(int32_t ordinal, JYPPX_CudaDeviceInfo* out_info)
{
    auto status = jyppx::cuda::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (ordinal < 0)
    {
        jyppx::cuda::set_cuda_error("cudaGetDeviceProperties", 0, "invalid-ordinal", "Device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaDeviceProp properties{};
    status = jyppx::cuda::map_cuda_status(cudaGetDeviceProperties(&properties, ordinal), "cudaGetDeviceProperties");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_info, 0, sizeof(*out_info));
    out_info->ordinal = ordinal;
    const std::string_view device_name(properties.name);
    const size_t name_length = (device_name.size() < (sizeof(out_info->name) - 1)) ? device_name.size() : (sizeof(out_info->name) - 1);
    std::memcpy(out_info->name, device_name.data(), name_length);
    out_info->name[name_length] = '\0';
    out_info->major = properties.major;
    out_info->minor = properties.minor;
    out_info->multi_processor_count = properties.multiProcessorCount;
    out_info->warp_size = properties.warpSize;
    out_info->max_threads_per_block = properties.maxThreadsPerBlock;
    out_info->can_map_host_memory = to_jyppx_bool(properties.canMapHostMemory != 0);
    out_info->integrated = to_jyppx_bool(properties.integrated != 0);
    out_info->total_global_memory = static_cast<uint64_t>(properties.totalGlobalMem);
    return JYPPX_STATUS_OK;
#else
    std::memset(out_info, 0, sizeof(*out_info));
    return jyppx::cuda::report_cuda_dependency_missing("CUDA device info query");
#endif
}

JYPPX_StatusCode jyppx_cuda_get_current_device(int32_t* out_device)
{
    auto status = jyppx::cuda::validate_output_pointer(out_device, "out_device");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    int device = 0;
    status = jyppx::cuda::map_cuda_status(cudaGetDevice(&device), "cudaGetDevice");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_device = device;
    return JYPPX_STATUS_OK;
#else
    *out_device = 0;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA current device query");
#endif
}

JYPPX_StatusCode jyppx_cuda_set_device(int32_t device)
{
    if (device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaSetDevice", 0, "invalid-device", "Device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaSetDevice(device), "cudaSetDevice");
#else
    (void)device;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA set device");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_synchronize(void)
{
#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA device synchronization");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_reset(void)
{
#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaDeviceReset(), "cudaDeviceReset");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA device reset");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_get_attribute(int32_t device, int32_t attribute, int32_t* out_value)
{
    auto status = jyppx::cuda::validate_output_pointer(out_value, "out_value");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaDeviceGetAttribute", 0, "invalid-device", "Device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (attribute <= 0)
    {
        jyppx::cuda::set_cuda_error("cudaDeviceGetAttribute", 0, "invalid-attribute", "CUDA device attribute must be a positive cudaDeviceAttr value.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    int value = 0;
    status = jyppx::cuda::map_cuda_status(cudaDeviceGetAttribute(&value, static_cast<cudaDeviceAttr>(attribute), device), "cudaDeviceGetAttribute");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_value = value;
    return JYPPX_STATUS_OK;
#else
    (void)device;
    (void)attribute;
    *out_value = 0;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA device attribute query");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_get_limit(int32_t limit, size_t* out_value)
{
    auto status = jyppx::cuda::validate_output_pointer(out_value, "out_value");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_value = 0;
    if (limit < 0)
    {
        jyppx::cuda::set_cuda_error("cudaDeviceGetLimit", 0, "invalid-limit", "CUDA device limit must be a non-negative cudaLimit value.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    size_t value = 0;
    status = jyppx::cuda::map_cuda_status(cudaDeviceGetLimit(&value, static_cast<cudaLimit>(limit)), "cudaDeviceGetLimit");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_value = value;
    return JYPPX_STATUS_OK;
#else
    (void)limit;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA device limit query");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_set_limit(int32_t limit, size_t value)
{
    if (limit < 0)
    {
        jyppx::cuda::set_cuda_error("cudaDeviceSetLimit", 0, "invalid-limit", "CUDA device limit must be a non-negative cudaLimit value.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaDeviceSetLimit(static_cast<cudaLimit>(limit), value), "cudaDeviceSetLimit");
#else
    (void)limit;
    (void)value;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA device limit set");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_get_cache_config(int32_t* out_config)
{
    auto status = jyppx::cuda::validate_output_pointer(out_config, "out_config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_config = 0;

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaFuncCache config = cudaFuncCachePreferNone;
    status = jyppx::cuda::map_cuda_status(cudaDeviceGetCacheConfig(&config), "cudaDeviceGetCacheConfig");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_config = static_cast<int32_t>(config);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA device cache config query");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_set_cache_config(int32_t config)
{
    if (config < 0 || config > 3)
    {
        jyppx::cuda::set_cuda_error("cudaDeviceSetCacheConfig", 0, "invalid-cache-config", "CUDA device cache config must be a cudaFuncCache value.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaDeviceSetCacheConfig(static_cast<cudaFuncCache>(config)), "cudaDeviceSetCacheConfig");
#else
    (void)config;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA device cache config set");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_get_shared_memory_config(int32_t* out_config)
{
    auto status = jyppx::cuda::validate_output_pointer(out_config, "out_config");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_config = 0;

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaSharedMemConfig config = cudaSharedMemBankSizeDefault;
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    status = jyppx::cuda::map_cuda_status(cudaDeviceGetSharedMemConfig(&config), "cudaDeviceGetSharedMemConfig");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_config = static_cast<int32_t>(config);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA shared memory bank config query");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_set_shared_memory_config(int32_t config)
{
    if (config < 0 || config > 2)
    {
        jyppx::cuda::set_cuda_error("cudaDeviceSetSharedMemConfig", 0, "invalid-shared-memory-config", "CUDA shared memory config must be a cudaSharedMemConfig value.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    const auto cuda_status = cudaDeviceSetSharedMemConfig(static_cast<cudaSharedMemConfig>(config));
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    return jyppx::cuda::map_cuda_status(cuda_status, "cudaDeviceSetSharedMemConfig");
#else
    (void)config;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA shared memory bank config set");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_get_pci_bus_id(int32_t device, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
    if (device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaDeviceGetPCIBusId", 0, "invalid-device", "Device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    char pci_bus_id[64]{};
    auto status = jyppx::cuda::map_cuda_status(cudaDeviceGetPCIBusId(pci_bus_id, static_cast<int>(sizeof(pci_bus_id)), device), "cudaDeviceGetPCIBusId");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    return copy_string_to_buffer(pci_bus_id, output_buffer, output_buffer_size, out_required_size);
#else
    (void)device;
    (void)output_buffer;
    (void)output_buffer_size;
    auto status = jyppx::cuda::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = 0;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA PCI bus id query");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_get_by_pci_bus_id(const char* pci_bus_id, int32_t* out_device)
{
    auto status = jyppx::cuda::validate_output_pointer(out_device, "out_device");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_device = -1;
    if (pci_bus_id == nullptr || pci_bus_id[0] == '\0')
    {
        jyppx::cuda::set_cuda_error("cudaDeviceGetByPCIBusId", 0, "invalid-pci-bus-id", "PCI bus id must not be null or empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    int device = -1;
    status = jyppx::cuda::map_cuda_status(cudaDeviceGetByPCIBusId(&device, pci_bus_id), "cudaDeviceGetByPCIBusId");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_device = device;
    return JYPPX_STATUS_OK;
#else
    (void)pci_bus_id;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA device lookup by PCI bus id");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_can_access_peer(int32_t device, int32_t peer_device, JYPPX_Boolean* out_can_access)
{
    auto status = jyppx::cuda::validate_output_pointer(out_can_access, "out_can_access");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (device < 0 || peer_device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaDeviceCanAccessPeer", 0, "invalid-device", "Device ordinals must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    int can_access = 0;
    status = jyppx::cuda::map_cuda_status(cudaDeviceCanAccessPeer(&can_access, device, peer_device), "cudaDeviceCanAccessPeer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_can_access = to_jyppx_bool(can_access != 0);
    return JYPPX_STATUS_OK;
#else
    (void)device;
    (void)peer_device;
    *out_can_access = JYPPX_FALSE;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA peer access query");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_get_p2p_attribute(int32_t attribute, int32_t source_device, int32_t destination_device, int32_t* out_value)
{
    auto status = jyppx::cuda::validate_output_pointer(out_value, "out_value");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_value = 0;
    if (attribute < 0 || source_device < 0 || destination_device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaDeviceGetP2PAttribute", 0, "invalid-p2p-arguments", "P2P attribute and device ordinals must be non-negative.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    int value = 0;
    status = jyppx::cuda::map_cuda_status(cudaDeviceGetP2PAttribute(&value, static_cast<cudaDeviceP2PAttr>(attribute), source_device, destination_device), "cudaDeviceGetP2PAttribute");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_value = value;
    return JYPPX_STATUS_OK;
#else
    (void)attribute;
    (void)source_device;
    (void)destination_device;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA P2P attribute query");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_enable_peer_access(int32_t peer_device, uint32_t flags)
{
    if (peer_device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaDeviceEnablePeerAccess", 0, "invalid-device", "Peer device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaDeviceEnablePeerAccess(peer_device, flags), "cudaDeviceEnablePeerAccess");
#else
    (void)peer_device;
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA peer access enable");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_disable_peer_access(int32_t peer_device)
{
    if (peer_device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaDeviceDisablePeerAccess", 0, "invalid-device", "Peer device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaDeviceDisablePeerAccess(peer_device), "cudaDeviceDisablePeerAccess");
#else
    (void)peer_device;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA peer access disable");
#endif
}

JYPPX_StatusCode jyppx_cuda_get_device_flags(uint32_t* out_flags)
{
    auto status = jyppx::cuda::validate_output_pointer(out_flags, "out_flags");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_flags = 0;

#if JYPPX_HAS_CUDA_TOOLKIT
    unsigned int flags = 0;
    status = jyppx::cuda::map_cuda_status(cudaGetDeviceFlags(&flags), "cudaGetDeviceFlags");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_flags = static_cast<uint32_t>(flags);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA device flags query");
#endif
}

JYPPX_StatusCode jyppx_cuda_set_device_flags(uint32_t flags)
{
#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaSetDeviceFlags(flags), "cudaSetDeviceFlags");
#else
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA device flags set");
#endif
}

JYPPX_StatusCode jyppx_cuda_init_device(int32_t device, uint32_t device_flags, uint32_t flags)
{
    if (device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaInitDevice", 0, "invalid-device", "CUDA device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    return jyppx::cuda::map_cuda_status(cudaInitDevice(device, device_flags, flags), "cudaInitDevice");
#else
    (void)device;
    (void)device_flags;
    (void)flags;
    jyppx::cuda::set_cuda_error("cudaInitDevice", 0, "cuda-version-not-supported", "cudaInitDevice requires CUDA runtime 12.0 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#endif
#else
    (void)device;
    (void)device_flags;
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA device initialization");
#endif
}

JYPPX_StatusCode jyppx_cuda_set_valid_devices(const int32_t* devices, uint32_t count)
{
    if (count == 0)
    {
        jyppx::cuda::set_cuda_error("cudaSetValidDevices", 0, "invalid-count", "At least one CUDA device ordinal is required.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (devices == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaSetValidDevices", 0, "invalid-devices", "CUDA valid-device ordinal array must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (count > static_cast<uint32_t>(std::numeric_limits<int>::max()))
    {
        jyppx::cuda::set_cuda_error("cudaSetValidDevices", 0, "invalid-count", "CUDA valid-device count exceeds the native runtime limit.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    std::vector<int> copied_devices;
    copied_devices.reserve(count);
    for (uint32_t index = 0; index < count; ++index)
    {
        const int32_t device = devices[index];
        if (device < 0)
        {
            jyppx::cuda::set_cuda_error("cudaSetValidDevices", 0, "invalid-device", "CUDA valid-device ordinals must be greater than or equal to zero.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }

        copied_devices.push_back(static_cast<int>(device));
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaSetValidDevices(copied_devices.data(), static_cast<int>(copied_devices.size())), "cudaSetValidDevices");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA valid-device set");
#endif
}

JYPPX_StatusCode jyppx_cuda_get_memory_info(JYPPX_CudaMemoryInfo* out_info)
{
    auto status = jyppx::cuda::validate_output_pointer(out_info, "out_info");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    status = jyppx::cuda::map_cuda_status(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    out_info->free_bytes = static_cast<uint64_t>(free_bytes);
    out_info->total_bytes = static_cast<uint64_t>(total_bytes);
    return JYPPX_STATUS_OK;
#else
    out_info->free_bytes = 0;
    out_info->total_bytes = 0;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memory info query");
#endif
}

JYPPX_StatusCode jyppx_cuda_get_last_error(int32_t* out_error_code)
{
    auto status = jyppx::cuda::validate_output_pointer(out_error_code, "out_error_code");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    *out_error_code = static_cast<int32_t>(cudaGetLastError());
    return JYPPX_STATUS_OK;
#else
    *out_error_code = 0;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA last error query");
#endif
}

JYPPX_StatusCode jyppx_cuda_peek_at_last_error(int32_t* out_error_code)
{
    auto status = jyppx::cuda::validate_output_pointer(out_error_code, "out_error_code");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    *out_error_code = static_cast<int32_t>(cudaPeekAtLastError());
    return JYPPX_STATUS_OK;
#else
    *out_error_code = 0;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA last error peek");
#endif
}

JYPPX_StatusCode jyppx_cuda_get_error_name(int32_t error_code, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
#if JYPPX_HAS_CUDA_TOOLKIT
    return copy_string_to_buffer(cudaGetErrorName(static_cast<cudaError_t>(error_code)), output_buffer, output_buffer_size, out_required_size);
#else
    (void)error_code;
    (void)output_buffer;
    (void)output_buffer_size;
    auto status = jyppx::cuda::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = 0;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA error name query");
#endif
}

JYPPX_StatusCode jyppx_cuda_get_error_string(int32_t error_code, char* output_buffer, size_t output_buffer_size, size_t* out_required_size)
{
#if JYPPX_HAS_CUDA_TOOLKIT
    return copy_string_to_buffer(cudaGetErrorString(static_cast<cudaError_t>(error_code)), output_buffer, output_buffer_size, out_required_size);
#else
    (void)error_code;
    (void)output_buffer;
    (void)output_buffer_size;
    auto status = jyppx::cuda::validate_output_pointer(out_required_size, "out_required_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_size = 0;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA error string query");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_create(uint32_t flags, JYPPX_CudaStream** out_stream)
{
    auto status = jyppx::cuda::validate_output_pointer(out_stream, "out_stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_stream = nullptr;

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream = new (std::nothrow) StreamObject{};
    if (stream == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    stream->base.magic = jyppx::cuda::kObjectMagic;
    stream->base.kind = ObjectKind::Stream;
    stream->flags = flags;
    stream->handle = nullptr;

    status = jyppx::cuda::map_cuda_status(cudaStreamCreateWithFlags(&stream->handle, flags), "cudaStreamCreateWithFlags");
    if (status != JYPPX_STATUS_OK)
    {
        delete stream;
        return status;
    }

    *out_stream = reinterpret_cast<JYPPX_CudaStream*>(stream);
    return JYPPX_STATUS_OK;
#else
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream creation");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_get_priority_range(int32_t* out_least_priority, int32_t* out_greatest_priority)
{
    auto status = jyppx::cuda::validate_output_pointer(out_least_priority, "out_least_priority");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_output_pointer(out_greatest_priority, "out_greatest_priority");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_least_priority = 0;
    *out_greatest_priority = 0;

#if JYPPX_HAS_CUDA_TOOLKIT
    int least_priority = 0;
    int greatest_priority = 0;
    status = jyppx::cuda::map_cuda_status(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority), "cudaDeviceGetStreamPriorityRange");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_least_priority = least_priority;
    *out_greatest_priority = greatest_priority;
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream priority range query");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_create_with_priority(uint32_t flags, int32_t priority, JYPPX_CudaStream** out_stream)
{
    auto status = jyppx::cuda::validate_output_pointer(out_stream, "out_stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_stream = nullptr;

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream = new (std::nothrow) StreamObject{};
    if (stream == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    stream->base.magic = jyppx::cuda::kObjectMagic;
    stream->base.kind = ObjectKind::Stream;
    stream->flags = flags;
    stream->handle = nullptr;

    status = jyppx::cuda::map_cuda_status(cudaStreamCreateWithPriority(&stream->handle, flags, priority), "cudaStreamCreateWithPriority");
    if (status != JYPPX_STATUS_OK)
    {
        delete stream;
        return status;
    }

    *out_stream = reinterpret_cast<JYPPX_CudaStream*>(stream);
    return JYPPX_STATUS_OK;
#else
    (void)flags;
    (void)priority;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream priority creation");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_get_flags(JYPPX_CudaStream* stream, uint32_t* out_flags)
{
    auto status = jyppx::cuda::validate_output_pointer(out_flags, "out_flags");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_flags = reinterpret_cast<StreamObject*>(stream)->flags;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_cuda_stream_get_priority(JYPPX_CudaStream* stream, int32_t* out_priority)
{
    auto status = jyppx::cuda::validate_output_pointer(out_priority, "out_priority");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_priority = 0;

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* handle = reinterpret_cast<StreamObject*>(stream);
    int priority = 0;
    status = jyppx::cuda::map_cuda_status(cudaStreamGetPriority(handle->handle, &priority), "cudaStreamGetPriority");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_priority = priority;
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream priority query");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_get_id(JYPPX_CudaStream* stream, uint64_t* out_stream_id)
{
    auto status = jyppx::cuda::validate_output_pointer(out_stream_id, "out_stream_id");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_stream_id = 0;
    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    unsigned long long stream_id = 0;
    status = jyppx::cuda::map_cuda_status(cudaStreamGetId(stream_object->handle, &stream_id), "cudaStreamGetId");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_stream_id = static_cast<uint64_t>(stream_id);
    return JYPPX_STATUS_OK;
#else
    jyppx::cuda::set_cuda_error("cudaStreamGetId", 0, "cuda-version-not-supported", "cudaStreamGetId requires CUDA runtime 12.0 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream id query");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_get_device(JYPPX_CudaStream* stream, int32_t* out_device)
{
    auto status = jyppx::cuda::validate_output_pointer(out_device, "out_device");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_device = -1;
    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    jyppx::cuda::set_cuda_error("cudaStreamGetDevice", 0, "cuda-runtime-symbol-not-available", "cudaStreamGetDevice is declared by newer CUDA headers but is not linkable in the current cudart import library; use cudaGetDevice through CudaDevice.Current as a fallback.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream device query");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_copy_attributes(JYPPX_CudaStream* destination, JYPPX_CudaStream* source)
{
    auto status = jyppx::cuda::validate_stream(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* destination_object = reinterpret_cast<StreamObject*>(destination);
    auto* source_object = reinterpret_cast<StreamObject*>(source);
    return jyppx::cuda::map_cuda_status(cudaStreamCopyAttributes(destination_object->handle, source_object->handle), "cudaStreamCopyAttributes");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream attribute copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_thread_exchange_stream_capture_mode(int32_t mode, int32_t* out_previous_mode)
{
    auto status = jyppx::cuda::validate_output_pointer(out_previous_mode, "out_previous_mode");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_previous_mode = 0;
    if (mode < 0 || mode > 2)
    {
        jyppx::cuda::set_cuda_error("cudaThreadExchangeStreamCaptureMode", 0, "invalid-capture-mode", "CUDA stream capture mode must be Global, ThreadLocal, or Relaxed.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaStreamCaptureMode capture_mode = static_cast<cudaStreamCaptureMode>(mode);
    status = jyppx::cuda::map_cuda_status(cudaThreadExchangeStreamCaptureMode(&capture_mode), "cudaThreadExchangeStreamCaptureMode");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_previous_mode = static_cast<int32_t>(capture_mode);
    return JYPPX_STATUS_OK;
#else
    (void)mode;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA thread stream capture mode exchange");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_query(JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* handle = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaStreamQuery(handle->handle), "cudaStreamQuery");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream query");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_wait_event(JYPPX_CudaStream* stream, JYPPX_CudaEvent* event_handle, uint32_t flags)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_event(event_handle, "event_handle");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    auto* event_object = reinterpret_cast<EventObject*>(event_handle);
    return jyppx::cuda::map_cuda_status(cudaStreamWaitEvent(stream_object->handle, event_object->handle, flags), "cudaStreamWaitEvent");
#else
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream wait event");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_synchronize(JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* handle = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaStreamSynchronize(handle->handle), "cudaStreamSynchronize");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream synchronization");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_destroy(JYPPX_CudaStream* stream)
{
    if (stream == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* handle = reinterpret_cast<StreamObject*>(stream);
    status = jyppx::cuda::map_cuda_status(cudaStreamDestroy(handle->handle), "cudaStreamDestroy");
    delete handle;
    return status;
#else
    delete reinterpret_cast<StreamObject*>(stream);
    return JYPPX_STATUS_OK;
#endif
}

#include "modules/graph/stream_capture_graph.inc"
#include "modules/graph/node_topology.inc"
#include "modules/graph/device_graph_memory.inc"
#include "modules/graph/child_graph_update.inc"
#include "modules/graph/owner_scoped_diagnostics.inc"
#include "modules/cuda_logs.inc"
#include "modules/stream_advanced.inc"
#include "modules/device_atomic_capabilities.inc"
#include "modules/kernel_launch.inc"

JYPPX_StatusCode jyppx_cuda_event_create(uint32_t flags, JYPPX_CudaEvent** out_event)
{
    auto status = jyppx::cuda::validate_output_pointer(out_event, "out_event");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_event = nullptr;

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* event_object = new (std::nothrow) EventObject{};
    if (event_object == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    event_object->base.magic = jyppx::cuda::kObjectMagic;
    event_object->base.kind = ObjectKind::Event;
    event_object->flags = flags;
    event_object->handle = nullptr;

    status = jyppx::cuda::map_cuda_status(cudaEventCreateWithFlags(&event_object->handle, flags), "cudaEventCreateWithFlags");
    if (status != JYPPX_STATUS_OK)
    {
        delete event_object;
        return status;
    }

    *out_event = reinterpret_cast<JYPPX_CudaEvent*>(event_object);
    return JYPPX_STATUS_OK;
#else
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA event creation");
#endif
}

JYPPX_StatusCode jyppx_cuda_event_get_flags(JYPPX_CudaEvent* event_handle, uint32_t* out_flags)
{
    auto status = jyppx::cuda::validate_output_pointer(out_flags, "out_flags");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_event(event_handle, "event_handle");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_flags = reinterpret_cast<EventObject*>(event_handle)->flags;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_cuda_event_query(JYPPX_CudaEvent* event_handle)
{
    auto status = jyppx::cuda::validate_event(event_handle, "event_handle");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* event_object = reinterpret_cast<EventObject*>(event_handle);
    return jyppx::cuda::map_cuda_status(cudaEventQuery(event_object->handle), "cudaEventQuery");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA event query");
#endif
}

JYPPX_StatusCode jyppx_cuda_event_record(JYPPX_CudaEvent* event_handle, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_event(event_handle, "event_handle");
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
    auto* event_object = reinterpret_cast<EventObject*>(event_handle);
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaEventRecord(event_object->handle, stream_object->handle), "cudaEventRecord");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA event record");
#endif
}

JYPPX_StatusCode jyppx_cuda_event_record_with_flags(JYPPX_CudaEvent* event_handle, JYPPX_CudaStream* stream, uint32_t flags)
{
    auto status = jyppx::cuda::validate_event(event_handle, "event_handle");
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
    auto* event_object = reinterpret_cast<EventObject*>(event_handle);
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaEventRecordWithFlags(event_object->handle, stream_object->handle, flags), "cudaEventRecordWithFlags");
#else
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA event record with flags");
#endif
}

JYPPX_StatusCode jyppx_cuda_event_synchronize(JYPPX_CudaEvent* event_handle)
{
    auto status = jyppx::cuda::validate_event(event_handle, "event_handle");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* event_object = reinterpret_cast<EventObject*>(event_handle);
    return jyppx::cuda::map_cuda_status(cudaEventSynchronize(event_object->handle), "cudaEventSynchronize");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA event synchronization");
#endif
}

JYPPX_StatusCode jyppx_cuda_event_elapsed_time(JYPPX_CudaEvent* start_event, JYPPX_CudaEvent* end_event, float* out_milliseconds)
{
    auto status = jyppx::cuda::validate_output_pointer(out_milliseconds, "out_milliseconds");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_event(start_event, "start_event");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_event(end_event, "end_event");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* start_object = reinterpret_cast<EventObject*>(start_event);
    auto* end_object = reinterpret_cast<EventObject*>(end_event);
    return jyppx::cuda::map_cuda_status(cudaEventElapsedTime(out_milliseconds, start_object->handle, end_object->handle), "cudaEventElapsedTime");
#else
    *out_milliseconds = 0.0F;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA event elapsed time");
#endif
}

JYPPX_StatusCode jyppx_cuda_event_destroy(JYPPX_CudaEvent* event_handle)
{
    if (event_handle == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    auto status = jyppx::cuda::validate_event(event_handle, "event_handle");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* event_object = reinterpret_cast<EventObject*>(event_handle);
    status = jyppx::cuda::map_cuda_status(cudaEventDestroy(event_object->handle), "cudaEventDestroy");
    delete event_object;
    return status;
#else
    delete reinterpret_cast<EventObject*>(event_handle);
    return JYPPX_STATUS_OK;
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_alloc(size_t size, JYPPX_CudaMemory** out_memory)
{
    auto status = jyppx::cuda::validate_output_pointer(out_memory, "out_memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_memory = nullptr;

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* memory = new (std::nothrow) MemoryObject{};
    if (memory == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    memory->base.magic = jyppx::cuda::kObjectMagic;
    memory->base.kind = ObjectKind::Memory;
    memory->pointer = nullptr;
    memory->size = size;
    memory->is_managed = false;

    status = jyppx::cuda::map_cuda_status(cudaMalloc(&memory->pointer, size), "cudaMalloc");
    if (status != JYPPX_STATUS_OK)
    {
        delete memory;
        return status;
    }

    *out_memory = reinterpret_cast<JYPPX_CudaMemory*>(memory);
    return JYPPX_STATUS_OK;
#else
    (void)size;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memory allocation");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_alloc_managed(size_t size, uint32_t flags, JYPPX_CudaMemory** out_memory)
{
    auto status = jyppx::cuda::validate_output_pointer(out_memory, "out_memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_memory = nullptr;

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* memory = new (std::nothrow) MemoryObject{};
    if (memory == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    memory->base.magic = jyppx::cuda::kObjectMagic;
    memory->base.kind = ObjectKind::Memory;
    memory->pointer = nullptr;
    memory->size = size;
    memory->is_managed = true;

    status = jyppx::cuda::map_cuda_status(cudaMallocManaged(&memory->pointer, size, flags), "cudaMallocManaged");
    if (status != JYPPX_STATUS_OK)
    {
        delete memory;
        return status;
    }

    *out_memory = reinterpret_cast<JYPPX_CudaMemory*>(memory);
    return JYPPX_STATUS_OK;
#else
    (void)size;
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA managed memory allocation");
#endif
}

#include "modules/memory/async_memory_pool.inc"

JYPPX_StatusCode jyppx_cuda_memory_get_size(JYPPX_CudaMemory* memory, size_t* out_size)
{
    auto status = jyppx::cuda::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_size = reinterpret_cast<MemoryObject*>(memory)->size;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_cuda_memory_get_device_pointer(JYPPX_CudaMemory* memory, void** out_pointer)
{
    auto status = jyppx::cuda::validate_output_pointer(out_pointer, "out_pointer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_pointer = reinterpret_cast<MemoryObject*>(memory)->pointer;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_cuda_memory_get_pointer_attributes(JYPPX_CudaMemory* memory, JYPPX_CudaPointerAttributes* out_attributes)
{
    auto status = jyppx::cuda::validate_output_pointer(out_attributes, "out_attributes");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_attributes, 0, sizeof(*out_attributes));
    status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(memory);

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaPointerAttributes attributes{};
    status = jyppx::cuda::map_cuda_status(cudaPointerGetAttributes(&attributes, memory_object->pointer), "cudaPointerGetAttributes");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    out_attributes->memory_type = static_cast<int32_t>(attributes.type);
    out_attributes->device = attributes.device;
    out_attributes->device_pointer = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(attributes.devicePointer));
    out_attributes->host_pointer = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(attributes.hostPointer));
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA pointer attributes query");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_copy_from_host(JYPPX_CudaMemory* memory, const void* source, size_t size)
{
    auto status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (source == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy", 0, "invalid-source", "Source pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(memory);
    if (size > memory_object->size)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy", 0, "size-out-of-range", "Source size exceeds allocated device memory.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaMemcpy(memory_object->pointer, source, size, cudaMemcpyHostToDevice), "cudaMemcpy(HtoD)");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA host-to-device copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_copy_to_host(JYPPX_CudaMemory* memory, void* destination, size_t size)
{
    auto status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (destination == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy", 0, "invalid-destination", "Destination pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(memory);
    if (size > memory_object->size)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy", 0, "size-out-of-range", "Destination size exceeds allocated device memory.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaMemcpy(destination, memory_object->pointer, size, cudaMemcpyDeviceToHost), "cudaMemcpy(DtoH)");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA device-to-host copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_copy_from_host_async(JYPPX_CudaMemory* memory, const void* source, size_t size, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_memory(memory, "memory");
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
        jyppx::cuda::set_cuda_error("cudaMemcpyAsync", 0, "invalid-source", "Source pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(memory);
    if (size > memory_object->size)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyAsync", 0, "size-out-of-range", "Source size exceeds allocated device memory.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaMemcpyAsync(memory_object->pointer, source, size, cudaMemcpyHostToDevice, stream_object->handle), "cudaMemcpyAsync(HtoD)");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async host-to-device copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_copy_to_host_async(JYPPX_CudaMemory* memory, void* destination, size_t size, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_memory(memory, "memory");
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
        jyppx::cuda::set_cuda_error("cudaMemcpyAsync", 0, "invalid-destination", "Destination pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(memory);
    if (size > memory_object->size)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyAsync", 0, "size-out-of-range", "Destination size exceeds allocated device memory.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaMemcpyAsync(destination, memory_object->pointer, size, cudaMemcpyDeviceToHost, stream_object->handle), "cudaMemcpyAsync(DtoH)");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async device-to-host copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_copy_device_to_device(JYPPX_CudaMemory* destination, JYPPX_CudaMemory* source, size_t size)
{
    auto status = jyppx::cuda::validate_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_memory(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* destination_object = reinterpret_cast<MemoryObject*>(destination);
    auto* source_object = reinterpret_cast<MemoryObject*>(source);
    if (size > destination_object->size || size > source_object->size)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpy", 0, "size-out-of-range", "Copy size exceeds source or destination device memory.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaMemcpy(destination_object->pointer, source_object->pointer, size, cudaMemcpyDeviceToDevice), "cudaMemcpy(DtoD)");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA device-to-device copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_copy_device_to_device_async(JYPPX_CudaMemory* destination, JYPPX_CudaMemory* source, size_t size, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_memory(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* destination_object = reinterpret_cast<MemoryObject*>(destination);
    auto* source_object = reinterpret_cast<MemoryObject*>(source);
    if (size > destination_object->size || size > source_object->size)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyAsync", 0, "size-out-of-range", "Copy size exceeds source or destination device memory.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaMemcpyAsync(destination_object->pointer, source_object->pointer, size, cudaMemcpyDeviceToDevice, stream_object->handle), "cudaMemcpyAsync(DtoD)");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async device-to-device copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_copy_default(JYPPX_CudaMemory* destination, JYPPX_CudaMemory* source, size_t size)
{
    auto status = jyppx::cuda::validate_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_memory(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* destination_object = reinterpret_cast<MemoryObject*>(destination);
    auto* source_object = reinterpret_cast<MemoryObject*>(source);
    if (size > destination_object->size || size > source_object->size)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyDefault", 0, "size-out-of-range", "Default copy size exceeds source or destination device memory.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaMemcpy(destination_object->pointer, source_object->pointer, size, cudaMemcpyDefault), "cudaMemcpy(Default)");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA default-kind memory copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_copy_default_async(JYPPX_CudaMemory* destination, JYPPX_CudaMemory* source, size_t size, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_memory(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* destination_object = reinterpret_cast<MemoryObject*>(destination);
    auto* source_object = reinterpret_cast<MemoryObject*>(source);
    if (size > destination_object->size || size > source_object->size)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyAsyncDefault", 0, "size-out-of-range", "Default async copy size exceeds source or destination device memory.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaMemcpyAsync(destination_object->pointer, source_object->pointer, size, cudaMemcpyDefault, stream_object->handle), "cudaMemcpyAsync(Default)");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async default-kind memory copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_copy_peer(JYPPX_CudaMemory* destination, int32_t destination_device, JYPPX_CudaMemory* source, int32_t source_device, size_t size)
{
    auto status = jyppx::cuda::validate_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_memory(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (destination_device < 0 || source_device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyPeer", 0, "invalid-device", "Source and destination device ordinals must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* destination_object = reinterpret_cast<MemoryObject*>(destination);
    auto* source_object = reinterpret_cast<MemoryObject*>(source);
    if (size > destination_object->size || size > source_object->size)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyPeer", 0, "size-out-of-range", "Peer copy size exceeds source or destination device memory.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaMemcpyPeer(destination_object->pointer, destination_device, source_object->pointer, source_device, size), "cudaMemcpyPeer");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA peer memory copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_copy_peer_async(JYPPX_CudaMemory* destination, int32_t destination_device, JYPPX_CudaMemory* source, int32_t source_device, size_t size, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_memory(source, "source");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (destination_device < 0 || source_device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyPeerAsync", 0, "invalid-device", "Source and destination device ordinals must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* destination_object = reinterpret_cast<MemoryObject*>(destination);
    auto* source_object = reinterpret_cast<MemoryObject*>(source);
    if (size > destination_object->size || size > source_object->size)
    {
        jyppx::cuda::set_cuda_error("cudaMemcpyPeerAsync", 0, "size-out-of-range", "Peer copy size exceeds source or destination device memory.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaMemcpyPeerAsync(destination_object->pointer, destination_device, source_object->pointer, source_device, size, stream_object->handle), "cudaMemcpyPeerAsync");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async peer memory copy");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_memset(JYPPX_CudaMemory* memory, int32_t value, size_t size)
{
    auto status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(memory);
    if (size > memory_object->size)
    {
        jyppx::cuda::set_cuda_error("cudaMemset", 0, "size-out-of-range", "Memset size exceeds allocated device memory.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaMemset(memory_object->pointer, value, size), "cudaMemset");
#else
    (void)value;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memory memset");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_memset_async(JYPPX_CudaMemory* memory, int32_t value, size_t size, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(memory);
    if (size > memory_object->size)
    {
        jyppx::cuda::set_cuda_error("cudaMemsetAsync", 0, "size-out-of-range", "Memset size exceeds allocated device memory.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaMemsetAsync(memory_object->pointer, value, size, stream_object->handle), "cudaMemsetAsync");
#else
    (void)value;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA async memory memset");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_prefetch_range_async(JYPPX_CudaMemory* memory, size_t offset, size_t count, int32_t destination_device, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (destination_device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemPrefetchAsync", 0, "invalid-device", "Destination device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(memory);
    const void* range_pointer = nullptr;
    status = validate_memory_range_query(memory_object, offset, count, "cudaMemPrefetchAsync", &range_pointer);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    const auto location = make_device_mem_location(destination_device);
    return jyppx::cuda::map_cuda_status(cudaMemPrefetchAsync(range_pointer, count, location, 0U, stream_object->handle), "cudaMemPrefetchAsync");
#else
    return jyppx::cuda::map_cuda_status(cudaMemPrefetchAsync(range_pointer, count, destination_device, stream_object->handle), "cudaMemPrefetchAsync");
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memory prefetch");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_prefetch_async(JYPPX_CudaMemory* memory, size_t size, int32_t destination_device, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_memory_prefetch_range_async(memory, 0, size, destination_device, stream);
}

JYPPX_StatusCode jyppx_cuda_memory_advise_range(JYPPX_CudaMemory* memory, size_t offset, size_t count, int32_t advice, int32_t device)
{
    auto status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (device < 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemAdvise", 0, "invalid-device", "Device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(memory);
    const void* range_pointer = nullptr;
    status = validate_memory_range_query(memory_object, offset, count, "cudaMemAdvise", &range_pointer);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    const auto location = make_device_mem_location(device);
    return jyppx::cuda::map_cuda_status(cudaMemAdvise(range_pointer, count, static_cast<cudaMemoryAdvise>(advice), location), "cudaMemAdvise");
#else
    return jyppx::cuda::map_cuda_status(cudaMemAdvise(range_pointer, count, static_cast<cudaMemoryAdvise>(advice), device), "cudaMemAdvise");
#endif
#else
    (void)advice;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memory advise");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_advise(JYPPX_CudaMemory* memory, size_t size, int32_t advice, int32_t device)
{
    return jyppx_cuda_memory_advise_range(memory, 0, size, advice, device);
}

JYPPX_StatusCode jyppx_cuda_memory_range_get_attribute(JYPPX_CudaMemory* memory, size_t offset, size_t count, int32_t attribute, int32_t* out_value)
{
    auto status = jyppx::cuda::validate_output_pointer(out_value, "out_value");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_value = 0;
    status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(memory);
    const void* range_pointer = nullptr;
    status = validate_memory_range_query(memory_object, offset, count, "cudaMemRangeGetAttribute", &range_pointer);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = validate_scalar_mem_range_attribute(attribute, "cudaMemRangeGetAttribute");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    int32_t value = 0;
    status = jyppx::cuda::map_cuda_status(
        cudaMemRangeGetAttribute(
            &value,
            get_scalar_mem_range_attribute_size(attribute),
            static_cast<cudaMemRangeAttribute>(attribute),
            range_pointer,
            count),
        "cudaMemRangeGetAttribute");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_value = value;
    return JYPPX_STATUS_OK;
#else
    (void)attribute;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memory range attribute query");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_range_get_attributes(
    JYPPX_CudaMemory* memory,
    size_t offset,
    size_t count,
    const int32_t* attributes,
    size_t attribute_count,
    JYPPX_CudaMemRangeAttributeValue* out_values)
{
    if (attribute_count == 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemRangeGetAttributes", 0, "invalid-attribute-count", "At least one CUDA memory range attribute must be requested.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (attribute_count > 16)
    {
        jyppx::cuda::set_cuda_error("cudaMemRangeGetAttributes", 0, "too-many-attributes", "CUDA memory range scalar batch query is limited to 16 attributes.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (attributes == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaMemRangeGetAttributes", 0, "invalid-attributes", "CUDA memory range attribute array must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto status = jyppx::cuda::validate_output_pointer(out_values, "out_values");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    std::memset(out_values, 0, sizeof(*out_values) * attribute_count);
    status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(memory);
    const void* range_pointer = nullptr;
    status = validate_memory_range_query(memory_object, offset, count, "cudaMemRangeGetAttributes", &range_pointer);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    for (size_t index = 0; index < attribute_count; ++index)
    {
        status = validate_scalar_mem_range_attribute(attributes[index], "cudaMemRangeGetAttributes");
        if (status != JYPPX_STATUS_OK)
        {
            return status;
        }
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    void** data = new (std::nothrow) void*[attribute_count];
    auto* data_sizes = new (std::nothrow) size_t[attribute_count];
    auto* cuda_attributes = new (std::nothrow) cudaMemRangeAttribute[attribute_count];
    auto* values = new (std::nothrow) int32_t[attribute_count];
    if (data == nullptr || data_sizes == nullptr || cuda_attributes == nullptr || values == nullptr)
    {
        delete[] data;
        delete[] data_sizes;
        delete[] cuda_attributes;
        delete[] values;
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    for (size_t index = 0; index < attribute_count; ++index)
    {
        values[index] = 0;
        data[index] = &values[index];
        data_sizes[index] = get_scalar_mem_range_attribute_size(attributes[index]);
        cuda_attributes[index] = static_cast<cudaMemRangeAttribute>(attributes[index]);
    }

    status = jyppx::cuda::map_cuda_status(
        cudaMemRangeGetAttributes(
            data,
            data_sizes,
            cuda_attributes,
            attribute_count,
            range_pointer,
            count),
        "cudaMemRangeGetAttributes");
    if (status == JYPPX_STATUS_OK)
    {
        for (size_t index = 0; index < attribute_count; ++index)
        {
            out_values[index].attribute = attributes[index];
            out_values[index].value = values[index];
        }
    }

    delete[] data;
    delete[] data_sizes;
    delete[] cuda_attributes;
    delete[] values;
    return status;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA memory range attributes query");
#endif
}

JYPPX_StatusCode jyppx_cuda_memory_range_get_accessed_by_count(JYPPX_CudaMemory* memory, size_t offset, size_t count, size_t* out_device_count)
{
    auto status = jyppx::cuda::validate_output_pointer(out_device_count, "out_device_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_device_count = 0;
    status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(memory);
    int32_t* devices = nullptr;
    size_t device_count = 0;
    status = query_memory_range_accessed_by_devices(memory_object, offset, count, &devices, &device_count);
    delete[] devices;
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_device_count = device_count;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_cuda_memory_range_copy_accessed_by_devices(
    JYPPX_CudaMemory* memory,
    size_t offset,
    size_t count,
    int32_t* output_devices,
    size_t output_device_count,
    size_t* out_required_count)
{
    auto status = jyppx::cuda::validate_output_pointer(out_required_count, "out_required_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_required_count = 0;
    if (output_devices == nullptr && output_device_count != 0)
    {
        jyppx::cuda::set_cuda_error("cudaMemRangeGetAttribute(AccessedBy)", 0, "invalid-output-buffer", "AccessedBy output buffer must not be null when output count is non-zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(memory);
    int32_t* devices = nullptr;
    size_t device_count = 0;
    status = query_memory_range_accessed_by_devices(memory_object, offset, count, &devices, &device_count);
    if (status != JYPPX_STATUS_OK)
    {
        delete[] devices;
        return status;
    }

    *out_required_count = device_count;
    if (output_devices == nullptr || output_device_count == 0)
    {
        delete[] devices;
        return JYPPX_STATUS_OK;
    }

    if (output_device_count < device_count)
    {
        delete[] devices;
        jyppx::cuda::set_cuda_error("cudaMemRangeGetAttribute(AccessedBy)", 0, "buffer-too-small", "AccessedBy output buffer is too small for the queried device list.");
        return JYPPX_STATUS_BUFFER_TOO_SMALL;
    }

    if (device_count != 0)
    {
        std::memcpy(output_devices, devices, sizeof(int32_t) * device_count);
    }

    delete[] devices;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_cuda_memory_free(JYPPX_CudaMemory* memory)
{
    if (memory == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    auto status = jyppx::cuda::validate_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(memory);
#if JYPPX_HAS_CUDA_TOOLKIT
    status = jyppx::cuda::map_cuda_status(cudaFree(memory_object->pointer), "cudaFree");
#else
    status = JYPPX_STATUS_OK;
#endif
    delete memory_object;
    return status;
}

#include "modules/memory/pitched_memory.inc"
#include "modules/memory/array_memory.inc"
#include "modules/memory/texture_surface_objects.inc"
#include "modules/memory/raw_memory_operations.inc"
#include "modules/memory/managed_memory_batch.inc"
#include "modules/deployment/kernel_library_metadata.inc"
#include "modules/deployment/primary_execution_context.inc"
#include "modules/deployment/official_token_aliases.inc"
#include "modules/deferred/twenty_third_batch_deferred.inc"
#include "modules/deferred/thirty_fifth_batch_stream_device_deferred.inc"
#include "modules/deferred/thirty_seventh_batch_graph_deferred.inc"
#include "modules/deferred/thirty_eighth_batch_other_deferred.inc"

JYPPX_StatusCode jyppx_cuda_pinned_memory_alloc(size_t size, JYPPX_CudaPinnedMemory** out_memory)
{
    return jyppx_cuda_pinned_memory_alloc_with_flags(size, 0, out_memory);
}

JYPPX_StatusCode jyppx_cuda_pinned_memory_alloc_with_flags(size_t size, uint32_t flags, JYPPX_CudaPinnedMemory** out_memory)
{
    auto status = jyppx::cuda::validate_output_pointer(out_memory, "out_memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_memory = nullptr;

    if (size == 0)
    {
        jyppx::cuda::set_cuda_error("cudaHostAlloc", 0, "invalid-size", "Pinned memory allocation size must be greater than zero.");
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
    memory->flags = flags;
    memory->is_registered = false;

    status = jyppx::cuda::map_cuda_status(cudaHostAlloc(&memory->pointer, size, flags), "cudaHostAlloc");
    if (status != JYPPX_STATUS_OK)
    {
        delete memory;
        return status;
    }

    *out_memory = reinterpret_cast<JYPPX_CudaPinnedMemory*>(memory);
    return JYPPX_STATUS_OK;
#else
    (void)size;
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA pinned host memory allocation");
#endif
}

JYPPX_StatusCode jyppx_cuda_pinned_memory_register(void* pointer, size_t size, uint32_t flags, JYPPX_CudaPinnedMemory** out_memory)
{
    auto status = jyppx::cuda::validate_output_pointer(out_memory, "out_memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_memory = nullptr;

    if (pointer == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaHostRegister", 0, "invalid-pointer", "Registered host memory pointer must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (size == 0)
    {
        jyppx::cuda::set_cuda_error("cudaHostRegister", 0, "invalid-size", "Registered host memory size must be greater than zero.");
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
    memory->pointer = pointer;
    memory->size = size;
    memory->flags = flags;
    memory->is_registered = true;

    status = jyppx::cuda::map_cuda_status(cudaHostRegister(pointer, size, flags), "cudaHostRegister");
    if (status != JYPPX_STATUS_OK)
    {
        delete memory;
        return status;
    }

    *out_memory = reinterpret_cast<JYPPX_CudaPinnedMemory*>(memory);
    return JYPPX_STATUS_OK;
#else
    (void)pointer;
    (void)size;
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA host memory registration");
#endif
}

JYPPX_StatusCode jyppx_cuda_pinned_memory_get_size(JYPPX_CudaPinnedMemory* memory, size_t* out_size)
{
    auto status = jyppx::cuda::validate_output_pointer(out_size, "out_size");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_pinned_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_size = reinterpret_cast<PinnedMemoryObject*>(memory)->size;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_cuda_pinned_memory_get_flags(JYPPX_CudaPinnedMemory* memory, uint32_t* out_flags)
{
    auto status = jyppx::cuda::validate_output_pointer(out_flags, "out_flags");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_pinned_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_flags = reinterpret_cast<PinnedMemoryObject*>(memory)->flags;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_cuda_pinned_memory_get_host_pointer(JYPPX_CudaPinnedMemory* memory, void** out_pointer)
{
    auto status = jyppx::cuda::validate_output_pointer(out_pointer, "out_pointer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_pinned_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_pointer = reinterpret_cast<PinnedMemoryObject*>(memory)->pointer;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_cuda_pinned_memory_get_mapped_device_pointer(JYPPX_CudaPinnedMemory* memory, void** out_pointer)
{
    auto status = jyppx::cuda::validate_output_pointer(out_pointer, "out_pointer");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_pinned_memory(memory, "memory");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_pointer = nullptr;
    auto* memory_object = reinterpret_cast<PinnedMemoryObject*>(memory);
    if ((memory_object->flags & 2U) == 0U)
    {
        jyppx::cuda::set_cuda_error("cudaHostGetDevicePointer", 0, "not-mapped", "Pinned memory must be allocated with cudaHostAllocMapped before a mapped device pointer can be queried.");
        return JYPPX_STATUS_NOT_SUPPORTED;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaHostGetDevicePointer(out_pointer, memory_object->pointer, 0), "cudaHostGetDevicePointer");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA mapped pinned memory pointer query");
#endif
}

JYPPX_StatusCode jyppx_cuda_pinned_memory_free(JYPPX_CudaPinnedMemory* memory)
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
#if JYPPX_HAS_CUDA_TOOLKIT
    status = memory_object->is_registered
        ? jyppx::cuda::map_cuda_status(cudaHostUnregister(memory_object->pointer), "cudaHostUnregister")
        : jyppx::cuda::map_cuda_status(cudaFreeHost(memory_object->pointer), "cudaFreeHost");
#else
    status = JYPPX_STATUS_OK;
#endif
    delete memory_object;
    return status;
}
