#include "object.hpp"

#include <sstream>

#include "../common/error_state.hpp"

namespace
{
template <typename THandle>
JYPPX_StatusCode validate_handle(const THandle* handle, const jyppx::cuda::ObjectKind expected_kind, const char* name)
{
    if (handle == nullptr)
    {
        std::ostringstream builder;
        builder << name << " handle must not be null.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_CUDA, builder.str().c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (handle->base.magic != jyppx::cuda::kObjectMagic)
    {
        std::ostringstream builder;
        builder << name << " does not point to a valid CUDA bridge handle.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_CUDA, builder.str().c_str());
        return JYPPX_STATUS_INVALID_STATE;
    }

    if (handle->base.kind != expected_kind)
    {
        std::ostringstream builder;
        builder << name << " has an unexpected CUDA object kind.";
        jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_CUDA, builder.str().c_str());
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}
}

namespace jyppx::cuda
{
JYPPX_StatusCode validate_output_pointer(void* pointer, const char* name)
{
    if (pointer != nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    std::ostringstream builder;
    builder << name << " output pointer must not be null.";
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_CUDA, builder.str().c_str());
    return JYPPX_STATUS_INVALID_ARGUMENT;
}

JYPPX_StatusCode validate_stream(const JYPPX_CudaStream* stream, const char* name)
{
    return validate_handle(reinterpret_cast<const StreamObject*>(stream), ObjectKind::Stream, name);
}

JYPPX_StatusCode validate_event(const JYPPX_CudaEvent* event_handle, const char* name)
{
    return validate_handle(reinterpret_cast<const EventObject*>(event_handle), ObjectKind::Event, name);
}

JYPPX_StatusCode validate_memory(const JYPPX_CudaMemory* memory, const char* name)
{
    return validate_handle(reinterpret_cast<const MemoryObject*>(memory), ObjectKind::Memory, name);
}

JYPPX_StatusCode validate_pinned_memory(const JYPPX_CudaPinnedMemory* memory, const char* name)
{
    return validate_handle(reinterpret_cast<const PinnedMemoryObject*>(memory), ObjectKind::PinnedMemory, name);
}

JYPPX_StatusCode validate_pitched_memory(const JYPPX_CudaPitchedMemory* memory, const char* name)
{
    return validate_handle(reinterpret_cast<const PitchedMemoryObject*>(memory), ObjectKind::PitchedMemory, name);
}

JYPPX_StatusCode validate_graph(const JYPPX_CudaGraph* graph, const char* name)
{
    return validate_handle(reinterpret_cast<const GraphObject*>(graph), ObjectKind::Graph, name);
}

JYPPX_StatusCode validate_graph_exec(const JYPPX_CudaGraphExec* graph_exec, const char* name)
{
    return validate_handle(reinterpret_cast<const GraphExecObject*>(graph_exec), ObjectKind::GraphExec, name);
}

JYPPX_StatusCode validate_graph_conditional_handle(const JYPPX_CudaGraphConditionalHandle* handle, const char* name)
{
    return validate_handle(reinterpret_cast<const GraphConditionalHandleObject*>(handle), ObjectKind::GraphConditionalHandle, name);
}

JYPPX_StatusCode validate_graph_conditional_node(const JYPPX_CudaGraphConditionalNode* node, const char* name)
{
    return validate_handle(reinterpret_cast<const GraphConditionalNodeObject*>(node), ObjectKind::GraphConditionalNode, name);
}

JYPPX_StatusCode validate_array(const JYPPX_CudaArray* array, const char* name)
{
    return validate_handle(reinterpret_cast<const ArrayObject*>(array), ObjectKind::Array, name);
}

JYPPX_StatusCode validate_mipmapped_array(const JYPPX_CudaMipmappedArray* array, const char* name)
{
    return validate_handle(reinterpret_cast<const MipmappedArrayObject*>(array), ObjectKind::MipmappedArray, name);
}

JYPPX_StatusCode validate_texture_object(const JYPPX_CudaTextureObject* texture, const char* name)
{
    return validate_handle(reinterpret_cast<const TextureObject*>(texture), ObjectKind::TextureObject, name);
}

JYPPX_StatusCode validate_surface_object(const JYPPX_CudaSurfaceObject* surface, const char* name)
{
    return validate_handle(reinterpret_cast<const SurfaceObject*>(surface), ObjectKind::SurfaceObject, name);
}

JYPPX_StatusCode validate_kernel_library(const JYPPX_CudaKernelLibrary* library, const char* name)
{
    return validate_handle(reinterpret_cast<const KernelLibraryObject*>(library), ObjectKind::KernelLibrary, name);
}

JYPPX_StatusCode validate_execution_context(const JYPPX_CudaExecutionContext* context, const char* name)
{
    return validate_handle(reinterpret_cast<const ExecutionContextObject*>(context), ObjectKind::ExecutionContext, name);
}

void set_cuda_error(const char* operation, const int32_t error_code, const char* error_name, const char* error_message)
{
    std::ostringstream builder;
    builder << operation << " failed with CUDA error " << error_code;
    if (error_name != nullptr && error_name[0] != '\0')
    {
        builder << " (" << error_name << ")";
    }
    if (error_message != nullptr && error_message[0] != '\0')
    {
        builder << ": " << error_message;
    }

    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_CUDA, builder.str().c_str());
}

#if JYPPX_HAS_CUDA_TOOLKIT
JYPPX_StatusCode map_cuda_status(const cudaError_t error_code, const char* operation)
{
    if (error_code == cudaSuccess)
    {
        return JYPPX_STATUS_OK;
    }

    set_cuda_error(operation, static_cast<int32_t>(error_code), cudaGetErrorName(error_code), cudaGetErrorString(error_code));

    switch (error_code)
    {
    case cudaErrorInvalidValue:
    case cudaErrorInvalidDevice:
    case cudaErrorInvalidPitchValue:
    case cudaErrorInvalidMemcpyDirection:
        return JYPPX_STATUS_INVALID_ARGUMENT;
    case cudaErrorMemoryAllocation:
        return JYPPX_STATUS_OUT_OF_MEMORY;
    case cudaErrorInsufficientDriver:
    case cudaErrorNoDevice:
        return JYPPX_STATUS_DEPENDENCY_MISSING;
    case cudaErrorNotReady:
        return JYPPX_STATUS_NOT_READY;
    default:
        return JYPPX_STATUS_RUNTIME_ERROR;
    }
}
#else
JYPPX_StatusCode map_cuda_status(const int32_t error_code, const char* operation)
{
    set_cuda_error(operation, error_code, "cuda-not-available", "CUDA Toolkit was not detected when the bridge was built.");
    return JYPPX_STATUS_DEPENDENCY_MISSING;
}
#endif

JYPPX_StatusCode report_cuda_dependency_missing(const char* feature_name)
{
    std::ostringstream builder;
    builder << feature_name << " is unavailable because the CUDA Toolkit was not detected when the bridge was built.";
    jyppx::common::set_last_error(JYPPX_ERROR_CATEGORY_CUDA, builder.str().c_str());
    return JYPPX_STATUS_DEPENDENCY_MISSING;
}
}
