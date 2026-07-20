#pragma once

#include <stddef.h>
#include <stdint.h>
#include <vector>

#include "jyppx/common/status.h"
#include "jyppx/cuda/types.h"

#if JYPPX_HAS_CUDA_TOOLKIT
#include <cuda_runtime_api.h>
#endif

namespace jyppx::cuda
{
enum class ObjectKind : uint32_t
{
    Stream = 1,
    Event = 2,
    Memory = 3,
    PinnedMemory = 4,
    PitchedMemory = 5,
    Graph = 6,
    GraphExec = 7,
    Array = 8,
    MipmappedArray = 9,
    TextureObject = 10,
    SurfaceObject = 11,
    KernelLibrary = 12,
    ExecutionContext = 13,
    GraphConditionalHandle = 14,
    GraphConditionalNode = 15
};

struct ObjectBase
{
    uint32_t magic;
    ObjectKind kind;
};

struct StreamObject
{
    ObjectBase base;
    uint32_t flags;
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaStream_t handle;
#else
    void* handle;
#endif
};

struct EventObject
{
    ObjectBase base;
    uint32_t flags;
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaEvent_t handle;
#else
    void* handle;
#endif
};

struct MemoryObject
{
    ObjectBase base;
    void* pointer;
    size_t size;
    bool is_managed;
    bool is_ipc_exportable;
};

struct PinnedMemoryObject
{
    ObjectBase base;
    void* pointer;
    size_t size;
    uint32_t flags;
    bool is_registered;
};

struct PitchedMemoryObject
{
    ObjectBase base;
    void* pointer;
    size_t pitch_bytes;
    size_t width_bytes;
    size_t height;
    size_t depth;
};

struct GraphObject
{
    ObjectBase base;
    size_t active_conditional_handles;
    size_t active_conditional_nodes;
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaGraph_t handle;
#else
    void* handle;
#endif
};

struct GraphConditionalHandleObject
{
    ObjectBase base;
    GraphObject* owner;
    uint64_t handle;
    uint32_t default_launch_value;
    uint32_t flags;
};

struct GraphConditionalNodeObject
{
    ObjectBase base;
    GraphObject* owner;
    uint32_t node_type;
    uint32_t body_count;
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaGraphNode_t node;
    cudaGraph_t* body_graphs;
#else
    void* node;
    void* body_graphs;
#endif
};

struct GraphExecObject
{
    ObjectBase base;
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaGraphExec_t handle;
#else
    void* handle;
#endif
};

struct ArrayObject
{
    ObjectBase base;
    bool owns_handle;
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaArray_t handle;
#else
    void* handle;
#endif
};

struct MipmappedArrayObject
{
    ObjectBase base;
    uint32_t levels;
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMipmappedArray_t handle;
#else
    void* handle;
#endif
};

struct TextureObject
{
    ObjectBase base;
    bool has_resource_view;
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaTextureObject_t handle;
#else
    uint64_t handle;
#endif
};

struct SurfaceObject
{
    ObjectBase base;
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaSurfaceObject_t handle;
#else
    uint64_t handle;
#endif
};

struct KernelLibraryObject
{
    ObjectBase base;
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12090
    cudaLibrary_t handle;
#else
    void* handle;
#endif
    std::vector<uint8_t> retained_code;
};

struct ExecutionContextObject
{
    ObjectBase base;
    int32_t device_ordinal;
    bool is_primary;
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    cudaExecutionContext_t handle;
#else
    void* handle;
#endif
};

constexpr uint32_t kObjectMagic = 0x4A595043U;

JYPPX_StatusCode validate_output_pointer(void* pointer, const char* name);
JYPPX_StatusCode validate_stream(const JYPPX_CudaStream* stream, const char* name);
JYPPX_StatusCode validate_event(const JYPPX_CudaEvent* event_handle, const char* name);
JYPPX_StatusCode validate_memory(const JYPPX_CudaMemory* memory, const char* name);
JYPPX_StatusCode validate_pinned_memory(const JYPPX_CudaPinnedMemory* memory, const char* name);
JYPPX_StatusCode validate_pitched_memory(const JYPPX_CudaPitchedMemory* memory, const char* name);
JYPPX_StatusCode validate_graph(const JYPPX_CudaGraph* graph, const char* name);
JYPPX_StatusCode validate_graph_exec(const JYPPX_CudaGraphExec* graph_exec, const char* name);
JYPPX_StatusCode validate_graph_conditional_handle(const JYPPX_CudaGraphConditionalHandle* handle, const char* name);
JYPPX_StatusCode validate_graph_conditional_node(const JYPPX_CudaGraphConditionalNode* node, const char* name);
JYPPX_StatusCode validate_array(const JYPPX_CudaArray* array, const char* name);
JYPPX_StatusCode validate_mipmapped_array(const JYPPX_CudaMipmappedArray* array, const char* name);
JYPPX_StatusCode validate_texture_object(const JYPPX_CudaTextureObject* texture, const char* name);
JYPPX_StatusCode validate_surface_object(const JYPPX_CudaSurfaceObject* surface, const char* name);
JYPPX_StatusCode validate_kernel_library(const JYPPX_CudaKernelLibrary* library, const char* name);
JYPPX_StatusCode validate_execution_context(const JYPPX_CudaExecutionContext* context, const char* name);

void set_cuda_error(const char* operation, int32_t error_code, const char* error_name, const char* error_message);

#if JYPPX_HAS_CUDA_TOOLKIT
JYPPX_StatusCode map_cuda_status(cudaError_t error_code, const char* operation);
#else
JYPPX_StatusCode map_cuda_status(int32_t error_code, const char* operation);
#endif

JYPPX_StatusCode report_cuda_dependency_missing(const char* feature_name);
}
