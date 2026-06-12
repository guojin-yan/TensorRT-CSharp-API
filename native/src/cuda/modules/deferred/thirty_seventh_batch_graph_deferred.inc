#define JYPPX_CUDA_GRAPH_DEFERRED(function_name, cuda_name, reason) \
JYPPX_StatusCode function_name(void) \
{ \
    jyppx::cuda::set_cuda_error(cuda_name, 0, "deferred-interface-boundary", reason); \
    return JYPPX_STATUS_NOT_IMPLEMENTED; \
}

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_kernel_node_get_attribute_deferred,
    "cudaGraphKernelNodeGetAttribute",
    "cudaGraphKernelNodeGetAttribute deferred: kernel node attribute storage is not safely modeled by the current bridge")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_kernel_node_get_params_deferred,
    "cudaGraphKernelNodeGetParams",
    "cudaGraphKernelNodeGetParams deferred: cudaKernelNodeParams contains raw kernel pointers and argument arrays that need a dedicated bridge")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_kernel_node_set_attribute_deferred,
    "cudaGraphKernelNodeSetAttribute",
    "cudaGraphKernelNodeSetAttribute deferred: kernel node attribute storage is not safely modeled by the current bridge")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_kernel_node_set_params_deferred,
    "cudaGraphKernelNodeSetParams",
    "cudaGraphKernelNodeSetParams deferred: cudaKernelNodeParams contains raw kernel pointers and argument arrays that need a dedicated bridge")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_mem_alloc_node_get_params_deferred,
    "cudaGraphMemAllocNodeGetParams",
    "cudaGraphMemAllocNodeGetParams deferred: graph allocation node parameters need an ownership-safe memory-pool bridge")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_memcpy_node_get_params_deferred,
    "cudaGraphMemcpyNodeGetParams",
    "cudaGraphMemcpyNodeGetParams deferred: cudaMemcpy3DParms contains raw host, device, array, and symbol pointers that need a dedicated bridge")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_memcpy_node_set_params_deferred,
    "cudaGraphMemcpyNodeSetParams",
    "cudaGraphMemcpyNodeSetParams deferred: cudaMemcpy3DParms contains raw host, device, array, and symbol pointers that need a dedicated bridge")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_memcpy_node_set_params_1d_deferred,
    "cudaGraphMemcpyNodeSetParams1D",
    "cudaGraphMemcpyNodeSetParams1D deferred: raw pointer and symbol targets require a dedicated graph memcpy bridge")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_memcpy_node_set_params_from_symbol_deferred,
    "cudaGraphMemcpyNodeSetParamsFromSymbol",
    "cudaGraphMemcpyNodeSetParamsFromSymbol deferred: raw CUDA symbol addresses require a dedicated symbol bridge")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_memcpy_node_set_params_to_symbol_deferred,
    "cudaGraphMemcpyNodeSetParamsToSymbol",
    "cudaGraphMemcpyNodeSetParamsToSymbol deferred: raw CUDA symbol addresses require a dedicated symbol bridge")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_mem_free_node_get_params_deferred,
    "cudaGraphMemFreeNodeGetParams",
    "cudaGraphMemFreeNodeGetParams deferred: graph free node memory ownership is not safely modeled by the current bridge")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_memset_node_get_params_deferred,
    "cudaGraphMemsetNodeGetParams",
    "cudaGraphMemsetNodeGetParams deferred: cudaMemsetParams needs a dedicated bridge for pitched and array targets")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_memset_node_set_params_deferred,
    "cudaGraphMemsetNodeSetParams",
    "cudaGraphMemsetNodeSetParams deferred: cudaMemsetParams needs a dedicated bridge for pitched and array targets")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_node_get_containing_graph_deferred,
    "cudaGraphNodeGetContainingGraph",
    "cudaGraphNodeGetContainingGraph deferred: returning a non-owning graph handle needs explicit bridge ownership semantics")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_node_get_dependencies_v2_deferred,
    "cudaGraphNodeGetDependencies_v2",
    "cudaGraphNodeGetDependencies_v2 deferred: edge data arrays are not safely modeled by the current bridge")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_node_get_dependent_nodes_v2_deferred,
    "cudaGraphNodeGetDependentNodes_v2",
    "cudaGraphNodeGetDependentNodes_v2 deferred: edge data arrays are not safely modeled by the current bridge")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_node_get_params_deferred,
    "cudaGraphNodeGetParams",
    "cudaGraphNodeGetParams deferred: the discriminated cudaGraphNodeParams union needs a dedicated bridge")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_remove_dependencies_v2_deferred,
    "cudaGraphRemoveDependencies_v2",
    "cudaGraphRemoveDependencies_v2 deferred: edge data arrays are not safely modeled by the current bridge")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_retain_user_object_deferred,
    "cudaGraphRetainUserObject",
    "cudaGraphRetainUserObject deferred: user-object lifetime and destructor callbacks need explicit managed ownership policy")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_graph_release_user_object_deferred,
    "cudaGraphReleaseUserObject",
    "cudaGraphReleaseUserObject deferred: user-object lifetime and destructor callbacks need explicit managed ownership policy")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_user_object_create_deferred,
    "cudaUserObjectCreate",
    "cudaUserObjectCreate deferred: destructor callback ownership is not safely modeled by the current bridge")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_user_object_retain_deferred,
    "cudaUserObjectRetain",
    "cudaUserObjectRetain deferred: user-object lifetime needs explicit managed ownership policy")

JYPPX_CUDA_GRAPH_DEFERRED(
    jyppx_cuda_user_object_release_deferred,
    "cudaUserObjectRelease",
    "cudaUserObjectRelease deferred: user-object lifetime needs explicit managed ownership policy")

#undef JYPPX_CUDA_GRAPH_DEFERRED
