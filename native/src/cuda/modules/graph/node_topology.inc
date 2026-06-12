#include <vector>

#if JYPPX_HAS_CUDA_TOOLKIT
cudaGraphNode_t jyppx_cuda_node_from_token(uintptr_t token)
{
    return reinterpret_cast<cudaGraphNode_t>(token);
}

uintptr_t jyppx_cuda_node_to_token(cudaGraphNode_t node)
{
    return reinterpret_cast<uintptr_t>(node);
}
#endif

JYPPX_StatusCode jyppx_cuda_validate_graph_node_token(uintptr_t node, const char* name)
{
    if (node == 0)
    {
        jyppx::cuda::set_cuda_error(name, 0, "invalid-graph-node", "CUDA graph node token must not be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_cuda_graph_add_empty_node_safe(JYPPX_CudaGraph* graph, uintptr_t* out_node)
{
    auto status = jyppx::cuda::validate_output_pointer(out_node, "out_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = 0;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    cudaGraphNode_t node = nullptr;
    status = jyppx::cuda::map_cuda_status(cudaGraphAddEmptyNode(&node, graph_object->handle, nullptr, 0), "cudaGraphAddEmptyNode");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = jyppx_cuda_node_to_token(node);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph empty node add");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_add_empty_node_after_safe(JYPPX_CudaGraph* graph, uintptr_t dependency_node, uintptr_t* out_node)
{
    auto status = jyppx_cuda_validate_graph_node_token(dependency_node, "dependency_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_node, "out_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = 0;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    cudaGraphNode_t dependency = jyppx_cuda_node_from_token(dependency_node);
    cudaGraphNode_t node = nullptr;
    status = jyppx::cuda::map_cuda_status(cudaGraphAddEmptyNode(&node, graph_object->handle, &dependency, 1), "cudaGraphAddEmptyNode");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = jyppx_cuda_node_to_token(node);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph empty node add with dependency");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_add_dependency_safe(JYPPX_CudaGraph* graph, uintptr_t from_node, uintptr_t to_node)
{
    auto status = jyppx_cuda_validate_graph_node_token(from_node, "from_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_cuda_validate_graph_node_token(to_node, "to_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    cudaGraphNode_t from = jyppx_cuda_node_from_token(from_node);
    cudaGraphNode_t to = jyppx_cuda_node_from_token(to_node);
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    return jyppx::cuda::map_cuda_status(cudaGraphAddDependencies(graph_object->handle, &from, &to, nullptr, 1), "cudaGraphAddDependencies");
#else
    return jyppx::cuda::map_cuda_status(cudaGraphAddDependencies(graph_object->handle, &from, &to, 1), "cudaGraphAddDependencies");
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph dependency add");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_remove_dependency_safe(JYPPX_CudaGraph* graph, uintptr_t from_node, uintptr_t to_node)
{
    auto status = jyppx_cuda_validate_graph_node_token(from_node, "from_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_cuda_validate_graph_node_token(to_node, "to_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    cudaGraphNode_t from = jyppx_cuda_node_from_token(from_node);
    cudaGraphNode_t to = jyppx_cuda_node_from_token(to_node);
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    return jyppx::cuda::map_cuda_status(cudaGraphRemoveDependencies(graph_object->handle, &from, &to, nullptr, 1), "cudaGraphRemoveDependencies");
#else
    return jyppx::cuda::map_cuda_status(cudaGraphRemoveDependencies(graph_object->handle, &from, &to, 1), "cudaGraphRemoveDependencies");
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph dependency remove");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_get_node_handle(JYPPX_CudaGraph* graph, size_t index, uintptr_t* out_node)
{
    auto status = jyppx::cuda::validate_output_pointer(out_node, "out_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = 0;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    size_t count = 0;
    status = jyppx::cuda::map_cuda_status(cudaGraphGetNodes(graph_object->handle, nullptr, &count), "cudaGraphGetNodes");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (index >= count)
    {
        jyppx::cuda::set_cuda_error("cudaGraphGetNodes", 0, "index-out-of-range", "CUDA graph node index is outside the current node count.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    std::vector<cudaGraphNode_t> nodes(count);
    status = jyppx::cuda::map_cuda_status(cudaGraphGetNodes(graph_object->handle, nodes.data(), &count), "cudaGraphGetNodes");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = jyppx_cuda_node_to_token(nodes[index]);
    return JYPPX_STATUS_OK;
#else
    (void)index;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph node handle query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_get_root_node_handle(JYPPX_CudaGraph* graph, size_t index, uintptr_t* out_node)
{
    auto status = jyppx::cuda::validate_output_pointer(out_node, "out_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = 0;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    size_t count = 0;
    status = jyppx::cuda::map_cuda_status(cudaGraphGetRootNodes(graph_object->handle, nullptr, &count), "cudaGraphGetRootNodes");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (index >= count)
    {
        jyppx::cuda::set_cuda_error("cudaGraphGetRootNodes", 0, "index-out-of-range", "CUDA graph root node index is outside the current root node count.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    std::vector<cudaGraphNode_t> nodes(count);
    status = jyppx::cuda::map_cuda_status(cudaGraphGetRootNodes(graph_object->handle, nodes.data(), &count), "cudaGraphGetRootNodes");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = jyppx_cuda_node_to_token(nodes[index]);
    return JYPPX_STATUS_OK;
#else
    (void)index;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph root node handle query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_get_edge_handle_pair(JYPPX_CudaGraph* graph, size_t index, uintptr_t* out_from_node, uintptr_t* out_to_node)
{
    auto status = jyppx::cuda::validate_output_pointer(out_from_node, "out_from_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_to_node, "out_to_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_from_node = 0;
    *out_to_node = 0;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    size_t count = 0;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    status = jyppx::cuda::map_cuda_status(cudaGraphGetEdges(graph_object->handle, nullptr, nullptr, nullptr, &count), "cudaGraphGetEdges");
#else
    status = jyppx::cuda::map_cuda_status(cudaGraphGetEdges(graph_object->handle, nullptr, nullptr, &count), "cudaGraphGetEdges");
#endif
    if (status != JYPPX_STATUS_OK) { return status; }
    if (index >= count)
    {
        jyppx::cuda::set_cuda_error("cudaGraphGetEdges", 0, "index-out-of-range", "CUDA graph edge index is outside the current edge count.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    std::vector<cudaGraphNode_t> from_nodes(count);
    std::vector<cudaGraphNode_t> to_nodes(count);
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    status = jyppx::cuda::map_cuda_status(cudaGraphGetEdges(graph_object->handle, from_nodes.data(), to_nodes.data(), nullptr, &count), "cudaGraphGetEdges");
#else
    status = jyppx::cuda::map_cuda_status(cudaGraphGetEdges(graph_object->handle, from_nodes.data(), to_nodes.data(), &count), "cudaGraphGetEdges");
#endif
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_from_node = jyppx_cuda_node_to_token(from_nodes[index]);
    *out_to_node = jyppx_cuda_node_to_token(to_nodes[index]);
    return JYPPX_STATUS_OK;
#else
    (void)index;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph edge handle pair query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_node_get_type_safe(uintptr_t node, int32_t* out_type)
{
    auto status = jyppx::cuda::validate_output_pointer(out_type, "out_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_type = 0;
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaGraphNodeType type{};
    status = jyppx::cuda::map_cuda_status(cudaGraphNodeGetType(jyppx_cuda_node_from_token(node), &type), "cudaGraphNodeGetType");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_type = static_cast<int32_t>(type);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph node type query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_node_get_dependency_count_safe(uintptr_t node, size_t* out_count)
{
    auto status = jyppx::cuda::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    return jyppx::cuda::map_cuda_status(cudaGraphNodeGetDependencies(jyppx_cuda_node_from_token(node), nullptr, nullptr, out_count), "cudaGraphNodeGetDependencies");
#else
    return jyppx::cuda::map_cuda_status(cudaGraphNodeGetDependencies(jyppx_cuda_node_from_token(node), nullptr, out_count), "cudaGraphNodeGetDependencies");
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph node dependency count query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_node_get_dependency_safe(uintptr_t node, size_t index, uintptr_t* out_dependency_node)
{
    auto status = jyppx::cuda::validate_output_pointer(out_dependency_node, "out_dependency_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_dependency_node = 0;
    size_t count = 0;
    status = jyppx_cuda_graph_node_get_dependency_count_safe(node, &count);
    if (status != JYPPX_STATUS_OK) { return status; }
    if (index >= count)
    {
        jyppx::cuda::set_cuda_error("cudaGraphNodeGetDependencies", 0, "index-out-of-range", "CUDA graph node dependency index is outside the dependency count.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_CUDA_TOOLKIT
    std::vector<cudaGraphNode_t> nodes(count);
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    status = jyppx::cuda::map_cuda_status(cudaGraphNodeGetDependencies(jyppx_cuda_node_from_token(node), nodes.data(), nullptr, &count), "cudaGraphNodeGetDependencies");
#else
    status = jyppx::cuda::map_cuda_status(cudaGraphNodeGetDependencies(jyppx_cuda_node_from_token(node), nodes.data(), &count), "cudaGraphNodeGetDependencies");
#endif
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_dependency_node = jyppx_cuda_node_to_token(nodes[index]);
    return JYPPX_STATUS_OK;
#else
    (void)index;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph node dependency query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_node_get_dependent_count_safe(uintptr_t node, size_t* out_count)
{
    auto status = jyppx::cuda::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    return jyppx::cuda::map_cuda_status(cudaGraphNodeGetDependentNodes(jyppx_cuda_node_from_token(node), nullptr, nullptr, out_count), "cudaGraphNodeGetDependentNodes");
#else
    return jyppx::cuda::map_cuda_status(cudaGraphNodeGetDependentNodes(jyppx_cuda_node_from_token(node), nullptr, out_count), "cudaGraphNodeGetDependentNodes");
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph node dependent count query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_node_get_dependent_safe(uintptr_t node, size_t index, uintptr_t* out_dependent_node)
{
    auto status = jyppx::cuda::validate_output_pointer(out_dependent_node, "out_dependent_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_dependent_node = 0;
    size_t count = 0;
    status = jyppx_cuda_graph_node_get_dependent_count_safe(node, &count);
    if (status != JYPPX_STATUS_OK) { return status; }
    if (index >= count)
    {
        jyppx::cuda::set_cuda_error("cudaGraphNodeGetDependentNodes", 0, "index-out-of-range", "CUDA graph node dependent index is outside the dependent count.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_CUDA_TOOLKIT
    std::vector<cudaGraphNode_t> nodes(count);
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    status = jyppx::cuda::map_cuda_status(cudaGraphNodeGetDependentNodes(jyppx_cuda_node_from_token(node), nodes.data(), nullptr, &count), "cudaGraphNodeGetDependentNodes");
#else
    status = jyppx::cuda::map_cuda_status(cudaGraphNodeGetDependentNodes(jyppx_cuda_node_from_token(node), nodes.data(), &count), "cudaGraphNodeGetDependentNodes");
#endif
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_dependent_node = jyppx_cuda_node_to_token(nodes[index]);
    return JYPPX_STATUS_OK;
#else
    (void)index;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph node dependent query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_node_find_in_clone_safe(uintptr_t original_node, JYPPX_CudaGraph* cloned_graph, uintptr_t* out_cloned_node)
{
    auto status = jyppx::cuda::validate_output_pointer(out_cloned_node, "out_cloned_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cloned_node = 0;
    status = jyppx_cuda_validate_graph_node_token(original_node, "original_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_graph(cloned_graph, "cloned_graph");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_object = reinterpret_cast<GraphObject*>(cloned_graph);
    cudaGraphNode_t cloned_node = nullptr;
    status = jyppx::cuda::map_cuda_status(cudaGraphNodeFindInClone(&cloned_node, jyppx_cuda_node_from_token(original_node), graph_object->handle), "cudaGraphNodeFindInClone");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_cloned_node = jyppx_cuda_node_to_token(cloned_node);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph node clone lookup");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_exec_node_set_enabled_safe(JYPPX_CudaGraphExec* graph_exec, uintptr_t node, JYPPX_Boolean enabled)
{
    auto status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_graph_exec(graph_exec, "graph_exec");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_exec_object = reinterpret_cast<GraphExecObject*>(graph_exec);
    return jyppx::cuda::map_cuda_status(cudaGraphNodeSetEnabled(graph_exec_object->handle, jyppx_cuda_node_from_token(node), enabled == JYPPX_TRUE ? 1U : 0U), "cudaGraphNodeSetEnabled");
#else
    (void)enabled;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph exec node enable set");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_exec_node_get_enabled_safe(JYPPX_CudaGraphExec* graph_exec, uintptr_t node, JYPPX_Boolean* out_enabled)
{
    auto status = jyppx::cuda::validate_output_pointer(out_enabled, "out_enabled");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_enabled = JYPPX_FALSE;
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_graph_exec(graph_exec, "graph_exec");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_exec_object = reinterpret_cast<GraphExecObject*>(graph_exec);
    unsigned int enabled = 0;
    status = jyppx::cuda::map_cuda_status(cudaGraphNodeGetEnabled(graph_exec_object->handle, jyppx_cuda_node_from_token(node), &enabled), "cudaGraphNodeGetEnabled");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_enabled = enabled != 0 ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph exec node enable query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_get_id_safe(JYPPX_CudaGraph* graph, uint32_t* out_id)
{
    auto status = jyppx::cuda::validate_output_pointer(out_id, "out_id");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_id = 0;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    unsigned int id = 0;
    status = jyppx::cuda::map_cuda_status(cudaGraphGetId(graph_object->handle, &id), "cudaGraphGetId");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_id = static_cast<uint32_t>(id);
    return JYPPX_STATUS_OK;
#else
    jyppx::cuda::set_cuda_error("cudaGraphGetId", 0, "cuda-version-not-supported", "cudaGraphGetId requires CUDA runtime 13.0 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph id query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_exec_get_id_safe(JYPPX_CudaGraphExec* graph_exec, uint32_t* out_id)
{
    auto status = jyppx::cuda::validate_output_pointer(out_id, "out_id");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_id = 0;
    status = jyppx::cuda::validate_graph_exec(graph_exec, "graph_exec");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    auto* graph_exec_object = reinterpret_cast<GraphExecObject*>(graph_exec);
    unsigned int id = 0;
    status = jyppx::cuda::map_cuda_status(cudaGraphExecGetId(graph_exec_object->handle, &id), "cudaGraphExecGetId");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_id = static_cast<uint32_t>(id);
    return JYPPX_STATUS_OK;
#else
    jyppx::cuda::set_cuda_error("cudaGraphExecGetId", 0, "cuda-version-not-supported", "cudaGraphExecGetId requires CUDA runtime 13.0 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph exec id query");
#endif
}
