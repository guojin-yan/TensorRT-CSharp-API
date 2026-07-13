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

JYPPX_StatusCode jyppx_cuda_validate_graph_edge_data(const JYPPX_CudaGraphEdgeData* edge_data, const char* name)
{
    if (edge_data == nullptr)
    {
        jyppx::cuda::set_cuda_error(name, 0, "invalid-graph-edge-data", "CUDA graph edge data must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (edge_data->type > 1)
    {
        jyppx::cuda::set_cuda_error(name, 0, "invalid-graph-edge-type", "CUDA graph edge dependency type must be Default or Programmatic.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (edge_data->reserved0 != 0 || edge_data->reserved1 != 0 || edge_data->reserved2 != 0 || edge_data->reserved3 != 0 || edge_data->reserved4 != 0)
    {
        jyppx::cuda::set_cuda_error(name, 0, "invalid-graph-edge-reserved", "CUDA graph edge data reserved bytes must be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_cuda_graph_edge_data_not_supported(const char* operation)
{
    jyppx::cuda::set_cuda_error(operation, 0, "cuda-version-not-supported", "CUDA graph edge data APIs require CUDA runtime 12.9 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12090
cudaGraphEdgeData jyppx_cuda_make_graph_edge_data(const JYPPX_CudaGraphEdgeData& edge_data)
{
    cudaGraphEdgeData value{};
    value.from_port = edge_data.from_port;
    value.to_port = edge_data.to_port;
    value.type = edge_data.type;
    value.reserved[0] = edge_data.reserved0;
    value.reserved[1] = edge_data.reserved1;
    value.reserved[2] = edge_data.reserved2;
    value.reserved[3] = edge_data.reserved3;
    value.reserved[4] = edge_data.reserved4;
    return value;
}

void jyppx_cuda_copy_graph_edge_data(const cudaGraphEdgeData& source, JYPPX_CudaGraphEdgeData* destination)
{
    destination->from_port = source.from_port;
    destination->to_port = source.to_port;
    destination->type = source.type;
    destination->reserved0 = source.reserved[0];
    destination->reserved1 = source.reserved[1];
    destination->reserved2 = source.reserved[2];
    destination->reserved3 = source.reserved[3];
    destination->reserved4 = source.reserved[4];
}
#endif

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

JYPPX_StatusCode jyppx_cuda_graph_add_dependency_v2_safe(
    JYPPX_CudaGraph* graph,
    uintptr_t from_node,
    uintptr_t to_node,
    const JYPPX_CudaGraphEdgeData* edge_data)
{
    auto status = jyppx_cuda_validate_graph_node_token(from_node, "from_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_cuda_validate_graph_node_token(to_node, "to_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_cuda_validate_graph_edge_data(edge_data, "edge_data");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12090
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    cudaGraphNode_t from = jyppx_cuda_node_from_token(from_node);
    cudaGraphNode_t to = jyppx_cuda_node_from_token(to_node);
    cudaGraphEdgeData cuda_edge_data = jyppx_cuda_make_graph_edge_data(*edge_data);
#if CUDART_VERSION >= 13000
    return jyppx::cuda::map_cuda_status(cudaGraphAddDependencies(graph_object->handle, &from, &to, &cuda_edge_data, 1), "cudaGraphAddDependencies");
#else
    return jyppx::cuda::map_cuda_status(cudaGraphAddDependencies_v2(graph_object->handle, &from, &to, &cuda_edge_data, 1), "cudaGraphAddDependencies_v2");
#endif
#else
    return jyppx_cuda_graph_edge_data_not_supported("cudaGraphAddDependencies_v2");
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph dependency edge data add");
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

JYPPX_StatusCode jyppx_cuda_graph_get_edges_v2_count_safe(JYPPX_CudaGraph* graph, size_t* out_count)
{
    auto status = jyppx::cuda::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    return jyppx::cuda::map_cuda_status(cudaGraphGetEdges(graph_object->handle, nullptr, nullptr, nullptr, out_count), "cudaGraphGetEdges");
#elif defined(CUDART_VERSION) && CUDART_VERSION >= 12090
    return jyppx::cuda::map_cuda_status(cudaGraphGetEdges_v2(graph_object->handle, nullptr, nullptr, nullptr, out_count), "cudaGraphGetEdges_v2");
#else
    return jyppx_cuda_graph_edge_data_not_supported("cudaGraphGetEdges_v2");
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph edge data edge count query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_get_edge_v2_safe(
    JYPPX_CudaGraph* graph,
    size_t index,
    uintptr_t* out_from_node,
    uintptr_t* out_to_node,
    JYPPX_CudaGraphEdgeData* out_edge_data)
{
    auto status = jyppx::cuda::validate_output_pointer(out_from_node, "out_from_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_to_node, "out_to_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_edge_data, "out_edge_data");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_from_node = 0;
    *out_to_node = 0;
    *out_edge_data = JYPPX_CudaGraphEdgeData{};
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }

    size_t count = 0;
    status = jyppx_cuda_graph_get_edges_v2_count_safe(graph, &count);
    if (status != JYPPX_STATUS_OK) { return status; }
    if (index >= count)
    {
        jyppx::cuda::set_cuda_error("cudaGraphGetEdges_v2", 0, "index-out-of-range", "CUDA graph edge index is outside the current edge count.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12090
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    std::vector<cudaGraphNode_t> from_nodes(count);
    std::vector<cudaGraphNode_t> to_nodes(count);
    std::vector<cudaGraphEdgeData> edge_data(count);
#if CUDART_VERSION >= 13000
    status = jyppx::cuda::map_cuda_status(cudaGraphGetEdges(graph_object->handle, from_nodes.data(), to_nodes.data(), edge_data.data(), &count), "cudaGraphGetEdges");
#else
    status = jyppx::cuda::map_cuda_status(cudaGraphGetEdges_v2(graph_object->handle, from_nodes.data(), to_nodes.data(), edge_data.data(), &count), "cudaGraphGetEdges_v2");
#endif
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_from_node = jyppx_cuda_node_to_token(from_nodes[index]);
    *out_to_node = jyppx_cuda_node_to_token(to_nodes[index]);
    jyppx_cuda_copy_graph_edge_data(edge_data[index], out_edge_data);
    return JYPPX_STATUS_OK;
#else
    (void)index;
    return jyppx_cuda_graph_edge_data_not_supported("cudaGraphGetEdges_v2");
#endif
#else
    (void)index;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph edge data edge query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_debug_dot_print_safe(JYPPX_CudaGraph* graph, const char* path, uint32_t flags)
{
    auto status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (path == nullptr || path[0] == '\0')
    {
        jyppx::cuda::set_cuda_error("cudaGraphDebugDotPrint", 0, "invalid-path", "CUDA graph debug DOT output path must not be null or empty.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    return jyppx::cuda::map_cuda_status(cudaGraphDebugDotPrint(graph_object->handle, path, flags), "cudaGraphDebugDotPrint");
#else
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph debug DOT export");
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

JYPPX_StatusCode jyppx_cuda_graph_memset_node_get_params_safe(uintptr_t node, JYPPX_CudaGraphMemsetNodeParams* out_params)
{
    auto status = jyppx::cuda::validate_output_pointer(out_params, "out_params");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_params = JYPPX_CudaGraphMemsetNodeParams{};
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemsetParams params{};
    status = jyppx::cuda::map_cuda_status(cudaGraphMemsetNodeGetParams(jyppx_cuda_node_from_token(node), &params), "cudaGraphMemsetNodeGetParams");
    if (status != JYPPX_STATUS_OK) { return status; }
    out_params->destination_address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(params.dst));
    out_params->pitch = static_cast<uint64_t>(params.pitch);
    out_params->value = params.value;
    out_params->element_size = params.elementSize;
    out_params->width = static_cast<uint64_t>(params.width);
    out_params->height = static_cast<uint64_t>(params.height);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph memset node parameter query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_memset_node_set_params_safe(uintptr_t node, JYPPX_CudaMemory* destination, uint32_t value, size_t count)
{
    auto status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (value > 0xFFU)
    {
        jyppx::cuda::set_cuda_error("cudaGraphMemsetNodeSetParams", 0, "invalid-memset-value", "CUDA graph memset node byte value must be between 0 and 255.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (count == 0)
    {
        jyppx::cuda::set_cuda_error("cudaGraphMemsetNodeSetParams", 0, "invalid-memset-size", "CUDA graph memset node byte count must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* memory_object = reinterpret_cast<MemoryObject*>(destination);
    if (count > memory_object->size)
    {
        jyppx::cuda::set_cuda_error("cudaGraphMemsetNodeSetParams", 0, "size-out-of-range", "CUDA graph memset node byte count exceeds allocated device memory.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemsetParams params{};
    params.dst = memory_object->pointer;
    params.pitch = count;
    params.value = value;
    params.elementSize = 1;
    params.width = count;
    params.height = 1;
    return jyppx::cuda::map_cuda_status(cudaGraphMemsetNodeSetParams(jyppx_cuda_node_from_token(node), &params), "cudaGraphMemsetNodeSetParams");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph memset node parameter update");
#endif
}

JYPPX_StatusCode jyppx_cuda_validate_graph_memcpy_1d_count(size_t count, size_t destination_size, size_t source_size, const char* operation)
{
    if (count == 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-memcpy-size", "CUDA graph memcpy node byte count must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (count > destination_size)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "destination-size-out-of-range", "CUDA graph memcpy node byte count exceeds destination allocation size.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (count > source_size)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "source-size-out-of-range", "CUDA graph memcpy node byte count exceeds source allocation size.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

#if JYPPX_HAS_CUDA_TOOLKIT
JYPPX_StatusCode jyppx_cuda_graph_add_memcpy_node_1d_impl(
    JYPPX_CudaGraph* graph,
    uintptr_t dependency_node,
    void* destination,
    const void* source,
    size_t count,
    cudaMemcpyKind kind,
    uintptr_t* out_node)
{
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    cudaGraphNode_t dependency = dependency_node == 0 ? nullptr : jyppx_cuda_node_from_token(dependency_node);
    const cudaGraphNode_t* dependencies = dependency_node == 0 ? nullptr : &dependency;
    size_t dependency_count = dependency_node == 0 ? 0U : 1U;
    cudaGraphNode_t node = nullptr;
    auto status = jyppx::cuda::map_cuda_status(
        cudaGraphAddMemcpyNode1D(&node, graph_object->handle, dependencies, dependency_count, destination, source, count, kind),
        "cudaGraphAddMemcpyNode1D");
    if (status != JYPPX_STATUS_OK) { return status; }

    *out_node = jyppx_cuda_node_to_token(node);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_cuda_graph_memcpy_node_set_params_1d_impl(uintptr_t node, void* destination, const void* source, size_t count, cudaMemcpyKind kind)
{
    return jyppx::cuda::map_cuda_status(
        cudaGraphMemcpyNodeSetParams1D(jyppx_cuda_node_from_token(node), destination, source, count, kind),
        "cudaGraphMemcpyNodeSetParams1D");
}

JYPPX_StatusCode jyppx_cuda_graph_exec_memcpy_node_set_params_1d_impl(
    JYPPX_CudaGraphExec* graph_exec,
    uintptr_t node,
    void* destination,
    const void* source,
    size_t count,
    cudaMemcpyKind kind)
{
    auto* exec_object = reinterpret_cast<GraphExecObject*>(graph_exec);
    return jyppx::cuda::map_cuda_status(
        cudaGraphExecMemcpyNodeSetParams1D(exec_object->handle, jyppx_cuda_node_from_token(node), destination, source, count, kind),
        "cudaGraphExecMemcpyNodeSetParams1D");
}
#endif

JYPPX_StatusCode jyppx_cuda_graph_add_memcpy_node_1d_device_to_device_safe(
    JYPPX_CudaGraph* graph,
    uintptr_t dependency_node,
    JYPPX_CudaMemory* destination,
    JYPPX_CudaMemory* source,
    size_t count,
    uintptr_t* out_node)
{
    auto status = jyppx::cuda::validate_output_pointer(out_node, "out_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = 0;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (dependency_node != 0)
    {
        status = jyppx_cuda_validate_graph_node_token(dependency_node, "dependency_node");
        if (status != JYPPX_STATUS_OK) { return status; }
    }
    status = jyppx::cuda::validate_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(source, "source");
    if (status != JYPPX_STATUS_OK) { return status; }

    auto* destination_object = reinterpret_cast<MemoryObject*>(destination);
    auto* source_object = reinterpret_cast<MemoryObject*>(source);
    status = jyppx_cuda_validate_graph_memcpy_1d_count(count, destination_object->size, source_object->size, "cudaGraphAddMemcpyNode1D");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx_cuda_graph_add_memcpy_node_1d_impl(graph, dependency_node, destination_object->pointer, source_object->pointer, count, cudaMemcpyDeviceToDevice, out_node);
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph memcpy 1D device-to-device node add");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_add_memcpy_node_1d_host_to_device_safe(
    JYPPX_CudaGraph* graph,
    uintptr_t dependency_node,
    JYPPX_CudaMemory* destination,
    JYPPX_CudaPinnedMemory* source,
    size_t count,
    uintptr_t* out_node)
{
    auto status = jyppx::cuda::validate_output_pointer(out_node, "out_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = 0;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (dependency_node != 0)
    {
        status = jyppx_cuda_validate_graph_node_token(dependency_node, "dependency_node");
        if (status != JYPPX_STATUS_OK) { return status; }
    }
    status = jyppx::cuda::validate_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_pinned_memory(source, "source");
    if (status != JYPPX_STATUS_OK) { return status; }

    auto* destination_object = reinterpret_cast<MemoryObject*>(destination);
    auto* source_object = reinterpret_cast<PinnedMemoryObject*>(source);
    status = jyppx_cuda_validate_graph_memcpy_1d_count(count, destination_object->size, source_object->size, "cudaGraphAddMemcpyNode1D");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx_cuda_graph_add_memcpy_node_1d_impl(graph, dependency_node, destination_object->pointer, source_object->pointer, count, cudaMemcpyHostToDevice, out_node);
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph memcpy 1D host-to-device node add");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_add_memcpy_node_1d_device_to_host_safe(
    JYPPX_CudaGraph* graph,
    uintptr_t dependency_node,
    JYPPX_CudaPinnedMemory* destination,
    JYPPX_CudaMemory* source,
    size_t count,
    uintptr_t* out_node)
{
    auto status = jyppx::cuda::validate_output_pointer(out_node, "out_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = 0;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (dependency_node != 0)
    {
        status = jyppx_cuda_validate_graph_node_token(dependency_node, "dependency_node");
        if (status != JYPPX_STATUS_OK) { return status; }
    }
    status = jyppx::cuda::validate_pinned_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(source, "source");
    if (status != JYPPX_STATUS_OK) { return status; }

    auto* destination_object = reinterpret_cast<PinnedMemoryObject*>(destination);
    auto* source_object = reinterpret_cast<MemoryObject*>(source);
    status = jyppx_cuda_validate_graph_memcpy_1d_count(count, destination_object->size, source_object->size, "cudaGraphAddMemcpyNode1D");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx_cuda_graph_add_memcpy_node_1d_impl(graph, dependency_node, destination_object->pointer, source_object->pointer, count, cudaMemcpyDeviceToHost, out_node);
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph memcpy 1D device-to-host node add");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_memcpy_node_get_params_safe(uintptr_t node, JYPPX_CudaGraphMemcpyNodeParams* out_params)
{
    auto status = jyppx::cuda::validate_output_pointer(out_params, "out_params");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_params = JYPPX_CudaGraphMemcpyNodeParams{};
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemcpy3DParms params{};
    status = jyppx::cuda::map_cuda_status(cudaGraphMemcpyNodeGetParams(jyppx_cuda_node_from_token(node), &params), "cudaGraphMemcpyNodeGetParams");
    if (status != JYPPX_STATUS_OK) { return status; }

    out_params->source_address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(params.srcPtr.ptr));
    out_params->destination_address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(params.dstPtr.ptr));
    out_params->source_pitch = static_cast<uint64_t>(params.srcPtr.pitch);
    out_params->destination_pitch = static_cast<uint64_t>(params.dstPtr.pitch);
    out_params->source_x_size = static_cast<uint64_t>(params.srcPtr.xsize);
    out_params->source_y_size = static_cast<uint64_t>(params.srcPtr.ysize);
    out_params->destination_x_size = static_cast<uint64_t>(params.dstPtr.xsize);
    out_params->destination_y_size = static_cast<uint64_t>(params.dstPtr.ysize);
    out_params->source_position_x = static_cast<uint64_t>(params.srcPos.x);
    out_params->source_position_y = static_cast<uint64_t>(params.srcPos.y);
    out_params->source_position_z = static_cast<uint64_t>(params.srcPos.z);
    out_params->destination_position_x = static_cast<uint64_t>(params.dstPos.x);
    out_params->destination_position_y = static_cast<uint64_t>(params.dstPos.y);
    out_params->destination_position_z = static_cast<uint64_t>(params.dstPos.z);
    out_params->width = static_cast<uint64_t>(params.extent.width);
    out_params->height = static_cast<uint64_t>(params.extent.height);
    out_params->depth = static_cast<uint64_t>(params.extent.depth);
    out_params->kind = static_cast<int32_t>(params.kind);
    out_params->source_is_array = params.srcArray == nullptr ? 0U : 1U;
    out_params->destination_is_array = params.dstArray == nullptr ? 0U : 1U;
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph memcpy node parameter query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_memcpy_node_set_params_1d_device_to_device_safe(
    uintptr_t node,
    JYPPX_CudaMemory* destination,
    JYPPX_CudaMemory* source,
    size_t count)
{
    auto status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(source, "source");
    if (status != JYPPX_STATUS_OK) { return status; }

    auto* destination_object = reinterpret_cast<MemoryObject*>(destination);
    auto* source_object = reinterpret_cast<MemoryObject*>(source);
    status = jyppx_cuda_validate_graph_memcpy_1d_count(count, destination_object->size, source_object->size, "cudaGraphMemcpyNodeSetParams1D");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx_cuda_graph_memcpy_node_set_params_1d_impl(node, destination_object->pointer, source_object->pointer, count, cudaMemcpyDeviceToDevice);
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph memcpy 1D device-to-device node update");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_memcpy_node_set_params_1d_host_to_device_safe(
    uintptr_t node,
    JYPPX_CudaMemory* destination,
    JYPPX_CudaPinnedMemory* source,
    size_t count)
{
    auto status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_pinned_memory(source, "source");
    if (status != JYPPX_STATUS_OK) { return status; }

    auto* destination_object = reinterpret_cast<MemoryObject*>(destination);
    auto* source_object = reinterpret_cast<PinnedMemoryObject*>(source);
    status = jyppx_cuda_validate_graph_memcpy_1d_count(count, destination_object->size, source_object->size, "cudaGraphMemcpyNodeSetParams1D");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx_cuda_graph_memcpy_node_set_params_1d_impl(node, destination_object->pointer, source_object->pointer, count, cudaMemcpyHostToDevice);
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph memcpy 1D host-to-device node update");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_memcpy_node_set_params_1d_device_to_host_safe(
    uintptr_t node,
    JYPPX_CudaPinnedMemory* destination,
    JYPPX_CudaMemory* source,
    size_t count)
{
    auto status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_pinned_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(source, "source");
    if (status != JYPPX_STATUS_OK) { return status; }

    auto* destination_object = reinterpret_cast<PinnedMemoryObject*>(destination);
    auto* source_object = reinterpret_cast<MemoryObject*>(source);
    status = jyppx_cuda_validate_graph_memcpy_1d_count(count, destination_object->size, source_object->size, "cudaGraphMemcpyNodeSetParams1D");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx_cuda_graph_memcpy_node_set_params_1d_impl(node, destination_object->pointer, source_object->pointer, count, cudaMemcpyDeviceToHost);
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph memcpy 1D device-to-host node update");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_exec_memcpy_node_set_params_1d_device_to_device_safe(
    JYPPX_CudaGraphExec* graph_exec,
    uintptr_t node,
    JYPPX_CudaMemory* destination,
    JYPPX_CudaMemory* source,
    size_t count)
{
    auto status = jyppx::cuda::validate_graph_exec(graph_exec, "graph_exec");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(source, "source");
    if (status != JYPPX_STATUS_OK) { return status; }

    auto* destination_object = reinterpret_cast<MemoryObject*>(destination);
    auto* source_object = reinterpret_cast<MemoryObject*>(source);
    status = jyppx_cuda_validate_graph_memcpy_1d_count(count, destination_object->size, source_object->size, "cudaGraphExecMemcpyNodeSetParams1D");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx_cuda_graph_exec_memcpy_node_set_params_1d_impl(graph_exec, node, destination_object->pointer, source_object->pointer, count, cudaMemcpyDeviceToDevice);
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph exec memcpy 1D device-to-device node update");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_exec_memcpy_node_set_params_1d_host_to_device_safe(
    JYPPX_CudaGraphExec* graph_exec,
    uintptr_t node,
    JYPPX_CudaMemory* destination,
    JYPPX_CudaPinnedMemory* source,
    size_t count)
{
    auto status = jyppx::cuda::validate_graph_exec(graph_exec, "graph_exec");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_pinned_memory(source, "source");
    if (status != JYPPX_STATUS_OK) { return status; }

    auto* destination_object = reinterpret_cast<MemoryObject*>(destination);
    auto* source_object = reinterpret_cast<PinnedMemoryObject*>(source);
    status = jyppx_cuda_validate_graph_memcpy_1d_count(count, destination_object->size, source_object->size, "cudaGraphExecMemcpyNodeSetParams1D");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx_cuda_graph_exec_memcpy_node_set_params_1d_impl(graph_exec, node, destination_object->pointer, source_object->pointer, count, cudaMemcpyHostToDevice);
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph exec memcpy 1D host-to-device node update");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_exec_memcpy_node_set_params_1d_device_to_host_safe(
    JYPPX_CudaGraphExec* graph_exec,
    uintptr_t node,
    JYPPX_CudaPinnedMemory* destination,
    JYPPX_CudaMemory* source,
    size_t count)
{
    auto status = jyppx::cuda::validate_graph_exec(graph_exec, "graph_exec");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_pinned_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_memory(source, "source");
    if (status != JYPPX_STATUS_OK) { return status; }

    auto* destination_object = reinterpret_cast<PinnedMemoryObject*>(destination);
    auto* source_object = reinterpret_cast<MemoryObject*>(source);
    status = jyppx_cuda_validate_graph_memcpy_1d_count(count, destination_object->size, source_object->size, "cudaGraphExecMemcpyNodeSetParams1D");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx_cuda_graph_exec_memcpy_node_set_params_1d_impl(graph_exec, node, destination_object->pointer, source_object->pointer, count, cudaMemcpyDeviceToHost);
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph exec memcpy 1D device-to-host node update");
#endif
}

constexpr int32_t kJyppxCudaKernelNodeAttributeCooperative = 2;
constexpr int32_t kJyppxCudaKernelNodeAttributeClusterDimension = 4;
constexpr int32_t kJyppxCudaKernelNodeAttributeClusterSchedulingPolicyPreference = 5;
constexpr int32_t kJyppxCudaKernelNodeAttributePriority = 8;

JYPPX_StatusCode jyppx_cuda_graph_kernel_node_attribute_not_supported(const char* operation, int32_t attribute)
{
    jyppx::cuda::set_cuda_error(operation, attribute, "unsupported-kernel-node-attribute", "Only cooperative, priority, cluster dimension, and cluster scheduling policy preference are supported by the scalar CUDA graph kernel node attribute bridge.");
    return JYPPX_STATUS_INVALID_ARGUMENT;
}

JYPPX_StatusCode jyppx_cuda_graph_kernel_node_attribute_version_not_supported(const char* operation, int32_t attribute)
{
    jyppx::cuda::set_cuda_error(operation, attribute, "cuda-version-not-supported", "This CUDA graph kernel node attribute requires CUDA runtime 11.8 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}

JYPPX_StatusCode jyppx_cuda_validate_graph_kernel_node_attribute_value(const JYPPX_CudaGraphKernelNodeAttributeValue* value, const char* operation)
{
    if (value == nullptr)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-kernel-node-attribute-value", "CUDA graph kernel node attribute value must not be null.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    if (value->reserved0 != 0 || value->reserved1 != 0 || value->reserved2 != 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-kernel-node-attribute-reserved", "CUDA graph kernel node attribute reserved fields must be zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    switch (value->attribute)
    {
    case kJyppxCudaKernelNodeAttributeCooperative:
        if ((value->int_value != 0 && value->int_value != 1) || value->x != 0 || value->y != 0 || value->z != 0)
        {
            jyppx::cuda::set_cuda_error(operation, value->attribute, "invalid-cooperative-attribute", "Cooperative kernel node attribute must use int_value 0 or 1 and zero vector fields.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }
        return JYPPX_STATUS_OK;

    case kJyppxCudaKernelNodeAttributePriority:
        if (value->x != 0 || value->y != 0 || value->z != 0)
        {
            jyppx::cuda::set_cuda_error(operation, value->attribute, "invalid-priority-attribute", "Priority kernel node attribute must use int_value and zero vector fields.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }
        return JYPPX_STATUS_OK;

    case kJyppxCudaKernelNodeAttributeClusterDimension:
        if (value->int_value != 0 || value->x == 0 || value->y == 0 || value->z == 0)
        {
            jyppx::cuda::set_cuda_error(operation, value->attribute, "invalid-cluster-dimension-attribute", "Cluster dimension kernel node attribute must use positive x/y/z fields and zero int_value.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }
        return JYPPX_STATUS_OK;

    case kJyppxCudaKernelNodeAttributeClusterSchedulingPolicyPreference:
        if (value->int_value < 0 || value->int_value > 2 || value->x != 0 || value->y != 0 || value->z != 0)
        {
            jyppx::cuda::set_cuda_error(operation, value->attribute, "invalid-cluster-scheduling-policy-attribute", "Cluster scheduling policy preference must use int_value 0, 1, or 2 and zero vector fields.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }
        return JYPPX_STATUS_OK;

    default:
        return jyppx_cuda_graph_kernel_node_attribute_not_supported(operation, value->attribute);
    }
}

JYPPX_StatusCode jyppx_cuda_graph_kernel_node_get_attribute_scalar_safe(
    uintptr_t node,
    int32_t attribute,
    JYPPX_CudaGraphKernelNodeAttributeValue* out_value)
{
    auto status = jyppx::cuda::validate_output_pointer(out_value, "out_value");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_value = JYPPX_CudaGraphKernelNodeAttributeValue{};
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaKernelNodeAttrValue native_value{};
    cudaGraphNode_t graph_node = jyppx_cuda_node_from_token(node);

    switch (attribute)
    {
    case kJyppxCudaKernelNodeAttributeCooperative:
        status = jyppx::cuda::map_cuda_status(cudaGraphKernelNodeGetAttribute(graph_node, cudaKernelNodeAttributeCooperative, &native_value), "cudaGraphKernelNodeGetAttribute");
        if (status != JYPPX_STATUS_OK) { return status; }
        out_value->attribute = attribute;
        out_value->int_value = native_value.cooperative;
        return JYPPX_STATUS_OK;

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
    case kJyppxCudaKernelNodeAttributePriority:
        status = jyppx::cuda::map_cuda_status(cudaGraphKernelNodeGetAttribute(graph_node, cudaKernelNodeAttributePriority, &native_value), "cudaGraphKernelNodeGetAttribute");
        if (status != JYPPX_STATUS_OK) { return status; }
        out_value->attribute = attribute;
        out_value->int_value = native_value.priority;
        return JYPPX_STATUS_OK;

    case kJyppxCudaKernelNodeAttributeClusterDimension:
        status = jyppx::cuda::map_cuda_status(cudaGraphKernelNodeGetAttribute(graph_node, cudaKernelNodeAttributeClusterDimension, &native_value), "cudaGraphKernelNodeGetAttribute");
        if (status != JYPPX_STATUS_OK) { return status; }
        out_value->attribute = attribute;
        out_value->x = native_value.clusterDim.x;
        out_value->y = native_value.clusterDim.y;
        out_value->z = native_value.clusterDim.z;
        return JYPPX_STATUS_OK;

    case kJyppxCudaKernelNodeAttributeClusterSchedulingPolicyPreference:
        status = jyppx::cuda::map_cuda_status(cudaGraphKernelNodeGetAttribute(graph_node, cudaKernelNodeAttributeClusterSchedulingPolicyPreference, &native_value), "cudaGraphKernelNodeGetAttribute");
        if (status != JYPPX_STATUS_OK) { return status; }
        out_value->attribute = attribute;
        out_value->int_value = static_cast<int32_t>(native_value.clusterSchedulingPolicyPreference);
        return JYPPX_STATUS_OK;
#else
    case kJyppxCudaKernelNodeAttributePriority:
    case kJyppxCudaKernelNodeAttributeClusterDimension:
    case kJyppxCudaKernelNodeAttributeClusterSchedulingPolicyPreference:
        return jyppx_cuda_graph_kernel_node_attribute_version_not_supported("cudaGraphKernelNodeGetAttribute", attribute);
#endif

    default:
        return jyppx_cuda_graph_kernel_node_attribute_not_supported("cudaGraphKernelNodeGetAttribute", attribute);
    }
#else
    (void)attribute;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph kernel node scalar attribute query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_kernel_node_set_attribute_scalar_safe(uintptr_t node, const JYPPX_CudaGraphKernelNodeAttributeValue* value)
{
    auto status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_cuda_validate_graph_kernel_node_attribute_value(value, "cudaGraphKernelNodeSetAttribute");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaKernelNodeAttrValue native_value{};
    cudaGraphNode_t graph_node = jyppx_cuda_node_from_token(node);

    switch (value->attribute)
    {
    case kJyppxCudaKernelNodeAttributeCooperative:
        native_value.cooperative = value->int_value;
        return jyppx::cuda::map_cuda_status(cudaGraphKernelNodeSetAttribute(graph_node, cudaKernelNodeAttributeCooperative, &native_value), "cudaGraphKernelNodeSetAttribute");

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
    case kJyppxCudaKernelNodeAttributePriority:
        native_value.priority = value->int_value;
        return jyppx::cuda::map_cuda_status(cudaGraphKernelNodeSetAttribute(graph_node, cudaKernelNodeAttributePriority, &native_value), "cudaGraphKernelNodeSetAttribute");

    case kJyppxCudaKernelNodeAttributeClusterDimension:
        native_value.clusterDim.x = value->x;
        native_value.clusterDim.y = value->y;
        native_value.clusterDim.z = value->z;
        return jyppx::cuda::map_cuda_status(cudaGraphKernelNodeSetAttribute(graph_node, cudaKernelNodeAttributeClusterDimension, &native_value), "cudaGraphKernelNodeSetAttribute");

    case kJyppxCudaKernelNodeAttributeClusterSchedulingPolicyPreference:
        native_value.clusterSchedulingPolicyPreference = static_cast<cudaClusterSchedulingPolicy>(value->int_value);
        return jyppx::cuda::map_cuda_status(cudaGraphKernelNodeSetAttribute(graph_node, cudaKernelNodeAttributeClusterSchedulingPolicyPreference, &native_value), "cudaGraphKernelNodeSetAttribute");
#else
    case kJyppxCudaKernelNodeAttributePriority:
    case kJyppxCudaKernelNodeAttributeClusterDimension:
    case kJyppxCudaKernelNodeAttributeClusterSchedulingPolicyPreference:
        return jyppx_cuda_graph_kernel_node_attribute_version_not_supported("cudaGraphKernelNodeSetAttribute", value->attribute);
#endif

    default:
        return jyppx_cuda_graph_kernel_node_attribute_not_supported("cudaGraphKernelNodeSetAttribute", value->attribute);
    }
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph kernel node scalar attribute update");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_add_event_node_safe(
    JYPPX_CudaGraph* graph,
    uintptr_t dependency_node,
    JYPPX_CudaEvent* event_handle,
    uintptr_t* out_node,
    bool record_node)
{
    auto status = jyppx::cuda::validate_output_pointer(out_node, "out_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = 0;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_event(event_handle, "event_handle");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (dependency_node != 0)
    {
        status = jyppx_cuda_validate_graph_node_token(dependency_node, "dependency_node");
        if (status != JYPPX_STATUS_OK) { return status; }
    }

#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11010
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    auto* event_object = reinterpret_cast<EventObject*>(event_handle);
    cudaGraphNode_t dependency = dependency_node == 0 ? nullptr : jyppx_cuda_node_from_token(dependency_node);
    const cudaGraphNode_t* dependencies = dependency_node == 0 ? nullptr : &dependency;
    size_t dependency_count = dependency_node == 0 ? 0U : 1U;
    cudaGraphNode_t node = nullptr;
    status = record_node
        ? jyppx::cuda::map_cuda_status(cudaGraphAddEventRecordNode(&node, graph_object->handle, dependencies, dependency_count, event_object->handle), "cudaGraphAddEventRecordNode")
        : jyppx::cuda::map_cuda_status(cudaGraphAddEventWaitNode(&node, graph_object->handle, dependencies, dependency_count, event_object->handle), "cudaGraphAddEventWaitNode");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = jyppx_cuda_node_to_token(node);
    return JYPPX_STATUS_OK;
#else
    jyppx::cuda::set_cuda_error(record_node ? "cudaGraphAddEventRecordNode" : "cudaGraphAddEventWaitNode", 0, "cuda-version-not-supported", "CUDA graph event nodes require CUDA runtime 11.1 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#endif
#else
    (void)record_node;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph event node add");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_add_event_record_node_safe(JYPPX_CudaGraph* graph, uintptr_t dependency_node, JYPPX_CudaEvent* event_handle, uintptr_t* out_node)
{
    return jyppx_cuda_graph_add_event_node_safe(graph, dependency_node, event_handle, out_node, true);
}

JYPPX_StatusCode jyppx_cuda_graph_add_event_wait_node_safe(JYPPX_CudaGraph* graph, uintptr_t dependency_node, JYPPX_CudaEvent* event_handle, uintptr_t* out_node)
{
    return jyppx_cuda_graph_add_event_node_safe(graph, dependency_node, event_handle, out_node, false);
}

JYPPX_StatusCode jyppx_cuda_graph_event_record_node_set_event_safe(uintptr_t node, JYPPX_CudaEvent* event_handle)
{
    auto status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_event(event_handle, "event_handle");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11010
    auto* event_object = reinterpret_cast<EventObject*>(event_handle);
    return jyppx::cuda::map_cuda_status(cudaGraphEventRecordNodeSetEvent(jyppx_cuda_node_from_token(node), event_object->handle), "cudaGraphEventRecordNodeSetEvent");
#else
    jyppx::cuda::set_cuda_error("cudaGraphEventRecordNodeSetEvent", 0, "cuda-version-not-supported", "CUDA graph event node updates require CUDA runtime 11.1 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph event record node update");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_event_record_node_has_event_safe(uintptr_t node, JYPPX_Boolean* out_has_event)
{
    auto status = jyppx::cuda::validate_output_pointer(out_has_event, "out_has_event");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_event = JYPPX_FALSE;
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11010
    cudaEvent_t event_handle{};
    status = jyppx::cuda::map_cuda_status(cudaGraphEventRecordNodeGetEvent(jyppx_cuda_node_from_token(node), &event_handle), "cudaGraphEventRecordNodeGetEvent");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_event = event_handle != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    jyppx::cuda::set_cuda_error("cudaGraphEventRecordNodeGetEvent", 0, "cuda-version-not-supported", "CUDA graph event node queries require CUDA runtime 11.1 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph event record node event presence query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_event_wait_node_set_event_safe(uintptr_t node, JYPPX_CudaEvent* event_handle)
{
    auto status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_event(event_handle, "event_handle");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11010
    auto* event_object = reinterpret_cast<EventObject*>(event_handle);
    return jyppx::cuda::map_cuda_status(cudaGraphEventWaitNodeSetEvent(jyppx_cuda_node_from_token(node), event_object->handle), "cudaGraphEventWaitNodeSetEvent");
#else
    jyppx::cuda::set_cuda_error("cudaGraphEventWaitNodeSetEvent", 0, "cuda-version-not-supported", "CUDA graph event node updates require CUDA runtime 11.1 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph event wait node update");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_event_wait_node_has_event_safe(uintptr_t node, JYPPX_Boolean* out_has_event)
{
    auto status = jyppx::cuda::validate_output_pointer(out_has_event, "out_has_event");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_event = JYPPX_FALSE;
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11010
    cudaEvent_t event_handle{};
    status = jyppx::cuda::map_cuda_status(cudaGraphEventWaitNodeGetEvent(jyppx_cuda_node_from_token(node), &event_handle), "cudaGraphEventWaitNodeGetEvent");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_event = event_handle != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    jyppx::cuda::set_cuda_error("cudaGraphEventWaitNodeGetEvent", 0, "cuda-version-not-supported", "CUDA graph event node queries require CUDA runtime 11.1 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph event wait node event presence query");
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

JYPPX_StatusCode jyppx_cuda_graph_node_get_dependencies_v2_count_safe(uintptr_t node, size_t* out_count)
{
    auto status = jyppx::cuda::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    return jyppx::cuda::map_cuda_status(cudaGraphNodeGetDependencies(jyppx_cuda_node_from_token(node), nullptr, nullptr, out_count), "cudaGraphNodeGetDependencies");
#elif defined(CUDART_VERSION) && CUDART_VERSION >= 12090
    return jyppx::cuda::map_cuda_status(cudaGraphNodeGetDependencies_v2(jyppx_cuda_node_from_token(node), nullptr, nullptr, out_count), "cudaGraphNodeGetDependencies_v2");
#else
    return jyppx_cuda_graph_edge_data_not_supported("cudaGraphNodeGetDependencies_v2");
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph node dependency edge data count query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_node_get_dependency_v2_safe(uintptr_t node, size_t index, uintptr_t* out_dependency_node, JYPPX_CudaGraphEdgeData* out_edge_data)
{
    auto status = jyppx::cuda::validate_output_pointer(out_dependency_node, "out_dependency_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_edge_data, "out_edge_data");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_dependency_node = 0;
    *out_edge_data = JYPPX_CudaGraphEdgeData{};

    size_t count = 0;
    status = jyppx_cuda_graph_node_get_dependencies_v2_count_safe(node, &count);
    if (status != JYPPX_STATUS_OK) { return status; }
    if (index >= count)
    {
        jyppx::cuda::set_cuda_error("cudaGraphNodeGetDependencies_v2", 0, "index-out-of-range", "CUDA graph node dependency index is outside the dependency count.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12090
    std::vector<cudaGraphNode_t> nodes(count);
    std::vector<cudaGraphEdgeData> edge_data(count);
#if CUDART_VERSION >= 13000
    status = jyppx::cuda::map_cuda_status(cudaGraphNodeGetDependencies(jyppx_cuda_node_from_token(node), nodes.data(), edge_data.data(), &count), "cudaGraphNodeGetDependencies");
#else
    status = jyppx::cuda::map_cuda_status(cudaGraphNodeGetDependencies_v2(jyppx_cuda_node_from_token(node), nodes.data(), edge_data.data(), &count), "cudaGraphNodeGetDependencies_v2");
#endif
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_dependency_node = jyppx_cuda_node_to_token(nodes[index]);
    jyppx_cuda_copy_graph_edge_data(edge_data[index], out_edge_data);
    return JYPPX_STATUS_OK;
#else
    (void)index;
    return jyppx_cuda_graph_edge_data_not_supported("cudaGraphNodeGetDependencies_v2");
#endif
#else
    (void)index;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph node dependency edge data query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_node_get_dependent_nodes_v2_count_safe(uintptr_t node, size_t* out_count)
{
    auto status = jyppx::cuda::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    return jyppx::cuda::map_cuda_status(cudaGraphNodeGetDependentNodes(jyppx_cuda_node_from_token(node), nullptr, nullptr, out_count), "cudaGraphNodeGetDependentNodes");
#elif defined(CUDART_VERSION) && CUDART_VERSION >= 12090
    return jyppx::cuda::map_cuda_status(cudaGraphNodeGetDependentNodes_v2(jyppx_cuda_node_from_token(node), nullptr, nullptr, out_count), "cudaGraphNodeGetDependentNodes_v2");
#else
    return jyppx_cuda_graph_edge_data_not_supported("cudaGraphNodeGetDependentNodes_v2");
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph node dependent edge data count query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_node_get_dependent_node_v2_safe(uintptr_t node, size_t index, uintptr_t* out_dependent_node, JYPPX_CudaGraphEdgeData* out_edge_data)
{
    auto status = jyppx::cuda::validate_output_pointer(out_dependent_node, "out_dependent_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_edge_data, "out_edge_data");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_dependent_node = 0;
    *out_edge_data = JYPPX_CudaGraphEdgeData{};

    size_t count = 0;
    status = jyppx_cuda_graph_node_get_dependent_nodes_v2_count_safe(node, &count);
    if (status != JYPPX_STATUS_OK) { return status; }
    if (index >= count)
    {
        jyppx::cuda::set_cuda_error("cudaGraphNodeGetDependentNodes_v2", 0, "index-out-of-range", "CUDA graph node dependent index is outside the dependent count.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12090
    std::vector<cudaGraphNode_t> nodes(count);
    std::vector<cudaGraphEdgeData> edge_data(count);
#if CUDART_VERSION >= 13000
    status = jyppx::cuda::map_cuda_status(cudaGraphNodeGetDependentNodes(jyppx_cuda_node_from_token(node), nodes.data(), edge_data.data(), &count), "cudaGraphNodeGetDependentNodes");
#else
    status = jyppx::cuda::map_cuda_status(cudaGraphNodeGetDependentNodes_v2(jyppx_cuda_node_from_token(node), nodes.data(), edge_data.data(), &count), "cudaGraphNodeGetDependentNodes_v2");
#endif
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_dependent_node = jyppx_cuda_node_to_token(nodes[index]);
    jyppx_cuda_copy_graph_edge_data(edge_data[index], out_edge_data);
    return JYPPX_STATUS_OK;
#else
    (void)index;
    return jyppx_cuda_graph_edge_data_not_supported("cudaGraphNodeGetDependentNodes_v2");
#endif
#else
    (void)index;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph node dependent edge data query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_remove_dependency_v2_safe(
    JYPPX_CudaGraph* graph,
    uintptr_t from_node,
    uintptr_t to_node,
    const JYPPX_CudaGraphEdgeData* edge_data)
{
    auto status = jyppx_cuda_validate_graph_node_token(from_node, "from_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_cuda_validate_graph_node_token(to_node, "to_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_cuda_validate_graph_edge_data(edge_data, "edge_data");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12090
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    cudaGraphNode_t from = jyppx_cuda_node_from_token(from_node);
    cudaGraphNode_t to = jyppx_cuda_node_from_token(to_node);
    cudaGraphEdgeData cuda_edge_data = jyppx_cuda_make_graph_edge_data(*edge_data);
#if CUDART_VERSION >= 13000
    return jyppx::cuda::map_cuda_status(cudaGraphRemoveDependencies(graph_object->handle, &from, &to, &cuda_edge_data, 1), "cudaGraphRemoveDependencies");
#else
    return jyppx::cuda::map_cuda_status(cudaGraphRemoveDependencies_v2(graph_object->handle, &from, &to, &cuda_edge_data, 1), "cudaGraphRemoveDependencies_v2");
#endif
#else
    return jyppx_cuda_graph_edge_data_not_supported("cudaGraphRemoveDependencies_v2");
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph dependency edge data remove");
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

JYPPX_StatusCode jyppx_cuda_graph_exec_event_record_node_set_event_safe(JYPPX_CudaGraphExec* graph_exec, uintptr_t node, JYPPX_CudaEvent* event_handle)
{
    auto status = jyppx::cuda::validate_graph_exec(graph_exec, "graph_exec");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_event(event_handle, "event_handle");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11010
    auto* graph_exec_object = reinterpret_cast<GraphExecObject*>(graph_exec);
    auto* event_object = reinterpret_cast<EventObject*>(event_handle);
    return jyppx::cuda::map_cuda_status(cudaGraphExecEventRecordNodeSetEvent(graph_exec_object->handle, jyppx_cuda_node_from_token(node), event_object->handle), "cudaGraphExecEventRecordNodeSetEvent");
#else
    jyppx::cuda::set_cuda_error("cudaGraphExecEventRecordNodeSetEvent", 0, "cuda-version-not-supported", "CUDA graph exec event node updates require CUDA runtime 11.1 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph exec event record node update");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_exec_event_wait_node_set_event_safe(JYPPX_CudaGraphExec* graph_exec, uintptr_t node, JYPPX_CudaEvent* event_handle)
{
    auto status = jyppx::cuda::validate_graph_exec(graph_exec, "graph_exec");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_event(event_handle, "event_handle");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11010
    auto* graph_exec_object = reinterpret_cast<GraphExecObject*>(graph_exec);
    auto* event_object = reinterpret_cast<EventObject*>(event_handle);
    return jyppx::cuda::map_cuda_status(cudaGraphExecEventWaitNodeSetEvent(graph_exec_object->handle, jyppx_cuda_node_from_token(node), event_object->handle), "cudaGraphExecEventWaitNodeSetEvent");
#else
    jyppx::cuda::set_cuda_error("cudaGraphExecEventWaitNodeSetEvent", 0, "cuda-version-not-supported", "CUDA graph exec event node updates require CUDA runtime 11.1 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph exec event wait node update");
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
