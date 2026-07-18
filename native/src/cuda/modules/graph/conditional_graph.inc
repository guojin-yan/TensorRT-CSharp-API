namespace
{
JYPPX_StatusCode validate_conditional_body_index(
    const jyppx::cuda::GraphConditionalNodeObject* node,
    uint32_t body_index,
    const char* operation)
{
    if (body_index < node->body_count)
    {
        return JYPPX_STATUS_OK;
    }

    jyppx::cuda::set_cuda_error(operation, 0, "invalid-body-index", "The conditional body index is outside the node body range.");
    return JYPPX_STATUS_INVALID_ARGUMENT;
}

JYPPX_StatusCode validate_conditional_body_graph(
    const jyppx::cuda::GraphConditionalNodeObject* node,
    uint32_t body_index,
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaGraph_t* out_graph,
#else
    void** out_graph,
#endif
    const char* operation)
{
    auto status = validate_conditional_body_index(node, body_index, operation);
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    if (node->body_graphs == nullptr || node->body_graphs[body_index] == nullptr)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "null-body-graph", "CUDA did not return a usable conditional body graph.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    *out_graph = node->body_graphs[body_index];
    return JYPPX_STATUS_OK;
#else
    (void)out_graph;
    jyppx::cuda::set_cuda_error(operation, 0, "cuda-version-not-supported", "Conditional graph bodies require CUDA runtime 12.3 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#endif
}

JYPPX_StatusCode create_conditional_handle_object(
    JYPPX_CudaGraph* graph,
    uint64_t handle,
    uint32_t default_launch_value,
    uint32_t flags,
    JYPPX_CudaGraphConditionalHandle** out_handle)
{
    auto* graph_object = reinterpret_cast<jyppx::cuda::GraphObject*>(graph);
    auto* handle_object = new (std::nothrow) jyppx::cuda::GraphConditionalHandleObject{};
    if (handle_object == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaGraphConditionalHandleCreate", 0, "out-of-memory", "Unable to allocate bridge metadata for the CUDA conditional handle.");
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    handle_object->base.magic = jyppx::cuda::kObjectMagic;
    handle_object->base.kind = jyppx::cuda::ObjectKind::GraphConditionalHandle;
    handle_object->owner = graph_object;
    handle_object->handle = handle;
    handle_object->default_launch_value = default_launch_value;
    handle_object->flags = flags;
    graph_object->active_conditional_handles++;
    *out_handle = reinterpret_cast<JYPPX_CudaGraphConditionalHandle*>(handle_object);
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode conditional_node_body_count(
    JYPPX_CudaGraphConditionalNode* node,
    uint32_t* out_body_count)
{
    auto status = jyppx::cuda::validate_graph_conditional_node(node, "node");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_body_count = reinterpret_cast<jyppx::cuda::GraphConditionalNodeObject*>(node)->body_count;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode conditional_body_count(
    JYPPX_CudaGraphConditionalNode* node,
    uint32_t body_index,
    int32_t selector,
    size_t* out_count)
{
    auto status = jyppx::cuda::validate_graph_conditional_node(node, "node");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* node_object = reinterpret_cast<jyppx::cuda::GraphConditionalNodeObject*>(node);
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaGraph_t body_graph = nullptr;
    status = validate_conditional_body_graph(node_object, body_index, &body_graph, "cudaGraphConditionalBodyCount");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    size_t count = 0;
    if (selector == 0)
    {
        status = jyppx::cuda::map_cuda_status(cudaGraphGetNodes(body_graph, nullptr, &count), "cudaGraphGetNodes(conditional body)");
    }
    else if (selector == 1)
    {
        status = jyppx::cuda::map_cuda_status(cudaGraphGetRootNodes(body_graph, nullptr, &count), "cudaGraphGetRootNodes(conditional body)");
    }
    else
    {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
        status = jyppx::cuda::map_cuda_status(cudaGraphGetEdges(body_graph, nullptr, nullptr, nullptr, &count), "cudaGraphGetEdges(conditional body)");
#else
        status = jyppx::cuda::map_cuda_status(cudaGraphGetEdges(body_graph, nullptr, nullptr, &count), "cudaGraphGetEdges(conditional body)");
#endif
    }

    if (status == JYPPX_STATUS_OK)
    {
        *out_count = count;
    }
    return status;
#else
    (void)node_object;
    (void)body_index;
    (void)selector;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA conditional body topology query");
#endif
}
}

JYPPX_StatusCode jyppx_cuda_graph_conditional_handle_create_safe(
    JYPPX_CudaGraph* graph,
    uint32_t default_launch_value,
    uint32_t flags,
    JYPPX_CudaGraphConditionalHandle** out_handle)
{
    auto status = jyppx::cuda::validate_output_pointer(out_handle, "out_handle");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_handle = nullptr;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12030
    auto* graph_object = reinterpret_cast<jyppx::cuda::GraphObject*>(graph);
    cudaGraphConditionalHandle handle = 0;
    status = jyppx::cuda::map_cuda_status(
        cudaGraphConditionalHandleCreate(&handle, graph_object->handle, default_launch_value, flags),
        "cudaGraphConditionalHandleCreate");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_conditional_handle_object(graph, static_cast<uint64_t>(handle), default_launch_value, flags, out_handle);
#elif JYPPX_HAS_CUDA_TOOLKIT
    (void)default_launch_value;
    (void)flags;
    jyppx::cuda::set_cuda_error("cudaGraphConditionalHandleCreate", 0, "cuda-version-not-supported", "CUDA conditional graph handles require CUDA runtime 12.3 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#else
    (void)default_launch_value;
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA conditional graph handle creation");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_conditional_handle_create_v2_safe(
    JYPPX_CudaGraph* graph,
    JYPPX_CudaExecutionContext* context,
    uint32_t default_launch_value,
    uint32_t flags,
    JYPPX_CudaGraphConditionalHandle** out_handle)
{
    auto status = jyppx::cuda::validate_output_pointer(out_handle, "out_handle");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_handle = nullptr;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    if (context != nullptr)
    {
        status = jyppx::cuda::validate_execution_context(context, "context");
        if (status != JYPPX_STATUS_OK) { return status; }
    }

    auto* graph_object = reinterpret_cast<jyppx::cuda::GraphObject*>(graph);
    auto* context_object = reinterpret_cast<jyppx::cuda::ExecutionContextObject*>(context);
    cudaGraphConditionalHandle handle = 0;
    cudaExecutionContext_t execution_context = context_object != nullptr ? context_object->handle : nullptr;
    status = jyppx::cuda::map_cuda_status(
        cudaGraphConditionalHandleCreate_v2(&handle, graph_object->handle, execution_context, default_launch_value, flags),
        "cudaGraphConditionalHandleCreate_v2");
    if (status != JYPPX_STATUS_OK) { return status; }
    return create_conditional_handle_object(graph, static_cast<uint64_t>(handle), default_launch_value, flags, out_handle);
#elif JYPPX_HAS_CUDA_TOOLKIT
    (void)context;
    (void)default_launch_value;
    (void)flags;
    jyppx::cuda::set_cuda_error("cudaGraphConditionalHandleCreate_v2", 0, "cuda-version-not-supported", "cudaGraphConditionalHandleCreate_v2 requires CUDA runtime 13.0 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#else
    (void)context;
    (void)default_launch_value;
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA conditional graph v2 handle creation");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_conditional_handle_destroy_safe(JYPPX_CudaGraphConditionalHandle* handle)
{
    if (handle == nullptr) { return JYPPX_STATUS_OK; }
    auto status = jyppx::cuda::validate_graph_conditional_handle(handle, "handle");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* handle_object = reinterpret_cast<jyppx::cuda::GraphConditionalHandleObject*>(handle);
    if (handle_object->owner != nullptr && handle_object->owner->active_conditional_handles > 0)
    {
        handle_object->owner->active_conditional_handles--;
    }
    delete handle_object;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_cuda_graph_add_conditional_node_safe(
    JYPPX_CudaGraph* graph,
    JYPPX_CudaGraphConditionalHandle* handle,
    uint32_t node_type,
    uint32_t body_count,
    uintptr_t dependency_node,
    JYPPX_CudaGraphConditionalNode** out_node)
{
    auto status = jyppx::cuda::validate_output_pointer(out_node, "out_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = nullptr;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_graph_conditional_handle(handle, "handle");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (dependency_node != 0)
    {
        status = jyppx_cuda_validate_graph_node_token(dependency_node, "dependency_node");
        if (status != JYPPX_STATUS_OK) { return status; }
    }

    auto* graph_object = reinterpret_cast<jyppx::cuda::GraphObject*>(graph);
    auto* handle_object = reinterpret_cast<jyppx::cuda::GraphConditionalHandleObject*>(handle);
    if (handle_object->owner != graph_object)
    {
        jyppx::cuda::set_cuda_error("cudaGraphAddNode(conditional)", 0, "handle-owner-mismatch", "The conditional handle is owned by a different CUDA graph.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (node_type > 2U || body_count == 0U)
    {
        jyppx::cuda::set_cuda_error("cudaGraphAddNode(conditional)", 0, "invalid-conditional-shape", "Conditional node type or body count is invalid.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if ((node_type == 0U && body_count > 2U) || (node_type == 1U && body_count != 1U))
    {
        jyppx::cuda::set_cuda_error("cudaGraphAddNode(conditional)", 0, "invalid-conditional-shape", "IF nodes accept one or two bodies and WHILE nodes accept exactly one body.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12030
    try
    {
        cudaGraphNodeParams params{};
        params.type = cudaGraphNodeTypeConditional;
        params.conditional.handle = static_cast<cudaGraphConditionalHandle>(handle_object->handle);
        params.conditional.type = static_cast<cudaGraphConditionalNodeType>(node_type);
        params.conditional.size = body_count;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
        params.conditional.ctx = nullptr;
#endif

        cudaGraphNode_t dependency = dependency_node == 0 ? nullptr : jyppx_cuda_node_from_token(dependency_node);
        const cudaGraphNode_t* dependencies = dependency_node == 0 ? nullptr : &dependency;
        cudaGraphNode_t node = nullptr;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
        status = jyppx::cuda::map_cuda_status(cudaGraphAddNode(&node, graph_object->handle, dependencies, nullptr, dependency_node == 0 ? 0U : 1U, &params), "cudaGraphAddNode(conditional)");
#else
        status = jyppx::cuda::map_cuda_status(cudaGraphAddNode(&node, graph_object->handle, dependencies, dependency_node == 0 ? 0U : 1U, &params), "cudaGraphAddNode(conditional)");
#endif
        if (status != JYPPX_STATUS_OK) { return status; }
        if (params.conditional.phGraph_out == nullptr)
        {
            cudaGraphDestroyNode(node);
            jyppx::cuda::set_cuda_error("cudaGraphAddNode(conditional)", 0, "null-body-graph", "CUDA did not return conditional body graphs.");
            return JYPPX_STATUS_INVALID_STATE;
        }

        auto* node_object = new (std::nothrow) jyppx::cuda::GraphConditionalNodeObject{};
        if (node_object == nullptr)
        {
            cudaGraphDestroyNode(node);
            jyppx::cuda::set_cuda_error("cudaGraphAddNode(conditional)", 0, "out-of-memory", "Unable to allocate bridge metadata for the CUDA conditional node.");
            return JYPPX_STATUS_OUT_OF_MEMORY;
        }

        node_object->base.magic = jyppx::cuda::kObjectMagic;
        node_object->base.kind = jyppx::cuda::ObjectKind::GraphConditionalNode;
        node_object->owner = graph_object;
        node_object->node_type = node_type;
        node_object->body_count = body_count;
        node_object->node = node;
        node_object->body_graphs = params.conditional.phGraph_out;
        graph_object->active_conditional_nodes++;
        *out_node = reinterpret_cast<JYPPX_CudaGraphConditionalNode*>(node_object);
        return JYPPX_STATUS_OK;
    }
    catch (const std::bad_alloc&)
    {
        return report_native_allocation_failure("cudaGraphAddNode(conditional)");
    }
    catch (...)
    {
        return report_native_exception("cudaGraphAddNode(conditional)");
    }
#elif JYPPX_HAS_CUDA_TOOLKIT
    (void)dependency_node;
    jyppx::cuda::set_cuda_error("cudaGraphAddNode(conditional)", 0, "cuda-version-not-supported", "Conditional graph nodes require CUDA runtime 12.3 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#else
    (void)dependency_node;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA conditional graph node creation");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_conditional_node_destroy_safe(JYPPX_CudaGraphConditionalNode* node)
{
    if (node == nullptr) { return JYPPX_STATUS_OK; }
    auto status = jyppx::cuda::validate_graph_conditional_node(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* node_object = reinterpret_cast<jyppx::cuda::GraphConditionalNodeObject*>(node);
    if (node_object->owner != nullptr && node_object->owner->active_conditional_nodes > 0)
    {
        node_object->owner->active_conditional_nodes--;
    }
    delete node_object;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_cuda_graph_conditional_node_get_node_token_safe(
    JYPPX_CudaGraphConditionalNode* node,
    uintptr_t* out_node)
{
    auto status = jyppx::cuda::validate_output_pointer(out_node, "out_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = 0;
    status = jyppx::cuda::validate_graph_conditional_node(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    *out_node = jyppx_cuda_node_to_token(reinterpret_cast<jyppx::cuda::GraphConditionalNodeObject*>(node)->node);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA conditional graph node token query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_conditional_node_get_body_count_safe(
    JYPPX_CudaGraphConditionalNode* node,
    uint32_t* out_body_count)
{
    auto status = jyppx::cuda::validate_output_pointer(out_body_count, "out_body_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_body_count = 0;
    return conditional_node_body_count(node, out_body_count);
}

JYPPX_StatusCode jyppx_cuda_graph_conditional_body_get_node_count_safe(
    JYPPX_CudaGraphConditionalNode* node,
    uint32_t body_index,
    size_t* out_count)
{
    auto status = jyppx::cuda::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
    return conditional_body_count(node, body_index, 0, out_count);
}

JYPPX_StatusCode jyppx_cuda_graph_conditional_body_get_root_node_count_safe(
    JYPPX_CudaGraphConditionalNode* node,
    uint32_t body_index,
    size_t* out_count)
{
    auto status = jyppx::cuda::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
    return conditional_body_count(node, body_index, 1, out_count);
}

JYPPX_StatusCode jyppx_cuda_graph_conditional_body_get_edge_count_safe(
    JYPPX_CudaGraphConditionalNode* node,
    uint32_t body_index,
    size_t* out_count)
{
    auto status = jyppx::cuda::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
    return conditional_body_count(node, body_index, 2, out_count);
}

JYPPX_StatusCode jyppx_cuda_graph_conditional_body_add_empty_node_safe(
    JYPPX_CudaGraphConditionalNode* node,
    uint32_t body_index,
    uintptr_t dependency_node,
    uintptr_t* out_body_node)
{
    auto status = jyppx::cuda::validate_output_pointer(out_body_node, "out_body_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_body_node = 0;
    status = jyppx::cuda::validate_graph_conditional_node(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (dependency_node != 0)
    {
        status = jyppx_cuda_validate_graph_node_token(dependency_node, "dependency_node");
        if (status != JYPPX_STATUS_OK) { return status; }
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* node_object = reinterpret_cast<jyppx::cuda::GraphConditionalNodeObject*>(node);
    cudaGraph_t body_graph = nullptr;
    status = validate_conditional_body_graph(node_object, body_index, &body_graph, "cudaGraphAddEmptyNode(conditional body)");
    if (status != JYPPX_STATUS_OK) { return status; }
    cudaGraphNode_t dependency = dependency_node == 0 ? nullptr : jyppx_cuda_node_from_token(dependency_node);
    const cudaGraphNode_t* dependencies = dependency_node == 0 ? nullptr : &dependency;
    cudaGraphNode_t body_node = nullptr;
    status = jyppx::cuda::map_cuda_status(
        cudaGraphAddEmptyNode(&body_node, body_graph, dependencies, dependency_node == 0 ? 0U : 1U),
        "cudaGraphAddEmptyNode(conditional body)");
    if (status == JYPPX_STATUS_OK)
    {
        *out_body_node = jyppx_cuda_node_to_token(body_node);
    }
    return status;
#else
    (void)node;
    (void)body_index;
    (void)dependency_node;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA conditional body empty node creation");
#endif
}
