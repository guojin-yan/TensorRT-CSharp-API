namespace
{
#if JYPPX_HAS_CUDA_TOOLKIT
JYPPX_StatusCode get_cuda_child_graph(uintptr_t node, cudaGraph_t* out_graph, const char* operation)
{
    auto status = jyppx::cuda::validate_output_pointer(out_graph, "out_graph");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_graph = nullptr;
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::map_cuda_status(
        cudaGraphChildGraphNodeGetGraph(jyppx_cuda_node_from_token(node), out_graph),
        operation);
    if (status != JYPPX_STATUS_OK) { return status; }
    if (*out_graph == nullptr)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "null-child-graph", "CUDA returned a null embedded child graph.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    return JYPPX_STATUS_OK;
}
#endif

JYPPX_StatusCode get_cuda_child_graph_count(uintptr_t node, int32_t selector, size_t* out_count)
{
    auto status = jyppx::cuda::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_count = 0;
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaGraph_t child_graph = nullptr;
    status = get_cuda_child_graph(node, &child_graph, "cudaGraphChildGraphNodeGetGraph");
    if (status != JYPPX_STATUS_OK) { return status; }
    size_t count = 0;
    if (selector == 0)
    {
        status = jyppx::cuda::map_cuda_status(cudaGraphGetNodes(child_graph, nullptr, &count), "cudaGraphGetNodes");
    }
    else if (selector == 1)
    {
        status = jyppx::cuda::map_cuda_status(cudaGraphGetRootNodes(child_graph, nullptr, &count), "cudaGraphGetRootNodes");
    }
    else
    {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
        status = jyppx::cuda::map_cuda_status(cudaGraphGetEdges(child_graph, nullptr, nullptr, nullptr, &count), "cudaGraphGetEdges");
#else
        status = jyppx::cuda::map_cuda_status(cudaGraphGetEdges(child_graph, nullptr, nullptr, &count), "cudaGraphGetEdges");
#endif
    }
    if (status == JYPPX_STATUS_OK) { *out_count = count; }
    return status;
#else
    (void)node;
    (void)selector;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA child graph count query");
#endif
}

#if JYPPX_HAS_CUDA_TOOLKIT
JYPPX_StatusCode copy_cuda_graph_node_type(
    cudaGraphNode_t node,
    JYPPX_Boolean* out_present,
    int32_t* out_node_type,
    const char* operation)
{
    *out_present = node != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    *out_node_type = -1;
    if (node == nullptr) { return JYPPX_STATUS_OK; }
    cudaGraphNodeType node_type{};
    auto status = jyppx::cuda::map_cuda_status(cudaGraphNodeGetType(node, &node_type), operation);
    if (status == JYPPX_STATUS_OK) { *out_node_type = static_cast<int32_t>(node_type); }
    return status;
}
#endif

JYPPX_StatusCode instantiate_cuda_graph_with_params(
    JYPPX_CudaGraph* graph,
    uint64_t flags,
    JYPPX_CudaStream* upload_stream,
    JYPPX_CudaGraphExec** out_graph_exec,
    int32_t* out_result,
    JYPPX_Boolean* out_has_error_node,
    int32_t* out_error_node_type)
{
    auto status = jyppx::cuda::validate_output_pointer(out_graph_exec, "out_graph_exec");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_result, "out_result");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_has_error_node, "out_has_error_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_error_node_type, "out_error_node_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_graph_exec = nullptr;
    *out_result = 0;
    *out_has_error_node = JYPPX_FALSE;
    *out_error_node_type = -1;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (upload_stream != nullptr)
    {
        status = jyppx::cuda::validate_stream(upload_stream, "upload_stream");
        if (status != JYPPX_STATUS_OK) { return status; }
    }
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    auto* graph_exec = new (std::nothrow) GraphExecObject{};
    if (graph_exec == nullptr) { return JYPPX_STATUS_OUT_OF_MEMORY; }
    graph_exec->base.magic = jyppx::cuda::kObjectMagic;
    graph_exec->base.kind = ObjectKind::GraphExec;
    graph_exec->handle = nullptr;
    cudaGraphInstantiateParams parameters{};
    parameters.flags = static_cast<unsigned long long>(flags);
    parameters.uploadStream = upload_stream != nullptr
        ? reinterpret_cast<StreamObject*>(upload_stream)->handle
        : nullptr;
    status = jyppx::cuda::map_cuda_status(
        cudaGraphInstantiateWithParams(&graph_exec->handle, graph_object->handle, &parameters),
        "cudaGraphInstantiateWithParams");
    *out_result = static_cast<int32_t>(parameters.result_out);
    auto metadata_status = copy_cuda_graph_node_type(
        parameters.errNode_out,
        out_has_error_node,
        out_error_node_type,
        "cudaGraphNodeGetType(instantiation error node)");
    if (status != JYPPX_STATUS_OK || metadata_status != JYPPX_STATUS_OK)
    {
        if (graph_exec->handle != nullptr) { (void)cudaGraphExecDestroy(graph_exec->handle); }
        delete graph_exec;
        return status != JYPPX_STATUS_OK ? status : metadata_status;
    }
    *out_graph_exec = reinterpret_cast<JYPPX_CudaGraphExec*>(graph_exec);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_CUDA_TOOLKIT
    (void)flags;
    (void)upload_stream;
    jyppx::cuda::set_cuda_error("cudaGraphInstantiateWithParams", 0, "cuda-version-not-supported", "CUDA graph parameterized instantiation requires CUDA runtime 12.0 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#else
    (void)flags;
    (void)upload_stream;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph parameterized instantiation");
#endif
}
}

JYPPX_StatusCode jyppx_cuda_graph_add_child_graph_node_safe(
    JYPPX_CudaGraph* graph,
    JYPPX_CudaGraph* child_graph,
    uintptr_t* out_node)
{
    auto status = jyppx::cuda::validate_output_pointer(out_node, "out_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = 0;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_graph(child_graph, "child_graph");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaGraphNode_t node = nullptr;
    status = jyppx::cuda::map_cuda_status(
        cudaGraphAddChildGraphNode(
            &node,
            reinterpret_cast<GraphObject*>(graph)->handle,
            nullptr,
            0,
            reinterpret_cast<GraphObject*>(child_graph)->handle),
        "cudaGraphAddChildGraphNode");
    if (status == JYPPX_STATUS_OK) { *out_node = jyppx_cuda_node_to_token(node); }
    return status;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA child graph node add");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_add_child_graph_node_after_safe(
    JYPPX_CudaGraph* graph,
    uintptr_t dependency_node,
    JYPPX_CudaGraph* child_graph,
    uintptr_t* out_node)
{
    auto status = jyppx_cuda_validate_graph_node_token(dependency_node, "dependency_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_node, "out_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = 0;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_graph(child_graph, "child_graph");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaGraphNode_t dependency = jyppx_cuda_node_from_token(dependency_node);
    cudaGraphNode_t node = nullptr;
    status = jyppx::cuda::map_cuda_status(
        cudaGraphAddChildGraphNode(
            &node,
            reinterpret_cast<GraphObject*>(graph)->handle,
            &dependency,
            1,
            reinterpret_cast<GraphObject*>(child_graph)->handle),
        "cudaGraphAddChildGraphNode");
    if (status == JYPPX_STATUS_OK) { *out_node = jyppx_cuda_node_to_token(node); }
    return status;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA child graph node add with dependency");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_child_graph_node_has_embedded_graph_safe(uintptr_t node, JYPPX_Boolean* out_has_graph)
{
    auto status = jyppx::cuda::validate_output_pointer(out_has_graph, "out_has_graph");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_has_graph = JYPPX_FALSE;
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaGraph_t child_graph = nullptr;
    status = get_cuda_child_graph(node, &child_graph, "cudaGraphChildGraphNodeGetGraph");
    if (status == JYPPX_STATUS_OK) { *out_has_graph = child_graph != nullptr ? JYPPX_TRUE : JYPPX_FALSE; }
    return status;
#else
    (void)node;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA child graph presence query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_child_graph_node_get_node_count_safe(uintptr_t node, size_t* out_count)
{
    return get_cuda_child_graph_count(node, 0, out_count);
}

JYPPX_StatusCode jyppx_cuda_graph_child_graph_node_get_root_node_count_safe(uintptr_t node, size_t* out_count)
{
    return get_cuda_child_graph_count(node, 1, out_count);
}

JYPPX_StatusCode jyppx_cuda_graph_child_graph_node_get_edge_count_safe(uintptr_t node, size_t* out_count)
{
    return get_cuda_child_graph_count(node, 2, out_count);
}

JYPPX_StatusCode jyppx_cuda_graph_exec_child_graph_node_set_params_safe(
    JYPPX_CudaGraphExec* graph_exec,
    uintptr_t node,
    JYPPX_CudaGraph* child_graph)
{
    auto status = jyppx::cuda::validate_graph_exec(graph_exec, "graph_exec");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_graph(child_graph, "child_graph");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(
        cudaGraphExecChildGraphNodeSetParams(
            reinterpret_cast<GraphExecObject*>(graph_exec)->handle,
            jyppx_cuda_node_from_token(node),
            reinterpret_cast<GraphObject*>(child_graph)->handle),
        "cudaGraphExecChildGraphNodeSetParams");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA executable child graph node update");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_exec_update_copied_metadata_safe(
    JYPPX_CudaGraphExec* graph_exec,
    JYPPX_CudaGraph* graph,
    int32_t* out_result,
    JYPPX_Boolean* out_has_error_node,
    int32_t* out_error_node_type,
    JYPPX_Boolean* out_has_error_from_node,
    int32_t* out_error_from_node_type)
{
    auto status = jyppx::cuda::validate_output_pointer(out_result, "out_result");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_has_error_node, "out_has_error_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_error_node_type, "out_error_node_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_has_error_from_node, "out_has_error_from_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_error_from_node_type, "out_error_from_node_type");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_result = 0;
    *out_has_error_node = JYPPX_FALSE;
    *out_error_node_type = -1;
    *out_has_error_from_node = JYPPX_FALSE;
    *out_error_from_node_type = -1;
    status = jyppx::cuda::validate_graph_exec(graph_exec, "graph_exec");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_exec_object = reinterpret_cast<GraphExecObject*>(graph_exec);
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    cudaGraphNode_t error_node = nullptr;
    cudaGraphNode_t error_from_node = nullptr;
    cudaError_t update_status = cudaSuccess;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info{};
    update_status = cudaGraphExecUpdate(graph_exec_object->handle, graph_object->handle, &result_info);
    *out_result = static_cast<int32_t>(result_info.result);
    error_node = result_info.errorNode;
    error_from_node = result_info.errorFromNode;
#else
    cudaGraphExecUpdateResult result{};
    update_status = cudaGraphExecUpdate(graph_exec_object->handle, graph_object->handle, &error_node, &result);
    *out_result = static_cast<int32_t>(result);
#endif
    if (update_status != cudaSuccess && update_status != cudaErrorGraphExecUpdateFailure)
    {
        return jyppx::cuda::map_cuda_status(update_status, "cudaGraphExecUpdate");
    }
    status = copy_cuda_graph_node_type(error_node, out_has_error_node, out_error_node_type, "cudaGraphNodeGetType(update error node)");
    if (status != JYPPX_STATUS_OK) { return status; }
    return copy_cuda_graph_node_type(error_from_node, out_has_error_from_node, out_error_from_node_type, "cudaGraphNodeGetType(update error from node)");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA executable graph update");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_instantiate_with_params_safe(
    JYPPX_CudaGraph* graph,
    uint64_t flags,
    JYPPX_CudaGraphExec** out_graph_exec,
    int32_t* out_result,
    JYPPX_Boolean* out_has_error_node,
    int32_t* out_error_node_type)
{
    return instantiate_cuda_graph_with_params(
        graph, flags, nullptr, out_graph_exec, out_result, out_has_error_node, out_error_node_type);
}

JYPPX_StatusCode jyppx_cuda_graph_instantiate_with_params_on_stream_safe(
    JYPPX_CudaGraph* graph,
    uint64_t flags,
    JYPPX_CudaStream* upload_stream,
    JYPPX_CudaGraphExec** out_graph_exec,
    int32_t* out_result,
    JYPPX_Boolean* out_has_error_node,
    int32_t* out_error_node_type)
{
    if (upload_stream == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaGraphInstantiateWithParams", 0, "invalid-upload-stream", "Upload stream must not be null for stream-based parameterized instantiation.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    return instantiate_cuda_graph_with_params(
        graph, flags, upload_stream, out_graph_exec, out_result, out_has_error_node, out_error_node_type);
}

JYPPX_StatusCode jyppx_cuda_graph_kernel_node_copy_attributes_safe(uintptr_t destination_node, uintptr_t source_node)
{
    auto status = jyppx_cuda_validate_graph_node_token(destination_node, "destination_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_cuda_validate_graph_node_token(source_node, "source_node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(
        cudaGraphKernelNodeCopyAttributes(
            jyppx_cuda_node_from_token(source_node),
            jyppx_cuda_node_from_token(destination_node)),
        "cudaGraphKernelNodeCopyAttributes");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph kernel node attribute copy");
#endif
}
