namespace
{
constexpr size_t kMaximumCaptureDependencyCount = 1000000;

JYPPX_StatusCode validate_graph_memset_inputs(
    JYPPX_CudaMemory* destination,
    uint32_t value,
    size_t count,
    const char* operation,
    MemoryObject** out_memory)
{
    auto status = jyppx::cuda::validate_output_pointer(out_memory, "out_memory");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_memory = nullptr;
    status = jyppx::cuda::validate_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (value > 0xFFU)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-memset-value", "CUDA graph memset byte value must be between 0 and 255.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (count == 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-memset-size", "CUDA graph memset byte count must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    auto* memory = reinterpret_cast<MemoryObject*>(destination);
    if (count > memory->size)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "size-out-of-range", "CUDA graph memset byte count exceeds the destination allocation size.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    *out_memory = memory;
    return JYPPX_STATUS_OK;
}

#if JYPPX_HAS_CUDA_TOOLKIT
cudaMemsetParams make_graph_memset_params(MemoryObject* destination, uint32_t value, size_t count)
{
    cudaMemsetParams params{};
    params.dst = destination->pointer;
    params.pitch = count;
    params.value = value;
    params.elementSize = 1;
    params.width = count;
    params.height = 1;
    return params;
}

JYPPX_StatusCode add_graph_memset_node(
    JYPPX_CudaGraph* graph,
    uintptr_t dependency_node,
    JYPPX_CudaMemory* destination,
    uint32_t value,
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
    MemoryObject* memory = nullptr;
    status = validate_graph_memset_inputs(destination, value, count, "cudaGraphAddMemsetNode", &memory);
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    cudaGraphNode_t dependency = dependency_node != 0 ? jyppx_cuda_node_from_token(dependency_node) : nullptr;
    cudaGraphNode_t node = nullptr;
    cudaMemsetParams params = make_graph_memset_params(memory, value, count);
    status = jyppx::cuda::map_cuda_status(
        cudaGraphAddMemsetNode(
            &node,
            graph_object->handle,
            dependency_node != 0 ? &dependency : nullptr,
            dependency_node != 0 ? 1U : 0U,
            &params),
        "cudaGraphAddMemsetNode");
    if (status == JYPPX_STATUS_OK) { *out_node = jyppx_cuda_node_to_token(node); }
    return status;
}

JYPPX_StatusCode copy_external_semaphore_snapshot(
    const uint32_t count,
    const void* semaphore_array,
    const void* parameter_array,
    JYPPX_CudaGraphExternalSemaphoreNodeParamsSnapshot* out_snapshot)
{
    out_snapshot->semaphore_count = count;
    out_snapshot->has_semaphore_array = semaphore_array != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    out_snapshot->has_parameter_array = parameter_array != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
}
#endif

JYPPX_StatusCode report_native_allocation_failure(const char* operation)
{
    jyppx::cuda::set_cuda_error(operation, 0, "native-allocation-failed", "The bridge could not allocate temporary storage for copied CUDA graph node tokens.");
    return JYPPX_STATUS_OUT_OF_MEMORY;
}

JYPPX_StatusCode report_native_exception(const char* operation)
{
    jyppx::cuda::set_cuda_error(operation, 0, "native-exception-caught", "The bridge caught an unexpected native exception before it could cross the C ABI.");
    return JYPPX_STATUS_RUNTIME_ERROR;
}
}

JYPPX_StatusCode jyppx_cuda_graph_add_memset_node_safe(
    JYPPX_CudaGraph* graph,
    JYPPX_CudaMemory* destination,
    uint32_t value,
    size_t count,
    uintptr_t* out_node)
{
#if JYPPX_HAS_CUDA_TOOLKIT
    return add_graph_memset_node(graph, 0, destination, value, count, out_node);
#else
    (void)graph; (void)destination; (void)value; (void)count; (void)out_node;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph memset node add");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_add_memset_node_after_safe(
    JYPPX_CudaGraph* graph,
    uintptr_t dependency_node,
    JYPPX_CudaMemory* destination,
    uint32_t value,
    size_t count,
    uintptr_t* out_node)
{
#if JYPPX_HAS_CUDA_TOOLKIT
    return add_graph_memset_node(graph, dependency_node, destination, value, count, out_node);
#else
    (void)graph; (void)dependency_node; (void)destination; (void)value; (void)count; (void)out_node;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph memset node add with dependency");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_exec_memset_node_set_params_safe(
    JYPPX_CudaGraphExec* graph_exec,
    uintptr_t node,
    JYPPX_CudaMemory* destination,
    uint32_t value,
    size_t count)
{
    auto status = jyppx::cuda::validate_graph_exec(graph_exec, "graph_exec");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    MemoryObject* memory = nullptr;
    status = validate_graph_memset_inputs(destination, value, count, "cudaGraphExecMemsetNodeSetParams", &memory);
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemsetParams params = make_graph_memset_params(memory, value, count);
    return jyppx::cuda::map_cuda_status(
        cudaGraphExecMemsetNodeSetParams(
            reinterpret_cast<GraphExecObject*>(graph_exec)->handle,
            jyppx_cuda_node_from_token(node),
            &params),
        "cudaGraphExecMemsetNodeSetParams");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA executable graph memset node parameter update");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_destroy_node_owner_scoped_safe(JYPPX_CudaGraph* graph, uintptr_t node)
{
    auto status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    try
    {
        auto* graph_object = reinterpret_cast<GraphObject*>(graph);
        size_t count = 0;
        status = jyppx::cuda::map_cuda_status(cudaGraphGetNodes(graph_object->handle, nullptr, &count), "cudaGraphGetNodes(node removal membership)");
        if (status != JYPPX_STATUS_OK) { return status; }
        std::vector<cudaGraphNode_t> nodes(count);
        if (count != 0)
        {
            status = jyppx::cuda::map_cuda_status(cudaGraphGetNodes(graph_object->handle, nodes.data(), &count), "cudaGraphGetNodes(node removal membership)");
            if (status != JYPPX_STATUS_OK) { return status; }
        }
        cudaGraphNode_t target = jyppx_cuda_node_from_token(node);
        bool found = false;
        for (size_t index = 0; index < count; ++index)
        {
            if (nodes[index] == target) { found = true; break; }
        }
        if (!found)
        {
            jyppx::cuda::set_cuda_error("cudaGraphDestroyNode", 0, "node-owner-mismatch", "The graph node token is not owned by the supplied CUDA graph.");
            return JYPPX_STATUS_INVALID_ARGUMENT;
        }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11040
        cudaGraphNodeType node_type{};
        status = jyppx::cuda::map_cuda_status(cudaGraphNodeGetType(target, &node_type), "cudaGraphNodeGetType(node removal membership)");
        if (status != JYPPX_STATUS_OK) { return status; }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
        if (node_type == cudaGraphNodeTypeConditional)
        {
            jyppx::cuda::set_cuda_error("cudaGraphDestroyNode", 0, "conditional-node-owner-active", "Conditional graph nodes are bridge-owned and cannot be destroyed through the generic node removal API.");
            return JYPPX_STATUS_INVALID_STATE;
        }
#endif
        if (node_type == cudaGraphNodeTypeMemAlloc || node_type == cudaGraphNodeTypeMemFree)
        {
            jyppx::cuda::set_cuda_error("cudaGraphDestroyNode", 0, "graph-memory-node-owner-active", "Graph memory allocation and free nodes are bridge-owned and cannot be destroyed through the generic node removal API.");
            return JYPPX_STATUS_INVALID_STATE;
        }
#endif
        return jyppx::cuda::map_cuda_status(cudaGraphDestroyNode(target), "cudaGraphDestroyNode");
    }
    catch (const std::bad_alloc&)
    {
        return report_native_allocation_failure("cudaGraphDestroyNode");
    }
    catch (...)
    {
        return report_native_exception("cudaGraphDestroyNode");
    }
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph owner-scoped node removal");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_kernel_node_get_params_snapshot_safe(
    uintptr_t node,
    JYPPX_CudaGraphKernelNodeParamsSnapshot* out_snapshot)
{
    auto status = jyppx::cuda::validate_output_pointer(out_snapshot, "out_snapshot");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_snapshot = JYPPX_CudaGraphKernelNodeParamsSnapshot{};
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaKernelNodeParams params{};
    status = jyppx::cuda::map_cuda_status(cudaGraphKernelNodeGetParams(jyppx_cuda_node_from_token(node), &params), "cudaGraphKernelNodeGetParams");
    if (status != JYPPX_STATUS_OK) { return status; }
    out_snapshot->grid_x = params.gridDim.x;
    out_snapshot->grid_y = params.gridDim.y;
    out_snapshot->grid_z = params.gridDim.z;
    out_snapshot->block_x = params.blockDim.x;
    out_snapshot->block_y = params.blockDim.y;
    out_snapshot->block_z = params.blockDim.z;
    out_snapshot->shared_memory_bytes = params.sharedMemBytes;
    out_snapshot->has_function = params.func != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    out_snapshot->has_kernel_params = params.kernelParams != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    out_snapshot->has_extra = params.extra != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph kernel node copied parameter snapshot");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_host_node_get_params_snapshot_safe(
    uintptr_t node,
    JYPPX_CudaGraphHostNodeParamsSnapshot* out_snapshot)
{
    auto status = jyppx::cuda::validate_output_pointer(out_snapshot, "out_snapshot");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_snapshot = JYPPX_CudaGraphHostNodeParamsSnapshot{};
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaHostNodeParams params{};
    status = jyppx::cuda::map_cuda_status(cudaGraphHostNodeGetParams(jyppx_cuda_node_from_token(node), &params), "cudaGraphHostNodeGetParams");
    if (status != JYPPX_STATUS_OK) { return status; }
    out_snapshot->has_callback = params.fn != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    out_snapshot->has_user_data = params.userData != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph host node copied parameter snapshot");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_mem_alloc_node_get_params_snapshot_safe(
    uintptr_t node,
    JYPPX_CudaGraphMemAllocNodeParamsSnapshot* out_snapshot)
{
    auto status = jyppx::cuda::validate_output_pointer(out_snapshot, "out_snapshot");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_snapshot = JYPPX_CudaGraphMemAllocNodeParamsSnapshot{};
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaMemAllocNodeParams params{};
    status = jyppx::cuda::map_cuda_status(cudaGraphMemAllocNodeGetParams(jyppx_cuda_node_from_token(node), &params), "cudaGraphMemAllocNodeGetParams");
    if (status != JYPPX_STATUS_OK) { return status; }
    out_snapshot->byte_count = static_cast<uint64_t>(params.bytesize);
    out_snapshot->access_descriptor_count = static_cast<uint64_t>(params.accessDescCount);
    out_snapshot->allocation_type = static_cast<int32_t>(params.poolProps.allocType);
    out_snapshot->handle_types = static_cast<uint64_t>(params.poolProps.handleTypes);
    out_snapshot->location_type = static_cast<int32_t>(params.poolProps.location.type);
    out_snapshot->location_id = params.poolProps.location.id;
    out_snapshot->has_access_descriptors = params.accessDescs != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    out_snapshot->has_device_pointer = params.dptr != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    out_snapshot->has_security_attributes = params.poolProps.win32SecurityAttributes != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph memory allocation node copied parameter snapshot");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_mem_free_node_get_params_snapshot_safe(
    uintptr_t node,
    JYPPX_CudaGraphMemFreeNodeParamsSnapshot* out_snapshot)
{
    auto status = jyppx::cuda::validate_output_pointer(out_snapshot, "out_snapshot");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_snapshot = JYPPX_CudaGraphMemFreeNodeParamsSnapshot{};
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    void* pointer = nullptr;
    status = jyppx::cuda::map_cuda_status(cudaGraphMemFreeNodeGetParams(jyppx_cuda_node_from_token(node), &pointer), "cudaGraphMemFreeNodeGetParams");
    if (status == JYPPX_STATUS_OK) { out_snapshot->has_device_pointer = pointer != nullptr ? JYPPX_TRUE : JYPPX_FALSE; }
    return status;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph memory free node copied parameter snapshot");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_external_semaphore_signal_node_get_params_snapshot_safe(
    uintptr_t node,
    JYPPX_CudaGraphExternalSemaphoreNodeParamsSnapshot* out_snapshot)
{
    auto status = jyppx::cuda::validate_output_pointer(out_snapshot, "out_snapshot");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_snapshot = JYPPX_CudaGraphExternalSemaphoreNodeParamsSnapshot{};
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaExternalSemaphoreSignalNodeParams params{};
    status = jyppx::cuda::map_cuda_status(cudaGraphExternalSemaphoresSignalNodeGetParams(jyppx_cuda_node_from_token(node), &params), "cudaGraphExternalSemaphoresSignalNodeGetParams");
    return status == JYPPX_STATUS_OK
        ? copy_external_semaphore_snapshot(params.numExtSems, params.extSemArray, params.paramsArray, out_snapshot)
        : status;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph external semaphore signal copied parameter snapshot");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_external_semaphore_wait_node_get_params_snapshot_safe(
    uintptr_t node,
    JYPPX_CudaGraphExternalSemaphoreNodeParamsSnapshot* out_snapshot)
{
    auto status = jyppx::cuda::validate_output_pointer(out_snapshot, "out_snapshot");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_snapshot = JYPPX_CudaGraphExternalSemaphoreNodeParamsSnapshot{};
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    cudaExternalSemaphoreWaitNodeParams params{};
    status = jyppx::cuda::map_cuda_status(cudaGraphExternalSemaphoresWaitNodeGetParams(jyppx_cuda_node_from_token(node), &params), "cudaGraphExternalSemaphoresWaitNodeGetParams");
    return status == JYPPX_STATUS_OK
        ? copy_external_semaphore_snapshot(params.numExtSems, params.extSemArray, params.paramsArray, out_snapshot)
        : status;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph external semaphore wait copied parameter snapshot");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_get_capture_summary_safe(
    JYPPX_CudaStream* stream,
    int32_t* out_capture_status,
    uint64_t* out_capture_id,
    JYPPX_Boolean* out_has_graph,
    size_t* out_dependency_count,
    JYPPX_Boolean* out_has_edge_data)
{
    auto status = jyppx::cuda::validate_output_pointer(out_capture_status, "out_capture_status");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_capture_id, "out_capture_id");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_has_graph, "out_has_graph");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_dependency_count, "out_dependency_count");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_has_edge_data, "out_has_edge_data");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_capture_status = 0;
    *out_capture_id = 0;
    *out_has_graph = JYPPX_FALSE;
    *out_dependency_count = 0;
    *out_has_edge_data = JYPPX_FALSE;
    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    unsigned long long capture_id = 0;
    cudaGraph_t graph = nullptr;
    const cudaGraphNode_t* dependencies = nullptr;
    size_t dependency_count = 0;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    const cudaGraphEdgeData* edge_data = nullptr;
    status = jyppx::cuda::map_cuda_status(
        cudaStreamGetCaptureInfo(stream_object->handle, &capture_status, &capture_id, &graph, &dependencies, &edge_data, &dependency_count),
        "cudaStreamGetCaptureInfo");
    *out_has_edge_data = edge_data != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
#elif defined(CUDART_VERSION) && CUDART_VERSION >= 12030
    const cudaGraphEdgeData* edge_data = nullptr;
    status = jyppx::cuda::map_cuda_status(
        cudaStreamGetCaptureInfo_v3(stream_object->handle, &capture_status, &capture_id, &graph, &dependencies, &edge_data, &dependency_count),
        "cudaStreamGetCaptureInfo_v3");
    *out_has_edge_data = edge_data != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
#elif defined(CUDART_VERSION) && CUDART_VERSION >= 11030
    status = jyppx::cuda::map_cuda_status(
        cudaStreamGetCaptureInfo_v2(stream_object->handle, &capture_status, &capture_id, &graph, &dependencies, &dependency_count),
        "cudaStreamGetCaptureInfo_v2");
#else
    status = jyppx::cuda::map_cuda_status(
        cudaStreamGetCaptureInfo(stream_object->handle, &capture_status, &capture_id),
        "cudaStreamGetCaptureInfo");
#endif
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_capture_status = static_cast<int32_t>(capture_status);
    *out_capture_id = static_cast<uint64_t>(capture_id);
    *out_has_graph = graph != nullptr ? JYPPX_TRUE : JYPPX_FALSE;
    *out_dependency_count = dependency_count;
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream copied capture summary");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_update_capture_dependencies_safe(
    JYPPX_CudaStream* stream,
    const uintptr_t* dependencies,
    size_t dependency_count,
    uint32_t mode)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (dependency_count > kMaximumCaptureDependencyCount)
    {
        jyppx::cuda::set_cuda_error("cudaStreamUpdateCaptureDependencies", 0, "dependency-count-too-large", "CUDA stream capture dependency count exceeds the bridge safety limit.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (dependency_count != 0 && dependencies == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaStreamUpdateCaptureDependencies", 0, "missing-dependency-array", "A non-zero CUDA stream capture dependency count requires a node-token array.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (mode > 1U)
    {
        jyppx::cuda::set_cuda_error("cudaStreamUpdateCaptureDependencies", 0, "invalid-dependency-mode", "CUDA stream capture dependency mode must be Add or Replace.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 11030
    try
    {
        std::vector<cudaGraphNode_t> nodes(dependency_count);
        for (size_t index = 0; index < dependency_count; ++index)
        {
            status = jyppx_cuda_validate_graph_node_token(dependencies[index], "dependency");
            if (status != JYPPX_STATUS_OK) { return status; }
            nodes[index] = jyppx_cuda_node_from_token(dependencies[index]);
        }
        auto* stream_object = reinterpret_cast<StreamObject*>(stream);
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
        return jyppx::cuda::map_cuda_status(
            cudaStreamUpdateCaptureDependencies(stream_object->handle, nodes.empty() ? nullptr : nodes.data(), nullptr, dependency_count, mode),
            "cudaStreamUpdateCaptureDependencies");
#else
        return jyppx::cuda::map_cuda_status(
            cudaStreamUpdateCaptureDependencies(stream_object->handle, nodes.empty() ? nullptr : nodes.data(), dependency_count, mode),
            "cudaStreamUpdateCaptureDependencies");
#endif
    }
    catch (const std::bad_alloc&)
    {
        return report_native_allocation_failure("cudaStreamUpdateCaptureDependencies");
    }
    catch (...)
    {
        return report_native_exception("cudaStreamUpdateCaptureDependencies");
    }
#elif JYPPX_HAS_CUDA_TOOLKIT
    (void)dependencies; (void)dependency_count; (void)mode;
    jyppx::cuda::set_cuda_error("cudaStreamUpdateCaptureDependencies", 0, "cuda-version-not-supported", "CUDA stream capture dependency updates require CUDA runtime 11.3 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#else
    (void)dependencies; (void)dependency_count; (void)mode;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream capture dependency update");
#endif
}
