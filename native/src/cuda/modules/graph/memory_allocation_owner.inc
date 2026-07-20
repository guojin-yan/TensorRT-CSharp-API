namespace
{
JYPPX_StatusCode validate_graph_memory_allocation_owner(
    JYPPX_CudaGraph* graph,
    JYPPX_CudaGraphMemoryAllocation* allocation,
    const char* operation,
    GraphObject** out_graph,
    GraphMemoryAllocationObject** out_allocation)
{
    auto status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_graph_memory_allocation(allocation, "allocation");
    if (status != JYPPX_STATUS_OK) { return status; }

    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    auto* allocation_object = reinterpret_cast<GraphMemoryAllocationObject*>(allocation);
    if (allocation_object->owner != graph_object)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "allocation-owner-mismatch", "The graph memory allocation belongs to a different CUDA graph.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (allocation_object->free_node_added)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "allocation-already-freed", "A memory free node has already been added for this graph allocation.");
        return JYPPX_STATUS_INVALID_STATE;
    }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 11040
    cudaGraphNodeType node_type{};
    status = jyppx::cuda::map_cuda_status(cudaGraphNodeGetType(allocation_object->allocation_node, &node_type), operation);
    if (status != JYPPX_STATUS_OK) { return status; }
    if (node_type != cudaGraphNodeTypeMemAlloc)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "allocation-node-type-mismatch", "The bridge allocation metadata no longer refers to a CUDA memory allocation node.");
        return JYPPX_STATUS_INVALID_STATE;
    }
#endif

    *out_graph = graph_object;
    *out_allocation = allocation_object;
    return JYPPX_STATUS_OK;
}

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 11040
size_t make_graph_allocation_dependencies(
    const GraphMemoryAllocationObject* allocation,
    const uintptr_t dependency_node,
    cudaGraphNode_t (&dependencies)[2])
{
    dependencies[0] = allocation->allocation_node;
    if (dependency_node == 0)
    {
        return 1;
    }

    cudaGraphNode_t dependency = jyppx_cuda_node_from_token(dependency_node);
    if (dependency == allocation->allocation_node)
    {
        return 1;
    }

    dependencies[1] = dependency;
    return 2;
}
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_memory_allocation_create_safe(
    JYPPX_CudaGraph* graph,
    uintptr_t dependency_node,
    size_t size,
    int32_t device_ordinal,
    JYPPX_CudaGraphMemoryAllocation** out_allocation)
{
    auto status = jyppx::cuda::validate_output_pointer(out_allocation, "out_allocation");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_allocation = nullptr;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (dependency_node != 0)
    {
        status = jyppx_cuda_validate_graph_node_token(dependency_node, "dependency_node");
        if (status != JYPPX_STATUS_OK) { return status; }
    }
    if (size == 0)
    {
        jyppx::cuda::set_cuda_error("cudaGraphAddMemAllocNode", 0, "invalid-allocation-size", "CUDA graph memory allocation size must be greater than zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (device_ordinal < 0)
    {
        jyppx::cuda::set_cuda_error("cudaGraphAddMemAllocNode", 0, "invalid-device", "CUDA graph memory allocation device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 11040
    int device_count = 0;
    status = jyppx::cuda::map_cuda_status(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount(graph memory allocation)");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (device_ordinal >= device_count)
    {
        jyppx::cuda::set_cuda_error("cudaGraphAddMemAllocNode", 0, "invalid-device", "CUDA graph memory allocation device ordinal is outside the available device range.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    cudaMemAllocNodeParams params{};
    params.poolProps.allocType = cudaMemAllocationTypePinned;
    params.poolProps.handleTypes = cudaMemHandleTypeNone;
    params.poolProps.location.type = cudaMemLocationTypeDevice;
    params.poolProps.location.id = device_ordinal;
    params.bytesize = size;

    cudaGraphNode_t dependency = dependency_node == 0 ? nullptr : jyppx_cuda_node_from_token(dependency_node);
    const cudaGraphNode_t* dependencies = dependency_node == 0 ? nullptr : &dependency;
    cudaGraphNode_t allocation_node = nullptr;
    status = jyppx::cuda::map_cuda_status(
        cudaGraphAddMemAllocNode(&allocation_node, graph_object->handle, dependencies, dependency_node == 0 ? 0U : 1U, &params),
        "cudaGraphAddMemAllocNode");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (allocation_node == nullptr || params.dptr == nullptr)
    {
        if (allocation_node != nullptr) { (void)cudaGraphDestroyNode(allocation_node); }
        jyppx::cuda::set_cuda_error("cudaGraphAddMemAllocNode", 0, "invalid-allocation-result", "CUDA returned a null graph allocation node or device address.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    auto* allocation = new (std::nothrow) GraphMemoryAllocationObject{};
    if (allocation == nullptr)
    {
        (void)cudaGraphDestroyNode(allocation_node);
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    allocation->base.magic = jyppx::cuda::kObjectMagic;
    allocation->base.kind = jyppx::cuda::ObjectKind::GraphMemoryAllocation;
    allocation->owner = graph_object;
    allocation->size = size;
    allocation->device_ordinal = device_ordinal;
    allocation->free_node_added = false;
    allocation->allocation_node = allocation_node;
    allocation->pointer = params.dptr;
    graph_object->active_memory_allocations++;
    *out_allocation = reinterpret_cast<JYPPX_CudaGraphMemoryAllocation*>(allocation);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_CUDA_TOOLKIT
    (void)dependency_node;
    (void)size;
    (void)device_ordinal;
    jyppx::cuda::set_cuda_error("cudaGraphAddMemAllocNode", 0, "cuda-version-not-supported", "CUDA graph memory allocation nodes require CUDA runtime 11.4 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#else
    (void)dependency_node;
    (void)size;
    (void)device_ordinal;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph memory allocation node creation");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_memory_allocation_destroy_safe(JYPPX_CudaGraphMemoryAllocation* allocation)
{
    if (allocation == nullptr) { return JYPPX_STATUS_OK; }
    auto status = jyppx::cuda::validate_graph_memory_allocation(allocation, "allocation");
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* allocation_object = reinterpret_cast<GraphMemoryAllocationObject*>(allocation);
    if (allocation_object->owner != nullptr && allocation_object->owner->active_memory_allocations > 0)
    {
        allocation_object->owner->active_memory_allocations--;
    }
    delete allocation_object;
    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode jyppx_cuda_graph_memory_allocation_add_memset_node_safe(
    JYPPX_CudaGraph* graph,
    JYPPX_CudaGraphMemoryAllocation* allocation,
    uintptr_t dependency_node,
    uint32_t value,
    size_t count,
    uintptr_t* out_node)
{
    auto status = jyppx::cuda::validate_output_pointer(out_node, "out_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = 0;
    if (dependency_node != 0)
    {
        status = jyppx_cuda_validate_graph_node_token(dependency_node, "dependency_node");
        if (status != JYPPX_STATUS_OK) { return status; }
    }
    GraphObject* graph_object = nullptr;
    GraphMemoryAllocationObject* allocation_object = nullptr;
    status = validate_graph_memory_allocation_owner(graph, allocation, "cudaGraphAddMemsetNode(graph allocation)", &graph_object, &allocation_object);
    if (status != JYPPX_STATUS_OK) { return status; }
    if (value > 0xFFU || count == 0 || count > allocation_object->size)
    {
        jyppx::cuda::set_cuda_error("cudaGraphAddMemsetNode(graph allocation)", 0, "invalid-memset-range", "Graph allocation memset requires a byte value and a non-zero count within the allocation size.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 11040
    cudaMemsetParams params{};
    params.dst = allocation_object->pointer;
    params.pitch = count;
    params.value = value;
    params.elementSize = 1;
    params.width = count;
    params.height = 1;
    cudaGraphNode_t dependencies[2]{};
    const size_t dependency_count = make_graph_allocation_dependencies(allocation_object, dependency_node, dependencies);
    cudaGraphNode_t node = nullptr;
    status = jyppx::cuda::map_cuda_status(
        cudaGraphAddMemsetNode(&node, graph_object->handle, dependencies, dependency_count, &params),
        "cudaGraphAddMemsetNode(graph allocation)");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = jyppx_cuda_node_to_token(node);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph allocation memset node creation");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_memory_allocation_add_device_to_host_node_safe(
    JYPPX_CudaGraph* graph,
    JYPPX_CudaGraphMemoryAllocation* allocation,
    uintptr_t dependency_node,
    JYPPX_CudaPinnedMemory* destination,
    size_t count,
    uintptr_t* out_node)
{
    auto status = jyppx::cuda::validate_output_pointer(out_node, "out_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = 0;
    if (dependency_node != 0)
    {
        status = jyppx_cuda_validate_graph_node_token(dependency_node, "dependency_node");
        if (status != JYPPX_STATUS_OK) { return status; }
    }
    status = jyppx::cuda::validate_pinned_memory(destination, "destination");
    if (status != JYPPX_STATUS_OK) { return status; }
    GraphObject* graph_object = nullptr;
    GraphMemoryAllocationObject* allocation_object = nullptr;
    status = validate_graph_memory_allocation_owner(graph, allocation, "cudaGraphAddMemcpyNode1D(graph allocation)", &graph_object, &allocation_object);
    if (status != JYPPX_STATUS_OK) { return status; }
    auto* destination_object = reinterpret_cast<PinnedMemoryObject*>(destination);
    if (count == 0 || count > allocation_object->size || count > destination_object->size)
    {
        jyppx::cuda::set_cuda_error("cudaGraphAddMemcpyNode1D(graph allocation)", 0, "invalid-memcpy-range", "Graph allocation copy requires a non-zero count within both source and destination sizes.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 11040
    cudaGraphNode_t dependencies[2]{};
    const size_t dependency_count = make_graph_allocation_dependencies(allocation_object, dependency_node, dependencies);
    cudaGraphNode_t node = nullptr;
    status = jyppx::cuda::map_cuda_status(
        cudaGraphAddMemcpyNode1D(&node, graph_object->handle, dependencies, dependency_count, destination_object->pointer, allocation_object->pointer, count, cudaMemcpyDeviceToHost),
        "cudaGraphAddMemcpyNode1D(graph allocation)");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = jyppx_cuda_node_to_token(node);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph allocation device-to-host node creation");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_memory_allocation_add_free_node_safe(
    JYPPX_CudaGraph* graph,
    JYPPX_CudaGraphMemoryAllocation* allocation,
    uintptr_t dependency_node,
    uintptr_t* out_node)
{
    auto status = jyppx::cuda::validate_output_pointer(out_node, "out_node");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node = 0;
    if (dependency_node != 0)
    {
        status = jyppx_cuda_validate_graph_node_token(dependency_node, "dependency_node");
        if (status != JYPPX_STATUS_OK) { return status; }
    }
    GraphObject* graph_object = nullptr;
    GraphMemoryAllocationObject* allocation_object = nullptr;
    status = validate_graph_memory_allocation_owner(graph, allocation, "cudaGraphAddMemFreeNode", &graph_object, &allocation_object);
    if (status != JYPPX_STATUS_OK) { return status; }

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 11040
    cudaGraphNode_t dependencies[2]{};
    const size_t dependency_count = make_graph_allocation_dependencies(allocation_object, dependency_node, dependencies);
    cudaGraphNode_t node = nullptr;
    status = jyppx::cuda::map_cuda_status(
        cudaGraphAddMemFreeNode(&node, graph_object->handle, dependencies, dependency_count, allocation_object->pointer),
        "cudaGraphAddMemFreeNode");
    if (status != JYPPX_STATUS_OK) { return status; }
    allocation_object->free_node_added = true;
    *out_node = jyppx_cuda_node_to_token(node);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph memory free node creation");
#endif
}
