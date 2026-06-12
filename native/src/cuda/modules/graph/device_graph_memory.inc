namespace
{
constexpr int32_t kCudaGraphMemAttrUsedCurrent = 0;
constexpr int32_t kCudaGraphMemAttrUsedHigh = 1;
constexpr int32_t kCudaGraphMemAttrReservedCurrent = 2;
constexpr int32_t kCudaGraphMemAttrReservedHigh = 3;

JYPPX_StatusCode validate_cuda_device_ordinal(const int32_t device, const char* operation)
{
    if (device < 0)
    {
        jyppx::cuda::set_cuda_error(operation, 0, "invalid-device", "CUDA device ordinal must be greater than or equal to zero.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

    return JYPPX_STATUS_OK;
}

JYPPX_StatusCode validate_cuda_graph_memory_attribute(const int32_t attribute, const char* operation)
{
    if (attribute >= kCudaGraphMemAttrUsedCurrent && attribute <= kCudaGraphMemAttrReservedHigh)
    {
        return JYPPX_STATUS_OK;
    }

    jyppx::cuda::set_cuda_error(operation, 0, "invalid-graph-memory-attribute", "CUDA graph memory attribute must be in the range [0, 3].");
    return JYPPX_STATUS_INVALID_ARGUMENT;
}

JYPPX_StatusCode validate_cuda_graph_memory_high_watermark_attribute(const int32_t attribute, const char* operation)
{
    if (attribute == kCudaGraphMemAttrUsedHigh || attribute == kCudaGraphMemAttrReservedHigh)
    {
        return JYPPX_STATUS_OK;
    }

    jyppx::cuda::set_cuda_error(operation, 0, "invalid-graph-memory-high-watermark", "Only CUDA graph memory high-watermark attributes can be reset.");
    return JYPPX_STATUS_INVALID_ARGUMENT;
}

JYPPX_StatusCode cuda_graph_node_identity_not_supported(const char* operation)
{
    jyppx::cuda::set_cuda_error(operation, 0, "cuda-version-not-supported", "CUDA graph node identity APIs require CUDA runtime 13.0 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
}
}

JYPPX_StatusCode jyppx_cuda_device_get_graph_memory_attribute(int32_t device, int32_t attribute, uint64_t* out_value)
{
    auto status = jyppx::cuda::validate_output_pointer(out_value, "out_value");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_value = 0;
    status = validate_cuda_device_ordinal(device, "cudaDeviceGetGraphMemAttribute");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_cuda_graph_memory_attribute(attribute, "cudaDeviceGetGraphMemAttribute");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    unsigned long long value = 0;
    status = jyppx::cuda::map_cuda_status(
        cudaDeviceGetGraphMemAttribute(device, static_cast<cudaGraphMemAttributeType>(attribute), &value),
        "cudaDeviceGetGraphMemAttribute");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_value = static_cast<uint64_t>(value);
    return JYPPX_STATUS_OK;
#else
    (void)device;
    (void)attribute;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph memory attribute query");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_trim_graph_memory(int32_t device)
{
    auto status = validate_cuda_device_ordinal(device, "cudaDeviceGraphMemTrim");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    return jyppx::cuda::map_cuda_status(cudaDeviceGraphMemTrim(device), "cudaDeviceGraphMemTrim");
#else
    (void)device;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph memory trim");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_reset_graph_memory_high_watermark(int32_t device, int32_t attribute)
{
    auto status = validate_cuda_device_ordinal(device, "cudaDeviceSetGraphMemAttribute");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = validate_cuda_graph_memory_high_watermark_attribute(attribute, "cudaDeviceSetGraphMemAttribute");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    unsigned long long value = 0;
    return jyppx::cuda::map_cuda_status(
        cudaDeviceSetGraphMemAttribute(device, static_cast<cudaGraphMemAttributeType>(attribute), &value),
        "cudaDeviceSetGraphMemAttribute");
#else
    (void)device;
    (void)attribute;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph memory high-watermark reset");
#endif
}

JYPPX_StatusCode jyppx_cuda_device_get_graph_memory_used_current(int32_t device, uint64_t* out_value)
{
    return jyppx_cuda_device_get_graph_memory_attribute(device, kCudaGraphMemAttrUsedCurrent, out_value);
}

JYPPX_StatusCode jyppx_cuda_device_get_graph_memory_used_high(int32_t device, uint64_t* out_value)
{
    return jyppx_cuda_device_get_graph_memory_attribute(device, kCudaGraphMemAttrUsedHigh, out_value);
}

JYPPX_StatusCode jyppx_cuda_device_get_graph_memory_reserved_current(int32_t device, uint64_t* out_value)
{
    return jyppx_cuda_device_get_graph_memory_attribute(device, kCudaGraphMemAttrReservedCurrent, out_value);
}

JYPPX_StatusCode jyppx_cuda_device_get_graph_memory_reserved_high(int32_t device, uint64_t* out_value)
{
    return jyppx_cuda_device_get_graph_memory_attribute(device, kCudaGraphMemAttrReservedHigh, out_value);
}

JYPPX_StatusCode jyppx_cuda_device_reset_graph_memory_used_high(int32_t device)
{
    return jyppx_cuda_device_reset_graph_memory_high_watermark(device, kCudaGraphMemAttrUsedHigh);
}

JYPPX_StatusCode jyppx_cuda_device_reset_graph_memory_reserved_high(int32_t device)
{
    return jyppx_cuda_device_reset_graph_memory_high_watermark(device, kCudaGraphMemAttrReservedHigh);
}

JYPPX_StatusCode jyppx_cuda_graph_node_is_in_graph_safe(JYPPX_CudaGraph* graph, uintptr_t node, JYPPX_Boolean* out_is_in_graph)
{
    auto status = jyppx::cuda::validate_output_pointer(out_is_in_graph, "out_is_in_graph");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_in_graph = JYPPX_FALSE;
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    cudaGraph_t containing_graph = nullptr;
    status = jyppx::cuda::map_cuda_status(cudaGraphNodeGetContainingGraph(jyppx_cuda_node_from_token(node), &containing_graph), "cudaGraphNodeGetContainingGraph");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_is_in_graph = containing_graph == graph_object->handle ? JYPPX_TRUE : JYPPX_FALSE;
    return JYPPX_STATUS_OK;
#else
    (void)graph;
    return cuda_graph_node_identity_not_supported("cudaGraphNodeGetContainingGraph");
#endif
#else
    (void)graph;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph node containing graph query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_node_get_local_id_safe(uintptr_t node, uint32_t* out_node_id)
{
    auto status = jyppx::cuda::validate_output_pointer(out_node_id, "out_node_id");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node_id = 0;
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    unsigned int node_id = 0;
    status = jyppx::cuda::map_cuda_status(cudaGraphNodeGetLocalId(jyppx_cuda_node_from_token(node), &node_id), "cudaGraphNodeGetLocalId");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_node_id = static_cast<uint32_t>(node_id);
    return JYPPX_STATUS_OK;
#else
    return cuda_graph_node_identity_not_supported("cudaGraphNodeGetLocalId");
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph node local id query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_node_get_tools_id_safe(uintptr_t node, uint64_t* out_tools_node_id)
{
    auto status = jyppx::cuda::validate_output_pointer(out_tools_node_id, "out_tools_node_id");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tools_node_id = 0;
    status = jyppx_cuda_validate_graph_node_token(node, "node");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    unsigned long long tools_node_id = 0;
    status = jyppx::cuda::map_cuda_status(cudaGraphNodeGetToolsId(jyppx_cuda_node_from_token(node), &tools_node_id), "cudaGraphNodeGetToolsId");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_tools_node_id = static_cast<uint64_t>(tools_node_id);
    return JYPPX_STATUS_OK;
#else
    return cuda_graph_node_identity_not_supported("cudaGraphNodeGetToolsId");
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph node tools id query");
#endif
}
