#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
extern "C" cudaError_t CUDARTAPI cudaStreamGetCaptureInfo_ptsz(
    cudaStream_t stream,
    cudaStreamCaptureStatus* capture_status,
    unsigned long long* capture_id);
#endif

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 11030 && CUDART_VERSION < 12030
extern "C" cudaError_t CUDARTAPI cudaStreamUpdateCaptureDependencies_ptsz(
    cudaStream_t stream,
    cudaGraphNode_t* dependencies,
    size_t dependency_count,
    unsigned int flags);
#endif

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12030 && CUDART_VERSION < 13000
extern "C" cudaError_t CUDARTAPI cudaStreamUpdateCaptureDependencies_v2(
    cudaStream_t stream,
    cudaGraphNode_t* dependencies,
    const cudaGraphEdgeData* edge_data,
    size_t dependency_count,
    unsigned int flags);
#endif

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12030
extern "C" cudaError_t CUDARTAPI cudaStreamBeginCaptureToGraph(
    cudaStream_t stream,
    cudaGraph_t graph,
    const cudaGraphNode_t* dependencies,
    const cudaGraphEdgeData* dependency_data,
    size_t dependency_count,
    cudaStreamCaptureMode mode);
#endif

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12030
static cudaGraphEdgeData make_stream_capture_edge_data(const JYPPX_CudaGraphEdgeData& source);
#endif

JYPPX_StatusCode jyppx_cuda_stream_begin_capture_to_graph_safe(
    JYPPX_CudaStream* stream,
    JYPPX_CudaGraph* graph,
    const uintptr_t* dependencies,
    const JYPPX_CudaGraphEdgeData* edge_data,
    size_t dependency_count,
    int32_t mode)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (dependency_count > kMaximumCaptureDependencyCount)
    {
        jyppx::cuda::set_cuda_error("cudaStreamBeginCaptureToGraph", 0, "dependency-count-too-large", "CUDA stream-to-graph capture dependency count exceeds the bridge safety limit.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (dependency_count != 0 && (dependencies == nullptr || edge_data == nullptr))
    {
        jyppx::cuda::set_cuda_error("cudaStreamBeginCaptureToGraph", 0, "missing-dependency-array", "A non-zero CUDA stream-to-graph capture dependency count requires node-token and edge-data arrays.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (mode < 0 || mode > 2)
    {
        jyppx::cuda::set_cuda_error("cudaStreamBeginCaptureToGraph", 0, "invalid-capture-mode", "CUDA stream capture mode must be Global, ThreadLocal, or Relaxed.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12030
    try
    {
        std::vector<cudaGraphNode_t> nodes(dependency_count);
        std::vector<cudaGraphEdgeData> edge_values(dependency_count);
        for (size_t index = 0; index < dependency_count; ++index)
        {
            status = jyppx_cuda_validate_graph_node_token(dependencies[index], "dependency");
            if (status != JYPPX_STATUS_OK) { return status; }
            status = jyppx_cuda_validate_graph_edge_data(&edge_data[index], "edge_data");
            if (status != JYPPX_STATUS_OK) { return status; }
            nodes[index] = jyppx_cuda_node_from_token(dependencies[index]);
            edge_values[index] = make_stream_capture_edge_data(edge_data[index]);
        }

        auto* stream_object = reinterpret_cast<StreamObject*>(stream);
        auto* graph_object = reinterpret_cast<GraphObject*>(graph);
        return jyppx::cuda::map_cuda_status(
            cudaStreamBeginCaptureToGraph(
                stream_object->handle,
                graph_object->handle,
                nodes.empty() ? nullptr : nodes.data(),
                edge_values.empty() ? nullptr : edge_values.data(),
                dependency_count,
                static_cast<cudaStreamCaptureMode>(mode)),
            "cudaStreamBeginCaptureToGraph");
    }
    catch (const std::bad_alloc&)
    {
        return report_native_allocation_failure("cudaStreamBeginCaptureToGraph");
    }
    catch (...)
    {
        return report_native_exception("cudaStreamBeginCaptureToGraph");
    }
#elif JYPPX_HAS_CUDA_TOOLKIT
    (void)dependencies; (void)edge_data; (void)dependency_count; (void)mode;
    jyppx::cuda::set_cuda_error("cudaStreamBeginCaptureToGraph", 0, "cuda-version-not-supported", "cudaStreamBeginCaptureToGraph requires CUDA runtime 12.3 or later and is not exposed by this toolkit.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#else
    (void)dependencies; (void)edge_data; (void)dependency_count; (void)mode;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream-to-graph capture begin");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_end_capture_into_graph_safe(
    JYPPX_CudaStream* stream,
    JYPPX_CudaGraph* graph)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    cudaGraph_t captured_graph = nullptr;
    status = jyppx::cuda::map_cuda_status(
        cudaStreamEndCapture(stream_object->handle, &captured_graph),
        "cudaStreamEndCapture");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (captured_graph != graph_object->handle)
    {
        if (captured_graph != nullptr)
        {
            (void)cudaGraphDestroy(captured_graph);
        }
        jyppx::cuda::set_cuda_error("cudaStreamEndCapture", 0, "capture-owner-mismatch", "CUDA returned a different graph handle; the owner-scoped capture session refused to create a second graph owner.");
        return JYPPX_STATUS_INVALID_STATE;
    }
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream-to-graph capture end");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_get_capture_info_ptsz_copied_scalars_safe(
    JYPPX_CudaStream* stream,
    int32_t* out_capture_status,
    uint64_t* out_capture_id)
{
    auto status = jyppx::cuda::validate_output_pointer(out_capture_status, "out_capture_status");
    if (status != JYPPX_STATUS_OK) { return status; }
    status = jyppx::cuda::validate_output_pointer(out_capture_id, "out_capture_id");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_capture_status = 0;
    *out_capture_id = 0;
    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK) { return status; }
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12000 && CUDART_VERSION < 13000
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    unsigned long long capture_id = 0;
    status = jyppx::cuda::map_cuda_status(
        cudaStreamGetCaptureInfo_ptsz(stream_object->handle, &capture_status, &capture_id),
        "cudaStreamGetCaptureInfo_ptsz");
    if (status != JYPPX_STATUS_OK) { return status; }
    *out_capture_status = static_cast<int32_t>(capture_status);
    *out_capture_id = static_cast<uint64_t>(capture_id);
    return JYPPX_STATUS_OK;
#elif JYPPX_HAS_CUDA_TOOLKIT
    jyppx::cuda::set_cuda_error(
        "cudaStreamGetCaptureInfo_ptsz",
        0,
        "cuda-version-not-supported",
        "cudaStreamGetCaptureInfo_ptsz requires CUDA runtime 12.x and is not exposed by this toolkit.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA per-thread stream capture info query");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_update_capture_dependencies_ptsz_safe(
    JYPPX_CudaStream* stream,
    const uintptr_t* dependencies,
    size_t dependency_count,
    uint32_t mode)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (dependency_count > kMaximumCaptureDependencyCount)
    {
        jyppx::cuda::set_cuda_error("cudaStreamUpdateCaptureDependencies_ptsz", 0, "dependency-count-too-large", "CUDA stream capture dependency count exceeds the bridge safety limit.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (dependency_count != 0 && dependencies == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaStreamUpdateCaptureDependencies_ptsz", 0, "missing-dependency-array", "A non-zero CUDA stream capture dependency count requires a node-token array.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (mode > 1U)
    {
        jyppx::cuda::set_cuda_error("cudaStreamUpdateCaptureDependencies_ptsz", 0, "invalid-dependency-mode", "CUDA stream capture dependency mode must be Add or Replace.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 11030 && CUDART_VERSION < 12030
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
        return jyppx::cuda::map_cuda_status(
            cudaStreamUpdateCaptureDependencies_ptsz(stream_object->handle, nodes.empty() ? nullptr : nodes.data(), dependency_count, mode),
            "cudaStreamUpdateCaptureDependencies_ptsz");
    }
    catch (const std::bad_alloc&)
    {
        return report_native_allocation_failure("cudaStreamUpdateCaptureDependencies_ptsz");
    }
    catch (...)
    {
        return report_native_exception("cudaStreamUpdateCaptureDependencies_ptsz");
    }
#elif JYPPX_HAS_CUDA_TOOLKIT
    (void)dependencies; (void)dependency_count; (void)mode;
    jyppx::cuda::set_cuda_error(
        "cudaStreamUpdateCaptureDependencies_ptsz",
        0,
        "cuda-version-not-supported",
        "cudaStreamUpdateCaptureDependencies_ptsz requires CUDA runtime 11.3 through 12.2 and is not exposed by this toolkit.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#else
    (void)dependencies; (void)dependency_count; (void)mode;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA per-thread stream capture dependency update");
#endif
}

#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12030
static cudaGraphEdgeData make_stream_capture_edge_data(const JYPPX_CudaGraphEdgeData& source)
{
    cudaGraphEdgeData value{};
    value.from_port = source.from_port;
    value.to_port = source.to_port;
    value.type = source.type;
    value.reserved[0] = source.reserved0;
    value.reserved[1] = source.reserved1;
    value.reserved[2] = source.reserved2;
    value.reserved[3] = source.reserved3;
    value.reserved[4] = source.reserved4;
    return value;
}
#endif

JYPPX_StatusCode jyppx_cuda_stream_update_capture_dependencies_v2_safe(
    JYPPX_CudaStream* stream,
    const uintptr_t* dependencies,
    const JYPPX_CudaGraphEdgeData* edge_data,
    size_t dependency_count,
    uint32_t mode)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK) { return status; }
    if (dependency_count > kMaximumCaptureDependencyCount)
    {
        jyppx::cuda::set_cuda_error("cudaStreamUpdateCaptureDependencies_v2", 0, "dependency-count-too-large", "CUDA stream capture dependency count exceeds the bridge safety limit.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (dependency_count != 0 && (dependencies == nullptr || edge_data == nullptr))
    {
        jyppx::cuda::set_cuda_error("cudaStreamUpdateCaptureDependencies_v2", 0, "missing-dependency-array", "A non-zero CUDA stream capture dependency count requires node-token and edge-data arrays.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
    if (mode > 1U)
    {
        jyppx::cuda::set_cuda_error("cudaStreamUpdateCaptureDependencies_v2", 0, "invalid-dependency-mode", "CUDA stream capture dependency mode must be Add or Replace.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }
#if JYPPX_HAS_CUDA_TOOLKIT && defined(CUDART_VERSION) && CUDART_VERSION >= 12030 && CUDART_VERSION < 13000
    try
    {
        std::vector<cudaGraphNode_t> nodes(dependency_count);
        std::vector<cudaGraphEdgeData> edge_values(dependency_count);
        for (size_t index = 0; index < dependency_count; ++index)
        {
            status = jyppx_cuda_validate_graph_node_token(dependencies[index], "dependency");
            if (status != JYPPX_STATUS_OK) { return status; }
            status = jyppx_cuda_validate_graph_edge_data(&edge_data[index], "edge_data");
            if (status != JYPPX_STATUS_OK) { return status; }
            nodes[index] = jyppx_cuda_node_from_token(dependencies[index]);
            edge_values[index] = make_stream_capture_edge_data(edge_data[index]);
        }

        auto* stream_object = reinterpret_cast<StreamObject*>(stream);
        return jyppx::cuda::map_cuda_status(
            cudaStreamUpdateCaptureDependencies_v2(
                stream_object->handle,
                nodes.empty() ? nullptr : nodes.data(),
                edge_values.empty() ? nullptr : edge_values.data(),
                dependency_count,
                mode),
            "cudaStreamUpdateCaptureDependencies_v2");
    }
    catch (const std::bad_alloc&)
    {
        return report_native_allocation_failure("cudaStreamUpdateCaptureDependencies_v2");
    }
    catch (...)
    {
        return report_native_exception("cudaStreamUpdateCaptureDependencies_v2");
    }
#elif JYPPX_HAS_CUDA_TOOLKIT
    (void)dependencies; (void)edge_data; (void)dependency_count; (void)mode;
    jyppx::cuda::set_cuda_error(
        "cudaStreamUpdateCaptureDependencies_v2",
        0,
        "cuda-version-not-supported",
        "cudaStreamUpdateCaptureDependencies_v2 requires CUDA runtime 12.3 through 12.9 and is not exposed by this toolkit.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#else
    (void)dependencies; (void)edge_data; (void)dependency_count; (void)mode;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream edge-data dependency update");
#endif
}
