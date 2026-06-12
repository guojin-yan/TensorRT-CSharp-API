JYPPX_StatusCode jyppx_cuda_stream_begin_capture(JYPPX_CudaStream* stream, int32_t mode)
{
    auto status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (mode < 0 || mode > 2)
    {
        jyppx::cuda::set_cuda_error("cudaStreamBeginCapture", 0, "invalid-capture-mode", "CUDA stream capture mode must be Global, ThreadLocal, or Relaxed.");
        return JYPPX_STATUS_INVALID_ARGUMENT;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaStreamBeginCapture(stream_object->handle, static_cast<cudaStreamCaptureMode>(mode)), "cudaStreamBeginCapture");
#else
    (void)mode;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream capture begin");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_end_capture(JYPPX_CudaStream* stream, JYPPX_CudaGraph** out_graph)
{
    auto status = jyppx::cuda::validate_output_pointer(out_graph, "out_graph");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_graph = nullptr;

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    cudaGraph_t raw_graph = nullptr;
    status = jyppx::cuda::map_cuda_status(cudaStreamEndCapture(stream_object->handle, &raw_graph), "cudaStreamEndCapture");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    if (raw_graph == nullptr)
    {
        jyppx::cuda::set_cuda_error("cudaStreamEndCapture", 0, "capture-invalidated", "CUDA stream capture ended without a graph; the capture sequence was likely invalidated.");
        return JYPPX_STATUS_INVALID_STATE;
    }

    auto* graph = new (std::nothrow) GraphObject{};
    if (graph == nullptr)
    {
        cudaGraphDestroy(raw_graph);
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    graph->base.magic = jyppx::cuda::kObjectMagic;
    graph->base.kind = ObjectKind::Graph;
    graph->handle = raw_graph;
    *out_graph = reinterpret_cast<JYPPX_CudaGraph*>(graph);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream capture end");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_is_capturing(JYPPX_CudaStream* stream, int32_t* out_capture_status)
{
    auto status = jyppx::cuda::validate_output_pointer(out_capture_status, "out_capture_status");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_capture_status = 0;

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    status = jyppx::cuda::map_cuda_status(cudaStreamIsCapturing(stream_object->handle, &capture_status), "cudaStreamIsCapturing");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_capture_status = static_cast<int32_t>(capture_status);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream capture status query");
#endif
}

JYPPX_StatusCode jyppx_cuda_stream_get_capture_info(JYPPX_CudaStream* stream, int32_t* out_capture_status, uint64_t* out_capture_id)
{
    auto status = jyppx::cuda::validate_output_pointer(out_capture_status, "out_capture_status");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_output_pointer(out_capture_id, "out_capture_id");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_capture_status = 0;
    *out_capture_id = 0;

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    unsigned long long capture_id = 0;
    status = jyppx::cuda::map_cuda_status(cudaStreamGetCaptureInfo(stream_object->handle, &capture_status, &capture_id), "cudaStreamGetCaptureInfo");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_capture_status = static_cast<int32_t>(capture_status);
    *out_capture_id = static_cast<uint64_t>(capture_id);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA stream capture info query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_create(uint32_t flags, JYPPX_CudaGraph** out_graph)
{
    auto status = jyppx::cuda::validate_output_pointer(out_graph, "out_graph");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_graph = nullptr;

#if JYPPX_HAS_CUDA_TOOLKIT
    cudaGraph_t raw_graph = nullptr;
    status = jyppx::cuda::map_cuda_status(cudaGraphCreate(&raw_graph, flags), "cudaGraphCreate");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* graph = new (std::nothrow) GraphObject{};
    if (graph == nullptr)
    {
        cudaGraphDestroy(raw_graph);
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    graph->base.magic = jyppx::cuda::kObjectMagic;
    graph->base.kind = ObjectKind::Graph;
    graph->handle = raw_graph;
    *out_graph = reinterpret_cast<JYPPX_CudaGraph*>(graph);
    return JYPPX_STATUS_OK;
#else
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph creation");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_clone(JYPPX_CudaGraph* graph, JYPPX_CudaGraph** out_clone)
{
    auto status = jyppx::cuda::validate_output_pointer(out_clone, "out_clone");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_clone = nullptr;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    cudaGraph_t raw_clone = nullptr;
    status = jyppx::cuda::map_cuda_status(cudaGraphClone(&raw_clone, graph_object->handle), "cudaGraphClone");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* clone = new (std::nothrow) GraphObject{};
    if (clone == nullptr)
    {
        cudaGraphDestroy(raw_clone);
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    clone->base.magic = jyppx::cuda::kObjectMagic;
    clone->base.kind = ObjectKind::Graph;
    clone->handle = raw_clone;
    *out_clone = reinterpret_cast<JYPPX_CudaGraph*>(clone);
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph clone");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_get_node_count(JYPPX_CudaGraph* graph, size_t* out_count)
{
    auto status = jyppx::cuda::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    size_t count = 0;
    status = jyppx::cuda::map_cuda_status(cudaGraphGetNodes(graph_object->handle, nullptr, &count), "cudaGraphGetNodes");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = count;
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph node count query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_get_root_node_count(JYPPX_CudaGraph* graph, size_t* out_count)
{
    auto status = jyppx::cuda::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    size_t count = 0;
    status = jyppx::cuda::map_cuda_status(cudaGraphGetRootNodes(graph_object->handle, nullptr, &count), "cudaGraphGetRootNodes");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = count;
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph root node count query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_get_edge_count(JYPPX_CudaGraph* graph, size_t* out_count)
{
    auto status = jyppx::cuda::validate_output_pointer(out_count, "out_count");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = 0;
    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    size_t count = 0;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    status = jyppx::cuda::map_cuda_status(cudaGraphGetEdges(graph_object->handle, nullptr, nullptr, nullptr, &count), "cudaGraphGetEdges");
#else
    status = jyppx::cuda::map_cuda_status(cudaGraphGetEdges(graph_object->handle, nullptr, nullptr, &count), "cudaGraphGetEdges");
#endif
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_count = count;
    return JYPPX_STATUS_OK;
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph edge count query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_instantiate(JYPPX_CudaGraph* graph, uint64_t flags, JYPPX_CudaGraphExec** out_graph_exec)
{
    auto status = jyppx::cuda::validate_output_pointer(out_graph_exec, "out_graph_exec");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_graph_exec = nullptr;

    status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
    auto* graph_exec = new (std::nothrow) GraphExecObject{};
    if (graph_exec == nullptr)
    {
        return JYPPX_STATUS_OUT_OF_MEMORY;
    }

    graph_exec->base.magic = jyppx::cuda::kObjectMagic;
    graph_exec->base.kind = ObjectKind::GraphExec;
    graph_exec->handle = nullptr;

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11040
    status = jyppx::cuda::map_cuda_status(cudaGraphInstantiateWithFlags(&graph_exec->handle, graph_object->handle, static_cast<unsigned long long>(flags)), "cudaGraphInstantiateWithFlags");
#else
    if (flags != 0)
    {
        delete graph_exec;
        jyppx::cuda::set_cuda_error("cudaGraphInstantiate", 0, "flags-not-supported", "Graph instantiate flags require CUDA runtime 11.4 or later.");
        return JYPPX_STATUS_NOT_SUPPORTED;
    }

    status = jyppx::cuda::map_cuda_status(cudaGraphInstantiate(&graph_exec->handle, graph_object->handle, nullptr, nullptr, 0), "cudaGraphInstantiate");
#endif
    if (status != JYPPX_STATUS_OK)
    {
        delete graph_exec;
        return status;
    }

    *out_graph_exec = reinterpret_cast<JYPPX_CudaGraphExec*>(graph_exec);
    return JYPPX_STATUS_OK;
#else
    (void)flags;
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph instantiation");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_exec_get_flags(JYPPX_CudaGraphExec* graph_exec, uint64_t* out_flags)
{
    auto status = jyppx::cuda::validate_output_pointer(out_flags, "out_flags");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_flags = 0;
    status = jyppx::cuda::validate_graph_exec(graph_exec, "graph_exec");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12010
    auto* graph_exec_object = reinterpret_cast<GraphExecObject*>(graph_exec);
    unsigned long long flags = 0;
    status = jyppx::cuda::map_cuda_status(cudaGraphExecGetFlags(graph_exec_object->handle, &flags), "cudaGraphExecGetFlags");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    *out_flags = static_cast<uint64_t>(flags);
    return JYPPX_STATUS_OK;
#else
    jyppx::cuda::set_cuda_error("cudaGraphExecGetFlags", 0, "cuda-version-not-supported", "cudaGraphExecGetFlags requires CUDA runtime 12.1 or later.");
    return JYPPX_STATUS_NOT_SUPPORTED;
#endif
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph exec flags query");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_exec_upload(JYPPX_CudaGraphExec* graph_exec, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_graph_exec(graph_exec, "graph_exec");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_exec_object = reinterpret_cast<GraphExecObject*>(graph_exec);
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaGraphUpload(graph_exec_object->handle, stream_object->handle), "cudaGraphUpload");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph exec upload");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_exec_launch(JYPPX_CudaGraphExec* graph_exec, JYPPX_CudaStream* stream)
{
    auto status = jyppx::cuda::validate_graph_exec(graph_exec, "graph_exec");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    status = jyppx::cuda::validate_stream(stream, "stream");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

#if JYPPX_HAS_CUDA_TOOLKIT
    auto* graph_exec_object = reinterpret_cast<GraphExecObject*>(graph_exec);
    auto* stream_object = reinterpret_cast<StreamObject*>(stream);
    return jyppx::cuda::map_cuda_status(cudaGraphLaunch(graph_exec_object->handle, stream_object->handle), "cudaGraphLaunch");
#else
    return jyppx::cuda::report_cuda_dependency_missing("CUDA graph launch");
#endif
}

JYPPX_StatusCode jyppx_cuda_graph_exec_destroy(JYPPX_CudaGraphExec* graph_exec)
{
    if (graph_exec == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    auto status = jyppx::cuda::validate_graph_exec(graph_exec, "graph_exec");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* graph_exec_object = reinterpret_cast<GraphExecObject*>(graph_exec);
#if JYPPX_HAS_CUDA_TOOLKIT
    status = jyppx::cuda::map_cuda_status(cudaGraphExecDestroy(graph_exec_object->handle), "cudaGraphExecDestroy");
#else
    status = JYPPX_STATUS_OK;
#endif
    delete graph_exec_object;
    return status;
}

JYPPX_StatusCode jyppx_cuda_graph_destroy(JYPPX_CudaGraph* graph)
{
    if (graph == nullptr)
    {
        return JYPPX_STATUS_OK;
    }

    auto status = jyppx::cuda::validate_graph(graph, "graph");
    if (status != JYPPX_STATUS_OK)
    {
        return status;
    }

    auto* graph_object = reinterpret_cast<GraphObject*>(graph);
#if JYPPX_HAS_CUDA_TOOLKIT
    status = jyppx::cuda::map_cuda_status(cudaGraphDestroy(graph_object->handle), "cudaGraphDestroy");
#else
    status = JYPPX_STATUS_OK;
#endif
    delete graph_object;
    return status;
}
