JYPPX_StatusCode jyppx_cuda_device_get_p2_p_attribute(int32_t attribute, int32_t source_device, int32_t destination_device, int32_t* out_value)
{
    return jyppx_cuda_device_get_p2p_attribute(attribute, source_device, destination_device, out_value);
}

JYPPX_StatusCode jyppx_cuda_device_get_shared_mem_config(int32_t* out_config)
{
    return jyppx_cuda_device_get_shared_memory_config(out_config);
}

JYPPX_StatusCode jyppx_cuda_device_set_shared_mem_config(int32_t config)
{
    return jyppx_cuda_device_set_shared_memory_config(config);
}

JYPPX_StatusCode jyppx_cuda_device_get_stream_priority_range(int32_t* out_least_priority, int32_t* out_greatest_priority)
{
    return jyppx_cuda_stream_get_priority_range(out_least_priority, out_greatest_priority);
}

JYPPX_StatusCode jyppx_cuda_event_elapsed_time_v2(JYPPX_CudaEvent* start_event, JYPPX_CudaEvent* end_event, float* out_milliseconds)
{
    return jyppx_cuda_event_elapsed_time(start_event, end_event, out_milliseconds);
}

JYPPX_StatusCode jyppx_cuda_graph_get_nodes(JYPPX_CudaGraph* graph, size_t* out_count)
{
    return jyppx_cuda_graph_get_node_count(graph, out_count);
}

JYPPX_StatusCode jyppx_cuda_graph_get_root_nodes(JYPPX_CudaGraph* graph, size_t* out_count)
{
    return jyppx_cuda_graph_get_root_node_count(graph, out_count);
}

JYPPX_StatusCode jyppx_cuda_graph_get_edges(JYPPX_CudaGraph* graph, size_t* out_count)
{
    return jyppx_cuda_graph_get_edge_count(graph, out_count);
}

JYPPX_StatusCode jyppx_cuda_graph_instantiate_with_flags(JYPPX_CudaGraph* graph, uint64_t flags, JYPPX_CudaGraphExec** out_graph_exec)
{
    return jyppx_cuda_graph_instantiate(graph, flags, out_graph_exec);
}

JYPPX_StatusCode jyppx_cuda_graph_launch(JYPPX_CudaGraphExec* graph_exec, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_graph_exec_launch(graph_exec, stream);
}

JYPPX_StatusCode jyppx_cuda_graph_upload(JYPPX_CudaGraphExec* graph_exec, JYPPX_CudaStream* stream)
{
    return jyppx_cuda_graph_exec_upload(graph_exec, stream);
}
