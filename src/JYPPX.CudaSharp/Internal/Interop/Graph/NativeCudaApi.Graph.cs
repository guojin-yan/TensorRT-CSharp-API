using System;
using JYPPX.CudaSharp.Internal.Handles;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static void BeginStreamCapture(SafeCudaStreamHandle stream, CudaStreamCaptureMode mode)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_begin_capture(stream, (int)mode));
    }

    public static SafeCudaGraphHandle EndStreamCapture(SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_end_capture(stream, out SafeCudaGraphHandle graph));
        return graph;
    }

    public static CudaStreamCaptureStatus GetStreamCaptureStatus(SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_is_capturing(stream, out int status));
        return (CudaStreamCaptureStatus)status;
    }

    public static CudaStreamCaptureInfo GetStreamCaptureInfo(SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_get_capture_info(stream, out int status, out ulong captureId));
        return new CudaStreamCaptureInfo((CudaStreamCaptureStatus)status, captureId);
    }

    public static SafeCudaGraphHandle CreateGraph(uint flags)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_create(flags, out SafeCudaGraphHandle graph));
        return graph;
    }

    public static SafeCudaGraphHandle CloneGraph(SafeCudaGraphHandle graph)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_clone(graph, out SafeCudaGraphHandle clone));
        return clone;
    }

    public static ulong GetGraphNodeCount(SafeCudaGraphHandle graph)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_get_node_count(graph, out UIntPtr count));
        return count.ToUInt64();
    }

    public static ulong GetGraphRootNodeCount(SafeCudaGraphHandle graph)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_get_root_node_count(graph, out UIntPtr count));
        return count.ToUInt64();
    }

    public static ulong GetGraphEdgeCount(SafeCudaGraphHandle graph)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_get_edge_count(graph, out UIntPtr count));
        return count.ToUInt64();
    }

    public static CudaGraphNode AddGraphEmptyNode(SafeCudaGraphHandle graph)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_add_empty_node_safe(graph, out UIntPtr node));
        return new CudaGraphNode(node);
    }

    public static CudaGraphNode AddGraphEmptyNodeAfter(SafeCudaGraphHandle graph, CudaGraphNode dependencyNode)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_add_empty_node_after_safe(graph, dependencyNode.Token, out UIntPtr node));
        return new CudaGraphNode(node);
    }

    public static void AddGraphDependency(SafeCudaGraphHandle graph, CudaGraphNode fromNode, CudaGraphNode toNode)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_add_dependency_safe(graph, fromNode.Token, toNode.Token));
    }

    public static void RemoveGraphDependency(SafeCudaGraphHandle graph, CudaGraphNode fromNode, CudaGraphNode toNode)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_remove_dependency_safe(graph, fromNode.Token, toNode.Token));
    }

    public static CudaGraphNode GetGraphNode(SafeCudaGraphHandle graph, ulong index)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_get_node_handle(graph, new UIntPtr(index), out UIntPtr node));
        return new CudaGraphNode(node);
    }

    public static CudaGraphNode GetGraphRootNode(SafeCudaGraphHandle graph, ulong index)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_get_root_node_handle(graph, new UIntPtr(index), out UIntPtr node));
        return new CudaGraphNode(node);
    }

    public static CudaGraphEdge GetGraphEdge(SafeCudaGraphHandle graph, ulong index)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_get_edge_handle_pair(graph, new UIntPtr(index), out UIntPtr from, out UIntPtr to));
        return new CudaGraphEdge(new CudaGraphNode(from), new CudaGraphNode(to));
    }

    public static CudaGraphNode FindGraphNodeInClone(CudaGraphNode originalNode, SafeCudaGraphHandle clonedGraph)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_node_find_in_clone_safe(originalNode.Token, clonedGraph, out UIntPtr clonedNode));
        return new CudaGraphNode(clonedNode);
    }

    public static CudaGraphNodeType GetGraphNodeType(CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_node_get_type_safe(node.Token, out int type));
        return (CudaGraphNodeType)type;
    }

    public static ulong GetGraphNodeDependencyCount(CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_node_get_dependency_count_safe(node.Token, out UIntPtr count));
        return count.ToUInt64();
    }

    public static CudaGraphNode GetGraphNodeDependency(CudaGraphNode node, ulong index)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_node_get_dependency_safe(node.Token, new UIntPtr(index), out UIntPtr dependencyNode));
        return new CudaGraphNode(dependencyNode);
    }

    public static ulong GetGraphNodeDependentCount(CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_node_get_dependent_count_safe(node.Token, out UIntPtr count));
        return count.ToUInt64();
    }

    public static CudaGraphNode GetGraphNodeDependent(CudaGraphNode node, ulong index)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_node_get_dependent_safe(node.Token, new UIntPtr(index), out UIntPtr dependentNode));
        return new CudaGraphNode(dependentNode);
    }

    public static uint GetGraphId(SafeCudaGraphHandle graph)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_get_id_safe(graph, out uint id));
        return id;
    }

    public static SafeCudaGraphExecHandle InstantiateGraph(SafeCudaGraphHandle graph, ulong flags)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_instantiate(graph, flags, out SafeCudaGraphExecHandle graphExec));
        return graphExec;
    }

    public static ulong GetGraphExecFlags(SafeCudaGraphExecHandle graphExec)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_exec_get_flags(graphExec, out ulong flags));
        return flags;
    }

    public static uint GetGraphExecId(SafeCudaGraphExecHandle graphExec)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_exec_get_id_safe(graphExec, out uint id));
        return id;
    }

    public static void SetGraphExecNodeEnabled(SafeCudaGraphExecHandle graphExec, CudaGraphNode node, bool enabled)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_exec_node_set_enabled_safe(graphExec, node.Token, enabled ? 1 : 0));
    }

    public static bool GetGraphExecNodeEnabled(SafeCudaGraphExecHandle graphExec, CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_exec_node_get_enabled_safe(graphExec, node.Token, out int enabled));
        return enabled != 0;
    }

    public static void UploadGraphExec(SafeCudaGraphExecHandle graphExec, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_exec_upload(graphExec, stream));
    }

    public static void LaunchGraphExec(SafeCudaGraphExecHandle graphExec, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_exec_launch(graphExec, stream));
    }
}
