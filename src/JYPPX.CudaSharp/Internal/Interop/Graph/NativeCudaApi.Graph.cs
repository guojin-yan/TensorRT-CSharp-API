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

    public static CudaGraphNode AddGraphEventRecordNode(SafeCudaGraphHandle graph, CudaGraphNode dependencyNode, SafeCudaEventHandle eventHandle)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_add_event_record_node_safe(graph, dependencyNode.Token, eventHandle, out UIntPtr node));
        return new CudaGraphNode(node);
    }

    public static CudaGraphNode AddGraphEventWaitNode(SafeCudaGraphHandle graph, CudaGraphNode dependencyNode, SafeCudaEventHandle eventHandle)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_add_event_wait_node_safe(graph, dependencyNode.Token, eventHandle, out UIntPtr node));
        return new CudaGraphNode(node);
    }

    public static CudaGraphNode AddGraphMemcpyNode1DDeviceToDevice(SafeCudaGraphHandle graph, CudaGraphNode dependencyNode, SafeCudaMemoryHandle destination, SafeCudaMemoryHandle source, int count)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_add_memcpy_node_1d_device_to_device_safe(graph, dependencyNode.Token, destination, source, new UIntPtr((uint)count), out UIntPtr node));
        return new CudaGraphNode(node);
    }

    public static CudaGraphNode AddGraphMemcpyNode1DHostToDevice(SafeCudaGraphHandle graph, CudaGraphNode dependencyNode, SafeCudaMemoryHandle destination, SafeCudaPinnedMemoryHandle source, int count)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_add_memcpy_node_1d_host_to_device_safe(graph, dependencyNode.Token, destination, source, new UIntPtr((uint)count), out UIntPtr node));
        return new CudaGraphNode(node);
    }

    public static CudaGraphNode AddGraphMemcpyNode1DDeviceToHost(SafeCudaGraphHandle graph, CudaGraphNode dependencyNode, SafeCudaPinnedMemoryHandle destination, SafeCudaMemoryHandle source, int count)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_add_memcpy_node_1d_device_to_host_safe(graph, dependencyNode.Token, destination, source, new UIntPtr((uint)count), out UIntPtr node));
        return new CudaGraphNode(node);
    }

    public static void AddGraphDependency(SafeCudaGraphHandle graph, CudaGraphNode fromNode, CudaGraphNode toNode)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_add_dependency_safe(graph, fromNode.Token, toNode.Token));
    }

    public static void AddGraphDependencyV2(SafeCudaGraphHandle graph, CudaGraphNode fromNode, CudaGraphNode toNode, CudaGraphEdgeData edgeData)
    {
        NativeCudaGraphEdgeData nativeEdgeData = edgeData.ToNative();
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_add_dependency_v2_safe(graph, fromNode.Token, toNode.Token, in nativeEdgeData));
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

    public static ulong GetGraphEdgeV2Count(SafeCudaGraphHandle graph)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_get_edges_v2_count_safe(graph, out UIntPtr count));
        return count.ToUInt64();
    }

    public static CudaGraphEdgeWithData GetGraphEdgeV2(SafeCudaGraphHandle graph, ulong index)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_get_edge_v2_safe(graph, new UIntPtr(index), out UIntPtr from, out UIntPtr to, out NativeCudaGraphEdgeData edgeData));
        return new CudaGraphEdgeWithData(new CudaGraphNode(from), new CudaGraphNode(to), new CudaGraphEdgeData(edgeData));
    }

    public static void ExportGraphDebugDot(SafeCudaGraphHandle graph, string path, CudaGraphDebugDotFlags flags)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("CUDA graph debug DOT output path must not be null, empty, or whitespace.", nameof(path));
        }

        if (path.IndexOf('\0') >= 0)
        {
            throw new ArgumentException("CUDA graph debug DOT output path must not contain null characters.", nameof(path));
        }

        using Utf8Interop.Utf8StringScope pathUtf8 = Utf8Interop.ToNativeString(path);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_debug_dot_print_safe(graph, pathUtf8.Pointer, (uint)flags));
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

    public static CudaGraphMemsetNodeParameters GetGraphMemsetNodeParameters(CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_memset_node_get_params_safe(node.Token, out NativeCudaGraphMemsetNodeParams parameters));
        return new CudaGraphMemsetNodeParameters(parameters);
    }

    public static CudaGraphMemcpyNodeParameters GetGraphMemcpyNodeParameters(CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_memcpy_node_get_params_safe(node.Token, out NativeCudaGraphMemcpyNodeParams parameters));
        return new CudaGraphMemcpyNodeParameters(parameters);
    }

    public static CudaGraphKernelNodeAttributeValue GetGraphKernelNodeAttribute(CudaGraphNode node, CudaGraphKernelNodeAttribute attribute)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_kernel_node_get_attribute_scalar_safe(node.Token, (int)attribute, out NativeCudaGraphKernelNodeAttributeValue value));
        return new CudaGraphKernelNodeAttributeValue(value);
    }

    public static void SetGraphMemsetNodeParameters(CudaGraphNode node, SafeCudaMemoryHandle destination, byte value, int count)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_memset_node_set_params_safe(node.Token, destination, value, new UIntPtr((uint)count)));
    }

    public static void SetGraphKernelNodeAttribute(CudaGraphNode node, CudaGraphKernelNodeAttributeValue value)
    {
        NativeCudaGraphKernelNodeAttributeValue nativeValue = value.ToNative();
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_kernel_node_set_attribute_scalar_safe(node.Token, in nativeValue));
    }

    public static void SetGraphMemcpyNodeParametersDeviceToDevice(CudaGraphNode node, SafeCudaMemoryHandle destination, SafeCudaMemoryHandle source, int count)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_memcpy_node_set_params_1d_device_to_device_safe(node.Token, destination, source, new UIntPtr((uint)count)));
    }

    public static void SetGraphMemcpyNodeParametersHostToDevice(CudaGraphNode node, SafeCudaMemoryHandle destination, SafeCudaPinnedMemoryHandle source, int count)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_memcpy_node_set_params_1d_host_to_device_safe(node.Token, destination, source, new UIntPtr((uint)count)));
    }

    public static void SetGraphMemcpyNodeParametersDeviceToHost(CudaGraphNode node, SafeCudaPinnedMemoryHandle destination, SafeCudaMemoryHandle source, int count)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_memcpy_node_set_params_1d_device_to_host_safe(node.Token, destination, source, new UIntPtr((uint)count)));
    }

    public static void SetGraphEventRecordNodeEvent(CudaGraphNode node, SafeCudaEventHandle eventHandle)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_event_record_node_set_event_safe(node.Token, eventHandle));
    }

    public static bool GraphEventRecordNodeHasEvent(CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_event_record_node_has_event_safe(node.Token, out int hasEvent));
        return hasEvent != 0;
    }

    public static void SetGraphEventWaitNodeEvent(CudaGraphNode node, SafeCudaEventHandle eventHandle)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_event_wait_node_set_event_safe(node.Token, eventHandle));
    }

    public static bool GraphEventWaitNodeHasEvent(CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_event_wait_node_has_event_safe(node.Token, out int hasEvent));
        return hasEvent != 0;
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

    public static ulong GetGraphNodeDependencyV2Count(CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_node_get_dependencies_v2_count_safe(node.Token, out UIntPtr count));
        return count.ToUInt64();
    }

    public static CudaGraphNodeDependency GetGraphNodeDependencyV2(CudaGraphNode node, ulong index)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_node_get_dependency_v2_safe(node.Token, new UIntPtr(index), out UIntPtr dependencyNode, out NativeCudaGraphEdgeData edgeData));
        return new CudaGraphNodeDependency(new CudaGraphNode(dependencyNode), new CudaGraphEdgeData(edgeData));
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

    public static ulong GetGraphNodeDependentV2Count(CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_node_get_dependent_nodes_v2_count_safe(node.Token, out UIntPtr count));
        return count.ToUInt64();
    }

    public static CudaGraphNodeDependency GetGraphNodeDependentV2(CudaGraphNode node, ulong index)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_node_get_dependent_node_v2_safe(node.Token, new UIntPtr(index), out UIntPtr dependentNode, out NativeCudaGraphEdgeData edgeData));
        return new CudaGraphNodeDependency(new CudaGraphNode(dependentNode), new CudaGraphEdgeData(edgeData));
    }

    public static void RemoveGraphDependencyV2(SafeCudaGraphHandle graph, CudaGraphNode fromNode, CudaGraphNode toNode, CudaGraphEdgeData edgeData)
    {
        NativeCudaGraphEdgeData nativeEdgeData = edgeData.ToNative();
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_remove_dependency_v2_safe(graph, fromNode.Token, toNode.Token, in nativeEdgeData));
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

    public static void SetGraphExecEventRecordNodeEvent(SafeCudaGraphExecHandle graphExec, CudaGraphNode node, SafeCudaEventHandle eventHandle)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_exec_event_record_node_set_event_safe(graphExec, node.Token, eventHandle));
    }

    public static void SetGraphExecEventWaitNodeEvent(SafeCudaGraphExecHandle graphExec, CudaGraphNode node, SafeCudaEventHandle eventHandle)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_exec_event_wait_node_set_event_safe(graphExec, node.Token, eventHandle));
    }

    public static void SetGraphExecMemcpyNodeParametersDeviceToDevice(SafeCudaGraphExecHandle graphExec, CudaGraphNode node, SafeCudaMemoryHandle destination, SafeCudaMemoryHandle source, int count)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_exec_memcpy_node_set_params_1d_device_to_device_safe(graphExec, node.Token, destination, source, new UIntPtr((uint)count)));
    }

    public static void SetGraphExecMemcpyNodeParametersHostToDevice(SafeCudaGraphExecHandle graphExec, CudaGraphNode node, SafeCudaMemoryHandle destination, SafeCudaPinnedMemoryHandle source, int count)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_exec_memcpy_node_set_params_1d_host_to_device_safe(graphExec, node.Token, destination, source, new UIntPtr((uint)count)));
    }

    public static void SetGraphExecMemcpyNodeParametersDeviceToHost(SafeCudaGraphExecHandle graphExec, CudaGraphNode node, SafeCudaPinnedMemoryHandle destination, SafeCudaMemoryHandle source, int count)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_exec_memcpy_node_set_params_1d_device_to_host_safe(graphExec, node.Token, destination, source, new UIntPtr((uint)count)));
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
