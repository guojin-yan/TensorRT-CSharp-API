using System;
using System.Collections.Generic;
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
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_get_capture_summary_safe(
            stream,
            out int status,
            out ulong captureId,
            out int hasGraph,
            out UIntPtr dependencyCount,
            out int hasEdgeData));
        return new CudaStreamCaptureInfo(
            (CudaStreamCaptureStatus)status,
            captureId,
            hasGraph != 0,
            dependencyCount.ToUInt64(),
            hasEdgeData != 0);
    }

    public static void UpdateStreamCaptureDependencies(
        SafeCudaStreamHandle stream,
        IReadOnlyList<CudaGraphNode> dependencies,
        CudaStreamCaptureDependencyMode mode)
    {
        if (dependencies.Count > 1000000)
        {
            throw new ArgumentOutOfRangeException(nameof(dependencies));
        }

        UIntPtr[] tokens = new UIntPtr[dependencies.Count];
        for (int index = 0; index < dependencies.Count; index++)
        {
            tokens[index] = dependencies[index].Token;
        }

        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_update_capture_dependencies_safe(
            stream,
            tokens,
            new UIntPtr((uint)tokens.Length),
            (uint)mode));
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

    public static CudaGraphNode AddGraphChildGraphNode(SafeCudaGraphHandle graph, SafeCudaGraphHandle childGraph)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_add_child_graph_node_safe(graph, childGraph, out UIntPtr node));
        return new CudaGraphNode(node);
    }

    public static CudaGraphNode AddGraphChildGraphNodeAfter(SafeCudaGraphHandle graph, CudaGraphNode dependencyNode, SafeCudaGraphHandle childGraph)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_add_child_graph_node_after_safe(graph, dependencyNode.Token, childGraph, out UIntPtr node));
        return new CudaGraphNode(node);
    }

    public static CudaGraphChildSnapshot GetGraphChildSnapshot(CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_child_graph_node_has_embedded_graph_safe(node.Token, out int hasEmbeddedGraph));
        if (hasEmbeddedGraph == 0)
        {
            return new CudaGraphChildSnapshot(false, 0, 0, 0);
        }

        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_child_graph_node_get_node_count_safe(node.Token, out UIntPtr nodeCount));
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_child_graph_node_get_root_node_count_safe(node.Token, out UIntPtr rootNodeCount));
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_child_graph_node_get_edge_count_safe(node.Token, out UIntPtr edgeCount));
        return new CudaGraphChildSnapshot(true, nodeCount.ToUInt64(), rootNodeCount.ToUInt64(), edgeCount.ToUInt64());
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

    public static CudaGraphNode AddGraphMemsetNode(SafeCudaGraphHandle graph, SafeCudaMemoryHandle destination, byte value, int count)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_add_memset_node_safe(
            graph,
            destination,
            value,
            new UIntPtr((uint)count),
            out UIntPtr node));
        return new CudaGraphNode(node);
    }

    public static CudaGraphNode AddGraphMemsetNodeAfter(SafeCudaGraphHandle graph, CudaGraphNode dependencyNode, SafeCudaMemoryHandle destination, byte value, int count)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_add_memset_node_after_safe(
            graph,
            dependencyNode.Token,
            destination,
            value,
            new UIntPtr((uint)count),
            out UIntPtr node));
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

    public static void RemoveGraphNode(SafeCudaGraphHandle graph, CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_destroy_node_owner_scoped_safe(graph, node.Token));
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

    public static CudaGraphKernelNodeParametersSnapshot GetGraphKernelNodeParametersSnapshot(CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_kernel_node_get_params_snapshot_safe(
            node.Token,
            out NativeCudaGraphKernelNodeParamsSnapshot snapshot));
        return new CudaGraphKernelNodeParametersSnapshot(snapshot);
    }

    public static CudaGraphHostNodeParametersSnapshot GetGraphHostNodeParametersSnapshot(CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_host_node_get_params_snapshot_safe(
            node.Token,
            out NativeCudaGraphHostNodeParamsSnapshot snapshot));
        return new CudaGraphHostNodeParametersSnapshot(snapshot);
    }

    public static CudaGraphMemoryAllocationNodeSnapshot GetGraphMemoryAllocationNodeSnapshot(CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_mem_alloc_node_get_params_snapshot_safe(
            node.Token,
            out NativeCudaGraphMemAllocNodeParamsSnapshot snapshot));
        return new CudaGraphMemoryAllocationNodeSnapshot(snapshot);
    }

    public static CudaGraphMemoryFreeNodeSnapshot GetGraphMemoryFreeNodeSnapshot(CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_mem_free_node_get_params_snapshot_safe(
            node.Token,
            out NativeCudaGraphMemFreeNodeParamsSnapshot snapshot));
        return new CudaGraphMemoryFreeNodeSnapshot(snapshot);
    }

    public static CudaGraphExternalSemaphoreNodeSnapshot GetGraphExternalSemaphoreSignalNodeSnapshot(CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_external_semaphore_signal_node_get_params_snapshot_safe(
            node.Token,
            out NativeCudaGraphExternalSemaphoreNodeParamsSnapshot snapshot));
        return new CudaGraphExternalSemaphoreNodeSnapshot(CudaGraphNodeType.ExternalSemaphoreSignal, snapshot);
    }

    public static CudaGraphExternalSemaphoreNodeSnapshot GetGraphExternalSemaphoreWaitNodeSnapshot(CudaGraphNode node)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_external_semaphore_wait_node_get_params_snapshot_safe(
            node.Token,
            out NativeCudaGraphExternalSemaphoreNodeParamsSnapshot snapshot));
        return new CudaGraphExternalSemaphoreNodeSnapshot(CudaGraphNodeType.ExternalSemaphoreWait, snapshot);
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

    public static SafeCudaGraphExecHandle InstantiateGraphWithParameters(SafeCudaGraphHandle graph, ulong flags)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_instantiate_with_params_safe(
            graph,
            flags,
            out SafeCudaGraphExecHandle graphExec,
            out _,
            out _,
            out _));
        return graphExec;
    }

    public static SafeCudaGraphExecHandle InstantiateGraphWithParameters(SafeCudaGraphHandle graph, ulong flags, SafeCudaStreamHandle uploadStream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_instantiate_with_params_on_stream_safe(
            graph,
            flags,
            uploadStream,
            out SafeCudaGraphExecHandle graphExec,
            out _,
            out _,
            out _));
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

    public static void SetGraphExecChildGraphNodeParameters(SafeCudaGraphExecHandle graphExec, CudaGraphNode node, SafeCudaGraphHandle childGraph)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_exec_child_graph_node_set_params_safe(graphExec, node.Token, childGraph));
    }

    public static void SetGraphExecMemsetNodeParameters(SafeCudaGraphExecHandle graphExec, CudaGraphNode node, SafeCudaMemoryHandle destination, byte value, int count)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_exec_memset_node_set_params_safe(
            graphExec,
            node.Token,
            destination,
            value,
            new UIntPtr((uint)count)));
    }

    public static CudaGraphExecUpdateSnapshot UpdateGraphExec(SafeCudaGraphExecHandle graphExec, SafeCudaGraphHandle graph)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_exec_update_copied_metadata_safe(
            graphExec,
            graph,
            out int result,
            out int hasErrorNode,
            out int errorNodeType,
            out int hasErrorFromNode,
            out int errorFromNodeType));
        return new CudaGraphExecUpdateSnapshot(
            (CudaGraphExecUpdateResult)result,
            hasErrorNode != 0 ? (CudaGraphNodeType?)errorNodeType : null,
            hasErrorFromNode != 0 ? (CudaGraphNodeType?)errorFromNodeType : null);
    }

    public static void CopyGraphKernelNodeAttributes(CudaGraphNode destinationNode, CudaGraphNode sourceNode)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_graph_kernel_node_copy_attributes_safe(destinationNode.Token, sourceNode.Token));
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
