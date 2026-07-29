using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaGraph
{
    /// <summary>
    /// Finds the clone-owned node that corresponds to an original graph node.
    /// 查找 cloned graph 中对应原始 graph 节点的 node token。
    /// </summary>
    /// <param name="originalNode">The node from the original graph. 原始 graph 中的节点。</param>
    /// <returns>The node token owned by this cloned graph. 由当前 cloned graph 拥有的 node token。</returns>
    public CudaGraphNode FindNodeInClone(CudaGraphNode originalNode)
    {
        return NativeCudaApi.FindGraphNodeInClone(originalNode, _handle);
    }

    /// <summary>
    /// Returns whether CUDA reports that a node belongs to this graph.
    /// 返回 CUDA 是否报告指定 node 属于当前 graph。
    /// </summary>
    /// <param name="node">The graph node token. graph node 的 token。</param>
    /// <returns><see langword="true"/> when the node's containing graph is this graph. 当 node 的 containing graph 是当前 graph 时返回 <see langword="true"/>。</returns>
    public bool ContainsNode(CudaGraphNode node)
    {
        return NativeCudaApi.IsGraphNodeInGraph(_handle, node);
    }

    /// <summary>
    /// Gets copied topology metadata for the embedded graph of a child-graph node.
    /// 获取 child-graph 节点中 embedded graph 的复制型拓扑元数据。
    /// </summary>
    /// <param name="node">A child-graph node token. Child-graph 节点 token。</param>
    /// <returns>A scalar-only embedded graph snapshot. 仅包含标量的 embedded graph 快照。</returns>
    public CudaGraphChildSnapshot GetChildGraphSnapshot(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphChildSnapshot(node);
    }

    /// <summary>
    /// Gets the CUDA graph node type for a graph-owned node token.
    /// 获取 graph-owned node token 的 CUDA graph node 类型。
    /// </summary>
    /// <param name="node">The node token. 节点 token。</param>
    /// <returns>The CUDA graph node type. CUDA graph node 类型。</returns>
    public static CudaGraphNodeType GetNodeType(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphNodeType(node);
    }

    /// <summary>
    /// Gets a copied topology snapshot for a CUDA graph node token.
    /// 获取 CUDA graph node token 的复制型拓扑快照。
    /// </summary>
    /// <param name="node">The graph node token. 图节点令牌。 </param>
    /// <returns>A snapshot containing node-level topology counters. 包含 node 级拓扑计数的快照。</returns>
    /// <remarks>
    /// The node token remains graph-owned; this API only copies scalar diagnostics from CUDA.
    /// node token 仍由 graph 拥有；该 API 仅从 CUDA 复制标量诊断信息。
    /// </remarks>
    public static CudaGraphNodeTopologySnapshot GetNodeTopologySnapshot(CudaGraphNode node)
    {
        return new CudaGraphNodeTopologySnapshot(
            GetNodeType(node),
            GetDependencyCount(node),
            GetDependencyWithEdgeDataCount(node),
            GetDependentCount(node),
            GetDependentWithEdgeDataCount(node));
    }

    /// <summary>
    /// Gets copied snapshots for node dependencies.
    /// 获取指定 node 依赖项的复制型快照列表。
    /// </summary>
    /// <param name="node">The graph node token. 图节点令牌。 </param>
    /// <param name="includeEdgeData">Whether to prefer CUDA edge-data queries. 是否优先使用 CUDA edge-data 查询。</param>
    /// <param name="maxDependencies">The maximum number of dependencies to copy. 要复制的最大依赖数量。</param>
    /// <returns>Bounded dependency snapshots. 有界依赖快照列表。</returns>
    public static IReadOnlyList<CudaGraphAdjacentNodeSnapshot> GetNodeDependencySnapshotList(
        CudaGraphNode node,
        bool includeEdgeData = true,
        ulong maxDependencies = ulong.MaxValue)
    {
        if (includeEdgeData)
        {
            ulong edgeDataCount = Math.Min(GetDependencyWithEdgeDataCount(node), maxDependencies);
            List<CudaGraphAdjacentNodeSnapshot> edgeDataSnapshots = new List<CudaGraphAdjacentNodeSnapshot>(ToListCapacity(edgeDataCount));
            for (ulong index = 0; index < edgeDataCount; index++)
            {
                CudaGraphNodeDependency dependency = GetDependencyWithEdgeData(node, index);
                edgeDataSnapshots.Add(new CudaGraphAdjacentNodeSnapshot(index, dependency.Node, dependency.EdgeData, true));
            }

            return edgeDataSnapshots;
        }

        ulong count = Math.Min(GetDependencyCount(node), maxDependencies);
        List<CudaGraphAdjacentNodeSnapshot> snapshots = new List<CudaGraphAdjacentNodeSnapshot>(ToListCapacity(count));
        for (ulong index = 0; index < count; index++)
        {
            snapshots.Add(new CudaGraphAdjacentNodeSnapshot(index, GetDependency(node, index), CudaGraphEdgeData.Default, false));
        }

        return snapshots;
    }

    /// <summary>
    /// Gets copied snapshots for node dependents.
    /// 获取依赖指定 node 的节点复制型快照列表。
    /// </summary>
    /// <param name="node">The graph node token. 图节点令牌。 </param>
    /// <param name="includeEdgeData">Whether to prefer CUDA edge-data queries. 是否优先使用 CUDA edge-data 查询。</param>
    /// <param name="maxDependents">The maximum number of dependents to copy. 要复制的最大 dependent 数量。</param>
    /// <returns>Bounded dependent snapshots. 有界 dependent 快照列表。</returns>
    public static IReadOnlyList<CudaGraphAdjacentNodeSnapshot> GetNodeDependentSnapshotList(
        CudaGraphNode node,
        bool includeEdgeData = true,
        ulong maxDependents = ulong.MaxValue)
    {
        if (includeEdgeData)
        {
            ulong edgeDataCount = Math.Min(GetDependentWithEdgeDataCount(node), maxDependents);
            List<CudaGraphAdjacentNodeSnapshot> edgeDataSnapshots = new List<CudaGraphAdjacentNodeSnapshot>(ToListCapacity(edgeDataCount));
            for (ulong index = 0; index < edgeDataCount; index++)
            {
                CudaGraphNodeDependency dependent = GetDependentWithEdgeData(node, index);
                edgeDataSnapshots.Add(new CudaGraphAdjacentNodeSnapshot(index, dependent.Node, dependent.EdgeData, true));
            }

            return edgeDataSnapshots;
        }

        ulong count = Math.Min(GetDependentCount(node), maxDependents);
        List<CudaGraphAdjacentNodeSnapshot> snapshots = new List<CudaGraphAdjacentNodeSnapshot>(ToListCapacity(count));
        for (ulong index = 0; index < count; index++)
        {
            snapshots.Add(new CudaGraphAdjacentNodeSnapshot(index, GetDependent(node, index), CudaGraphEdgeData.Default, false));
        }

        return snapshots;
    }

    /// <summary>
    /// Gets copied scalar parameters from a CUDA graph memset node.
    /// 从 CUDA graph memset 节点获取复制出的标量参数。
    /// </summary>
    /// <param name="node">The memset node token. memset 节点 token。</param>
    /// <returns>Copied memset node parameters. 复制出的 memset node 参数。</returns>
    /// <remarks>
    /// The returned destination address is diagnostic only and does not transfer memory ownership.
    /// 返回的目标地址仅用于诊断，不转移内存所有权。
    /// </remarks>
    public static CudaGraphMemsetNodeParameters GetMemsetNodeParameters(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphMemsetNodeParameters(node);
    }

    /// <summary>
    /// Gets copied diagnostic parameters from a CUDA graph memcpy node.
    /// 从 CUDA graph memcpy 节点获取复制出的诊断参数。
    /// </summary>
    /// <param name="node">The memcpy node token. memcpy 节点 token。</param>
    /// <returns>Copied memcpy node parameters. 复制出的 memcpy node 参数。</returns>
    /// <remarks>
    /// Returned addresses are diagnostic numeric values only and do not transfer memory ownership.
    /// 返回的地址仅为诊断数值，不转移内存所有权。
    /// </remarks>
    public static CudaGraphMemcpyNodeParameters GetMemcpyNodeParameters(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphMemcpyNodeParameters(node);
    }

    /// <summary>
    /// Gets a copied, pointer-free typed descriptor for supported CUDA graph node parameters.
    /// 获取支持的 CUDA graph node 参数的复制型、无指针 typed descriptor。
    /// </summary>
    /// <param name="node">The graph-owned node token. Graph 拥有的 node token。</param>
    /// <returns>A copied node parameter descriptor that never exposes borrowed CUDA handles or native pointers. 不暴露 borrowed CUDA handle 或原生指针的复制型 node 参数 descriptor。</returns>
    /// <remarks>
    /// This method intentionally collapses event nodes to event-presence booleans and memcpy/memset nodes to copied scalar metadata.
    /// 该方法有意将 event node 折叠为 event-presence 布尔值，并将 memcpy/memset node 折叠为复制出的标量 metadata。
    /// </remarks>
    public static CudaGraphNodeParamsDescriptor GetNodeParamsDescriptor(CudaGraphNode node)
    {
        CudaGraphNodeType nodeType = GetNodeType(node);
        return nodeType switch
        {
            CudaGraphNodeType.Empty => CudaGraphNodeParamsDescriptor.Empty(),
            CudaGraphNodeType.Memset => CudaGraphNodeParamsDescriptor.Memset(GetMemsetNodeParameters(node)),
            CudaGraphNodeType.Memcpy => CudaGraphNodeParamsDescriptor.Memcpy(GetMemcpyNodeParameters(node)),
            CudaGraphNodeType.EventRecord => CudaGraphNodeParamsDescriptor.EventRecord(EventRecordNodeHasEvent(node)),
            CudaGraphNodeType.WaitEvent => CudaGraphNodeParamsDescriptor.EventWait(EventWaitNodeHasEvent(node)),
            _ => CudaGraphNodeParamsDescriptor.Unsupported(nodeType)
        };
    }

    /// <summary>
    /// Gets a copied scalar attribute descriptor from a CUDA graph kernel node.
    /// 从 CUDA graph kernel 节点获取复制出的标量 attribute descriptor。
    /// </summary>
    /// <param name="node">The kernel node token. kernel 节点 token。</param>
    /// <param name="attribute">The supported scalar attribute. 支持的标量 attribute。</param>
    /// <returns>The copied attribute descriptor. 复制出的 attribute descriptor。</returns>
    /// <remarks>
    /// This method supports only scalar or scalar-vector attributes with stable ABI: cooperative, priority, cluster dimension, and cluster scheduling policy preference.
    /// 该方法只支持 ABI 稳定的标量或标量向量 attribute：cooperative、priority、cluster dimension 和 cluster scheduling policy preference。
    /// </remarks>
    public static CudaGraphKernelNodeAttributeValue GetKernelNodeAttribute(CudaGraphNode node, CudaGraphKernelNodeAttribute attribute)
    {
        ValidateKernelNodeAttribute(attribute, nameof(attribute));
        return NativeCudaApi.GetGraphKernelNodeAttribute(node, attribute);
    }

    /// <summary>Gets copied scalar parameters for a kernel node without exposing function or argument pointers. 获取 kernel 节点的复制型标量参数，不暴露 function 或 argument pointer。</summary>
    public static CudaGraphKernelNodeParametersSnapshot GetKernelNodeParametersSnapshot(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphKernelNodeParametersSnapshot(node);
    }

    /// <summary>Gets callback-presence metadata for a host node without exposing callback pointers. 获取 host 节点的 callback 存在性元数据，不暴露 callback pointer。</summary>
    public static CudaGraphHostNodeParametersSnapshot GetHostNodeParametersSnapshot(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphHostNodeParametersSnapshot(node);
    }

    /// <summary>Gets copied scalar metadata for a graph memory-allocation node. 获取 graph memory-allocation 节点的复制型标量元数据。</summary>
    public static CudaGraphMemoryAllocationNodeSnapshot GetMemoryAllocationNodeSnapshot(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphMemoryAllocationNodeSnapshot(node);
    }

    /// <summary>Gets pointer-presence metadata for a graph memory-free node. 获取 graph memory-free 节点的 pointer 存在性元数据。</summary>
    public static CudaGraphMemoryFreeNodeSnapshot GetMemoryFreeNodeSnapshot(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphMemoryFreeNodeSnapshot(node);
    }

    /// <summary>Gets copied count metadata for an external-semaphore signal node. 获取 external-semaphore signal 节点的复制型 count 元数据。</summary>
    public static CudaGraphExternalSemaphoreNodeSnapshot GetExternalSemaphoreSignalNodeSnapshot(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphExternalSemaphoreSignalNodeSnapshot(node);
    }

    /// <summary>Gets copied count metadata for an external-semaphore wait node. 获取 external-semaphore wait 节点的复制型 count 元数据。</summary>
    public static CudaGraphExternalSemaphoreNodeSnapshot GetExternalSemaphoreWaitNodeSnapshot(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphExternalSemaphoreWaitNodeSnapshot(node);
    }

    /// <summary>
    /// Copies CUDA kernel-node attributes from one graph-owned node to another.
    /// 将 CUDA kernel node attribute 从一个 graph-owned 节点复制到另一个节点。
    /// </summary>
    /// <param name="destinationNode">The destination kernel node. 目标 kernel 节点。</param>
    /// <param name="sourceNode">The source kernel node. 源 kernel 节点。</param>
    public static void CopyKernelNodeAttributes(CudaGraphNode destinationNode, CudaGraphNode sourceNode)
    {
        NativeCudaApi.CopyGraphKernelNodeAttributes(destinationNode, sourceNode);
    }

    /// <summary>
    /// Gets whether a CUDA graph kernel node is marked cooperative.
    /// 获取 CUDA graph kernel 节点是否标记为 cooperative。
    /// </summary>
    /// <param name="node">The kernel node token. kernel 节点 token。</param>
    public static bool GetKernelNodeCooperative(CudaGraphNode node)
    {
        return GetKernelNodeAttribute(node, CudaGraphKernelNodeAttribute.Cooperative).CooperativeEnabled;
    }

    /// <summary>
    /// Gets a CUDA graph kernel node priority.
    /// 获取 CUDA graph kernel 节点 priority。
    /// </summary>
    /// <param name="node">The kernel node token. kernel 节点 token。</param>
    public static int GetKernelNodePriority(CudaGraphNode node)
    {
        return GetKernelNodeAttribute(node, CudaGraphKernelNodeAttribute.Priority).IntValue;
    }

    /// <summary>
    /// Gets a CUDA graph kernel node cluster dimension descriptor.
    /// 获取 CUDA graph kernel 节点 cluster dimension descriptor。
    /// </summary>
    /// <param name="node">The kernel node token. kernel 节点 token。</param>
    public static CudaGraphKernelNodeAttributeValue GetKernelNodeClusterDimension(CudaGraphNode node)
    {
        return GetKernelNodeAttribute(node, CudaGraphKernelNodeAttribute.ClusterDimension);
    }

    /// <summary>
    /// Gets a CUDA graph kernel node cluster scheduling policy preference.
    /// 获取 CUDA graph kernel 节点 cluster scheduling policy preference。
    /// </summary>
    /// <param name="node">The kernel node token. kernel 节点 token。</param>
    public static CudaClusterSchedulingPolicyPreference GetKernelNodeClusterSchedulingPolicy(CudaGraphNode node)
    {
        return GetKernelNodeAttribute(node, CudaGraphKernelNodeAttribute.ClusterSchedulingPolicyPreference).ClusterSchedulingPolicyPreference;
    }

}
