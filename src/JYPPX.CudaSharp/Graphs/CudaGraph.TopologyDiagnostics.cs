using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaGraph
{
    /// <summary>
    /// Adds a dependency edge between two nodes in this graph.
    /// 在当前 graph 的两个节点之间添加依赖边。
    /// </summary>
    /// <param name="fromNode">The source node. 源节点。</param>
    /// <param name="toNode">The destination node. 目标节点。</param>
    public void AddDependency(CudaGraphNode fromNode, CudaGraphNode toNode)
    {
        NativeCudaApi.AddGraphDependency(_handle, fromNode, toNode);
    }

    /// <summary>
    /// Adds a dependency edge with explicit CUDA edge data.
    /// 使用显式 CUDA edge data 添加两个节点之间的依赖边。
    /// </summary>
    /// <param name="fromNode">The source node. 源节点。</param>
    /// <param name="toNode">The destination node. 目标节点。</param>
    /// <param name="edgeData">The CUDA graph edge data. CUDA graph 边数据。</param>
    /// <remarks>
    /// This API uses graph-owned node tokens and copied edge-data values; no native pointer is exposed or retained by public C#.
    /// 该 API 使用 graph-owned node token 和已复制的 edge-data 值；public C# 不暴露或保留原生指针。
    /// </remarks>
    public void AddDependency(CudaGraphNode fromNode, CudaGraphNode toNode, CudaGraphEdgeData edgeData)
    {
        NativeCudaApi.AddGraphDependencyV2(_handle, fromNode, toNode, edgeData);
    }

    /// <summary>
    /// Removes a dependency edge between two nodes in this graph.
    /// 删除当前 graph 的两个节点之间的依赖边。
    /// </summary>
    /// <param name="fromNode">The source node. 源节点。</param>
    /// <param name="toNode">The destination node. 目标节点。</param>
    public void RemoveDependency(CudaGraphNode fromNode, CudaGraphNode toNode)
    {
        NativeCudaApi.RemoveGraphDependency(_handle, fromNode, toNode);
    }

    /// <summary>
    /// Removes a node after verifying that it belongs to this graph.
    /// 在确认节点属于当前 graph 后将其删除。
    /// </summary>
    public void RemoveNode(CudaGraphNode node)
    {
        NativeCudaApi.RemoveGraphNode(_handle, node);
    }

    /// <summary>
    /// Removes a dependency edge with explicit CUDA edge data.
    /// 使用显式 CUDA edge data 删除两个节点之间的依赖边。
    /// </summary>
    /// <param name="fromNode">The source node. 源节点。</param>
    /// <param name="toNode">The destination node. 目标节点。</param>
    /// <param name="edgeData">The CUDA graph edge data. CUDA graph 边数据。</param>
    public void RemoveDependency(CudaGraphNode fromNode, CudaGraphNode toNode, CudaGraphEdgeData edgeData)
    {
        NativeCudaApi.RemoveGraphDependencyV2(_handle, fromNode, toNode, edgeData);
    }

    /// <summary>
    /// Gets a node token by index from this graph.
    /// 按索引获取当前 graph 中的节点 token。
    /// </summary>
    /// <param name="index">The zero-based node index. 从零开始的节点索引。</param>
    /// <returns>A graph-owned node token. 由 graph 拥有的 node token。</returns>
    public CudaGraphNode GetNode(ulong index)
    {
        return NativeCudaApi.GetGraphNode(_handle, index);
    }

    /// <summary>
    /// Gets a root node token by index from this graph.
    /// 按索引获取当前 graph 中的 root node token。
    /// </summary>
    /// <param name="index">The zero-based root node index. 从零开始的 root node 索引。</param>
    /// <returns>A graph-owned node token. 由 graph 拥有的 node token。</returns>
    public CudaGraphNode GetRootNode(ulong index)
    {
        return NativeCudaApi.GetGraphRootNode(_handle, index);
    }

    /// <summary>
    /// Gets a dependency edge by index from this graph.
    /// 按索引获取当前 graph 中的依赖边。
    /// </summary>
    /// <param name="index">The zero-based edge index. 从零开始的边索引。</param>
    /// <returns>The graph edge. Graph 依赖边。</returns>
    public CudaGraphEdge GetEdge(ulong index)
    {
        return NativeCudaApi.GetGraphEdge(_handle, index);
    }

    /// <summary>
    /// Gets the number of dependency edges available through the CUDA edge-data query.
    /// 获取可通过 CUDA edge-data 查询访问的依赖边数量。
    /// </summary>
    public ulong GetEdgeWithEdgeDataCount()
    {
        return NativeCudaApi.GetGraphEdgeV2Count(_handle);
    }

    /// <summary>
    /// Gets a copied topology snapshot for this CUDA graph.
    /// 获取当前 CUDA graph 的复制型拓扑快照。
    /// </summary>
    /// <returns>A snapshot containing graph-level topology counters. 包含 graph 级拓扑计数的快照。</returns>
    /// <remarks>
    /// The snapshot contains copied scalar values only and does not expose borrowed CUDA pointers.
    /// 该快照仅包含复制出的标量值，不暴露借用的 CUDA 指针。
    /// </remarks>
    public CudaGraphTopologySnapshot GetTopologySnapshot()
    {
        return new CudaGraphTopologySnapshot(
            NodeCount,
            RootNodeCount,
            EdgeCount,
            GetEdgeWithEdgeDataCount());
    }

    /// <summary>
    /// Gets copied snapshots for graph nodes.
    /// 获取 graph node 的复制型快照列表。
    /// </summary>
    /// <param name="maxNodes">The maximum number of nodes to copy. 要复制的最大节点数量。</param>
    /// <returns>Bounded node snapshots. 有界 node 快照列表。</returns>
    public IReadOnlyList<CudaGraphNodeSnapshot> GetNodeSnapshotList(ulong maxNodes = ulong.MaxValue)
    {
        ulong count = Math.Min(NodeCount, maxNodes);
        List<CudaGraphNodeSnapshot> snapshots = new List<CudaGraphNodeSnapshot>(ToListCapacity(count));
        for (ulong index = 0; index < count; index++)
        {
            CudaGraphNode node = GetNode(index);
            snapshots.Add(new CudaGraphNodeSnapshot(index, node, GetNodeTopologySnapshot(node)));
        }

        return snapshots;
    }

    /// <summary>
    /// Gets copied snapshots for root graph nodes.
    /// 获取 root graph node 的复制型快照列表。
    /// </summary>
    /// <param name="maxNodes">The maximum number of root nodes to copy. 要复制的最大 root 节点数量。</param>
    /// <returns>Bounded root-node snapshots. 有界 root-node 快照列表。</returns>
    public IReadOnlyList<CudaGraphNodeSnapshot> GetRootNodeSnapshotList(ulong maxNodes = ulong.MaxValue)
    {
        ulong count = Math.Min(RootNodeCount, maxNodes);
        List<CudaGraphNodeSnapshot> snapshots = new List<CudaGraphNodeSnapshot>(ToListCapacity(count));
        for (ulong index = 0; index < count; index++)
        {
            CudaGraphNode node = GetRootNode(index);
            snapshots.Add(new CudaGraphNodeSnapshot(index, node, GetNodeTopologySnapshot(node)));
        }

        return snapshots;
    }

    /// <summary>
    /// Gets copied snapshots for graph dependency edges.
    /// 获取 graph 依赖边的复制型快照列表。
    /// </summary>
    /// <param name="includeEdgeData">Whether to prefer CUDA edge-data queries. 是否优先使用 CUDA edge-data 查询。</param>
    /// <param name="maxEdges">The maximum number of edges to copy. 要复制的最大边数量。</param>
    /// <returns>Bounded edge snapshots. 有界 edge 快照列表。</returns>
    public IReadOnlyList<CudaGraphEdgeSnapshot> GetEdgeSnapshotList(bool includeEdgeData = true, ulong maxEdges = ulong.MaxValue)
    {
        if (includeEdgeData)
        {
            ulong edgeDataCount = Math.Min(GetEdgeWithEdgeDataCount(), maxEdges);
            List<CudaGraphEdgeSnapshot> edgeDataSnapshots = new List<CudaGraphEdgeSnapshot>(ToListCapacity(edgeDataCount));
            for (ulong index = 0; index < edgeDataCount; index++)
            {
                CudaGraphEdgeWithData edge = GetEdgeWithEdgeData(index);
                edgeDataSnapshots.Add(new CudaGraphEdgeSnapshot(index, edge.From, edge.To, edge.EdgeData, true));
            }

            return edgeDataSnapshots;
        }

        ulong count = Math.Min(EdgeCount, maxEdges);
        List<CudaGraphEdgeSnapshot> snapshots = new List<CudaGraphEdgeSnapshot>(ToListCapacity(count));
        for (ulong index = 0; index < count; index++)
        {
            CudaGraphEdge edge = GetEdge(index);
            snapshots.Add(new CudaGraphEdgeSnapshot(index, edge.From, edge.To, CudaGraphEdgeData.Default, false));
        }

        return snapshots;
    }

    /// <summary>
    /// Gets a bounded copied diagnostic snapshot for this graph.
    /// 获取当前 graph 的有界复制型诊断快照。
    /// </summary>
    /// <param name="maxNodes">The maximum number of graph nodes to copy. 要复制的最大 graph node 数量。</param>
    /// <param name="maxRootNodes">The maximum number of root nodes to copy. 要复制的最大 root node 数量。</param>
    /// <param name="maxEdges">The maximum number of dependency edges to copy. 要复制的最大依赖边数量。</param>
    /// <returns>A bounded copied graph diagnostic snapshot. 有界复制型 graph 诊断快照。</returns>
    public CudaGraphDiagnosticSnapshot GetDiagnosticSnapshot(
        ulong maxNodes = 64,
        ulong maxRootNodes = 64,
        ulong maxEdges = 64)
    {
        return new CudaGraphDiagnosticSnapshot(
            GetTopologySnapshot(),
            GetNodeSnapshotList(maxNodes),
            GetRootNodeSnapshotList(maxRootNodes),
            GetEdgeSnapshotList(includeEdgeData: true, maxEdges: maxEdges));
    }

    /// <summary>
    /// Gets a dependency edge and CUDA edge data by index from this graph.
    /// 按索引获取当前 graph 中的依赖边和 CUDA edge data。
    /// </summary>
    /// <param name="index">The zero-based edge index. 从零开始的边索引。</param>
    /// <returns>The graph edge and edge data. Graph 依赖边和 edge data。</returns>
    public CudaGraphEdgeWithData GetEdgeWithEdgeData(ulong index)
    {
        return NativeCudaApi.GetGraphEdgeV2(_handle, index);
    }

    /// <summary>
    /// Exports this CUDA graph to a Graphviz DOT file for diagnostics.
    /// 将当前 CUDA graph 导出为 Graphviz DOT 诊断文件。
    /// </summary>
    /// <param name="path">The output file path. 输出文件路径。</param>
    /// <param name="flags">CUDA graph debug DOT flags. CUDA graph debug DOT 标志。</param>
    public void ExportDebugDot(string path, CudaGraphDebugDotFlags flags = CudaGraphDebugDotFlags.None)
    {
        NativeCudaApi.ExportGraphDebugDot(_handle, path, flags);
    }

}
