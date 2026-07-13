using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Managed wrapper around a captured CUDA graph.
/// 已捕获 CUDA graph 的托管封装。
/// </summary>
public sealed class CudaGraph : IDisposable
{
    private readonly SafeCudaGraphHandle _handle;

    internal CudaGraph(SafeCudaGraphHandle handle)
    {
        _handle = handle ?? throw new ArgumentNullException(nameof(handle));
    }

    internal SafeCudaGraphHandle Handle => _handle;

    /// <summary>
    /// Creates an empty CUDA graph.
    /// 创建一个空 CUDA graph。
    /// </summary>
    /// <param name="flags">CUDA graph creation flags. CUDA graph 创建标志。</param>
    /// <returns>A managed CUDA graph wrapper. 托管 CUDA graph 封装。</returns>
    public static CudaGraph Create(uint flags = 0)
    {
        NativeBridgeLoader.EnsureInitialized();
        return new CudaGraph(NativeCudaApi.CreateGraph(flags));
    }

    /// <summary>
    /// Gets the number of nodes currently in this graph.
    /// 获取当前 graph 中的 node 数量。
    /// </summary>
    public ulong NodeCount => NativeCudaApi.GetGraphNodeCount(_handle);

    /// <summary>
    /// Gets the number of root nodes currently in this graph.
    /// 获取当前 graph 中的 root node 数量。
    /// </summary>
    public ulong RootNodeCount => NativeCudaApi.GetGraphRootNodeCount(_handle);

    /// <summary>
    /// Gets the number of dependency edges currently in this graph.
    /// 获取当前 graph 中的依赖 edge 数量。
    /// </summary>
    public ulong EdgeCount => NativeCudaApi.GetGraphEdgeCount(_handle);

    /// <summary>
    /// Gets the CUDA graph identifier when supported by the loaded CUDA runtime.
    /// 在当前 CUDA runtime 支持时获取 CUDA graph 标识符。
    /// </summary>
    public uint Id => NativeCudaApi.GetGraphId(_handle);

    /// <summary>
    /// Clones this CUDA graph into a new managed graph wrapper.
    /// 将当前 CUDA graph 克隆为新的托管 graph 封装。
    /// </summary>
    /// <returns>A cloned CUDA graph. 克隆后的 CUDA graph。</returns>
    public CudaGraph Clone()
    {
        return new CudaGraph(NativeCudaApi.CloneGraph(_handle));
    }

    /// <summary>
    /// Adds an empty node to this graph.
    /// 向当前 graph 添加一个空节点。
    /// </summary>
    /// <returns>A graph-owned node token. 由 graph 拥有的 node token。</returns>
    public CudaGraphNode AddEmptyNode()
    {
        return NativeCudaApi.AddGraphEmptyNode(_handle);
    }

    /// <summary>
    /// Adds an empty node that depends on an existing node.
    /// 添加一个依赖现有节点的空节点。
    /// </summary>
    /// <param name="dependencyNode">The dependency node. 依赖节点。</param>
    /// <returns>A graph-owned node token. 由 graph 拥有的 node token。</returns>
    public CudaGraphNode AddEmptyNodeAfter(CudaGraphNode dependencyNode)
    {
        return NativeCudaApi.AddGraphEmptyNodeAfter(_handle, dependencyNode);
    }

    /// <summary>
    /// Adds an event-record node to this graph.
    /// 向当前 graph 添加 event-record 节点。
    /// </summary>
    /// <param name="eventHandle">The caller-owned CUDA event used by the graph node. graph 节点使用的调用方拥有的 CUDA event。</param>
    /// <returns>A graph-owned event-record node token. 由 graph 拥有的 event-record node token。</returns>
    /// <remarks>
    /// The event must remain alive while the graph or any executable graph created from it may use this node.
    /// 该 event 必须在 graph 或由该 graph 创建的 executable graph 可能使用该节点期间保持存活。
    /// </remarks>
    public CudaGraphNode AddEventRecordNode(CudaEvent eventHandle)
    {
        return AddEventRecordNodeAfter(default, eventHandle);
    }

    /// <summary>
    /// Adds an event-record node that depends on an existing graph node.
    /// 添加一个依赖现有节点的 event-record 节点。
    /// </summary>
    /// <param name="dependencyNode">The dependency node. 依赖节点。</param>
    /// <param name="eventHandle">The caller-owned CUDA event used by the graph node. graph 节点使用的调用方拥有的 CUDA event。</param>
    /// <returns>A graph-owned event-record node token. 由 graph 拥有的 event-record node token。</returns>
    /// <remarks>
    /// The event must remain alive while the graph or any executable graph created from it may use this node.
    /// 该 event 必须在 graph 或由该 graph 创建的 executable graph 可能使用该节点期间保持存活。
    /// </remarks>
    public CudaGraphNode AddEventRecordNodeAfter(CudaGraphNode dependencyNode, CudaEvent eventHandle)
    {
        if (eventHandle == null)
        {
            throw new ArgumentNullException(nameof(eventHandle));
        }

        return NativeCudaApi.AddGraphEventRecordNode(_handle, dependencyNode, eventHandle.Handle);
    }

    /// <summary>
    /// Adds an event-wait node to this graph.
    /// 向当前 graph 添加 event-wait 节点。
    /// </summary>
    /// <param name="eventHandle">The caller-owned CUDA event waited on by the graph node. graph 节点等待的调用方拥有的 CUDA event。</param>
    /// <returns>A graph-owned event-wait node token. 由 graph 拥有的 event-wait node token。</returns>
    /// <remarks>
    /// The event must remain alive while the graph or any executable graph created from it may use this node.
    /// 该 event 必须在 graph 或由该 graph 创建的 executable graph 可能使用该节点期间保持存活。
    /// </remarks>
    public CudaGraphNode AddEventWaitNode(CudaEvent eventHandle)
    {
        return AddEventWaitNodeAfter(default, eventHandle);
    }

    /// <summary>
    /// Adds an event-wait node that depends on an existing graph node.
    /// 添加一个依赖现有节点的 event-wait 节点。
    /// </summary>
    /// <param name="dependencyNode">The dependency node. 依赖节点。</param>
    /// <param name="eventHandle">The caller-owned CUDA event waited on by the graph node. graph 节点等待的调用方拥有的 CUDA event。</param>
    /// <returns>A graph-owned event-wait node token. 由 graph 拥有的 event-wait node token。</returns>
    /// <remarks>
    /// The event must remain alive while the graph or any executable graph created from it may use this node.
    /// 该 event 必须在 graph 或由该 graph 创建的 executable graph 可能使用该节点期间保持存活。
    /// </remarks>
    public CudaGraphNode AddEventWaitNodeAfter(CudaGraphNode dependencyNode, CudaEvent eventHandle)
    {
        if (eventHandle == null)
        {
            throw new ArgumentNullException(nameof(eventHandle));
        }

        return NativeCudaApi.AddGraphEventWaitNode(_handle, dependencyNode, eventHandle.Handle);
    }

    /// <summary>
    /// Adds a 1D device-to-device memcpy node to this graph.
    /// 向当前 graph 添加 1D device-to-device memcpy 节点。
    /// </summary>
    /// <param name="destination">The destination device memory owner. 目标设备内存所有者。</param>
    /// <param name="source">The source device memory owner. 源设备内存所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <returns>A graph-owned memcpy node token. 由 graph 拥有的 memcpy node token。</returns>
    /// <remarks>
    /// The memory owners must remain alive while the graph or any executable graph created from it may use this node.
    /// 内存所有者必须在 graph 或由该 graph 创建的 executable graph 可能使用该节点期间保持存活。
    /// </remarks>
    public CudaGraphNode AddDeviceToDeviceMemcpyNode(CudaMemory destination, CudaMemory source, int count)
    {
        return AddDeviceToDeviceMemcpyNodeAfter(default, destination, source, count);
    }

    /// <summary>
    /// Adds a dependent 1D device-to-device memcpy node to this graph.
    /// 向当前 graph 添加带依赖的 1D device-to-device memcpy 节点。
    /// </summary>
    /// <param name="dependencyNode">The dependency node, or the default token for no dependency. 依赖节点；默认 token 表示无依赖。</param>
    /// <param name="destination">The destination device memory owner. 目标设备内存所有者。</param>
    /// <param name="source">The source device memory owner. 源设备内存所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <returns>A graph-owned memcpy node token. 由 graph 拥有的 memcpy node token。</returns>
    public CudaGraphNode AddDeviceToDeviceMemcpyNodeAfter(CudaGraphNode dependencyNode, CudaMemory destination, CudaMemory source, int count)
    {
        ValidateDeviceMemory(destination, nameof(destination));
        ValidateDeviceMemory(source, nameof(source));
        ValidateMemcpyCount(count, destination.SizeInBytes, source.SizeInBytes, nameof(count));
        return NativeCudaApi.AddGraphMemcpyNode1DDeviceToDevice(_handle, dependencyNode, destination.Handle, source.Handle, count);
    }

    /// <summary>
    /// Adds a 1D pinned-host-to-device memcpy node to this graph.
    /// 向当前 graph 添加 1D pinned-host-to-device memcpy 节点。
    /// </summary>
    /// <param name="destination">The destination device memory owner. 目标设备内存所有者。</param>
    /// <param name="source">The source pinned host memory owner. 源 pinned host memory 所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <returns>A graph-owned memcpy node token. 由 graph 拥有的 memcpy node token。</returns>
    /// <remarks>
    /// The memory owners must remain alive while the graph or any executable graph created from it may use this node.
    /// 内存所有者必须在 graph 或由该 graph 创建的 executable graph 可能使用该节点期间保持存活。
    /// </remarks>
    public CudaGraphNode AddHostToDeviceMemcpyNode(CudaMemory destination, CudaPinnedMemory source, int count)
    {
        return AddHostToDeviceMemcpyNodeAfter(default, destination, source, count);
    }

    /// <summary>
    /// Adds a dependent 1D pinned-host-to-device memcpy node to this graph.
    /// 向当前 graph 添加带依赖的 1D pinned-host-to-device memcpy 节点。
    /// </summary>
    /// <param name="dependencyNode">The dependency node, or the default token for no dependency. 依赖节点；默认 token 表示无依赖。</param>
    /// <param name="destination">The destination device memory owner. 目标设备内存所有者。</param>
    /// <param name="source">The source pinned host memory owner. 源 pinned host memory 所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <returns>A graph-owned memcpy node token. 由 graph 拥有的 memcpy node token。</returns>
    public CudaGraphNode AddHostToDeviceMemcpyNodeAfter(CudaGraphNode dependencyNode, CudaMemory destination, CudaPinnedMemory source, int count)
    {
        ValidateDeviceMemory(destination, nameof(destination));
        ValidatePinnedMemory(source, nameof(source));
        ValidateMemcpyCount(count, destination.SizeInBytes, source.SizeInBytes, nameof(count));
        return NativeCudaApi.AddGraphMemcpyNode1DHostToDevice(_handle, dependencyNode, destination.Handle, source.Handle, count);
    }

    /// <summary>
    /// Adds a 1D device-to-pinned-host memcpy node to this graph.
    /// 向当前 graph 添加 1D device-to-pinned-host memcpy 节点。
    /// </summary>
    /// <param name="destination">The destination pinned host memory owner. 目标 pinned host memory 所有者。</param>
    /// <param name="source">The source device memory owner. 源设备内存所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <returns>A graph-owned memcpy node token. 由 graph 拥有的 memcpy node token。</returns>
    /// <remarks>
    /// The memory owners must remain alive while the graph or any executable graph created from it may use this node.
    /// 内存所有者必须在 graph 或由该 graph 创建的 executable graph 可能使用该节点期间保持存活。
    /// </remarks>
    public CudaGraphNode AddDeviceToHostMemcpyNode(CudaPinnedMemory destination, CudaMemory source, int count)
    {
        return AddDeviceToHostMemcpyNodeAfter(default, destination, source, count);
    }

    /// <summary>
    /// Adds a dependent 1D device-to-pinned-host memcpy node to this graph.
    /// 向当前 graph 添加带依赖的 1D device-to-pinned-host memcpy 节点。
    /// </summary>
    /// <param name="dependencyNode">The dependency node, or the default token for no dependency. 依赖节点；默认 token 表示无依赖。</param>
    /// <param name="destination">The destination pinned host memory owner. 目标 pinned host memory 所有者。</param>
    /// <param name="source">The source device memory owner. 源设备内存所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <returns>A graph-owned memcpy node token. 由 graph 拥有的 memcpy node token。</returns>
    public CudaGraphNode AddDeviceToHostMemcpyNodeAfter(CudaGraphNode dependencyNode, CudaPinnedMemory destination, CudaMemory source, int count)
    {
        ValidatePinnedMemory(destination, nameof(destination));
        ValidateDeviceMemory(source, nameof(source));
        ValidateMemcpyCount(count, destination.SizeInBytes, source.SizeInBytes, nameof(count));
        return NativeCudaApi.AddGraphMemcpyNode1DDeviceToHost(_handle, dependencyNode, destination.Handle, source.Handle, count);
    }

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
    /// <param name="node">The graph node token. graph node token。</param>
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
    /// <param name="node">The graph node token. graph node token。</param>
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
    /// <param name="node">The graph node token. graph node token。</param>
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

    /// <summary>
    /// Updates a CUDA graph memset node using a managed device-memory owner and byte-count descriptor.
    /// 使用托管设备内存所有者和字节数 descriptor 更新 CUDA graph memset 节点。
    /// </summary>
    /// <param name="node">The memset node token. memset 节点 token。</param>
    /// <param name="destination">The destination device memory owner. 目标设备内存所有者。</param>
    /// <param name="value">The byte value used by the memset node. memset 节点使用的字节填充值。</param>
    /// <param name="count">The number of bytes to fill. 要填充的字节数。</param>
    /// <remarks>
    /// This method does not accept raw device pointers. The native bridge derives the destination pointer from <paramref name="destination"/> and validates <paramref name="count"/> against the allocation size.
    /// 该方法不接受裸设备指针。Native bridge 会从 <paramref name="destination"/> 推导目标指针，并按分配大小校验 <paramref name="count"/>。
    /// </remarks>
    public static void SetMemsetNodeParameters(CudaGraphNode node, CudaMemory destination, byte value, int count)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (count <= 0 || count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.SetGraphMemsetNodeParameters(node, destination.Handle, value, count);
    }

    /// <summary>
    /// Updates a CUDA graph kernel node scalar attribute using a typed descriptor.
    /// 使用 typed descriptor 更新 CUDA graph kernel 节点的标量 attribute。
    /// </summary>
    /// <param name="node">The kernel node token. kernel 节点 token。</param>
    /// <param name="value">The scalar attribute descriptor. 标量 attribute descriptor。</param>
    /// <remarks>
    /// This method does not accept raw kernel parameters, function pointers, callback user data, or CUDA's native union.
    /// 该方法不接受裸 kernel 参数、function pointer、callback user data 或 CUDA 原生 union。
    /// </remarks>
    public static void SetKernelNodeAttribute(CudaGraphNode node, CudaGraphKernelNodeAttributeValue value)
    {
        ValidateKernelNodeAttribute(value.Attribute, nameof(value));
        NativeCudaApi.SetGraphKernelNodeAttribute(node, value);
    }

    /// <summary>
    /// Updates a CUDA graph kernel node cooperative flag.
    /// 更新 CUDA graph kernel 节点 cooperative 标志。
    /// </summary>
    /// <param name="node">The kernel node token. kernel 节点 token。</param>
    /// <param name="enabled">Whether cooperative launch is enabled. 是否启用 cooperative launch。</param>
    public static void SetKernelNodeCooperative(CudaGraphNode node, bool enabled)
    {
        SetKernelNodeAttribute(node, CudaGraphKernelNodeAttributeValue.Cooperative(enabled));
    }

    /// <summary>
    /// Updates a CUDA graph kernel node priority.
    /// 更新 CUDA graph kernel 节点 priority。
    /// </summary>
    /// <param name="node">The kernel node token. kernel 节点 token。</param>
    /// <param name="priority">The kernel execution priority. Kernel 执行优先级。</param>
    public static void SetKernelNodePriority(CudaGraphNode node, int priority)
    {
        SetKernelNodeAttribute(node, CudaGraphKernelNodeAttributeValue.Priority(priority));
    }

    /// <summary>
    /// Updates a CUDA graph kernel node cluster dimension.
    /// 更新 CUDA graph kernel 节点 cluster dimension。
    /// </summary>
    /// <param name="node">The kernel node token. kernel 节点 token。</param>
    /// <param name="x">The cluster X dimension. Cluster X 维度。</param>
    /// <param name="y">The cluster Y dimension. Cluster Y 维度。</param>
    /// <param name="z">The cluster Z dimension. Cluster Z 维度。</param>
    public static void SetKernelNodeClusterDimension(CudaGraphNode node, uint x, uint y, uint z)
    {
        SetKernelNodeAttribute(node, CudaGraphKernelNodeAttributeValue.ClusterDimension(x, y, z));
    }

    /// <summary>
    /// Updates a CUDA graph kernel node cluster scheduling policy preference.
    /// 更新 CUDA graph kernel 节点 cluster scheduling policy preference。
    /// </summary>
    /// <param name="node">The kernel node token. kernel 节点 token。</param>
    /// <param name="policy">The cluster scheduling policy preference. Cluster 调度策略偏好。</param>
    public static void SetKernelNodeClusterSchedulingPolicy(CudaGraphNode node, CudaClusterSchedulingPolicyPreference policy)
    {
        SetKernelNodeAttribute(node, CudaGraphKernelNodeAttributeValue.ClusterSchedulingPolicy(policy));
    }

    /// <summary>
    /// Updates a CUDA graph memcpy node for a 1D device-to-device copy.
    /// 将 CUDA graph memcpy 节点更新为 1D device-to-device 复制。
    /// </summary>
    /// <param name="node">The memcpy node token. memcpy 节点 token。</param>
    /// <param name="destination">The destination device memory owner. 目标设备内存所有者。</param>
    /// <param name="source">The source device memory owner. 源设备内存所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <remarks>
    /// This method derives native pointers from managed memory owners and validates <paramref name="count"/> against both allocations.
    /// 该方法从托管内存所有者推导 native 指针，并按两个分配的大小校验 <paramref name="count"/>。
    /// </remarks>
    public static void SetDeviceToDeviceMemcpyNodeParameters(CudaGraphNode node, CudaMemory destination, CudaMemory source, int count)
    {
        ValidateDeviceMemory(destination, nameof(destination));
        ValidateDeviceMemory(source, nameof(source));
        ValidateMemcpyCount(count, destination.SizeInBytes, source.SizeInBytes, nameof(count));
        NativeCudaApi.SetGraphMemcpyNodeParametersDeviceToDevice(node, destination.Handle, source.Handle, count);
    }

    /// <summary>
    /// Updates a CUDA graph memcpy node for a 1D pinned-host-to-device copy.
    /// 将 CUDA graph memcpy 节点更新为 1D pinned-host-to-device 复制。
    /// </summary>
    /// <param name="node">The memcpy node token. memcpy 节点 token。</param>
    /// <param name="destination">The destination device memory owner. 目标设备内存所有者。</param>
    /// <param name="source">The source pinned host memory owner. 源 pinned host memory 所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <remarks>
    /// This method derives native pointers from managed memory owners and validates <paramref name="count"/> against both allocations.
    /// 该方法从托管内存所有者推导 native 指针，并按两个分配的大小校验 <paramref name="count"/>。
    /// </remarks>
    public static void SetHostToDeviceMemcpyNodeParameters(CudaGraphNode node, CudaMemory destination, CudaPinnedMemory source, int count)
    {
        ValidateDeviceMemory(destination, nameof(destination));
        ValidatePinnedMemory(source, nameof(source));
        ValidateMemcpyCount(count, destination.SizeInBytes, source.SizeInBytes, nameof(count));
        NativeCudaApi.SetGraphMemcpyNodeParametersHostToDevice(node, destination.Handle, source.Handle, count);
    }

    /// <summary>
    /// Updates a CUDA graph memcpy node for a 1D device-to-pinned-host copy.
    /// 将 CUDA graph memcpy 节点更新为 1D device-to-pinned-host 复制。
    /// </summary>
    /// <param name="node">The memcpy node token. memcpy 节点 token。</param>
    /// <param name="destination">The destination pinned host memory owner. 目标 pinned host memory 所有者。</param>
    /// <param name="source">The source device memory owner. 源设备内存所有者。</param>
    /// <param name="count">The byte count to copy. 要复制的字节数。</param>
    /// <remarks>
    /// This method derives native pointers from managed memory owners and validates <paramref name="count"/> against both allocations.
    /// 该方法从托管内存所有者推导 native 指针，并按两个分配的大小校验 <paramref name="count"/>。
    /// </remarks>
    public static void SetDeviceToHostMemcpyNodeParameters(CudaGraphNode node, CudaPinnedMemory destination, CudaMemory source, int count)
    {
        ValidatePinnedMemory(destination, nameof(destination));
        ValidateDeviceMemory(source, nameof(source));
        ValidateMemcpyCount(count, destination.SizeInBytes, source.SizeInBytes, nameof(count));
        NativeCudaApi.SetGraphMemcpyNodeParametersDeviceToHost(node, destination.Handle, source.Handle, count);
    }

    /// <summary>
    /// Updates an event-record node to use a caller-owned CUDA event.
    /// 将 event-record 节点更新为使用调用方拥有的 CUDA event。
    /// </summary>
    /// <param name="node">The event-record node token. event-record 节点 token。</param>
    /// <param name="eventHandle">The caller-owned CUDA event. 调用方拥有的 CUDA event。</param>
    /// <remarks>
    /// The event must remain alive while the graph or any executable graph created from it may use this node.
    /// 该 event 必须在 graph 或由该 graph 创建的 executable graph 可能使用该节点期间保持存活。
    /// </remarks>
    public static void SetEventRecordNodeEvent(CudaGraphNode node, CudaEvent eventHandle)
    {
        if (eventHandle == null)
        {
            throw new ArgumentNullException(nameof(eventHandle));
        }

        NativeCudaApi.SetGraphEventRecordNodeEvent(node, eventHandle.Handle);
    }

    /// <summary>
    /// Gets whether an event-record node currently references a CUDA event without exposing the borrowed event handle.
    /// 查询 event-record 节点当前是否引用 CUDA event，但不暴露 borrowed event handle。
    /// </summary>
    /// <param name="node">The graph-owned event-record node token. Graph 拥有的 event-record node token。</param>
    /// <returns><see langword="true"/> when CUDA reports a non-null event for the node. CUDA 报告该节点具有非空 event 时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// This is a copied scalar snapshot over <c>cudaGraphEventRecordNodeGetEvent</c>; it intentionally does not return <see cref="CudaEvent"/> or a native pointer.
    /// 这是基于 <c>cudaGraphEventRecordNodeGetEvent</c> 的复制型标量快照；有意不返回 <see cref="CudaEvent"/> 或原生指针。
    /// </remarks>
    public static bool EventRecordNodeHasEvent(CudaGraphNode node)
    {
        return NativeCudaApi.GraphEventRecordNodeHasEvent(node);
    }

    /// <summary>
    /// Updates an event-wait node to use a caller-owned CUDA event.
    /// 将 event-wait 节点更新为使用调用方拥有的 CUDA event。
    /// </summary>
    /// <param name="node">The event-wait node token. event-wait 节点 token。</param>
    /// <param name="eventHandle">The caller-owned CUDA event. 调用方拥有的 CUDA event。</param>
    /// <remarks>
    /// The event must remain alive while the graph or any executable graph created from it may use this node.
    /// 该 event 必须在 graph 或由该 graph 创建的 executable graph 可能使用该节点期间保持存活。
    /// </remarks>
    public static void SetEventWaitNodeEvent(CudaGraphNode node, CudaEvent eventHandle)
    {
        if (eventHandle == null)
        {
            throw new ArgumentNullException(nameof(eventHandle));
        }

        NativeCudaApi.SetGraphEventWaitNodeEvent(node, eventHandle.Handle);
    }

    /// <summary>
    /// Gets whether an event-wait node currently references a CUDA event without exposing the borrowed event handle.
    /// 查询 event-wait 节点当前是否引用 CUDA event，但不暴露 borrowed event handle。
    /// </summary>
    /// <param name="node">The graph-owned event-wait node token. Graph 拥有的 event-wait node token。</param>
    /// <returns><see langword="true"/> when CUDA reports a non-null event for the node. CUDA 报告该节点具有非空 event 时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// This is a copied scalar snapshot over <c>cudaGraphEventWaitNodeGetEvent</c>; it intentionally does not return <see cref="CudaEvent"/> or a native pointer.
    /// 这是基于 <c>cudaGraphEventWaitNodeGetEvent</c> 的复制型标量快照；有意不返回 <see cref="CudaEvent"/> 或原生指针。
    /// </remarks>
    public static bool EventWaitNodeHasEvent(CudaGraphNode node)
    {
        return NativeCudaApi.GraphEventWaitNodeHasEvent(node);
    }

    /// <summary>
    /// Gets the CUDA runtime local id for a graph node when supported by the loaded runtime.
    /// 在当前 CUDA runtime 支持时获取 graph node 的 local id。
    /// </summary>
    /// <param name="node">The graph node token. graph node 的 token。</param>
    /// <returns>The CUDA graph node local id. CUDA graph node 的 local id。</returns>
    public static uint GetNodeLocalId(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphNodeLocalId(node);
    }

    /// <summary>
    /// Gets the CUDA tools id for a graph node when supported by the loaded runtime.
    /// 在当前 CUDA runtime 支持时获取 graph node 的 tools id。
    /// </summary>
    /// <param name="node">The graph node token. graph node 的 token。</param>
    /// <returns>The CUDA graph node tools id. CUDA graph node 的 tools id。</returns>
    public static ulong GetNodeToolsId(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphNodeToolsId(node);
    }

    /// <summary>
    /// Gets the number of dependencies for a graph node.
    /// 获取指定 graph node 的依赖数量。
    /// </summary>
    /// <param name="node">The node token. 节点 token。</param>
    /// <returns>The dependency count. 依赖数量。</returns>
    public static ulong GetDependencyCount(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphNodeDependencyCount(node);
    }

    /// <summary>
    /// Gets a dependency node by index for a graph node.
    /// 按索引获取指定 graph node 的依赖节点。
    /// </summary>
    /// <param name="node">The node token. 节点 token。</param>
    /// <param name="index">The zero-based dependency index. 从零开始的依赖索引。</param>
    /// <returns>The dependency node token. 依赖节点 token。</returns>
    public static CudaGraphNode GetDependency(CudaGraphNode node, ulong index)
    {
        return NativeCudaApi.GetGraphNodeDependency(node, index);
    }

    /// <summary>
    /// Gets the number of dependencies with CUDA edge data for a graph node.
    /// 获取指定 graph node 带 CUDA edge data 的依赖数量。
    /// </summary>
    /// <param name="node">The node token. 节点 token。</param>
    /// <returns>The dependency count. 依赖数量。</returns>
    public static ulong GetDependencyWithEdgeDataCount(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphNodeDependencyV2Count(node);
    }

    /// <summary>
    /// Gets a dependency node and CUDA edge data by index for a graph node.
    /// 按索引获取指定 graph node 的依赖节点和 CUDA edge data。
    /// </summary>
    /// <param name="node">The node token. 节点 token。</param>
    /// <param name="index">The zero-based dependency index. 从零开始的依赖索引。</param>
    /// <returns>The dependency node and edge data. 依赖节点和 edge data。</returns>
    public static CudaGraphNodeDependency GetDependencyWithEdgeData(CudaGraphNode node, ulong index)
    {
        return NativeCudaApi.GetGraphNodeDependencyV2(node, index);
    }

    /// <summary>
    /// Gets the number of dependent nodes for a graph node.
    /// 获取依赖指定 graph node 的节点数量。
    /// </summary>
    /// <param name="node">The node token. 节点 token。</param>
    /// <returns>The dependent count. dependent 节点数量。</returns>
    public static ulong GetDependentCount(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphNodeDependentCount(node);
    }

    /// <summary>
    /// Gets a dependent node by index for a graph node.
    /// 按索引获取依赖指定 graph node 的节点。
    /// </summary>
    /// <param name="node">The node token. 节点 token。</param>
    /// <param name="index">The zero-based dependent index. 从零开始的 dependent 节点索引。</param>
    /// <returns>The dependent node token. dependent 节点 token。</returns>
    public static CudaGraphNode GetDependent(CudaGraphNode node, ulong index)
    {
        return NativeCudaApi.GetGraphNodeDependent(node, index);
    }

    /// <summary>
    /// Gets the number of dependent nodes with CUDA edge data for a graph node.
    /// 获取依赖指定 graph node 且带 CUDA edge data 的节点数量。
    /// </summary>
    /// <param name="node">The node token. 节点 token。</param>
    /// <returns>The dependent node count. dependent 节点数量。</returns>
    public static ulong GetDependentWithEdgeDataCount(CudaGraphNode node)
    {
        return NativeCudaApi.GetGraphNodeDependentV2Count(node);
    }

    /// <summary>
    /// Gets a dependent node and CUDA edge data by index for a graph node.
    /// 按索引获取依赖指定 graph node 的节点和 CUDA edge data。
    /// </summary>
    /// <param name="node">The node token. 节点 token。</param>
    /// <param name="index">The zero-based dependent index. 从零开始的 dependent 节点索引。</param>
    /// <returns>The dependent node and edge data. dependent 节点和 edge data。</returns>
    public static CudaGraphNodeDependency GetDependentWithEdgeData(CudaGraphNode node, ulong index)
    {
        return NativeCudaApi.GetGraphNodeDependentV2(node, index);
    }

    /// <summary>
    /// Instantiates this graph as an executable CUDA graph.
    /// 将当前 graph 实例化为可执行 CUDA graph。
    /// </summary>
    /// <param name="flags">The CUDA graph-instantiation flags. CUDA graph 实例化标志。</param>
    /// <returns>A managed executable graph wrapper. 可执行 graph 的托管封装。</returns>
    public CudaGraphExec Instantiate(ulong flags = 0)
    {
        return new CudaGraphExec(NativeCudaApi.InstantiateGraph(_handle, flags));
    }

    internal static void ValidateDeviceMemory(CudaMemory memory, string parameterName)
    {
        if (memory == null)
        {
            throw new ArgumentNullException(parameterName);
        }
    }

    internal static void ValidatePinnedMemory(CudaPinnedMemory memory, string parameterName)
    {
        if (memory == null)
        {
            throw new ArgumentNullException(parameterName);
        }
    }

    internal static void ValidateMemcpyCount(int count, int destinationSize, int sourceSize, string parameterName)
    {
        if (count <= 0 || count > destinationSize || count > sourceSize)
        {
            throw new ArgumentOutOfRangeException(parameterName);
        }
    }

    private static void ValidateKernelNodeAttribute(CudaGraphKernelNodeAttribute attribute, string parameterName)
    {
        switch (attribute)
        {
            case CudaGraphKernelNodeAttribute.Cooperative:
            case CudaGraphKernelNodeAttribute.Priority:
            case CudaGraphKernelNodeAttribute.ClusterDimension:
            case CudaGraphKernelNodeAttribute.ClusterSchedulingPolicyPreference:
                return;
            default:
                throw new ArgumentOutOfRangeException(parameterName);
        }
    }

    private static int ToListCapacity(ulong count)
    {
        return count > int.MaxValue ? int.MaxValue : (int)count;
    }

    /// <summary>
    /// Releases the captured CUDA graph handle.
    /// 释放已捕获的 CUDA graph 句柄。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}
