using System;
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
    /// Instantiates this graph as an executable CUDA graph.
    /// 将当前 graph 实例化为可执行 CUDA graph。
    /// </summary>
    /// <param name="flags">The CUDA graph-instantiation flags. CUDA graph 实例化标志。</param>
    /// <returns>A managed executable graph wrapper. 可执行 graph 的托管封装。</returns>
    public CudaGraphExec Instantiate(ulong flags = 0)
    {
        return new CudaGraphExec(NativeCudaApi.InstantiateGraph(_handle, flags));
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
