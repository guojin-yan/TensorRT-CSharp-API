namespace JYPPX.CudaSharp;

/// <summary>
/// Captures a copied, ownership-neutral topology summary for a CUDA graph.
/// 捕获 CUDA graph 的复制型、无所有权转移拓扑摘要。
/// </summary>
public sealed class CudaGraphTopologySnapshot
{
    /// <summary>
    /// Initializes a CUDA graph topology snapshot.
    /// 初始化 CUDA graph 拓扑快照。
    /// </summary>
    /// <param name="nodeCount">The graph node count. graph 节点数量。</param>
    /// <param name="rootNodeCount">The graph root-node count. graph root 节点数量。</param>
    /// <param name="edgeCount">The graph dependency edge count. graph 依赖边数量。</param>
    /// <param name="edgeWithDataCount">The dependency edge count available through CUDA edge-data queries. 可通过 CUDA edge-data 查询访问的依赖边数量。</param>
    public CudaGraphTopologySnapshot(ulong nodeCount, ulong rootNodeCount, ulong edgeCount, ulong edgeWithDataCount)
    {
        NodeCount = nodeCount;
        RootNodeCount = rootNodeCount;
        EdgeCount = edgeCount;
        EdgeWithDataCount = edgeWithDataCount;
    }

    /// <summary>
    /// Gets the graph node count.
    /// 获取 graph 节点数量。
    /// </summary>
    public ulong NodeCount { get; }

    /// <summary>
    /// Gets the graph root-node count.
    /// 获取 graph root 节点数量。
    /// </summary>
    public ulong RootNodeCount { get; }

    /// <summary>
    /// Gets the graph dependency edge count.
    /// 获取 graph 依赖边数量。
    /// </summary>
    public ulong EdgeCount { get; }

    /// <summary>
    /// Gets the dependency edge count available through CUDA edge-data queries.
    /// 获取可通过 CUDA edge-data 查询访问的依赖边数量。
    /// </summary>
    public ulong EdgeWithDataCount { get; }

    /// <summary>
    /// Formats this snapshot for diagnostics.
    /// 将该快照格式化为诊断字符串。
    /// </summary>
    public override string ToString() =>
        $"Nodes={NodeCount}, Roots={RootNodeCount}, Edges={EdgeCount}, EdgeDataEdges={EdgeWithDataCount}";
}
