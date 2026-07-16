namespace JYPPX.CudaSharp;

/// <summary>
/// Captures copied topology metadata for an embedded CUDA child graph.
/// 捕获 CUDA embedded child graph 的复制型拓扑元数据。
/// </summary>
public sealed class CudaGraphChildSnapshot
{
    /// <summary>
    /// Initializes an embedded child-graph snapshot.
    /// 初始化 embedded child-graph 快照。
    /// </summary>
    public CudaGraphChildSnapshot(bool hasEmbeddedGraph, ulong nodeCount, ulong rootNodeCount, ulong edgeCount)
    {
        HasEmbeddedGraph = hasEmbeddedGraph;
        NodeCount = nodeCount;
        RootNodeCount = rootNodeCount;
        EdgeCount = edgeCount;
    }

    /// <summary>Gets whether the child node has an embedded graph. 获取 child 节点是否包含 embedded graph。</summary>
    public bool HasEmbeddedGraph { get; }

    /// <summary>Gets the copied embedded node count. 获取复制出的 embedded node 数量。</summary>
    public ulong NodeCount { get; }

    /// <summary>Gets the copied embedded root-node count. 获取复制出的 embedded root node 数量。</summary>
    public ulong RootNodeCount { get; }

    /// <summary>Gets the copied embedded edge count. 获取复制出的 embedded edge 数量。</summary>
    public ulong EdgeCount { get; }

    /// <summary>Formats this snapshot for diagnostics. 将该快照格式化为诊断字符串。</summary>
    public override string ToString() =>
        $"HasGraph={HasEmbeddedGraph}, Nodes={NodeCount}, Roots={RootNodeCount}, Edges={EdgeCount}";
}
