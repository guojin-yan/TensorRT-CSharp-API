namespace JYPPX.CudaSharp;

/// <summary>
/// Captures copied diagnostics for a CUDA graph node entry.
/// 捕获 CUDA graph node 条目的复制型诊断信息。
/// </summary>
public sealed class CudaGraphNodeSnapshot
{
    /// <summary>
    /// Initializes a CUDA graph node snapshot.
    /// 初始化 CUDA graph node 快照。
    /// </summary>
    /// <param name="index">The zero-based node index within the queried list. 查询列表中的从零开始节点索引。</param>
    /// <param name="node">The graph-owned node value token. graph 拥有的 node 值 token。</param>
    /// <param name="topology">The copied node topology snapshot. 复制出的 node 拓扑快照。</param>
    public CudaGraphNodeSnapshot(ulong index, CudaGraphNode node, CudaGraphNodeTopologySnapshot topology)
    {
        Index = index;
        Node = node;
        NodeText = node.ToString();
        Topology = topology;
    }

    /// <summary>
    /// Gets the zero-based node index within the queried list.
    /// 获取查询列表中的从零开始节点索引。
    /// </summary>
    public ulong Index { get; }

    /// <summary>
    /// Gets the graph-owned node value token.
    /// 获取 graph 拥有的 node 值 token。
    /// </summary>
    public CudaGraphNode Node { get; }

    /// <summary>
    /// Gets a diagnostic text representation of the node token.
    /// 获取 node token 的诊断文本表示。
    /// </summary>
    public string NodeText { get; }

    /// <summary>
    /// Gets the copied node topology snapshot.
    /// 获取复制出的 node 拓扑快照。
    /// </summary>
    public CudaGraphNodeTopologySnapshot Topology { get; }

    /// <summary>
    /// Formats this snapshot for diagnostics.
    /// 将该快照格式化为诊断字符串。
    /// </summary>
    public override string ToString() => $"Index={Index}, Node={NodeText}, {Topology}";
}
