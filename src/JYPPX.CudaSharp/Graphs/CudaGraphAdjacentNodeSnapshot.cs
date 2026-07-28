namespace JYPPX.CudaSharp;

/// <summary>
/// Captures copied diagnostics for an adjacent CUDA graph node.
/// 捕获相邻 CUDA graph node 的复制型诊断信息。
/// </summary>
public sealed class CudaGraphAdjacentNodeSnapshot
{
    /// <summary>
    /// Initializes an adjacent CUDA graph node snapshot without edge-data metadata.
    /// 初始化不含 edge-data 元数据的相邻 CUDA graph node 快照。
    /// </summary>
    /// <param name="index">The zero-based adjacent node index. 从零开始的相邻节点索引。</param>
    /// <param name="node">The adjacent graph-owned node value token. 相邻的 graph-owned node 值 token。</param>
    /// <param name="edgeData">The copied CUDA edge data when available. 可用时复制出的 CUDA edge data。</param>
    /// <param name="hasEdgeData">Whether <paramref name="edgeData"/> is available. <paramref name="edgeData"/> 是否可用。</param>
    public CudaGraphAdjacentNodeSnapshot(ulong index, CudaGraphNode node, CudaGraphEdgeData edgeData, bool hasEdgeData)
    {
        Index = index;
        Node = node;
        NodeText = node.ToString();
        EdgeData = edgeData;
        HasEdgeData = hasEdgeData;
        EdgeDataState = hasEdgeData ? "Available" : "NotRequestedOrUnsupported";
    }

    /// <summary>
    /// Gets the zero-based adjacent node index.
    /// 获取从零开始的相邻节点索引。
    /// </summary>
    public ulong Index { get; }

    /// <summary>
    /// Gets the adjacent graph-owned node value token.
    /// 获取相邻的 graph-owned node 值 token。
    /// </summary>
    public CudaGraphNode Node { get; }

    /// <summary>
    /// Gets a diagnostic text representation of the node token.
    /// 获取 node token 的诊断文本表示。
    /// </summary>
    public string NodeText { get; }

    /// <summary>
    /// Gets whether copied CUDA edge data is available.
    /// 获取是否存在复制出的 CUDA edge data。
    /// </summary>
    public bool HasEdgeData { get; }

    /// <summary>
    /// Gets the copied CUDA edge data when available.
    /// 获取可用时复制出的 CUDA edge data。
    /// </summary>
    public CudaGraphEdgeData EdgeData { get; }

    /// <summary>
    /// Gets a diagnostic state for edge-data availability.
    /// 获取 edge-data 可用性的诊断状态。
    /// </summary>
    public string EdgeDataState { get; }

    /// <summary>
    /// Formats this snapshot for diagnostics.
    /// 将该快照格式化为诊断字符串。
    /// </summary>
    public override string ToString() =>
        HasEdgeData
            ? $"Index={Index}, Node={NodeText}, EdgeData={EdgeData}"
            : $"Index={Index}, Node={NodeText}, EdgeDataState={EdgeDataState}";
}
