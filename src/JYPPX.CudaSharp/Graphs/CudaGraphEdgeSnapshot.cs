namespace JYPPX.CudaSharp;

/// <summary>
/// Captures copied diagnostics for a CUDA graph dependency edge.
/// 捕获 CUDA graph 依赖边的复制型诊断信息。
/// </summary>
public sealed class CudaGraphEdgeSnapshot
{
    /// <summary>
    /// Initializes a CUDA graph edge snapshot.
    /// 初始化 CUDA graph edge 快照。
    /// </summary>
    /// <param name="index">The zero-based edge index. 从零开始的边索引。</param>
    /// <param name="from">The source graph-owned node value token. 源 graph-owned node 值 token。</param>
    /// <param name="to">The destination graph-owned node value token. 目标 graph-owned node 值 token。</param>
    /// <param name="edgeData">The copied CUDA edge data when available. 可用时复制出的 CUDA edge data。</param>
    /// <param name="hasEdgeData">Whether <paramref name="edgeData"/> is available. <paramref name="edgeData"/> 是否可用。</param>
    public CudaGraphEdgeSnapshot(ulong index, CudaGraphNode from, CudaGraphNode to, CudaGraphEdgeData edgeData, bool hasEdgeData)
    {
        Index = index;
        From = from;
        To = to;
        FromText = from.ToString();
        ToText = to.ToString();
        EdgeData = edgeData;
        HasEdgeData = hasEdgeData;
        EdgeDataState = hasEdgeData ? "Available" : "NotRequestedOrUnsupported";
    }

    /// <summary>
    /// Gets the zero-based edge index.
    /// 获取从零开始的边索引。
    /// </summary>
    public ulong Index { get; }

    /// <summary>
    /// Gets the source graph-owned node value token.
    /// 获取源 graph-owned node 值 token。
    /// </summary>
    public CudaGraphNode From { get; }

    /// <summary>
    /// Gets the destination graph-owned node value token.
    /// 获取目标 graph-owned node 值 token。
    /// </summary>
    public CudaGraphNode To { get; }

    /// <summary>
    /// Gets a diagnostic text representation of the source node token.
    /// 获取源 node token 的诊断文本表示。
    /// </summary>
    public string FromText { get; }

    /// <summary>
    /// Gets a diagnostic text representation of the destination node token.
    /// 获取目标 node token 的诊断文本表示。
    /// </summary>
    public string ToText { get; }

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
            ? $"Index={Index}, Edge={FromText}->{ToText}, EdgeData={EdgeData}"
            : $"Index={Index}, Edge={FromText}->{ToText}, EdgeDataState={EdgeDataState}";
}
