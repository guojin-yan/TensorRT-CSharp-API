namespace JYPPX.CudaSharp;

/// <summary>
/// Describes a dependency edge between two CUDA graph nodes.
/// 描述两个 CUDA graph node 之间的一条依赖边。
/// </summary>
public readonly struct CudaGraphEdge
{
    /// <summary>
    /// Initializes a dependency edge between two CUDA graph nodes.
    /// 使用两个 CUDA graph 节点初始化一条依赖边。
    /// </summary>
    /// <param name="from">The source node. 源节点。</param>
    /// <param name="to">The destination node. 目标节点。</param>
    public CudaGraphEdge(CudaGraphNode from, CudaGraphNode to)
    {
        From = from;
        To = to;
    }

    /// <summary>
    /// Gets the source node of the dependency edge.
    /// 获取依赖边的源 node。
    /// </summary>
    public CudaGraphNode From { get; }

    /// <summary>
    /// Gets the destination node of the dependency edge.
    /// 获取依赖边的目标 node。
    /// </summary>
    public CudaGraphNode To { get; }

    /// <summary>
    /// Formats the dependency edge for diagnostics.
    /// 将依赖边格式化为便于诊断的字符串。
    /// </summary>
    public override string ToString() => $"{From}->{To}";
}

/// <summary>
/// Describes a CUDA graph dependency edge together with CUDA edge data.
/// 描述带 CUDA edge data 的 CUDA graph 依赖边。
/// </summary>
public readonly struct CudaGraphEdgeWithData
{
    /// <summary>
    /// Initializes a dependency edge with CUDA edge data.
    /// 使用两个 CUDA graph 节点和 edge data 初始化一条依赖边。
    /// </summary>
    /// <param name="from">The source node. 源节点。</param>
    /// <param name="to">The destination node. 目标节点。</param>
    /// <param name="edgeData">The CUDA graph edge data. CUDA graph 边数据。</param>
    public CudaGraphEdgeWithData(CudaGraphNode from, CudaGraphNode to, CudaGraphEdgeData edgeData)
    {
        From = from;
        To = to;
        EdgeData = edgeData;
    }

    /// <summary>
    /// Gets the source node of the dependency edge.
    /// 获取依赖边的源 node。
    /// </summary>
    public CudaGraphNode From { get; }

    /// <summary>
    /// Gets the destination node of the dependency edge.
    /// 获取依赖边的目标 node。
    /// </summary>
    public CudaGraphNode To { get; }

    /// <summary>
    /// Gets the CUDA graph edge data.
    /// 获取 CUDA graph edge data。
    /// </summary>
    public CudaGraphEdgeData EdgeData { get; }

    /// <summary>
    /// Formats the dependency edge for diagnostics.
    /// 将依赖边格式化为便于诊断的字符串。
    /// </summary>
    public override string ToString() => $"{From}->{To} [{EdgeData}]";
}
