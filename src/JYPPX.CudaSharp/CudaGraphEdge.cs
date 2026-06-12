namespace JYPPX.CudaSharp;

/// <summary>
/// Describes a dependency edge between two CUDA graph nodes.
/// 描述两个 CUDA graph node 之间的一条依赖边。
/// </summary>
public readonly struct CudaGraphEdge
{
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

    public override string ToString() => $"{From}->{To}";
}
