namespace JYPPX.CudaSharp;

/// <summary>
/// Captures copied topology counters for a CUDA graph node.
/// 捕获 CUDA graph node 的复制型拓扑计数。
/// </summary>
public sealed class CudaGraphNodeTopologySnapshot
{
    /// <summary>
    /// Initializes a CUDA graph node topology snapshot.
    /// 初始化 CUDA graph node 拓扑快照。
    /// </summary>
    /// <param name="nodeType">The CUDA graph node type. CUDA graph node 类型。</param>
    /// <param name="dependencyCount">The dependency count. 依赖数量。</param>
    /// <param name="dependencyWithEdgeDataCount">The dependency count available through CUDA edge-data queries. 可通过 CUDA edge-data 查询访问的依赖数量。</param>
    /// <param name="dependentCount">The dependent-node count. 依赖当前节点的节点数量。</param>
    /// <param name="dependentWithEdgeDataCount">The dependent-node count available through CUDA edge-data queries. 可通过 CUDA edge-data 查询访问的 dependent 节点数量。</param>
    public CudaGraphNodeTopologySnapshot(
        CudaGraphNodeType nodeType,
        ulong dependencyCount,
        ulong dependencyWithEdgeDataCount,
        ulong dependentCount,
        ulong dependentWithEdgeDataCount)
    {
        NodeType = nodeType;
        DependencyCount = dependencyCount;
        DependencyWithEdgeDataCount = dependencyWithEdgeDataCount;
        DependentCount = dependentCount;
        DependentWithEdgeDataCount = dependentWithEdgeDataCount;
    }

    /// <summary>
    /// Gets the CUDA graph node type.
    /// 获取 CUDA graph node 类型。
    /// </summary>
    public CudaGraphNodeType NodeType { get; }

    /// <summary>
    /// Gets the dependency count.
    /// 获取依赖数量。
    /// </summary>
    public ulong DependencyCount { get; }

    /// <summary>
    /// Gets the dependency count available through CUDA edge-data queries.
    /// 获取可通过 CUDA edge-data 查询访问的依赖数量。
    /// </summary>
    public ulong DependencyWithEdgeDataCount { get; }

    /// <summary>
    /// Gets the dependent-node count.
    /// 获取依赖当前节点的节点数量。
    /// </summary>
    public ulong DependentCount { get; }

    /// <summary>
    /// Gets the dependent-node count available through CUDA edge-data queries.
    /// 获取可通过 CUDA edge-data 查询访问的 dependent 节点数量。
    /// </summary>
    public ulong DependentWithEdgeDataCount { get; }

    /// <summary>
    /// Formats this snapshot for diagnostics.
    /// 将该快照格式化为诊断字符串。
    /// </summary>
    public override string ToString() =>
        $"Type={NodeType}, Dependencies={DependencyCount}, DependencyEdgeData={DependencyWithEdgeDataCount}, Dependents={DependentCount}, DependentEdgeData={DependentWithEdgeDataCount}";
}
