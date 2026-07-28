namespace JYPPX.CudaSharp;

/// <summary>
/// Captures copied executable-state diagnostics for a CUDA graph node.
/// 捕获 CUDA graph executable 中某个 node 的复制型执行状态诊断。
/// </summary>
public sealed class CudaGraphExecNodeStateSnapshot
{
    /// <summary>
    /// Initializes a CUDA graph executable node-state snapshot.
    /// 初始化 CUDA graph executable node 状态快照。
    /// </summary>
    /// <param name="enabled">Whether the node is enabled in the executable graph. executable graph 中该节点是否启用。</param>
    /// <param name="flags">The executable graph flags. 可执行图标志。</param>
    public CudaGraphExecNodeStateSnapshot(bool enabled, ulong flags)
        : this(0, default, enabled, flags, hasNode: false)
    {
    }

    /// <summary>
    /// Initializes a CUDA graph executable node-state snapshot with node diagnostics.
    /// 使用 node 诊断信息初始化 CUDA graph executable node 状态快照。
    /// </summary>
    /// <param name="index">The zero-based node index within the queried list. 查询列表中的从零开始节点索引。</param>
    /// <param name="node">The graph-owned node value token. graph 拥有的 node 值 token。</param>
    /// <param name="enabled">Whether the node is enabled in the executable graph. executable graph 中该节点是否启用。</param>
    /// <param name="flags">The executable graph flags. 可执行图标志。</param>
    public CudaGraphExecNodeStateSnapshot(ulong index, CudaGraphNode node, bool enabled, ulong flags)
        : this(index, node, enabled, flags, hasNode: true)
    {
    }

    private CudaGraphExecNodeStateSnapshot(ulong index, CudaGraphNode node, bool enabled, ulong flags, bool hasNode)
    {
        Index = index;
        Node = node;
        NodeText = hasNode ? node.ToString() : string.Empty;
        HasNode = hasNode;
        Enabled = enabled;
        Flags = flags;
    }

    /// <summary>
    /// Gets the zero-based node index within the queried list.
    /// 获取查询列表中的从零开始节点索引。
    /// </summary>
    public ulong Index { get; }

    /// <summary>
    /// Gets whether this snapshot includes the graph-owned node value token.
    /// 获取该快照是否包含 graph-owned node 值 token。
    /// </summary>
    public bool HasNode { get; }

    /// <summary>
    /// Gets the graph-owned node value token when available.
    /// 获取可用时的 graph-owned node 值 token。
    /// </summary>
    public CudaGraphNode Node { get; }

    /// <summary>
    /// Gets a diagnostic text representation of the node token when available.
    /// 获取可用时的 node token 诊断文本表示。
    /// </summary>
    public string NodeText { get; }

    /// <summary>
    /// Gets whether the node is enabled in the executable graph.
    /// 获取 executable graph 中该节点是否启用。
    /// </summary>
    public bool Enabled { get; }

    /// <summary>
    /// Gets the executable graph flags.
    /// 获取 executable graph flags。
    /// </summary>
    public ulong Flags { get; }

    /// <summary>
    /// Formats this snapshot for diagnostics.
    /// 将该快照格式化为诊断字符串。
    /// </summary>
    public override string ToString() =>
        HasNode
            ? $"Index={Index}, Node={NodeText}, Enabled={Enabled}, Flags={Flags}"
            : $"Enabled={Enabled}, Flags={Flags}";
}
