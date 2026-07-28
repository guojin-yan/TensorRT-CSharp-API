namespace JYPPX.CudaSharp;

/// <summary>
/// Captures pointer-free metadata from a CUDA executable-graph update.
/// 捕获 CUDA executable graph 更新产生的不含指针的元数据。
/// </summary>
public sealed class CudaGraphExecUpdateSnapshot
{
    /// <summary>Initializes an executable-graph update snapshot. 初始化 executable graph 更新快照。</summary>
    public CudaGraphExecUpdateSnapshot(
        CudaGraphExecUpdateResult result,
        CudaGraphNodeType? errorNodeType,
        CudaGraphNodeType? errorFromNodeType)
    {
        Result = result;
        ErrorNodeType = errorNodeType;
        ErrorFromNodeType = errorFromNodeType;
    }

    /// <summary>Gets the CUDA update result. 获取 CUDA 更新结果。</summary>
    public CudaGraphExecUpdateResult Result { get; }

    /// <summary>Gets whether the update succeeded. 获取更新是否成功。</summary>
    public bool Succeeded => Result == CudaGraphExecUpdateResult.Success;

    /// <summary>Gets the copied error-node type when available. 获取可用时复制出的错误节点类型。</summary>
    public CudaGraphNodeType? ErrorNodeType { get; }

    /// <summary>Gets the copied source error-node type when available. 获取可用时复制出的源错误节点类型。</summary>
    public CudaGraphNodeType? ErrorFromNodeType { get; }

    /// <summary>Formats this snapshot for diagnostics. 将该快照格式化为诊断字符串。</summary>
    public override string ToString() =>
        $"Result={Result}, ErrorNodeType={ErrorNodeType?.ToString() ?? "None"}, ErrorFromNodeType={ErrorFromNodeType?.ToString() ?? "None"}";
}
