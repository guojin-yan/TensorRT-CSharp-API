using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Captures copied count and presence metadata from an external-semaphore graph node.
/// 捕获 external-semaphore graph 节点的复制型 count 和存在性元数据。
/// </summary>
public sealed class CudaGraphExternalSemaphoreNodeSnapshot
{
    internal CudaGraphExternalSemaphoreNodeSnapshot(CudaGraphNodeType nodeType, NativeCudaGraphExternalSemaphoreNodeParamsSnapshot native)
    {
        NodeType = nodeType;
        SemaphoreCount = native.SemaphoreCount;
        HasSemaphoreArray = native.HasSemaphoreArray != 0;
        HasParameterArray = native.HasParameterArray != 0;
    }

    /// <summary>Gets the signal or wait node type. 获取 signal 或 wait 节点类型。</summary>
    public CudaGraphNodeType NodeType { get; }
    /// <summary>Gets the copied semaphore count. 获取复制的 semaphore 数量。</summary>
    public uint SemaphoreCount { get; }
    /// <summary>Gets whether the native node had a semaphore array. 获取 native 节点是否包含 semaphore 数组。</summary>
    public bool HasSemaphoreArray { get; }
    /// <summary>Gets whether the native node had a parameter array. 获取 native 节点是否包含参数数组。</summary>
    public bool HasParameterArray { get; }

    /// <inheritdoc />
    public override string ToString() =>
        $"NodeType={NodeType}, SemaphoreCount={SemaphoreCount}, HasSemaphoreArray={HasSemaphoreArray}, HasParameterArray={HasParameterArray}";
}
