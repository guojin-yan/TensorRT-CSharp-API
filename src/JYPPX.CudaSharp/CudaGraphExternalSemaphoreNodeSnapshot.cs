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

    public CudaGraphNodeType NodeType { get; }
    public uint SemaphoreCount { get; }
    public bool HasSemaphoreArray { get; }
    public bool HasParameterArray { get; }

    /// <inheritdoc />
    public override string ToString() =>
        $"NodeType={NodeType}, SemaphoreCount={SemaphoreCount}, HasSemaphoreArray={HasSemaphoreArray}, HasParameterArray={HasParameterArray}";
}
