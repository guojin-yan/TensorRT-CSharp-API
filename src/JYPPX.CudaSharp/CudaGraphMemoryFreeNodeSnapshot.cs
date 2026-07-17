using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Captures whether a CUDA graph memory-free node has a configured device pointer without exposing it.
/// 捕获 CUDA graph memory-free 节点是否配置 device pointer，但不暴露该 pointer。
/// </summary>
public sealed class CudaGraphMemoryFreeNodeSnapshot
{
    internal CudaGraphMemoryFreeNodeSnapshot(NativeCudaGraphMemFreeNodeParamsSnapshot native)
    {
        HasDevicePointer = native.HasDevicePointer != 0;
    }

    public bool HasDevicePointer { get; }

    /// <inheritdoc />
    public override string ToString() => $"HasDevicePointer={HasDevicePointer}";
}
