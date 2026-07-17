using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Captures pointer-free scalar parameters from a CUDA graph kernel node.
/// 捕获 CUDA graph kernel 节点中不含指针的标量参数。
/// </summary>
public sealed class CudaGraphKernelNodeParametersSnapshot
{
    internal CudaGraphKernelNodeParametersSnapshot(NativeCudaGraphKernelNodeParamsSnapshot native)
    {
        GridX = native.GridX;
        GridY = native.GridY;
        GridZ = native.GridZ;
        BlockX = native.BlockX;
        BlockY = native.BlockY;
        BlockZ = native.BlockZ;
        SharedMemoryBytes = native.SharedMemoryBytes;
        HasFunction = native.HasFunction != 0;
        HasKernelParameters = native.HasKernelParams != 0;
        HasExtraParameters = native.HasExtra != 0;
    }

    public uint GridX { get; }
    public uint GridY { get; }
    public uint GridZ { get; }
    public uint BlockX { get; }
    public uint BlockY { get; }
    public uint BlockZ { get; }
    public uint SharedMemoryBytes { get; }
    public bool HasFunction { get; }
    public bool HasKernelParameters { get; }
    public bool HasExtraParameters { get; }

    /// <inheritdoc />
    public override string ToString() =>
        $"Grid={GridX}x{GridY}x{GridZ}, Block={BlockX}x{BlockY}x{BlockZ}, SharedMemory={SharedMemoryBytes}, HasFunction={HasFunction}, HasKernelParameters={HasKernelParameters}, HasExtraParameters={HasExtraParameters}";
}
