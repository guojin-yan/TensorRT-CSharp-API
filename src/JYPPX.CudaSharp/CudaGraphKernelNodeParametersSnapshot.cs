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

    /// <summary>Gets the grid size along X. 获取 grid 的 X 维大小。</summary>
    public uint GridX { get; }
    /// <summary>Gets the grid size along Y. 获取 grid 的 Y 维大小。</summary>
    public uint GridY { get; }
    /// <summary>Gets the grid size along Z. 获取 grid 的 Z 维大小。</summary>
    public uint GridZ { get; }
    /// <summary>Gets the block size along X. 获取 block 的 X 维大小。</summary>
    public uint BlockX { get; }
    /// <summary>Gets the block size along Y. 获取 block 的 Y 维大小。</summary>
    public uint BlockY { get; }
    /// <summary>Gets the block size along Z. 获取 block 的 Z 维大小。</summary>
    public uint BlockZ { get; }
    /// <summary>Gets the dynamic shared-memory size in bytes. 获取动态 shared memory 的字节数。</summary>
    public uint SharedMemoryBytes { get; }
    /// <summary>Gets whether the native parameters contained a function. 获取 native 参数是否包含函数。</summary>
    public bool HasFunction { get; }
    /// <summary>Gets whether the native parameters contained kernel arguments. 获取 native 参数是否包含 kernel 参数。</summary>
    public bool HasKernelParameters { get; }
    /// <summary>Gets whether the native parameters contained an extra-argument array. 获取 native 参数是否包含额外参数数组。</summary>
    public bool HasExtraParameters { get; }

    /// <inheritdoc />
    public override string ToString() =>
        $"Grid={GridX}x{GridY}x{GridZ}, Block={BlockX}x{BlockY}x{BlockZ}, SharedMemory={SharedMemoryBytes}, HasFunction={HasFunction}, HasKernelParameters={HasKernelParameters}, HasExtraParameters={HasExtraParameters}";
}
