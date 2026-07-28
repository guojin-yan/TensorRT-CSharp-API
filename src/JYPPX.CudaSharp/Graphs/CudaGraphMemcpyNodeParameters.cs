using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Describes a CUDA graph memcpy node using copied diagnostic values.
/// 使用复制出的诊断值描述 CUDA graph memcpy 节点。
/// </summary>
/// <remarks>
/// Address properties are diagnostic numeric values borrowed from CUDA. They are not owning memory handles and cannot be used for memory access.
/// 地址属性是从 CUDA 读取的诊断数值，不是拥有所有权的内存句柄，也不能用于内存访问。
/// </remarks>
public readonly struct CudaGraphMemcpyNodeParameters
{
    internal CudaGraphMemcpyNodeParameters(NativeCudaGraphMemcpyNodeParams native)
    {
        SourceAddress = native.SourceAddress;
        DestinationAddress = native.DestinationAddress;
        SourcePitch = native.SourcePitch;
        DestinationPitch = native.DestinationPitch;
        SourceXSize = native.SourceXSize;
        SourceYSize = native.SourceYSize;
        DestinationXSize = native.DestinationXSize;
        DestinationYSize = native.DestinationYSize;
        SourcePositionX = native.SourcePositionX;
        SourcePositionY = native.SourcePositionY;
        SourcePositionZ = native.SourcePositionZ;
        DestinationPositionX = native.DestinationPositionX;
        DestinationPositionY = native.DestinationPositionY;
        DestinationPositionZ = native.DestinationPositionZ;
        Width = native.Width;
        Height = native.Height;
        Depth = native.Depth;
        Kind = (CudaMemcpyKind)native.Kind;
        SourceIsArray = native.SourceIsArray != 0;
        DestinationIsArray = native.DestinationIsArray != 0;
    }

    /// <summary>
    /// Gets the source address as a diagnostic value.
    /// 获取源地址的诊断数值。
    /// </summary>
    public ulong SourceAddress { get; }

    /// <summary>
    /// Gets the destination address as a diagnostic value.
    /// 获取目标地址的诊断数值。
    /// </summary>
    public ulong DestinationAddress { get; }

    /// <summary>
    /// Gets the source pitch in bytes.
    /// 获取源 pitch，单位为字节。
    /// </summary>
    public ulong SourcePitch { get; }

    /// <summary>
    /// Gets the destination pitch in bytes.
    /// 获取目标 pitch，单位为字节。
    /// </summary>
    public ulong DestinationPitch { get; }

    /// <summary>
    /// Gets the source row length descriptor.
    /// 获取源 row length descriptor。
    /// </summary>
    public ulong SourceXSize { get; }

    /// <summary>
    /// Gets the source layer height descriptor.
    /// 获取源 layer height descriptor。
    /// </summary>
    public ulong SourceYSize { get; }

    /// <summary>
    /// Gets the destination row length descriptor.
    /// 获取目标 row length descriptor。
    /// </summary>
    public ulong DestinationXSize { get; }

    /// <summary>
    /// Gets the destination layer height descriptor.
    /// 获取目标 layer height descriptor。
    /// </summary>
    public ulong DestinationYSize { get; }

    /// <summary>
    /// Gets the source X offset.
    /// 获取源 X 偏移。
    /// </summary>
    public ulong SourcePositionX { get; }

    /// <summary>
    /// Gets the source Y offset.
    /// 获取源 Y 偏移。
    /// </summary>
    public ulong SourcePositionY { get; }

    /// <summary>
    /// Gets the source Z offset.
    /// 获取源 Z 偏移。
    /// </summary>
    public ulong SourcePositionZ { get; }

    /// <summary>
    /// Gets the destination X offset.
    /// 获取目标 X 偏移。
    /// </summary>
    public ulong DestinationPositionX { get; }

    /// <summary>
    /// Gets the destination Y offset.
    /// 获取目标 Y 偏移。
    /// </summary>
    public ulong DestinationPositionY { get; }

    /// <summary>
    /// Gets the destination Z offset.
    /// 获取目标 Z 偏移。
    /// </summary>
    public ulong DestinationPositionZ { get; }

    /// <summary>
    /// Gets the copied extent width in bytes for 1D nodes.
    /// 获取 1D 节点的复制宽度，单位为字节。
    /// </summary>
    public ulong Width { get; }

    /// <summary>
    /// Gets the copied extent height.
    /// 获取复制高度。
    /// </summary>
    public ulong Height { get; }

    /// <summary>
    /// Gets the copied extent depth.
    /// 获取复制深度。
    /// </summary>
    public ulong Depth { get; }

    /// <summary>
    /// Gets the CUDA memcpy direction reported by CUDA.
    /// 获取 CUDA 报告的 memcpy 方向。
    /// </summary>
    public CudaMemcpyKind Kind { get; }

    /// <summary>
    /// Gets whether CUDA reported an array source.
    /// 获取 CUDA 是否报告源为 array。
    /// </summary>
    public bool SourceIsArray { get; }

    /// <summary>
    /// Gets whether CUDA reported an array destination.
    /// 获取 CUDA 是否报告目标为 array。
    /// </summary>
    public bool DestinationIsArray { get; }

    /// <summary>
    /// Formats the copied memcpy node parameters for diagnostics.
    /// 将复制出的 memcpy node 参数格式化为诊断字符串。
    /// </summary>
    public override string ToString() =>
        $"Kind={Kind}, Source=0x{SourceAddress:X}, Destination=0x{DestinationAddress:X}, Extent={Width}x{Height}x{Depth}, SourceArray={SourceIsArray}, DestinationArray={DestinationIsArray}";
}
