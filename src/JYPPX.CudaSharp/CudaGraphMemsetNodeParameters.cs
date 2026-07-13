using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Describes a CUDA graph memset node using copied scalar parameters.
/// 使用复制出的标量参数描述 CUDA graph memset 节点。
/// </summary>
/// <remarks>
/// <see cref="DestinationAddress"/> is a diagnostic address value borrowed from CUDA; it is not an owning memory handle.
/// <see cref="DestinationAddress"/> 是从 CUDA 读取的诊断地址值，不是拥有所有权的内存句柄。
/// </remarks>
public readonly struct CudaGraphMemsetNodeParameters
{
    internal CudaGraphMemsetNodeParameters(NativeCudaGraphMemsetNodeParams native)
    {
        DestinationAddress = native.DestinationAddress;
        Pitch = native.Pitch;
        Value = native.Value;
        ElementSize = native.ElementSize;
        Width = native.Width;
        Height = native.Height;
    }

    /// <summary>
    /// Gets the destination device address as a diagnostic value.
    /// 获取目标设备地址的诊断数值。
    /// </summary>
    public ulong DestinationAddress { get; }

    /// <summary>
    /// Gets the destination pitch in bytes.
    /// 获取目标 pitch，单位为字节。
    /// </summary>
    public ulong Pitch { get; }

    /// <summary>
    /// Gets the memset value.
    /// 获取 memset 填充值。
    /// </summary>
    public uint Value { get; }

    /// <summary>
    /// Gets the element size in bytes.
    /// 获取元素大小，单位为字节。
    /// </summary>
    public uint ElementSize { get; }

    /// <summary>
    /// Gets the row width in elements.
    /// 获取行宽，单位为元素。
    /// </summary>
    public ulong Width { get; }

    /// <summary>
    /// Gets the row count.
    /// 获取行数。
    /// </summary>
    public ulong Height { get; }

    /// <summary>
    /// Formats the copied memset node parameters for diagnostics.
    /// 将复制出的 memset node 参数格式化为诊断字符串。
    /// </summary>
    public override string ToString() =>
        $"Destination=0x{DestinationAddress:X}, Pitch={Pitch}, Value={Value}, ElementSize={ElementSize}, Width={Width}, Height={Height}";
}
