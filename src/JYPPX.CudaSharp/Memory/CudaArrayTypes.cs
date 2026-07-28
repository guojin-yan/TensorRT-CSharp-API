using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Describes the channel bit layout for a CUDA array. 描述 CUDA array 的通道位布局。
/// </summary>
public readonly struct CudaChannelFormatDescriptor
{
    /// <summary>
    /// Initializes a channel descriptor with explicit component bit sizes. 使用显式分量位宽初始化通道描述符。
    /// </summary>
    public CudaChannelFormatDescriptor(int x, int y, int z, int w, CudaChannelFormatKind formatKind)
    {
        X = x;
        Y = y;
        Z = z;
        W = w;
        FormatKind = formatKind;
    }

    /// <summary>
    /// Gets the bit size of the X component. 获取 X 分量的位宽。
    /// </summary>
    public int X { get; }
    /// <summary>
    /// Gets the bit size of the Y component. 获取 Y 分量的位宽。
    /// </summary>
    public int Y { get; }
    /// <summary>
    /// Gets the bit size of the Z component. 获取 Z 分量的位宽。
    /// </summary>
    public int Z { get; }
    /// <summary>
    /// Gets the bit size of the W component. 获取 W 分量的位宽。
    /// </summary>
    public int W { get; }
    /// <summary>
    /// Gets the logical format kind for the channel descriptor. 获取该通道描述符的逻辑格式类型。
    /// </summary>
    public CudaChannelFormatKind FormatKind { get; }

    /// <summary>
    /// Gets a single-channel 32-bit floating-point descriptor. 获取单通道 32 位浮点描述符。
    /// </summary>
    public static CudaChannelFormatDescriptor Float32 => new CudaChannelFormatDescriptor(32, 0, 0, 0, CudaChannelFormatKind.Float);
    /// <summary>
    /// Gets a single-channel 8-bit unsigned descriptor. 获取单通道 8 位无符号整数描述符。
    /// </summary>
    public static CudaChannelFormatDescriptor UInt8 => new CudaChannelFormatDescriptor(8, 0, 0, 0, CudaChannelFormatKind.Unsigned);

    internal NativeCudaChannelFormatDesc ToNative()
    {
        return new NativeCudaChannelFormatDesc
        {
            X = X,
            Y = Y,
            Z = Z,
            W = W,
            FormatKind = (int)FormatKind
        };
    }

    internal static CudaChannelFormatDescriptor FromNative(NativeCudaChannelFormatDesc descriptor)
    {
        return new CudaChannelFormatDescriptor(
            descriptor.X,
            descriptor.Y,
            descriptor.Z,
            descriptor.W,
            (CudaChannelFormatKind)descriptor.FormatKind);
    }

    /// <summary>
    /// Formats the channel descriptor for diagnostics.
    /// 将通道描述符格式化为便于诊断的字符串。
    /// </summary>
    public override string ToString()
    {
        return $"{FormatKind}[{X},{Y},{Z},{W}]";
    }
}

/// <summary>
/// Represents the width, height, and depth of a CUDA array extent. 表示 CUDA array 的宽度、高度和深度范围。
/// </summary>
public readonly struct CudaArrayExtent
{
    /// <summary>
    /// Initializes a CUDA array extent. 初始化 CUDA array 的范围信息。
    /// </summary>
    public CudaArrayExtent(ulong width, ulong height, ulong depth)
    {
        Width = width;
        Height = height;
        Depth = depth;
    }

    /// <summary>
    /// Gets the width of the extent. 获取范围的宽度。
    /// </summary>
    public ulong Width { get; }
    /// <summary>
    /// Gets the height of the extent. 获取范围的高度。
    /// </summary>
    public ulong Height { get; }
    /// <summary>
    /// Gets the depth of the extent. 获取范围的深度。
    /// </summary>
    public ulong Depth { get; }

    internal NativeCudaArrayExtent ToNative()
    {
        return new NativeCudaArrayExtent
        {
            Width = Width,
            Height = Height,
            Depth = Depth
        };
    }

    internal static CudaArrayExtent FromNative(NativeCudaArrayExtent extent)
    {
        return new CudaArrayExtent(extent.Width, extent.Height, extent.Depth);
    }

    /// <summary>
    /// Formats the extent as width x height x depth text.
    /// 将范围格式化为 width x height x depth 文本。
    /// </summary>
    public override string ToString()
    {
        return $"{Width}x{Height}x{Depth}";
    }
}

/// <summary>
/// Provides immutable metadata about a CUDA array. 提供 CUDA array 的不可变元数据。
/// </summary>
public readonly struct CudaArrayInfo
{
    internal CudaArrayInfo(CudaChannelFormatDescriptor channel, CudaArrayExtent extent, CudaArrayCreationFlags flags)
    {
        Channel = channel;
        Extent = extent;
        Flags = flags;
    }

    /// <summary>
    /// Gets the channel descriptor. 获取通道描述符。
    /// </summary>
    public CudaChannelFormatDescriptor Channel { get; }
    /// <summary>
    /// Gets the array extent. 获取 array 的范围信息。
    /// </summary>
    public CudaArrayExtent Extent { get; }
    /// <summary>
    /// Gets the creation flags used for the array. 获取创建该 array 时使用的标志。
    /// </summary>
    public CudaArrayCreationFlags Flags { get; }

    internal static CudaArrayInfo FromNative(NativeCudaArrayInfo info)
    {
        return new CudaArrayInfo(
            CudaChannelFormatDescriptor.FromNative(info.Channel),
            CudaArrayExtent.FromNative(info.Extent),
            (CudaArrayCreationFlags)info.Flags);
    }

    /// <summary>
    /// Formats array metadata for diagnostics.
    /// 将 array 元数据格式化为便于诊断的字符串。
    /// </summary>
    public override string ToString()
    {
        return $"{Extent} {Channel} Flags={Flags}";
    }
}

/// <summary>
/// Describes device-memory requirements for a CUDA array. 描述 CUDA array 的设备内存需求。
/// </summary>
public readonly struct CudaArrayMemoryRequirements
{
    internal CudaArrayMemoryRequirements(ulong sizeBytes, ulong alignmentBytes)
    {
        SizeBytes = sizeBytes;
        AlignmentBytes = alignmentBytes;
    }

    /// <summary>
    /// Gets the required memory size in bytes. 获取所需内存大小（字节）。
    /// </summary>
    public ulong SizeBytes { get; }
    /// <summary>
    /// Gets the required alignment in bytes. 获取所需对齐（字节）。
    /// </summary>
    public ulong AlignmentBytes { get; }

    internal static CudaArrayMemoryRequirements FromNative(NativeCudaArrayMemoryRequirements requirements)
    {
        return new CudaArrayMemoryRequirements(requirements.Size, requirements.Alignment);
    }

    /// <summary>
    /// Formats memory requirements for diagnostics.
    /// 将内存需求格式化为便于诊断的字符串。
    /// </summary>
    public override string ToString()
    {
        return $"Size={SizeBytes}, Alignment={AlignmentBytes}";
    }
}

/// <summary>
/// Describes sparse-tile metadata for a CUDA array. 描述 CUDA array 的稀疏 tile 元数据。
/// </summary>
public readonly struct CudaArraySparseProperties
{
    internal CudaArraySparseProperties(uint tileWidth, uint tileHeight, uint tileDepth, uint mipTailFirstLevel, ulong mipTailSizeBytes, CudaArraySparseFlags flags)
    {
        TileWidth = tileWidth;
        TileHeight = tileHeight;
        TileDepth = tileDepth;
        MipTailFirstLevel = mipTailFirstLevel;
        MipTailSizeBytes = mipTailSizeBytes;
        Flags = flags;
    }

    /// <summary>
    /// Gets the tile width. 获取 tile 宽度。
    /// </summary>
    public uint TileWidth { get; }
    /// <summary>
    /// Gets the tile height. 获取 tile 高度。
    /// </summary>
    public uint TileHeight { get; }
    /// <summary>
    /// Gets the tile depth. 获取 tile 深度。
    /// </summary>
    public uint TileDepth { get; }
    /// <summary>
    /// Gets the first mip-tail level. 获取首个 mip-tail level。
    /// </summary>
    public uint MipTailFirstLevel { get; }
    /// <summary>
    /// Gets the mip-tail size in bytes. 获取 mip-tail 大小（字节）。
    /// </summary>
    public ulong MipTailSizeBytes { get; }
    /// <summary>
    /// Gets sparse-array flags. 获取 sparse array 标志。
    /// </summary>
    public CudaArraySparseFlags Flags { get; }

    internal static CudaArraySparseProperties FromNative(NativeCudaArraySparseProperties properties)
    {
        return new CudaArraySparseProperties(
            properties.TileWidth,
            properties.TileHeight,
            properties.TileDepth,
            properties.MipTailFirstLevel,
            properties.MipTailSize,
            (CudaArraySparseFlags)properties.Flags);
    }

    /// <summary>
    /// Formats sparse-array properties for diagnostics.
    /// 将稀疏 array 属性格式化为便于诊断的字符串。
    /// </summary>
    public override string ToString()
    {
        return $"Tile={TileWidth}x{TileHeight}x{TileDepth}, MipTailFirstLevel={MipTailFirstLevel}, MipTailSize={MipTailSizeBytes}, Flags={Flags}";
    }
}
