using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public readonly struct CudaChannelFormatDescriptor
{
    public CudaChannelFormatDescriptor(int x, int y, int z, int w, CudaChannelFormatKind formatKind)
    {
        X = x;
        Y = y;
        Z = z;
        W = w;
        FormatKind = formatKind;
    }

    public int X { get; }
    public int Y { get; }
    public int Z { get; }
    public int W { get; }
    public CudaChannelFormatKind FormatKind { get; }

    public static CudaChannelFormatDescriptor Float32 => new CudaChannelFormatDescriptor(32, 0, 0, 0, CudaChannelFormatKind.Float);
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

    public override string ToString()
    {
        return $"{FormatKind}[{X},{Y},{Z},{W}]";
    }
}

public readonly struct CudaArrayExtent
{
    public CudaArrayExtent(ulong width, ulong height, ulong depth)
    {
        Width = width;
        Height = height;
        Depth = depth;
    }

    public ulong Width { get; }
    public ulong Height { get; }
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

    public override string ToString()
    {
        return $"{Width}x{Height}x{Depth}";
    }
}

public readonly struct CudaArrayInfo
{
    internal CudaArrayInfo(CudaChannelFormatDescriptor channel, CudaArrayExtent extent, CudaArrayCreationFlags flags)
    {
        Channel = channel;
        Extent = extent;
        Flags = flags;
    }

    public CudaChannelFormatDescriptor Channel { get; }
    public CudaArrayExtent Extent { get; }
    public CudaArrayCreationFlags Flags { get; }

    internal static CudaArrayInfo FromNative(NativeCudaArrayInfo info)
    {
        return new CudaArrayInfo(
            CudaChannelFormatDescriptor.FromNative(info.Channel),
            CudaArrayExtent.FromNative(info.Extent),
            (CudaArrayCreationFlags)info.Flags);
    }

    public override string ToString()
    {
        return $"{Extent} {Channel} Flags={Flags}";
    }
}

public readonly struct CudaArrayMemoryRequirements
{
    internal CudaArrayMemoryRequirements(ulong sizeBytes, ulong alignmentBytes)
    {
        SizeBytes = sizeBytes;
        AlignmentBytes = alignmentBytes;
    }

    public ulong SizeBytes { get; }
    public ulong AlignmentBytes { get; }

    internal static CudaArrayMemoryRequirements FromNative(NativeCudaArrayMemoryRequirements requirements)
    {
        return new CudaArrayMemoryRequirements(requirements.Size, requirements.Alignment);
    }

    public override string ToString()
    {
        return $"Size={SizeBytes}, Alignment={AlignmentBytes}";
    }
}

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

    public uint TileWidth { get; }
    public uint TileHeight { get; }
    public uint TileDepth { get; }
    public uint MipTailFirstLevel { get; }
    public ulong MipTailSizeBytes { get; }
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

    public override string ToString()
    {
        return $"Tile={TileWidth}x{TileHeight}x{TileDepth}, MipTailFirstLevel={MipTailFirstLevel}, MipTailSize={MipTailSizeBytes}, Flags={Flags}";
    }
}
