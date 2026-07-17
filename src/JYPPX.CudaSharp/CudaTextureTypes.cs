using System;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>Specifies CUDA texture addressing behavior. 指定 CUDA texture 地址处理行为。</summary>
public enum CudaTextureAddressMode
{
    /// <summary>Wraps coordinates. 循环坐标。</summary>
    Wrap = 0,
    /// <summary>Clamps coordinates. 截断坐标。</summary>
    Clamp = 1,
    /// <summary>Mirrors coordinates. 镜像坐标。</summary>
    Mirror = 2,
    /// <summary>Uses the border color. 使用边界颜色。</summary>
    Border = 3
}

/// <summary>Specifies CUDA texture filtering. 指定 CUDA texture 过滤方式。</summary>
public enum CudaTextureFilterMode
{
    /// <summary>Uses point sampling. 使用点采样。</summary>
    Point = 0,
    /// <summary>Uses linear filtering. 使用线性过滤。</summary>
    Linear = 1
}

/// <summary>Specifies CUDA texture read conversion. 指定 CUDA texture 读取转换。</summary>
public enum CudaTextureReadMode
{
    /// <summary>Returns the stored element type. 返回存储的元素类型。</summary>
    ElementType = 0,
    /// <summary>Returns normalized floating-point values. 返回归一化浮点值。</summary>
    NormalizedFloat = 1
}

/// <summary>Identifies the CUDA resource kind copied from a descriptor. 标识从 descriptor 复制出的 CUDA 资源类型。</summary>
public enum CudaResourceType
{
    /// <summary>CUDA array resource. CUDA array 资源。</summary>
    Array = 0,
    /// <summary>CUDA mipmapped array resource. CUDA mipmapped array 资源。</summary>
    MipmappedArray = 1,
    /// <summary>Linear device-memory resource. 线性设备内存资源。</summary>
    Linear = 2,
    /// <summary>Pitched two-dimensional device-memory resource. 带 pitch 的二维设备内存资源。</summary>
    Pitch2D = 3
}

/// <summary>Identifies a CUDA texture resource-view format. 标识 CUDA texture resource-view 格式。</summary>
public enum CudaTextureResourceViewFormat
{
    None = 0x00,
    UnsignedChar1 = 0x01,
    UnsignedChar2 = 0x02,
    UnsignedChar4 = 0x03,
    SignedChar1 = 0x04,
    SignedChar2 = 0x05,
    SignedChar4 = 0x06,
    UnsignedShort1 = 0x07,
    UnsignedShort2 = 0x08,
    UnsignedShort4 = 0x09,
    SignedShort1 = 0x0A,
    SignedShort2 = 0x0B,
    SignedShort4 = 0x0C,
    UnsignedInt1 = 0x0D,
    UnsignedInt2 = 0x0E,
    UnsignedInt4 = 0x0F,
    SignedInt1 = 0x10,
    SignedInt2 = 0x11,
    SignedInt4 = 0x12,
    Half1 = 0x13,
    Half2 = 0x14,
    Half4 = 0x15,
    Float1 = 0x16,
    Float2 = 0x17,
    Float4 = 0x18,
    UnsignedBlockCompressed1 = 0x19,
    UnsignedBlockCompressed2 = 0x1A,
    UnsignedBlockCompressed3 = 0x1B,
    UnsignedBlockCompressed4 = 0x1C,
    SignedBlockCompressed4 = 0x1D,
    UnsignedBlockCompressed5 = 0x1E,
    SignedBlockCompressed5 = 0x1F,
    UnsignedBlockCompressed6H = 0x20,
    SignedBlockCompressed6H = 0x21,
    UnsignedBlockCompressed7 = 0x22
}

/// <summary>Defines pointer-free CUDA texture sampling parameters. 定义无指针的 CUDA texture 采样参数。</summary>
public readonly struct CudaTextureDescriptor
{
    /// <summary>Initializes a CUDA texture descriptor. 初始化 CUDA texture descriptor。</summary>
    public CudaTextureDescriptor(
        CudaTextureAddressMode addressModeX,
        CudaTextureAddressMode addressModeY = CudaTextureAddressMode.Clamp,
        CudaTextureAddressMode addressModeZ = CudaTextureAddressMode.Clamp,
        CudaTextureFilterMode filterMode = CudaTextureFilterMode.Point,
        CudaTextureReadMode readMode = CudaTextureReadMode.ElementType,
        bool normalizedCoordinates = false,
        bool srgb = false,
        uint maxAnisotropy = 1,
        CudaTextureFilterMode mipmapFilterMode = CudaTextureFilterMode.Point,
        float mipmapLevelBias = 0,
        float minMipmapLevelClamp = 0,
        float maxMipmapLevelClamp = 0,
        bool disableTrilinearOptimization = false,
        bool seamlessCubemap = false,
        float borderColorR = 0,
        float borderColorG = 0,
        float borderColorB = 0,
        float borderColorA = 0)
    {
        AddressModeX = addressModeX;
        AddressModeY = addressModeY;
        AddressModeZ = addressModeZ;
        FilterMode = filterMode;
        ReadMode = readMode;
        NormalizedCoordinates = normalizedCoordinates;
        Srgb = srgb;
        MaxAnisotropy = maxAnisotropy;
        MipmapFilterMode = mipmapFilterMode;
        MipmapLevelBias = mipmapLevelBias;
        MinMipmapLevelClamp = minMipmapLevelClamp;
        MaxMipmapLevelClamp = maxMipmapLevelClamp;
        DisableTrilinearOptimization = disableTrilinearOptimization;
        SeamlessCubemap = seamlessCubemap;
        BorderColorR = borderColorR;
        BorderColorG = borderColorG;
        BorderColorB = borderColorB;
        BorderColorA = borderColorA;
        Validate();
    }

    /// <summary>Gets a conservative point-sampling descriptor. 获取保守的点采样 descriptor。</summary>
    public static CudaTextureDescriptor Default => new CudaTextureDescriptor(CudaTextureAddressMode.Clamp);

    public CudaTextureAddressMode AddressModeX { get; }
    public CudaTextureAddressMode AddressModeY { get; }
    public CudaTextureAddressMode AddressModeZ { get; }
    public CudaTextureFilterMode FilterMode { get; }
    public CudaTextureReadMode ReadMode { get; }
    public bool NormalizedCoordinates { get; }
    public bool Srgb { get; }
    public uint MaxAnisotropy { get; }
    public CudaTextureFilterMode MipmapFilterMode { get; }
    public float MipmapLevelBias { get; }
    public float MinMipmapLevelClamp { get; }
    public float MaxMipmapLevelClamp { get; }
    public bool DisableTrilinearOptimization { get; }
    public bool SeamlessCubemap { get; }
    public float BorderColorR { get; }
    public float BorderColorG { get; }
    public float BorderColorB { get; }
    public float BorderColorA { get; }

    internal NativeCudaTextureDescriptor ToNative()
    {
        Validate();
        return new NativeCudaTextureDescriptor
        {
            AddressModeX = (int)AddressModeX,
            AddressModeY = (int)AddressModeY,
            AddressModeZ = (int)AddressModeZ,
            FilterMode = (int)FilterMode,
            ReadMode = (int)ReadMode,
            Srgb = Srgb ? 1 : 0,
            BorderColorR = BorderColorR,
            BorderColorG = BorderColorG,
            BorderColorB = BorderColorB,
            BorderColorA = BorderColorA,
            NormalizedCoordinates = NormalizedCoordinates ? 1 : 0,
            MaxAnisotropy = MaxAnisotropy,
            MipmapFilterMode = (int)MipmapFilterMode,
            MipmapLevelBias = MipmapLevelBias,
            MinMipmapLevelClamp = MinMipmapLevelClamp,
            MaxMipmapLevelClamp = MaxMipmapLevelClamp,
            DisableTrilinearOptimization = DisableTrilinearOptimization ? 1 : 0,
            SeamlessCubemap = SeamlessCubemap ? 1 : 0
        };
    }

    internal static CudaTextureDescriptor FromNative(NativeCudaTextureDescriptor descriptor)
    {
        return new CudaTextureDescriptor(
            (CudaTextureAddressMode)descriptor.AddressModeX,
            (CudaTextureAddressMode)descriptor.AddressModeY,
            (CudaTextureAddressMode)descriptor.AddressModeZ,
            (CudaTextureFilterMode)descriptor.FilterMode,
            (CudaTextureReadMode)descriptor.ReadMode,
            descriptor.NormalizedCoordinates != 0,
            descriptor.Srgb != 0,
            descriptor.MaxAnisotropy,
            (CudaTextureFilterMode)descriptor.MipmapFilterMode,
            descriptor.MipmapLevelBias,
            descriptor.MinMipmapLevelClamp,
            descriptor.MaxMipmapLevelClamp,
            descriptor.DisableTrilinearOptimization != 0,
            descriptor.SeamlessCubemap != 0,
            descriptor.BorderColorR,
            descriptor.BorderColorG,
            descriptor.BorderColorB,
            descriptor.BorderColorA);
    }

    private void Validate()
    {
        if (!Enum.IsDefined(typeof(CudaTextureAddressMode), AddressModeX) ||
            !Enum.IsDefined(typeof(CudaTextureAddressMode), AddressModeY) ||
            !Enum.IsDefined(typeof(CudaTextureAddressMode), AddressModeZ))
        {
            throw new ArgumentOutOfRangeException(nameof(AddressModeX));
        }
        if (!Enum.IsDefined(typeof(CudaTextureFilterMode), FilterMode) ||
            !Enum.IsDefined(typeof(CudaTextureFilterMode), MipmapFilterMode))
        {
            throw new ArgumentOutOfRangeException(nameof(FilterMode));
        }
        if (!Enum.IsDefined(typeof(CudaTextureReadMode), ReadMode))
        {
            throw new ArgumentOutOfRangeException(nameof(ReadMode));
        }
        if (MaxAnisotropy > 16)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxAnisotropy));
        }
        if (!IsFinite(MipmapLevelBias) || !IsFinite(MinMipmapLevelClamp) || !IsFinite(MaxMipmapLevelClamp) ||
            !IsFinite(BorderColorR) || !IsFinite(BorderColorG) || !IsFinite(BorderColorB) || !IsFinite(BorderColorA) ||
            MinMipmapLevelClamp > MaxMipmapLevelClamp)
        {
            throw new ArgumentOutOfRangeException(nameof(MinMipmapLevelClamp));
        }
    }

    private static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

    /// <summary>Formats the copied descriptor for diagnostics. 格式化复制型 descriptor。</summary>
    public override string ToString() =>
        $"Address=[{AddressModeX},{AddressModeY},{AddressModeZ}] Filter={FilterMode} Read={ReadMode} Normalized={NormalizedCoordinates} Srgb={Srgb} Anisotropy={MaxAnisotropy} Seamless={SeamlessCubemap}";
}

/// <summary>Contains a pointer-free copied CUDA resource descriptor. 包含无指针的 CUDA resource descriptor 副本。</summary>
public readonly struct CudaResourceDescriptorSnapshot
{
    internal CudaResourceDescriptorSnapshot(NativeCudaResourceDescriptorSnapshot snapshot)
    {
        ResourceType = (CudaResourceType)snapshot.ResourceType;
        HasArray = snapshot.HasArray != 0;
        HasMipmappedArray = snapshot.HasMipmappedArray != 0;
        HasDevicePointer = snapshot.HasDevicePointer != 0;
        SizeInBytes = snapshot.SizeInBytes;
        Width = snapshot.Width;
        Height = snapshot.Height;
        PitchInBytes = snapshot.PitchInBytes;
    }

    public CudaResourceType ResourceType { get; }
    public bool HasArray { get; }
    public bool HasMipmappedArray { get; }
    public bool HasDevicePointer { get; }
    public ulong SizeInBytes { get; }
    public ulong Width { get; }
    public ulong Height { get; }
    public ulong PitchInBytes { get; }

    public override string ToString() =>
        $"Type={ResourceType} Array={HasArray} Mipmapped={HasMipmappedArray} DevicePointer={HasDevicePointer} Size={SizeInBytes} Extent={Width}x{Height} Pitch={PitchInBytes}";
}

/// <summary>Contains a copied CUDA texture resource-view descriptor. 包含复制型 CUDA texture resource-view descriptor。</summary>
public readonly struct CudaTextureResourceViewSnapshot
{
    internal CudaTextureResourceViewSnapshot(NativeCudaTextureResourceViewSnapshot snapshot)
    {
        IsSpecified = snapshot.IsSpecified != 0;
        Format = (CudaTextureResourceViewFormat)snapshot.Format;
        Width = snapshot.Width;
        Height = snapshot.Height;
        Depth = snapshot.Depth;
        FirstMipmapLevel = snapshot.FirstMipmapLevel;
        LastMipmapLevel = snapshot.LastMipmapLevel;
        FirstLayer = snapshot.FirstLayer;
        LastLayer = snapshot.LastLayer;
    }

    /// <summary>Gets whether creation supplied an explicit resource-view descriptor. 获取创建时是否提供了显式 resource-view descriptor。</summary>
    public bool IsSpecified { get; }
    public CudaTextureResourceViewFormat Format { get; }
    public ulong Width { get; }
    public ulong Height { get; }
    public ulong Depth { get; }
    public uint FirstMipmapLevel { get; }
    public uint LastMipmapLevel { get; }
    public uint FirstLayer { get; }
    public uint LastLayer { get; }

    public override string ToString() =>
        $"Specified={IsSpecified} Format={Format} Extent={Width}x{Height}x{Depth} Mips={FirstMipmapLevel}-{LastMipmapLevel} Layers={FirstLayer}-{LastLayer}";
}
