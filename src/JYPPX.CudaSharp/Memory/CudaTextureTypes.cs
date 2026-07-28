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
    /// <summary>No explicit resource-view format. 未指定显式 resource-view 格式。</summary>
    None = 0x00,
    /// <summary>One-component unsigned 8-bit integer view. 单通道无符号 8 位整数视图。</summary>
    UnsignedChar1 = 0x01,
    /// <summary>Two-component unsigned 8-bit integer view. 双通道无符号 8 位整数视图。</summary>
    UnsignedChar2 = 0x02,
    /// <summary>Four-component unsigned 8-bit integer view. 四通道无符号 8 位整数视图。</summary>
    UnsignedChar4 = 0x03,
    /// <summary>One-component signed 8-bit integer view. 单通道有符号 8 位整数视图。</summary>
    SignedChar1 = 0x04,
    /// <summary>Two-component signed 8-bit integer view. 双通道有符号 8 位整数视图。</summary>
    SignedChar2 = 0x05,
    /// <summary>Four-component signed 8-bit integer view. 四通道有符号 8 位整数视图。</summary>
    SignedChar4 = 0x06,
    /// <summary>One-component unsigned 16-bit integer view. 单通道无符号 16 位整数视图。</summary>
    UnsignedShort1 = 0x07,
    /// <summary>Two-component unsigned 16-bit integer view. 双通道无符号 16 位整数视图。</summary>
    UnsignedShort2 = 0x08,
    /// <summary>Four-component unsigned 16-bit integer view. 四通道无符号 16 位整数视图。</summary>
    UnsignedShort4 = 0x09,
    /// <summary>One-component signed 16-bit integer view. 单通道有符号 16 位整数视图。</summary>
    SignedShort1 = 0x0A,
    /// <summary>Two-component signed 16-bit integer view. 双通道有符号 16 位整数视图。</summary>
    SignedShort2 = 0x0B,
    /// <summary>Four-component signed 16-bit integer view. 四通道有符号 16 位整数视图。</summary>
    SignedShort4 = 0x0C,
    /// <summary>One-component unsigned 32-bit integer view. 单通道无符号 32 位整数视图。</summary>
    UnsignedInt1 = 0x0D,
    /// <summary>Two-component unsigned 32-bit integer view. 双通道无符号 32 位整数视图。</summary>
    UnsignedInt2 = 0x0E,
    /// <summary>Four-component unsigned 32-bit integer view. 四通道无符号 32 位整数视图。</summary>
    UnsignedInt4 = 0x0F,
    /// <summary>One-component signed 32-bit integer view. 单通道有符号 32 位整数视图。</summary>
    SignedInt1 = 0x10,
    /// <summary>Two-component signed 32-bit integer view. 双通道有符号 32 位整数视图。</summary>
    SignedInt2 = 0x11,
    /// <summary>Four-component signed 32-bit integer view. 四通道有符号 32 位整数视图。</summary>
    SignedInt4 = 0x12,
    /// <summary>One-component half-precision floating-point view. 单通道半精度浮点视图。</summary>
    Half1 = 0x13,
    /// <summary>Two-component half-precision floating-point view. 双通道半精度浮点视图。</summary>
    Half2 = 0x14,
    /// <summary>Four-component half-precision floating-point view. 四通道半精度浮点视图。</summary>
    Half4 = 0x15,
    /// <summary>One-component single-precision floating-point view. 单通道单精度浮点视图。</summary>
    Float1 = 0x16,
    /// <summary>Two-component single-precision floating-point view. 双通道单精度浮点视图。</summary>
    Float2 = 0x17,
    /// <summary>Four-component single-precision floating-point view. 四通道单精度浮点视图。</summary>
    Float4 = 0x18,
    /// <summary>Unsigned BC1 block-compressed view. 无符号 BC1 块压缩视图。</summary>
    UnsignedBlockCompressed1 = 0x19,
    /// <summary>Unsigned BC2 block-compressed view. 无符号 BC2 块压缩视图。</summary>
    UnsignedBlockCompressed2 = 0x1A,
    /// <summary>Unsigned BC3 block-compressed view. 无符号 BC3 块压缩视图。</summary>
    UnsignedBlockCompressed3 = 0x1B,
    /// <summary>Unsigned BC4 block-compressed view. 无符号 BC4 块压缩视图。</summary>
    UnsignedBlockCompressed4 = 0x1C,
    /// <summary>Signed BC4 block-compressed view. 有符号 BC4 块压缩视图。</summary>
    SignedBlockCompressed4 = 0x1D,
    /// <summary>Unsigned BC5 block-compressed view. 无符号 BC5 块压缩视图。</summary>
    UnsignedBlockCompressed5 = 0x1E,
    /// <summary>Signed BC5 block-compressed view. 有符号 BC5 块压缩视图。</summary>
    SignedBlockCompressed5 = 0x1F,
    /// <summary>Unsigned BC6H block-compressed view. 无符号 BC6H 块压缩视图。</summary>
    UnsignedBlockCompressed6H = 0x20,
    /// <summary>Signed BC6H block-compressed view. 有符号 BC6H 块压缩视图。</summary>
    SignedBlockCompressed6H = 0x21,
    /// <summary>Unsigned BC7 block-compressed view. 无符号 BC7 块压缩视图。</summary>
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

    /// <summary>Gets the X-axis addressing mode. 获取 X 轴寻址模式。</summary>
    public CudaTextureAddressMode AddressModeX { get; }
    /// <summary>Gets the Y-axis addressing mode. 获取 Y 轴寻址模式。</summary>
    public CudaTextureAddressMode AddressModeY { get; }
    /// <summary>Gets the Z-axis addressing mode. 获取 Z 轴寻址模式。</summary>
    public CudaTextureAddressMode AddressModeZ { get; }
    /// <summary>Gets the base-level filtering mode. 获取基础 mip level 的过滤模式。</summary>
    public CudaTextureFilterMode FilterMode { get; }
    /// <summary>Gets the texture read-conversion mode. 获取 texture 读取转换模式。</summary>
    public CudaTextureReadMode ReadMode { get; }
    /// <summary>Gets whether texture coordinates are normalized. 获取 texture 坐标是否归一化。</summary>
    public bool NormalizedCoordinates { get; }
    /// <summary>Gets whether sRGB conversion is enabled. 获取是否启用 sRGB 转换。</summary>
    public bool Srgb { get; }
    /// <summary>Gets the maximum anisotropy value. 获取最大各向异性值。</summary>
    public uint MaxAnisotropy { get; }
    /// <summary>Gets the mipmap filtering mode. 获取 mipmap 过滤模式。</summary>
    public CudaTextureFilterMode MipmapFilterMode { get; }
    /// <summary>Gets the mipmap level-of-detail bias. 获取 mipmap 细节级别偏移。</summary>
    public float MipmapLevelBias { get; }
    /// <summary>Gets the minimum mipmap level clamp. 获取最小 mipmap level 限制。</summary>
    public float MinMipmapLevelClamp { get; }
    /// <summary>Gets the maximum mipmap level clamp. 获取最大 mipmap level 限制。</summary>
    public float MaxMipmapLevelClamp { get; }
    /// <summary>Gets whether trilinear optimization is disabled. 获取是否禁用三线性优化。</summary>
    public bool DisableTrilinearOptimization { get; }
    /// <summary>Gets whether seamless cubemap filtering is enabled. 获取是否启用无缝 cubemap 过滤。</summary>
    public bool SeamlessCubemap { get; }
    /// <summary>Gets the red border-color component. 获取边界颜色的红色分量。</summary>
    public float BorderColorR { get; }
    /// <summary>Gets the green border-color component. 获取边界颜色的绿色分量。</summary>
    public float BorderColorG { get; }
    /// <summary>Gets the blue border-color component. 获取边界颜色的蓝色分量。</summary>
    public float BorderColorB { get; }
    /// <summary>Gets the alpha border-color component. 获取边界颜色的透明度分量。</summary>
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

    /// <summary>Gets the copied CUDA resource type. 获取复制的 CUDA 资源类型。</summary>
    public CudaResourceType ResourceType { get; }
    /// <summary>Gets whether an array handle was present without exposing it. 获取是否存在 array handle，但不暴露其值。</summary>
    public bool HasArray { get; }
    /// <summary>Gets whether a mipmapped-array handle was present without exposing it. 获取是否存在 mipmapped-array handle，但不暴露其值。</summary>
    public bool HasMipmappedArray { get; }
    /// <summary>Gets whether a device pointer was present without exposing it. 获取是否存在 device pointer，但不暴露其值。</summary>
    public bool HasDevicePointer { get; }
    /// <summary>Gets the linear resource size in bytes. 获取线性资源的字节数。</summary>
    public ulong SizeInBytes { get; }
    /// <summary>Gets the copied resource width. 获取复制的资源宽度。</summary>
    public ulong Width { get; }
    /// <summary>Gets the copied resource height. 获取复制的资源高度。</summary>
    public ulong Height { get; }
    /// <summary>Gets the copied row pitch in bytes. 获取复制的行 pitch 字节数。</summary>
    public ulong PitchInBytes { get; }

    /// <summary>Formats the copied resource descriptor for diagnostics. 格式化复制型资源 descriptor 以供诊断。</summary>
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
    /// <summary>Gets the copied resource-view format. 获取复制的 resource-view 格式。</summary>
    public CudaTextureResourceViewFormat Format { get; }
    /// <summary>Gets the resource-view width. 获取 resource-view 宽度。</summary>
    public ulong Width { get; }
    /// <summary>Gets the resource-view height. 获取 resource-view 高度。</summary>
    public ulong Height { get; }
    /// <summary>Gets the resource-view depth. 获取 resource-view 深度。</summary>
    public ulong Depth { get; }
    /// <summary>Gets the first visible mipmap level. 获取第一个可见 mipmap level。</summary>
    public uint FirstMipmapLevel { get; }
    /// <summary>Gets the last visible mipmap level. 获取最后一个可见 mipmap level。</summary>
    public uint LastMipmapLevel { get; }
    /// <summary>Gets the first visible array layer. 获取第一个可见 array layer。</summary>
    public uint FirstLayer { get; }
    /// <summary>Gets the last visible array layer. 获取最后一个可见 array layer。</summary>
    public uint LastLayer { get; }

    /// <summary>Formats the copied resource view for diagnostics. 格式化复制型 resource view 以供诊断。</summary>
    public override string ToString() =>
        $"Specified={IsSpecified} Format={Format} Extent={Width}x{Height}x{Depth} Mips={FirstMipmapLevel}-{LastMipmapLevel} Layers={FirstLayer}-{LastLayer}";
}
