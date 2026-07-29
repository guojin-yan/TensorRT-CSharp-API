using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Channel-format kinds supported by CUDA arrays and textures.
/// CUDA array 与 texture 支持的通道格式类型。
/// </summary>
public enum CudaChannelFormatKind
{
    /// <summary>
    /// Signed integer channel format.
    /// 有符号整数通道格式。
    /// </summary>
    Signed = 0,
    /// <summary>
    /// Unsigned integer channel format.
    /// 无符号整数通道格式。
    /// </summary>
    Unsigned = 1,
    /// <summary>
    /// Floating-point channel format.
    /// 浮点通道格式。
    /// </summary>
    Float = 2,
    /// <summary>
    /// No channel format is specified.
    /// 未指定通道格式。
    /// </summary>
    None = 3,
    /// <summary>
    /// NV12 planar format.
    /// NV12 平面格式。
    /// </summary>
    NV12 = 4,
    /// <summary>
    /// Unsigned normalized 8-bit single-channel format.
    /// 无符号归一化 8 位单通道格式。
    /// </summary>
    UnsignedNormalized8X1 = 5,
    /// <summary>
    /// Unsigned normalized 8-bit two-channel format.
    /// 无符号归一化 8 位双通道格式。
    /// </summary>
    UnsignedNormalized8X2 = 6,
    /// <summary>
    /// Unsigned normalized 8-bit four-channel format.
    /// 无符号归一化 8 位四通道格式。
    /// </summary>
    UnsignedNormalized8X4 = 7,
    /// <summary>
    /// Unsigned normalized 16-bit single-channel format.
    /// 无符号归一化 16 位单通道格式。
    /// </summary>
    UnsignedNormalized16X1 = 8,
    /// <summary>
    /// Unsigned normalized 16-bit two-channel format.
    /// 无符号归一化 16 位双通道格式。
    /// </summary>
    UnsignedNormalized16X2 = 9,
    /// <summary>
    /// Unsigned normalized 16-bit four-channel format.
    /// 无符号归一化 16 位四通道格式。
    /// </summary>
    UnsignedNormalized16X4 = 10,
    /// <summary>
    /// Signed normalized 8-bit single-channel format.
    /// 有符号归一化 8 位单通道格式。
    /// </summary>
    SignedNormalized8X1 = 11,
    /// <summary>
    /// Signed normalized 8-bit two-channel format.
    /// 有符号归一化 8 位双通道格式。
    /// </summary>
    SignedNormalized8X2 = 12,
    /// <summary>
    /// Signed normalized 8-bit four-channel format.
    /// 有符号归一化 8 位四通道格式。
    /// </summary>
    SignedNormalized8X4 = 13,
    /// <summary>
    /// Signed normalized 16-bit single-channel format.
    /// 有符号归一化 16 位单通道格式。
    /// </summary>
    SignedNormalized16X1 = 14,
    /// <summary>
    /// Signed normalized 16-bit two-channel format.
    /// 有符号归一化 16 位双通道格式。
    /// </summary>
    SignedNormalized16X2 = 15,
    /// <summary>
    /// Signed normalized 16-bit four-channel format.
    /// 有符号归一化 16 位四通道格式。
    /// </summary>
    SignedNormalized16X4 = 16,
    /// <summary>
    /// Unsigned block-compressed format BC1.
    /// 无符号块压缩 BC1 格式。
    /// </summary>
    UnsignedBlockCompressed1 = 17,
    /// <summary>
    /// Unsigned block-compressed sRGB BC1 format.
    /// 无符号 sRGB 块压缩 BC1 格式。
    /// </summary>
    UnsignedBlockCompressed1SRgb = 18,
    /// <summary>
    /// Unsigned block-compressed format BC2.
    /// 无符号块压缩 BC2 格式。
    /// </summary>
    UnsignedBlockCompressed2 = 19,
    /// <summary>
    /// Unsigned block-compressed sRGB BC2 format.
    /// 无符号 sRGB 块压缩 BC2 格式。
    /// </summary>
    UnsignedBlockCompressed2SRgb = 20,
    /// <summary>
    /// Unsigned block-compressed format BC3.
    /// 无符号块压缩 BC3 格式。
    /// </summary>
    UnsignedBlockCompressed3 = 21,
    /// <summary>
    /// Unsigned block-compressed sRGB BC3 format.
    /// 无符号 sRGB 块压缩 BC3 格式。
    /// </summary>
    UnsignedBlockCompressed3SRgb = 22,
    /// <summary>
    /// Unsigned block-compressed format BC4.
    /// 无符号块压缩 BC4 格式。
    /// </summary>
    UnsignedBlockCompressed4 = 23,
    /// <summary>
    /// Signed block-compressed format BC4.
    /// 有符号块压缩 BC4 格式。
    /// </summary>
    SignedBlockCompressed4 = 24,
    /// <summary>
    /// Unsigned block-compressed format BC5.
    /// 无符号块压缩 BC5 格式。
    /// </summary>
    UnsignedBlockCompressed5 = 25,
    /// <summary>
    /// Signed block-compressed format BC5.
    /// 有符号块压缩 BC5 格式。
    /// </summary>
    SignedBlockCompressed5 = 26,
    /// <summary>
    /// Unsigned block-compressed format BC6H.
    /// 无符号块压缩 BC6H 格式。
    /// </summary>
    UnsignedBlockCompressed6H = 27,
    /// <summary>
    /// Signed block-compressed format BC6H.
    /// 有符号块压缩 BC6H 格式。
    /// </summary>
    SignedBlockCompressed6H = 28,
    /// <summary>
    /// Unsigned block-compressed format BC7.
    /// 无符号块压缩 BC7 格式。
    /// </summary>
    UnsignedBlockCompressed7 = 29,
    /// <summary>
    /// Unsigned block-compressed sRGB BC7 format.
    /// 无符号 sRGB 块压缩 BC7 格式。
    /// </summary>
    UnsignedBlockCompressed7SRgb = 30,
    /// <summary>
    /// Unsigned normalized 10:10:10:2 packed format.
    /// 无符号归一化 10:10:10:2 打包格式。
    /// </summary>
    UnsignedNormalized1010102 = 31
}
/// <summary>
/// Flags that control CUDA array creation behavior.
/// 控制 CUDA array 创建行为的标志。
/// </summary>
[Flags]
public enum CudaArrayCreationFlags : uint
{
    /// <summary>
    /// Uses CUDA default array-creation behavior.
    /// 使用 CUDA 默认的 array 创建行为。
    /// </summary>
    Default = 0x00,
    /// <summary>
    /// Creates a layered CUDA array.
    /// 创建 layered CUDA array。
    /// </summary>
    Layered = 0x01,
    /// <summary>
    /// Enables surface load/store support.
    /// 启用 surface load/store 支持。
    /// </summary>
    SurfaceLoadStore = 0x02,
    /// <summary>
    /// Creates a cubemap-compatible CUDA array.
    /// 创建兼容 cubemap 的 CUDA array。
    /// </summary>
    Cubemap = 0x04,
    /// <summary>
    /// Enables texture gather support.
    /// 启用 texture gather 支持。
    /// </summary>
    TextureGather = 0x08,
    /// <summary>
    /// Marks the array as a color attachment.
    /// 将该 array 标记为 color attachment。
    /// </summary>
    ColorAttachment = 0x20,
    /// <summary>
    /// Creates a sparse CUDA array.
    /// 创建 sparse CUDA array。
    /// </summary>
    Sparse = 0x40,
    /// <summary>
    /// Enables deferred mapping for sparse arrays.
    /// 为 sparse array 启用 deferred mapping。
    /// </summary>
    DeferredMapping = 0x80
}

/// <summary>
/// Flags that describe sparse CUDA array behavior.
/// 描述 sparse CUDA array 行为的标志。
/// </summary>
[Flags]
public enum CudaArraySparseFlags : uint
{
    /// <summary>
    /// No sparse-array flags are set.
    /// 不设置任何 sparse array 标志。
    /// </summary>
    None = 0,
    /// <summary>
    /// Uses a single mip tail for the sparse array.
    /// 为 sparse array 使用单个 mip tail。
    /// </summary>
    SingleMipTail = 0x01
}
