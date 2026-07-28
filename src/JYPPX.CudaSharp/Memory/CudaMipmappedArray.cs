using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Managed wrapper around a CUDA mipmapped array allocation.
/// CUDA mipmapped array 分配的托管封装。
/// </summary>
public sealed class CudaMipmappedArray : IDisposable
{
    private readonly SafeCudaMipmappedArrayHandle _handle;

    /// <summary>
    /// Allocates a CUDA mipmapped array.
    /// 分配一个 CUDA mipmapped array。
    /// </summary>
    /// <param name="descriptor">The channel layout descriptor. 通道布局描述符。</param>
    /// <param name="extent">The mip level extent. mip level 范围。</param>
    /// <param name="levelCount">The number of mip levels. mip level 数量。</param>
    /// <param name="flags">The CUDA array creation flags. CUDA array 创建标志。</param>
    public CudaMipmappedArray(CudaChannelFormatDescriptor descriptor, CudaArrayExtent extent, uint levelCount, CudaArrayCreationFlags flags = CudaArrayCreationFlags.Default)
    {
        if (extent.Width == 0 || extent.Height == 0 || extent.Depth == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(extent));
        }

        if (levelCount == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(levelCount));
        }

        NativeBridgeLoader.EnsureInitialized();
        _handle = NativeCudaApi.AllocateMipmappedArray(descriptor, extent, levelCount, flags);
        Descriptor = descriptor;
        Extent = extent;
        LevelCount = levelCount;
        Flags = flags;
    }

    internal SafeCudaMipmappedArrayHandle Handle => _handle;

    /// <summary>
    /// Gets the channel-format descriptor for each mip level.
    /// 获取每个 mip level 的通道格式描述符。
    /// </summary>
    public CudaChannelFormatDescriptor Descriptor { get; }
    /// <summary>
    /// Gets the logical extent of the base mip level.
    /// 获取基础 mip level 的逻辑范围。
    /// </summary>
    public CudaArrayExtent Extent { get; }
    /// <summary>
    /// Gets the number of mip levels in the allocation.
    /// 获取该分配中的 mip level 数量。
    /// </summary>
    public uint LevelCount { get; }
    /// <summary>
    /// Gets the CUDA array creation flags.
    /// 获取 CUDA array 创建标志。
    /// </summary>
    public CudaArrayCreationFlags Flags { get; }

    /// <summary>
    /// Gets metadata for a specific mip level.
    /// 获取指定 mip level 的元数据。
    /// </summary>
    /// <param name="level">The zero-based mip level. 从零开始的 mip level。</param>
    /// <returns>The mip-level metadata. mip level 元数据。</returns>
    public CudaArrayInfo GetLevelInfo(uint level)
    {
        if (level >= LevelCount)
        {
            throw new ArgumentOutOfRangeException(nameof(level));
        }

        return CudaArrayInfo.FromNative(NativeCudaApi.GetMipmappedArrayLevelInfo(_handle, level));
    }

    /// <summary>
    /// Queries device memory requirements for this mipmapped array.
    /// 查询当前 mipmapped array 的设备内存需求。
    /// </summary>
    /// <param name="device">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>The memory requirements for the requested device. 指定设备对应的内存需求。</returns>
    public CudaArrayMemoryRequirements GetMemoryRequirements(int device)
    {
        return CudaArrayMemoryRequirements.FromNative(NativeCudaApi.GetMipmappedArrayMemoryRequirements(_handle, device));
    }

    /// <summary>
    /// Tries to query device memory requirements without throwing.
    /// 尝试在不抛异常的情况下查询设备内存需求。
    /// </summary>
    /// <param name="device">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <param name="requirements">The returned memory requirements on success. 成功时返回的内存需求。</param>
    /// <param name="diagnostic">The diagnostic message on failure. 失败时的诊断消息。</param>
    /// <returns><see langword="true"/> on success. 成功时返回 <see langword="true"/>。</returns>
    public bool TryGetMemoryRequirements(int device, out CudaArrayMemoryRequirements requirements, out string diagnostic)
    {
        try
        {
            requirements = GetMemoryRequirements(device);
            diagnostic = string.Empty;
            return true;
        }
        catch (CudaException exception)
        {
            requirements = default;
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Gets sparse-allocation properties for this mipmapped array.
    /// 获取当前 mipmapped array 的稀疏分配属性。
    /// </summary>
    /// <returns>The sparse-array properties. 稀疏 array 属性。</returns>
    public CudaArraySparseProperties GetSparseProperties()
    {
        return CudaArraySparseProperties.FromNative(NativeCudaApi.GetMipmappedArraySparseProperties(_handle));
    }

    /// <summary>
    /// Tries to query sparse-allocation properties without throwing.
    /// 尝试在不抛异常的情况下查询稀疏分配属性。
    /// </summary>
    /// <param name="properties">The returned sparse properties on success. 成功时返回的稀疏属性。</param>
    /// <param name="diagnostic">The diagnostic message on failure. 失败时的诊断消息。</param>
    /// <returns><see langword="true"/> on success. 成功时返回 <see langword="true"/>。</returns>
    public bool TryGetSparseProperties(out CudaArraySparseProperties properties, out string diagnostic)
    {
        try
        {
            properties = GetSparseProperties();
            diagnostic = string.Empty;
            return true;
        }
        catch (CudaException exception)
        {
            properties = default;
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Releases the mipmapped-array handle.
    /// 释放 mipmapped array 句柄。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}
