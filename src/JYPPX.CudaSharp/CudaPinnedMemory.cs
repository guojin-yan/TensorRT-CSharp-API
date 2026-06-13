using System;
using System.Runtime.InteropServices;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Managed wrapper around CUDA pinned host memory.
/// CUDA pinned host memory 的托管封装。
/// </summary>
public sealed class CudaPinnedMemory : IDisposable
{
    private readonly SafeCudaPinnedMemoryHandle _handle;

    /// <summary>
    /// Allocates pinned host memory with default CUDA flags.
    /// 使用默认 CUDA 标志分配 pinned host memory。
    /// </summary>
    /// <param name="sizeInBytes">The allocation size in bytes. 分配大小，单位为字节。</param>
    public CudaPinnedMemory(int sizeInBytes)
        : this(sizeInBytes, CudaPinnedMemoryAllocationFlags.Default)
    {
    }

    /// <summary>
    /// Allocates pinned host memory with explicit CUDA flags.
    /// 使用显式 CUDA 标志分配 pinned host memory。
    /// </summary>
    /// <param name="sizeInBytes">The allocation size in bytes. 分配大小，单位为字节。</param>
    /// <param name="flags">The pinned-memory allocation flags. pinned memory 分配标志。</param>
    public CudaPinnedMemory(int sizeInBytes, CudaPinnedMemoryAllocationFlags flags)
    {
        if (sizeInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInBytes));
        }

        NativeBridgeLoader.EnsureInitialized();
        _handle = flags == CudaPinnedMemoryAllocationFlags.Default
            ? NativeCudaApi.AllocatePinnedMemory(sizeInBytes)
            : NativeCudaApi.AllocatePinnedMemory(sizeInBytes, flags);
        SizeInBytes = checked((int)NativeCudaApi.GetPinnedMemorySize(_handle));
    }

    internal SafeCudaPinnedMemoryHandle Handle => _handle;

    internal IntPtr Pointer => NativeCudaApi.GetPinnedMemoryPointer(_handle);

    /// <summary>
    /// Gets the allocation size in bytes.
    /// 获取分配大小，单位为字节。
    /// </summary>
    public int SizeInBytes { get; }

    /// <summary>
    /// Gets the CUDA flags used for the pinned allocation.
    /// 获取该 pinned 分配使用的 CUDA 标志。
    /// </summary>
    public CudaPinnedMemoryAllocationFlags Flags => NativeCudaApi.GetPinnedMemoryFlags(_handle);

    /// <summary>
    /// Gets whether the pinned allocation is mapped into device address space.
    /// 获取该 pinned 分配是否已映射到设备地址空间。
    /// </summary>
    public bool IsMapped => (Flags & CudaPinnedMemoryAllocationFlags.Mapped) != 0;

    /// <summary>
    /// Gets the mapped device pointer address when the allocation is mapped.
    /// 在分配已映射时获取其对应的设备指针地址。
    /// </summary>
    public ulong MappedDevicePointerAddress => unchecked((ulong)NativeCudaApi.GetPinnedMemoryMappedDevicePointer(_handle).ToInt64());

    /// <summary>
    /// Copies a byte array into the pinned allocation.
    /// 将字节数组复制到 pinned 分配中。
    /// </summary>
    /// <param name="source">The source byte array. 源字节数组。</param>
    public void CopyFrom(byte[] source)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        CopyFrom(source, source.Length);
    }

    /// <summary>
    /// Copies a float array into the pinned allocation.
    /// 将浮点数组复制到 pinned 分配中。
    /// </summary>
    /// <param name="source">The source float array. 源浮点数组。</param>
    public void CopyFrom(float[] source)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        int byteCount = checked(source.Length * sizeof(float));
        byte[] bytes = new byte[byteCount];
        Buffer.BlockCopy(source, 0, bytes, 0, byteCount);
        CopyFrom(bytes, byteCount);
    }

    /// <summary>
    /// Copies a specific byte count into the pinned allocation.
    /// 将指定字节数复制到 pinned 分配中。
    /// </summary>
    /// <param name="source">The source byte array. 源字节数组。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    public void CopyFrom(byte[] source, int count)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        ValidateCount(count, source.Length, nameof(count));
        Marshal.Copy(source, 0, Pointer, count);
    }

    /// <summary>
    /// Copies the pinned allocation into a byte array.
    /// 将 pinned 分配复制到字节数组中。
    /// </summary>
    /// <param name="destination">The destination byte array. 目标字节数组。</param>
    public void CopyTo(byte[] destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        CopyTo(destination, destination.Length);
    }

    /// <summary>
    /// Copies the pinned allocation into a float array.
    /// 将 pinned 分配复制到浮点数组中。
    /// </summary>
    /// <param name="destination">The destination float array. 目标浮点数组。</param>
    public void CopyTo(float[] destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        int byteCount = checked(destination.Length * sizeof(float));
        byte[] bytes = new byte[byteCount];
        CopyTo(bytes, byteCount);
        Buffer.BlockCopy(bytes, 0, destination, 0, byteCount);
    }

    /// <summary>
    /// Copies a specific byte count from the pinned allocation into a byte array.
    /// 将 pinned 分配中的指定字节数复制到字节数组中。
    /// </summary>
    /// <param name="destination">The destination byte array. 目标字节数组。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    public void CopyTo(byte[] destination, int count)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        ValidateCount(count, destination.Length, nameof(count));
        Marshal.Copy(Pointer, destination, 0, count);
    }

    /// <summary>
    /// Materializes a byte array from the pinned allocation.
    /// 从 pinned 分配中生成字节数组。
    /// </summary>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    /// <returns>A managed byte array copy. 托管字节数组副本。</returns>
    public byte[] ToArray(int count)
    {
        ValidateCount(count, count, nameof(count));
        byte[] data = new byte[count];
        CopyTo(data, count);
        return data;
    }

    /// <summary>
    /// Materializes a float array from the pinned allocation.
    /// 从 pinned 分配中生成浮点数组。
    /// </summary>
    /// <param name="elementCount">The number of single-precision elements to copy. 要复制的单精度元素数量。</param>
    /// <returns>A managed float array copy. 托管浮点数组副本。</returns>
    public float[] ToSingleArray(int elementCount)
    {
        if (elementCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(elementCount));
        }

        int byteCount = checked(elementCount * sizeof(float));
        ValidateCount(byteCount, byteCount, nameof(elementCount));
        float[] data = new float[elementCount];
        CopyTo(data);
        return data;
    }

    /// <summary>
    /// Releases the pinned allocation.
    /// 释放 pinned 分配。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ValidateCount(int count, int managedBufferLength, string parameterName)
    {
        if (count < 0 || count > SizeInBytes || count > managedBufferLength)
        {
            throw new ArgumentOutOfRangeException(parameterName);
        }
    }
}
