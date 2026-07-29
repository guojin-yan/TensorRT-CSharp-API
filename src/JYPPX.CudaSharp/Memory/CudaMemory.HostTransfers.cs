using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public partial class CudaMemory
{
    /// <summary>
    /// Copies a byte array into device memory.
    /// 将字节数组复制到设备内存中。
    /// </summary>
    /// <param name="source">The source byte array. 源字节数组。</param>
    public void CopyFrom(byte[] source)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        if (source.Length > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(source));
        }

        NativeCudaApi.CopyFromHost(_handle, source, source.Length);
    }

    /// <summary>
    /// Copies a float array into device memory.
    /// 将浮点数组复制到设备内存中。
    /// </summary>
    /// <param name="source">The source float array. 源浮点数组。</param>
    public void CopyFrom(float[] source)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        int byteCount = checked(source.Length * sizeof(float));
        if (byteCount > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(source));
        }

        byte[] bytes = new byte[byteCount];
        Buffer.BlockCopy(source, 0, bytes, 0, byteCount);
        CopyFrom(bytes);
    }

    /// <summary>
    /// Asynchronously copies a pinned host buffer into device memory.
    /// 将 pinned host 缓冲区异步复制到设备内存中。
    /// </summary>
    /// <param name="source">The source pinned host buffer. 源 pinned host 缓冲区。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyFromAsync(CudaPinnedMemory source, int count, CudaStream stream)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        ValidateCount(count, nameof(count));
        if (count > source.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyFromPinnedHostAsync(_handle, source.Handle, count, stream.Handle);
    }

    /// <summary>
    /// Asynchronously copies the full pinned-host buffer into this device allocation.
    /// 异步将整个 pinned host 缓冲区复制到当前设备内存。
    /// </summary>
    /// <param name="source">The source pinned host buffer. 源 pinned host 缓冲区。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyFromAsync(CudaPinnedMemory source, CudaStream stream)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        CopyFromAsync(source, Math.Min(source.SizeInBytes, SizeInBytes), stream);
    }

    /// <summary>
    /// Copies device memory into a byte array.
    /// 将设备内存复制到字节数组中。
    /// </summary>
    /// <param name="destination">The destination byte array. 目标字节数组。</param>
    public void CopyTo(byte[] destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (destination.Length > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(destination));
        }

        NativeCudaApi.CopyToHost(_handle, destination, destination.Length);
    }

    /// <summary>
    /// Copies device memory into a float array.
    /// 将设备内存复制到浮点数组中。
    /// </summary>
    /// <param name="destination">The destination float array. 目标浮点数组。</param>
    public void CopyTo(float[] destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        int byteCount = checked(destination.Length * sizeof(float));
        if (byteCount > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(destination));
        }

        byte[] bytes = new byte[byteCount];
        CopyTo(bytes);
        Buffer.BlockCopy(bytes, 0, destination, 0, byteCount);
    }

    /// <summary>
    /// Asynchronously copies device memory into a pinned host buffer.
    /// 将设备内存异步复制到 pinned host 缓冲区中。
    /// </summary>
    /// <param name="destination">The destination pinned host buffer. 目标 pinned host 缓冲区。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyToAsync(CudaPinnedMemory destination, int count, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        ValidateCount(count, nameof(count));
        if (count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyToPinnedHostAsync(_handle, destination.Handle, count, stream.Handle);
    }

    /// <summary>
    /// Asynchronously copies this allocation into a full pinned-host buffer.
    /// 异步将当前设备内存复制到整个 pinned host 缓冲区。
    /// </summary>
    /// <param name="destination">The destination pinned host buffer. 目标 pinned host 缓冲区。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyToAsync(CudaPinnedMemory destination, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        CopyToAsync(destination, Math.Min(destination.SizeInBytes, SizeInBytes), stream);
    }

}
