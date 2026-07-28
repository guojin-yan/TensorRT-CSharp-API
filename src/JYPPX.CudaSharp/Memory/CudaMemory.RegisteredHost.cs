using System;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public partial class CudaMemory
{
    /// <summary>
    /// Asynchronously copies registered host memory into device memory.
    /// 将 registered host memory 异步复制到设备内存中。
    /// </summary>
    /// <param name="source">The source registered host buffer. 源 registered host 缓冲区。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyFromAsync(CudaRegisteredHostMemory source, int count, CudaStream stream)
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
    /// Asynchronously copies device memory into registered host memory.
    /// 将设备内存异步复制到 registered host memory 中。
    /// </summary>
    /// <param name="destination">The destination registered host buffer. 目标 registered host 缓冲区。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyToAsync(CudaRegisteredHostMemory destination, int count, CudaStream stream)
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
}
