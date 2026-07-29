using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public partial class CudaMemory
{
    /// <summary>
    /// Copies bytes from this allocation to another device allocation.
    /// 将当前分配中的字节复制到另一个设备分配中。
    /// </summary>
    /// <param name="destination">The destination device allocation. 目标设备内存分配。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    public void CopyTo(CudaMemory destination, int count)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        ValidateCount(count, nameof(count));
        if (count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyDeviceToDevice(destination._handle, _handle, count);
    }

    /// <summary>
    /// Copies bytes from this allocation to another allocation using the smaller allocation size.
    /// 按两个分配中较小的大小，将当前分配复制到另一个分配。
    /// </summary>
    /// <param name="destination">The destination device allocation. 目标设备内存分配。</param>
    public void CopyTo(CudaMemory destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        CopyTo(destination, Math.Min(SizeInBytes, destination.SizeInBytes));
    }

    /// <summary>
    /// Asynchronously copies bytes to another device allocation.
    /// 异步将字节复制到另一个设备分配中。
    /// </summary>
    /// <param name="destination">The destination device allocation. 目标设备内存分配。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyToAsync(CudaMemory destination, int count, CudaStream stream)
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

        NativeCudaApi.CopyDeviceToDeviceAsync(destination._handle, _handle, count, stream.Handle);
    }

    /// <summary>
    /// Asynchronously copies bytes to another allocation using the smaller allocation size.
    /// 按两个分配中较小的大小，异步将当前分配复制到另一个分配。
    /// </summary>
    /// <param name="destination">The destination device allocation. 目标设备内存分配。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyToAsync(CudaMemory destination, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        CopyToAsync(destination, Math.Min(SizeInBytes, destination.SizeInBytes), stream);
    }

    /// <summary>
    /// Copies bytes to another CUDA allocation and lets CUDA infer the copy direction.
    /// 将字节复制到另一个 CUDA 分配，并让 CUDA 自动推断复制方向。
    /// </summary>
    /// <param name="destination">The destination allocation. 目标设备内存分配。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    public void CopyToAuto(CudaMemory destination, int count)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        ValidateCount(count, nameof(count));
        if (count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyDefault(destination._handle, _handle, count);
    }

    /// <summary>
    /// Asynchronously copies bytes to another CUDA allocation and lets CUDA infer the copy direction.
    /// 异步将字节复制到另一个 CUDA 分配，并让 CUDA 自动推断复制方向。
    /// </summary>
    /// <param name="destination">The destination allocation. 目标设备内存分配。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyToAutoAsync(CudaMemory destination, int count, CudaStream stream)
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

        NativeCudaApi.CopyDefaultAsync(destination._handle, _handle, count, stream.Handle);
    }

    /// <summary>
    /// Copies bytes from this allocation to a destination allocation on another CUDA device.
    /// 将当前分配中的字节复制到另一个 CUDA 设备上的目标分配。
    /// </summary>
    /// <param name="destination">The destination allocation. 目标设备内存分配。</param>
    /// <param name="sourceDevice">The CUDA device ordinal that owns this source allocation. 拥有当前源分配的 CUDA 设备序号。</param>
    /// <param name="destinationDevice">The CUDA device ordinal that owns the destination allocation. 拥有目标分配的 CUDA 设备序号。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    public void CopyToPeer(CudaMemory destination, int sourceDevice, int destinationDevice, int count)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        ValidateCount(count, nameof(count));
        if (count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyPeer(destination._handle, destinationDevice, _handle, sourceDevice, count);
    }

    /// <summary>
    /// Asynchronously copies bytes from this allocation to another CUDA device using a stream.
    /// 使用 stream 将当前分配中的字节异步复制到另一个 CUDA 设备。
    /// </summary>
    /// <param name="destination">The destination allocation. 目标设备内存分配。</param>
    /// <param name="sourceDevice">The CUDA device ordinal that owns this source allocation. 拥有当前源分配的 CUDA 设备序号。</param>
    /// <param name="destinationDevice">The CUDA device ordinal that owns the destination allocation. 拥有目标分配的 CUDA 设备序号。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序该复制操作的 CUDA stream。</param>
    public void CopyToPeerAsync(CudaMemory destination, int sourceDevice, int destinationDevice, int count, CudaStream stream)
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

        NativeCudaApi.CopyPeerAsync(destination._handle, destinationDevice, _handle, sourceDevice, count, stream.Handle);
    }

}
