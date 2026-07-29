using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaPitchedMemory
{
    /// <summary>
    /// Copies a logical 2D host buffer into this pitched allocation.
    /// 将逻辑 2D 主机缓冲区复制到当前 pitched 分配。
    /// </summary>
    /// <param name="source">The host source buffer. 主机源缓冲区。</param>
    /// <param name="sourcePitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    public void CopyFrom2D(byte[] source, int sourcePitch)
    {
        CopyFrom2D(source, sourcePitch, WidthInBytes, Height);
    }

    /// <summary>
    /// Copies a logical 2D host subregion into this pitched allocation.
    /// 将逻辑 2D 主机子区域复制到当前 pitched 分配。
    /// </summary>
    /// <param name="source">The host source buffer. 主机源缓冲区。</param>
    /// <param name="sourcePitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    /// <param name="widthInBytes">The copied row width in bytes. 要复制的每行字节数。</param>
    /// <param name="height">The number of copied rows. 要复制的行数。</param>
    public void CopyFrom2D(byte[] source, int sourcePitch, int widthInBytes, int height)
    {
        Validate2DRegion(sourcePitch, widthInBytes, height, nameof(sourcePitch));
        NativeCudaApi.CopyPitchedFromHost2D(_handle, source, sourcePitch, widthInBytes, height);
    }

    /// <summary>
    /// Copies this pitched allocation into a logical 2D host buffer.
    /// 将当前 pitched 分配复制到逻辑 2D 主机缓冲区。
    /// </summary>
    /// <param name="destination">The host destination buffer. 主机目标缓冲区。</param>
    /// <param name="destinationPitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    public void CopyTo2D(byte[] destination, int destinationPitch)
    {
        CopyTo2D(destination, destinationPitch, WidthInBytes, Height);
    }

    /// <summary>
    /// Copies a logical 2D device subregion into a host buffer.
    /// 将逻辑 2D 设备子区域复制到主机缓冲区。
    /// </summary>
    /// <param name="destination">The host destination buffer. 主机目标缓冲区。</param>
    /// <param name="destinationPitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    /// <param name="widthInBytes">The copied row width in bytes. 要复制的每行字节数。</param>
    /// <param name="height">The number of copied rows. 要复制的行数。</param>
    public void CopyTo2D(byte[] destination, int destinationPitch, int widthInBytes, int height)
    {
        Validate2DRegion(destinationPitch, widthInBytes, height, nameof(destinationPitch));
        NativeCudaApi.CopyPitchedToHost2D(_handle, destination, destinationPitch, widthInBytes, height);
    }

    /// <summary>
    /// Copies the overlapping 2D region to another pitched allocation.
    /// 将重叠的 2D 区域复制到另一个 pitched 分配。
    /// </summary>
    /// <param name="destination">The destination pitched allocation. 目标 pitched 分配。</param>
    public void CopyTo(CudaPitchedMemory destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        int width = Math.Min(WidthInBytes, destination.WidthInBytes);
        int height = Math.Min(Height, destination.Height);
        NativeCudaApi.CopyPitchedDeviceToDevice2D(destination._handle, _handle, width, height);
    }

    /// <summary>
    /// Asynchronously copies pinned 2D host memory into this pitched allocation.
    /// 异步将 pinned 2D 主机内存复制到当前 pitched 分配。
    /// </summary>
    /// <param name="source">The pinned host source buffer. pinned 主机源缓冲区。</param>
    /// <param name="sourcePitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyFrom2DAsync(CudaPinnedMemory source, int sourcePitch, CudaStream stream)
    {
        CopyFrom2DAsync(source, sourcePitch, WidthInBytes, Height, stream);
    }

    /// <summary>
    /// Asynchronously copies a pinned 2D host subregion into this pitched allocation.
    /// 异步将 pinned 2D 主机子区域复制到当前 pitched 分配。
    /// </summary>
    /// <param name="source">The pinned host source buffer. pinned 主机源缓冲区。</param>
    /// <param name="sourcePitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    /// <param name="widthInBytes">The copied row width in bytes. 要复制的每行字节数。</param>
    /// <param name="height">The number of copied rows. 要复制的行数。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyFrom2DAsync(CudaPinnedMemory source, int sourcePitch, int widthInBytes, int height, CudaStream stream)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        Validate2DRegion(sourcePitch, widthInBytes, height, nameof(sourcePitch));
        ValidatePinnedHostBuffer(source, sourcePitch, height, nameof(source));
        NativeCudaApi.CopyPitchedFromPinnedHost2DAsync(_handle, source.Handle, sourcePitch, widthInBytes, height, stream.Handle);
    }

    /// <summary>
    /// Asynchronously copies this pitched allocation into pinned 2D host memory.
    /// 异步将当前 pitched 分配复制到 pinned 2D 主机内存。
    /// </summary>
    /// <param name="destination">The pinned host destination buffer. pinned 主机目标缓冲区。</param>
    /// <param name="destinationPitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyTo2DAsync(CudaPinnedMemory destination, int destinationPitch, CudaStream stream)
    {
        CopyTo2DAsync(destination, destinationPitch, WidthInBytes, Height, stream);
    }

    /// <summary>
    /// Asynchronously copies a logical 2D device subregion into pinned host memory.
    /// 异步将逻辑 2D 设备子区域复制到 pinned 主机内存。
    /// </summary>
    /// <param name="destination">The pinned host destination buffer. pinned 主机目标缓冲区。</param>
    /// <param name="destinationPitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    /// <param name="widthInBytes">The copied row width in bytes. 要复制的每行字节数。</param>
    /// <param name="height">The number of copied rows. 要复制的行数。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyTo2DAsync(CudaPinnedMemory destination, int destinationPitch, int widthInBytes, int height, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        Validate2DRegion(destinationPitch, widthInBytes, height, nameof(destinationPitch));
        ValidatePinnedHostBuffer(destination, destinationPitch, height, nameof(destination));
        NativeCudaApi.CopyPitchedToPinnedHost2DAsync(_handle, destination.Handle, destinationPitch, widthInBytes, height, stream.Handle);
    }

    /// <summary>
    /// Asynchronously copies the overlapping 2D region to another pitched allocation.
    /// 异步将重叠的 2D 区域复制到另一个 pitched 分配。
    /// </summary>
    /// <param name="destination">The destination pitched allocation. 目标 pitched 分配。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyToAsync(CudaPitchedMemory destination, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        int width = Math.Min(WidthInBytes, destination.WidthInBytes);
        int height = Math.Min(Height, destination.Height);
        NativeCudaApi.CopyPitchedDeviceToDevice2DAsync(destination._handle, _handle, width, height, stream.Handle);
    }

}
