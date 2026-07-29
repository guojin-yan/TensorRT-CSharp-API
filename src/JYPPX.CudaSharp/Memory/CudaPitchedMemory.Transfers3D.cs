using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaPitchedMemory
{
    /// <summary>
    /// Copies a logical 3D host buffer into this pitched device allocation.
    /// 将逻辑 3D 主机缓冲区复制到当前 pitched 设备分配。
    /// </summary>
    /// <param name="source">The host source buffer. 主机源缓冲区。</param>
    /// <param name="sourcePitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    /// <param name="height">The height of one slice. 单个 slice 的高度。</param>
    /// <param name="depth">The number of slices. slice 数量。</param>
    public void CopyFrom3D(byte[] source, int sourcePitch, int height, int depth)
    {
        CopyFrom3D(source, sourcePitch, WidthInBytes, height, depth);
    }

    /// <summary>
    /// Copies a logical 3D host subregion into this pitched device allocation.
    /// 将逻辑 3D 主机子区域复制到当前 pitched 设备分配。
    /// </summary>
    /// <param name="source">The host source buffer. 主机源缓冲区。</param>
    /// <param name="sourcePitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    /// <param name="widthInBytes">The copied row width in bytes. 要复制的每行字节数。</param>
    /// <param name="height">The height of one slice. 单个 slice 的高度。</param>
    /// <param name="depth">The number of slices. slice 数量。</param>
    public void CopyFrom3D(byte[] source, int sourcePitch, int widthInBytes, int height, int depth)
    {
        Validate3DRegion(sourcePitch, widthInBytes, height, depth, nameof(sourcePitch));
        Validate3DExtent(widthInBytes, height, depth);
        NativeCudaApi.CopyPitchedFromHost3D(_handle, source, sourcePitch, widthInBytes, height, depth);
    }

    /// <summary>
    /// Copies this pitched device allocation into a logical 3D host buffer.
    /// 将当前 pitched 设备分配复制到逻辑 3D 主机缓冲区。
    /// </summary>
    /// <param name="destination">The host destination buffer. 主机目标缓冲区。</param>
    /// <param name="destinationPitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    /// <param name="height">The height of one slice. 单个 slice 的高度。</param>
    /// <param name="depth">The number of slices. slice 数量。</param>
    public void CopyTo3D(byte[] destination, int destinationPitch, int height, int depth)
    {
        CopyTo3D(destination, destinationPitch, WidthInBytes, height, depth);
    }

    /// <summary>
    /// Copies a logical 3D device subregion into a host buffer.
    /// 将逻辑 3D 设备子区域复制到主机缓冲区。
    /// </summary>
    /// <param name="destination">The host destination buffer. 主机目标缓冲区。</param>
    /// <param name="destinationPitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    /// <param name="widthInBytes">The copied row width in bytes. 要复制的每行字节数。</param>
    /// <param name="height">The height of one slice. 单个 slice 的高度。</param>
    /// <param name="depth">The number of slices. slice 数量。</param>
    public void CopyTo3D(byte[] destination, int destinationPitch, int widthInBytes, int height, int depth)
    {
        Validate3DRegion(destinationPitch, widthInBytes, height, depth, nameof(destinationPitch));
        Validate3DExtent(widthInBytes, height, depth);
        NativeCudaApi.CopyPitchedToHost3D(_handle, destination, destinationPitch, widthInBytes, height, depth);
    }

    /// <summary>
    /// Copies a logical 3D region from this pitched allocation to another pitched allocation.
    /// 将当前 pitched 分配中的逻辑 3D 区域复制到另一个 pitched 分配。
    /// </summary>
    /// <param name="destination">The destination pitched allocation. 目标 pitched 分配。</param>
    /// <param name="widthInBytes">The copied row width in bytes. 要复制的每行字节数。</param>
    /// <param name="height">The height of one slice. 单个 slice 的高度。</param>
    /// <param name="depth">The number of slices. slice 数量。</param>
    public void CopyTo3D(CudaPitchedMemory destination, int widthInBytes, int height, int depth)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        Validate3DExtent(widthInBytes, height, depth);
        destination.Validate3DExtent(widthInBytes, height, depth);
        NativeCudaApi.CopyPitchedDeviceToDevice3D(destination._handle, _handle, widthInBytes, height, depth);
    }

    /// <summary>
    /// Asynchronously copies pinned logical 3D host memory into this pitched device allocation.
    /// 异步将 pinned 逻辑 3D 主机内存复制到当前 pitched 设备分配。
    /// </summary>
    /// <param name="source">The pinned host source memory. pinned 主机源内存。</param>
    /// <param name="sourcePitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    /// <param name="height">The height of one slice. 单个 slice 的高度。</param>
    /// <param name="depth">The number of slices. slice 数量。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyFrom3DAsync(CudaPinnedMemory source, int sourcePitch, int height, int depth, CudaStream stream)
    {
        CopyFrom3DAsync(source, sourcePitch, WidthInBytes, height, depth, stream);
    }

    /// <summary>
    /// Asynchronously copies pinned logical 3D host memory into this pitched device allocation.
    /// 异步将 pinned 逻辑 3D 主机内存复制到当前 pitched 设备分配。
    /// </summary>
    /// <param name="source">The pinned host source memory. pinned 主机源内存。</param>
    /// <param name="sourcePitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    /// <param name="widthInBytes">The copied row width in bytes. 要复制的每行字节数。</param>
    /// <param name="height">The height of one slice. 单个 slice 的高度。</param>
    /// <param name="depth">The number of slices. slice 数量。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyFrom3DAsync(CudaPinnedMemory source, int sourcePitch, int widthInBytes, int height, int depth, CudaStream stream)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        Validate3DRegion(sourcePitch, widthInBytes, height, depth, nameof(sourcePitch));
        Validate3DExtent(widthInBytes, height, depth);
        ValidatePinnedHostBuffer(source, sourcePitch, height, depth, nameof(source));
        NativeCudaApi.CopyPitchedFromPinnedHost3DAsync(_handle, source.Handle, sourcePitch, widthInBytes, height, depth, stream.Handle);
    }

    /// <summary>
    /// Asynchronously copies this pitched device allocation into pinned logical 3D host memory.
    /// 异步将当前 pitched 设备分配复制到 pinned 逻辑 3D 主机内存。
    /// </summary>
    /// <param name="destination">The pinned host destination memory. pinned 主机目标内存。</param>
    /// <param name="destinationPitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    /// <param name="height">The height of one slice. 单个 slice 的高度。</param>
    /// <param name="depth">The number of slices. slice 数量。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyTo3DAsync(CudaPinnedMemory destination, int destinationPitch, int height, int depth, CudaStream stream)
    {
        CopyTo3DAsync(destination, destinationPitch, WidthInBytes, height, depth, stream);
    }

    /// <summary>
    /// Asynchronously copies a logical 3D device subregion into pinned host memory.
    /// 异步将逻辑 3D 设备子区域复制到 pinned 主机内存。
    /// </summary>
    /// <param name="destination">The pinned host destination memory. pinned 主机目标内存。</param>
    /// <param name="destinationPitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    /// <param name="widthInBytes">The copied row width in bytes. 要复制的每行字节数。</param>
    /// <param name="height">The height of one slice. 单个 slice 的高度。</param>
    /// <param name="depth">The number of slices. slice 数量。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyTo3DAsync(CudaPinnedMemory destination, int destinationPitch, int widthInBytes, int height, int depth, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        Validate3DRegion(destinationPitch, widthInBytes, height, depth, nameof(destinationPitch));
        Validate3DExtent(widthInBytes, height, depth);
        ValidatePinnedHostBuffer(destination, destinationPitch, height, depth, nameof(destination));
        NativeCudaApi.CopyPitchedToPinnedHost3DAsync(_handle, destination.Handle, destinationPitch, widthInBytes, height, depth, stream.Handle);
    }

    /// <summary>
    /// Asynchronously copies a logical 3D region to another pitched allocation.
    /// 异步将逻辑 3D 区域复制到另一个 pitched 分配。
    /// </summary>
    /// <param name="destination">The destination pitched allocation. 目标 pitched 分配。</param>
    /// <param name="widthInBytes">The copied row width in bytes. 要复制的每行字节数。</param>
    /// <param name="height">The height of one slice. 单个 slice 的高度。</param>
    /// <param name="depth">The number of slices. slice 数量。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyTo3DAsync(CudaPitchedMemory destination, int widthInBytes, int height, int depth, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        Validate3DExtent(widthInBytes, height, depth);
        destination.Validate3DExtent(widthInBytes, height, depth);
        NativeCudaApi.CopyPitchedDeviceToDevice3DAsync(destination._handle, _handle, widthInBytes, height, depth, stream.Handle);
    }

}
