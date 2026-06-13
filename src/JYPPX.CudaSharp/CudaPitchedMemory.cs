using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Managed wrapper around a CUDA pitched device allocation for 2D tensor/image-like data.
/// 面向二维张量或图像类数据的 CUDA pitched 设备内存托管封装。
/// </summary>
public sealed class CudaPitchedMemory : IDisposable
{
    private readonly SafeCudaPitchedMemoryHandle _handle;

    /// <summary>
    /// Allocates CUDA pitched device memory for a 2D logical region.
    /// 为二维逻辑区域分配 CUDA pitched 设备内存。
    /// </summary>
    /// <param name="widthInBytes">The logical row width in bytes. 逻辑每行宽度，单位为字节。</param>
    /// <param name="height">The logical height in rows. 逻辑高度，单位为行。</param>
    public CudaPitchedMemory(int widthInBytes, int height)
    {
        if (widthInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(widthInBytes));
        }

        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }

        NativeBridgeLoader.EnsureInitialized();
        _handle = NativeCudaApi.AllocatePitchedMemory(widthInBytes, height);
        NativeCudaPitchedMemoryInfo info = NativeCudaApi.GetPitchedMemoryInfo(_handle);
        WidthInBytes = checked((int)info.WidthBytes);
        Height = checked((int)info.Height);
        PitchInBytes = checked((int)info.PitchBytes);
    }

    internal SafeCudaPitchedMemoryHandle Handle => _handle;

    /// <summary>
    /// Gets the logical width in bytes.
    /// 获取逻辑宽度，单位为字节。
    /// </summary>
    public int WidthInBytes { get; }
    /// <summary>
    /// Gets the logical height in rows.
    /// 获取逻辑高度，单位为行。
    /// </summary>
    public int Height { get; }
    /// <summary>
    /// Gets the allocated pitch in bytes.
    /// 获取实际分配的 pitch，单位为字节。
    /// </summary>
    public int PitchInBytes { get; }

    /// <summary>
    /// Allocates pitched CUDA memory for a logical 3D region.
    /// 为逻辑 3D 区域分配 CUDA pitched memory。
    /// </summary>
    /// <param name="widthInBytes">The width of each row in bytes. 每行宽度，单位为字节。</param>
    /// <param name="height">The height of each slice. 每个 slice 的高度。</param>
    /// <param name="depth">The number of slices. slice 数量。</param>
    /// <returns>A pitched allocation large enough for <paramref name="height"/> * <paramref name="depth"/> rows. 足以容纳 height * depth 行的 pitched allocation。</returns>
    public static CudaPitchedMemory Allocate3D(int widthInBytes, int height, int depth)
    {
        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }

        if (depth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(depth));
        }

        return new CudaPitchedMemory(widthInBytes, checked(height * depth));
    }

    /// <summary>
    /// Fills the full logical 2D allocation with a byte value.
    /// 使用指定字节值填充完整的逻辑二维分配区域。
    /// </summary>
    /// <param name="value">The byte value used by CUDA memset. CUDA memset 使用的字节值。</param>
    public void Fill2D(byte value)
    {
        Fill2D(value, WidthInBytes, Height);
    }

    /// <summary>
    /// Fills a logical 2D subregion with a byte value.
    /// 使用指定字节值填充逻辑二维子区域。
    /// </summary>
    /// <param name="value">The byte value used by CUDA memset. CUDA memset 使用的字节值。</param>
    /// <param name="widthInBytes">The row width to fill in bytes. 要填充的每行字节数。</param>
    /// <param name="height">The number of rows to fill. 要填充的行数。</param>
    public void Fill2D(byte value, int widthInBytes, int height)
    {
        Validate2DExtent(widthInBytes, height);
        NativeCudaApi.FillPitched2D(_handle, value, widthInBytes, height);
    }

    /// <summary>
    /// Asynchronously fills the full logical 2D allocation with a byte value.
    /// 使用指定字节值异步填充完整的逻辑二维分配区域。
    /// </summary>
    /// <param name="value">The byte value used by CUDA memset. CUDA memset 使用的字节值。</param>
    /// <param name="stream">The CUDA stream that orders the fill. 用于排序填充操作的 CUDA stream。</param>
    public void Fill2DAsync(byte value, CudaStream stream)
    {
        Fill2DAsync(value, WidthInBytes, Height, stream);
    }

    /// <summary>
    /// Asynchronously fills a logical 2D subregion with a byte value.
    /// 使用指定字节值异步填充逻辑二维子区域。
    /// </summary>
    /// <param name="value">The byte value used by CUDA memset. CUDA memset 使用的字节值。</param>
    /// <param name="widthInBytes">The row width to fill in bytes. 要填充的每行字节数。</param>
    /// <param name="height">The number of rows to fill. 要填充的行数。</param>
    /// <param name="stream">The CUDA stream that orders the fill. 用于排序填充操作的 CUDA stream。</param>
    public void Fill2DAsync(byte value, int widthInBytes, int height, CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        Validate2DExtent(widthInBytes, height);
        NativeCudaApi.FillPitched2DAsync(_handle, value, widthInBytes, height, stream.Handle);
    }

    /// <summary>
    /// Fills a logical 3D region with a byte value.
    /// 使用指定字节值填充逻辑 3D 区域。
    /// </summary>
    /// <param name="value">The byte value used by CUDA memset. CUDA memset 使用的字节值。</param>
    /// <param name="height">The height of one logical slice. 单个逻辑 slice 的高度。</param>
    /// <param name="depth">The number of logical slices. 逻辑 slice 数量。</param>
    public void Fill3D(byte value, int height, int depth)
    {
        Fill3D(value, WidthInBytes, height, depth);
    }

    /// <summary>
    /// Fills a logical 3D subregion with a byte value.
    /// 使用指定字节值填充逻辑 3D 子区域。
    /// </summary>
    /// <param name="value">The byte value used by CUDA memset. CUDA memset 使用的字节值。</param>
    /// <param name="widthInBytes">The row width to fill in bytes. 要填充的每行字节数。</param>
    /// <param name="height">The height of one logical slice. 单个逻辑 slice 的高度。</param>
    /// <param name="depth">The number of logical slices. 逻辑 slice 数量。</param>
    public void Fill3D(byte value, int widthInBytes, int height, int depth)
    {
        Validate3DExtent(widthInBytes, height, depth);
        NativeCudaApi.FillPitched3D(_handle, value, widthInBytes, height, depth);
    }

    /// <summary>
    /// Asynchronously fills a logical 3D region with a byte value.
    /// 异步使用指定字节值填充逻辑 3D 区域。
    /// </summary>
    /// <param name="value">The byte value used by CUDA memset. CUDA memset 使用的字节值。</param>
    /// <param name="height">The height of one logical slice. 单个逻辑 slice 的高度。</param>
    /// <param name="depth">The number of logical slices. 逻辑 slice 数量。</param>
    /// <param name="stream">The CUDA stream that orders the fill. 用于排序填充操作的 CUDA stream。</param>
    public void Fill3DAsync(byte value, int height, int depth, CudaStream stream)
    {
        Fill3DAsync(value, WidthInBytes, height, depth, stream);
    }

    /// <summary>
    /// Asynchronously fills a logical 3D subregion with a byte value.
    /// 异步使用指定字节值填充逻辑 3D 子区域。
    /// </summary>
    /// <param name="value">The byte value used by CUDA memset. CUDA memset 使用的字节值。</param>
    /// <param name="widthInBytes">The row width to fill in bytes. 要填充的每行字节数。</param>
    /// <param name="height">The height of one logical slice. 单个逻辑 slice 的高度。</param>
    /// <param name="depth">The number of logical slices. 逻辑 slice 数量。</param>
    /// <param name="stream">The CUDA stream that orders the fill. 用于排序填充操作的 CUDA stream。</param>
    public void Fill3DAsync(byte value, int widthInBytes, int height, int depth, CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        Validate3DExtent(widthInBytes, height, depth);
        NativeCudaApi.FillPitched3DAsync(_handle, value, widthInBytes, height, depth, stream.Handle);
    }

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

    /// <summary>
    /// Copies this pitched allocation to a newly allocated logical 2D host buffer.
    /// 将当前 pitched 分配复制到新建的逻辑 2D 主机缓冲区。
    /// </summary>
    /// <param name="destinationPitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    /// <returns>The copied host buffer. 复制得到的主机缓冲区。</returns>
    public byte[] ToArray2D(int destinationPitch)
    {
        ValidateHostPitch(destinationPitch, WidthInBytes, nameof(destinationPitch));
        byte[] data = new byte[checked(destinationPitch * Height)];
        CopyTo2D(data, destinationPitch);
        return data;
    }

    /// <summary>
    /// Copies this pitched allocation to a newly allocated logical 3D host buffer.
    /// 将当前 pitched 分配复制到新建的逻辑 3D 主机缓冲区。
    /// </summary>
    /// <param name="destinationPitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    /// <param name="height">The height of one slice. 单个 slice 的高度。</param>
    /// <param name="depth">The number of slices. slice 数量。</param>
    /// <returns>The copied host buffer. 复制得到的主机缓冲区。</returns>
    public byte[] ToArray3D(int destinationPitch, int height, int depth)
    {
        Validate3DRegion(destinationPitch, WidthInBytes, height, depth, nameof(destinationPitch));
        Validate3DExtent(WidthInBytes, height, depth);
        byte[] data = new byte[checked(destinationPitch * height * depth)];
        CopyTo3D(data, destinationPitch, WidthInBytes, height, depth);
        return data;
    }

    /// <summary>
    /// Releases the pitched allocation.
    /// 释放 pitched 分配。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }

    private void Validate2DRegion(int hostPitch, int widthInBytes, int height, string pitchParameterName)
    {
        Validate2DExtent(widthInBytes, height);
        ValidateHostPitch(hostPitch, widthInBytes, pitchParameterName);
    }

    private void Validate2DExtent(int widthInBytes, int height)
    {
        if (widthInBytes <= 0 || widthInBytes > WidthInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(widthInBytes));
        }

        if (height <= 0 || height > Height)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }
    }

    private void Validate3DExtent(int widthInBytes, int height, int depth)
    {
        if (widthInBytes <= 0 || widthInBytes > WidthInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(widthInBytes));
        }

        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }

        if (depth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(depth));
        }

        int flattenedHeight = checked(height * depth);
        if (flattenedHeight > Height)
        {
            throw new ArgumentOutOfRangeException(nameof(depth), "Logical 3D height * depth exceeds the allocated pitched memory height.");
        }
    }

    private static void Validate3DRegion(int hostPitch, int widthInBytes, int height, int depth, string pitchParameterName)
    {
        if (widthInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(widthInBytes));
        }

        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }

        if (depth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(depth));
        }

        ValidateHostPitch(hostPitch, widthInBytes, pitchParameterName);
        _ = checked(hostPitch * height * depth);
    }

    private static void ValidateHostPitch(int pitch, int widthInBytes, string parameterName)
    {
        if (pitch < widthInBytes)
        {
            throw new ArgumentOutOfRangeException(parameterName);
        }
    }

    private static void ValidatePinnedHostBuffer(CudaPinnedMemory memory, int pitch, int height, string parameterName)
    {
        int requiredBytes = checked(pitch * height);
        if (memory.SizeInBytes < requiredBytes)
        {
            throw new ArgumentOutOfRangeException(parameterName, "Pinned host buffer is too small for the requested 2D copy region.");
        }
    }

    private static void ValidatePinnedHostBuffer(CudaPinnedMemory memory, int pitch, int height, int depth, string parameterName)
    {
        int requiredBytes = checked(pitch * height * depth);
        if (memory.SizeInBytes < requiredBytes)
        {
            throw new ArgumentOutOfRangeException(parameterName, "Pinned host buffer is too small for the requested 3D copy region.");
        }
    }
}
