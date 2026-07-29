using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Managed wrapper around a CUDA pitched device allocation for 2D tensor/image-like data.
/// 面向二维张量或图像类数据的 CUDA pitched 设备内存托管封装。
/// </summary>
public sealed partial class CudaPitchedMemory : IDisposable
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
