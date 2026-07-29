using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Wraps a CUDA array allocation and its copy helpers. 封装 CUDA array 分配对象及其复制辅助方法。
/// </summary>
public sealed partial class CudaArray : IDisposable
{
    private readonly SafeCudaArrayHandle _handle;

    /// <summary>
    /// Allocates a 1D or 2D CUDA array. 分配一个一维或二维 CUDA array。
    /// </summary>
    public CudaArray(CudaChannelFormatDescriptor descriptor, ulong width, ulong height = 0, CudaArrayCreationFlags flags = CudaArrayCreationFlags.Default)
    {
        if (width == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width));
        }

        NativeBridgeLoader.EnsureInitialized();
        _handle = NativeCudaApi.AllocateArray(descriptor, width, height, flags);
        Descriptor = descriptor;
        Extent = new CudaArrayExtent(width, height, 0);
        Flags = flags;
    }

    private CudaArray(SafeCudaArrayHandle handle, CudaChannelFormatDescriptor descriptor, CudaArrayExtent extent, CudaArrayCreationFlags flags)
    {
        _handle = handle;
        Descriptor = descriptor;
        Extent = extent;
        Flags = flags;
    }

    internal SafeCudaArrayHandle Handle => _handle;

    /// <summary>
    /// Gets the logical channel descriptor used for this array. 获取该 array 使用的逻辑通道描述符。
    /// </summary>
    public CudaChannelFormatDescriptor Descriptor { get; }
    /// <summary>
    /// Gets the array extent. 获取 array 的范围信息。
    /// </summary>
    public CudaArrayExtent Extent { get; }
    /// <summary>
    /// Gets the creation flags supplied when the array was created. 获取创建该 array 时使用的标志。
    /// </summary>
    public CudaArrayCreationFlags Flags { get; }

    /// <summary>
    /// Gets native metadata reported by CUDA for this array. 获取 CUDA 为该 array 报告的原生元数据。
    /// </summary>
    public CudaArrayInfo Info => CudaArrayInfo.FromNative(NativeCudaApi.GetArrayInfo(_handle));

    /// <summary>
    /// Gets the channel descriptor copied from CUDA. 获取从 CUDA 读取的通道描述符。
    /// </summary>
    public CudaChannelFormatDescriptor ChannelDescriptor => CudaChannelFormatDescriptor.FromNative(NativeCudaApi.GetArrayChannelDescriptor(_handle));

    /// <summary>
    /// Allocates a 3D CUDA array. 分配一个三维 CUDA array。
    /// </summary>
    public static CudaArray Create3D(CudaChannelFormatDescriptor descriptor, CudaArrayExtent extent, CudaArrayCreationFlags flags = CudaArrayCreationFlags.Default)
    {
        if (extent.Width == 0 || extent.Height == 0 || extent.Depth == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(extent));
        }

        NativeBridgeLoader.EnsureInitialized();
        SafeCudaArrayHandle handle = NativeCudaApi.Allocate3DArray(descriptor, extent, flags);
        return new CudaArray(handle, descriptor, extent, flags);
    }

    /// <summary>
    /// Releases the native CUDA array handle. 释放原生 CUDA array 句柄。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }

    private static void ValidateStream(CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }
    }

    private static void ValidateByteCount(int byteCount)
    {
        if (byteCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(byteCount));
        }
    }

    private static void Validate2DExtent(int widthBytes, int height)
    {
        if (widthBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(widthBytes));
        }

        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }

        _ = checked(widthBytes * height);
    }

    private static void Validate2DRegion(int pitch, int widthBytes, int height, string pitchParameterName)
    {
        Validate2DExtent(widthBytes, height);

        if (pitch < widthBytes)
        {
            throw new ArgumentOutOfRangeException(pitchParameterName);
        }

        _ = checked(pitch * height);
    }

    private static void Validate3DExtent(int widthBytes, int height, int depth)
    {
        Validate2DExtent(widthBytes, height);

        if (depth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(depth));
        }

        _ = checked(widthBytes * height * depth);
    }

    private static void Validate3DRegion(int pitch, int pitchedWidthBytes, int pitchedHeight, int widthBytes, int height, int depth, string pitchParameterName)
    {
        Validate3DExtent(widthBytes, height, depth);

        if (pitch < widthBytes)
        {
            throw new ArgumentOutOfRangeException(pitchParameterName);
        }

        if (pitchedWidthBytes < widthBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(pitchedWidthBytes));
        }

        if (pitchedHeight < height)
        {
            throw new ArgumentOutOfRangeException(nameof(pitchedHeight));
        }

        _ = checked(pitch * pitchedHeight * depth);
    }

    private static void ValidatePinnedBuffer(CudaPinnedMemory memory, int byteCount, string parameterName)
    {
        if (memory == null)
        {
            throw new ArgumentNullException(parameterName);
        }

        ValidateByteCount(byteCount);
        if (memory.SizeInBytes < byteCount)
        {
            throw new ArgumentOutOfRangeException(parameterName);
        }
    }

    private static void ValidatePinned2D(CudaPinnedMemory memory, int pitch, int widthBytes, int height, string parameterName)
    {
        if (memory == null)
        {
            throw new ArgumentNullException(parameterName);
        }

        Validate2DRegion(pitch, widthBytes, height, nameof(pitch));
        int requiredBytes = checked(pitch * height);
        if (memory.SizeInBytes < requiredBytes)
        {
            throw new ArgumentOutOfRangeException(parameterName);
        }
    }

    private static void ValidatePinned3D(CudaPinnedMemory memory, int pitch, int pitchedWidthBytes, int pitchedHeight, int widthBytes, int height, int depth, string parameterName)
    {
        if (memory == null)
        {
            throw new ArgumentNullException(parameterName);
        }

        Validate3DRegion(pitch, pitchedWidthBytes, pitchedHeight, widthBytes, height, depth, nameof(pitch));
        int requiredBytes = checked(pitch * pitchedHeight * depth);
        if (memory.SizeInBytes < requiredBytes)
        {
            throw new ArgumentOutOfRangeException(parameterName);
        }
    }
}
