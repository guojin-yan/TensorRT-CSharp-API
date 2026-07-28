using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Wraps a CUDA array allocation and its copy helpers. 封装 CUDA array 分配对象及其复制辅助方法。
/// </summary>
public sealed class CudaArray : IDisposable
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
    /// Queries the device-memory requirements for this array on the selected device. 查询该 array 在指定设备上的显存需求。
    /// </summary>
    public CudaArrayMemoryRequirements GetMemoryRequirements(int device)
    {
        return CudaArrayMemoryRequirements.FromNative(NativeCudaApi.GetArrayMemoryRequirements(_handle, device));
    }

    /// <summary>
    /// Attempts to query device-memory requirements without surfacing a CUDA exception. 尝试查询显存需求而不直接抛出 CUDA 异常。
    /// </summary>
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
    /// Queries sparse-array metadata for this array. 查询该 array 的 sparse 元数据。
    /// </summary>
    public CudaArraySparseProperties GetSparseProperties()
    {
        return CudaArraySparseProperties.FromNative(NativeCudaApi.GetArraySparseProperties(_handle));
    }

    /// <summary>
    /// Attempts to query sparse-array metadata without surfacing a CUDA exception. 尝试查询 sparse 元数据而不直接抛出 CUDA 异常。
    /// </summary>
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
    /// Copies the full source byte array into this CUDA array. 将整个源字节数组复制到当前 CUDA array。
    /// </summary>
    public void CopyFrom(byte[] source)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        CopyFrom(source, source.Length);
    }

    /// <summary>
    /// Copies a fixed number of bytes into this CUDA array. 将固定字节数复制到当前 CUDA array。
    /// </summary>
    public void CopyFrom(byte[] source, int byteCount)
    {
        CopyFrom(source, 0, 0, byteCount);
    }

    /// <summary>
    /// Copies bytes into this CUDA array at the specified 2D offset. 在指定二维偏移处把字节复制到当前 CUDA array。
    /// </summary>
    public void CopyFrom(byte[] source, int destinationXBytes, int destinationY, int byteCount)
    {
        ValidateByteCount(byteCount);
        NativeCudaApi.CopyToArray(_handle, destinationXBytes, destinationY, source, byteCount);
    }

    /// <summary>
    /// Copies the full contents of this CUDA array into a managed byte array. 将当前 CUDA array 的全部内容复制到托管字节数组。
    /// </summary>
    public void CopyTo(byte[] destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        CopyTo(destination, destination.Length);
    }

    /// <summary>
    /// Copies a fixed number of bytes from this CUDA array into a managed byte array. 将固定字节数从当前 CUDA array 复制到托管字节数组。
    /// </summary>
    public void CopyTo(byte[] destination, int byteCount)
    {
        CopyTo(destination, 0, 0, byteCount);
    }

    /// <summary>
    /// Copies bytes from this CUDA array at the specified 2D offset into a managed byte array. 从指定二维偏移把字节复制到托管字节数组。
    /// </summary>
    public void CopyTo(byte[] destination, int sourceXBytes, int sourceY, int byteCount)
    {
        ValidateByteCount(byteCount);
        NativeCudaApi.CopyFromArray(destination, _handle, sourceXBytes, sourceY, byteCount);
    }

    /// <summary>
    /// Copies bytes from this CUDA array into another CUDA array. 将当前 CUDA array 的字节复制到另一个 CUDA array。
    /// </summary>
    public void CopyTo(CudaArray destination, int byteCount)
    {
        CopyTo(destination, 0, 0, 0, 0, byteCount);
    }

    /// <summary>
    /// Copies bytes between two CUDA arrays using explicit source and destination offsets. 使用显式源/目标偏移在两个 CUDA array 之间复制字节。
    /// </summary>
    public void CopyTo(CudaArray destination, int destinationXBytes, int destinationY, int sourceXBytes, int sourceY, int byteCount)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        ValidateByteCount(byteCount);
        NativeCudaApi.CopyArrayToArray(destination._handle, destinationXBytes, destinationY, _handle, sourceXBytes, sourceY, byteCount);
    }

    /// <summary>
    /// Copies bytes from pinned host memory into this CUDA array asynchronously. 以异步方式把 pinned host memory 复制到当前 CUDA array。
    /// </summary>
    public void CopyFromAsync(CudaPinnedMemory source, int byteCount, CudaStream stream)
    {
        CopyFromAsync(source, 0, 0, byteCount, stream);
    }

    /// <summary>
    /// Copies bytes from pinned host memory into this CUDA array asynchronously at the specified 2D offset. 在指定二维偏移处异步复制 pinned host memory。
    /// </summary>
    public void CopyFromAsync(CudaPinnedMemory source, int destinationXBytes, int destinationY, int byteCount, CudaStream stream)
    {
        ValidatePinnedBuffer(source, byteCount, nameof(source));
        ValidateStream(stream);
        NativeCudaApi.CopyToArrayAsync(_handle, destinationXBytes, destinationY, source.Handle, byteCount, stream.Handle);
    }

    /// <summary>
    /// Copies bytes from this CUDA array into pinned host memory asynchronously. 以异步方式把当前 CUDA array 复制到 pinned host memory。
    /// </summary>
    public void CopyToAsync(CudaPinnedMemory destination, int byteCount, CudaStream stream)
    {
        CopyToAsync(destination, 0, 0, byteCount, stream);
    }

    /// <summary>
    /// Copies bytes from this CUDA array into pinned host memory asynchronously using an explicit 2D source offset. 使用显式二维源偏移异步复制到 pinned host memory。
    /// </summary>
    public void CopyToAsync(CudaPinnedMemory destination, int sourceXBytes, int sourceY, int byteCount, CudaStream stream)
    {
        ValidatePinnedBuffer(destination, byteCount, nameof(destination));
        ValidateStream(stream);
        NativeCudaApi.CopyFromArrayAsync(destination.Handle, _handle, sourceXBytes, sourceY, byteCount, stream.Handle);
    }

    /// <summary>
    /// Copies a 2D managed byte buffer into this CUDA array. 将二维托管字节缓冲区复制到当前 CUDA array。
    /// </summary>
    public void CopyFrom2D(byte[] source, int sourcePitch, int widthBytes, int height)
    {
        CopyFrom2D(source, sourcePitch, 0, 0, widthBytes, height);
    }

    /// <summary>
    /// Copies a 2D managed byte buffer into this CUDA array at the specified destination offset. 在指定目标偏移处复制二维托管字节缓冲区。
    /// </summary>
    public void CopyFrom2D(byte[] source, int sourcePitch, int destinationXBytes, int destinationY, int widthBytes, int height)
    {
        Validate2DRegion(sourcePitch, widthBytes, height, nameof(sourcePitch));
        NativeCudaApi.Copy2DToArray(_handle, destinationXBytes, destinationY, source, sourcePitch, widthBytes, height);
    }

    /// <summary>
    /// Copies a 2D region from this CUDA array into a managed byte buffer. 将当前 CUDA array 的二维区域复制到托管字节缓冲区。
    /// </summary>
    public void CopyTo2D(byte[] destination, int destinationPitch, int widthBytes, int height)
    {
        CopyTo2D(destination, destinationPitch, 0, 0, widthBytes, height);
    }

    /// <summary>
    /// Copies a 2D region from this CUDA array into a managed byte buffer using an explicit source offset. 使用显式源偏移复制二维区域到托管字节缓冲区。
    /// </summary>
    public void CopyTo2D(byte[] destination, int destinationPitch, int sourceXBytes, int sourceY, int widthBytes, int height)
    {
        Validate2DRegion(destinationPitch, widthBytes, height, nameof(destinationPitch));
        NativeCudaApi.Copy2DFromArray(destination, destinationPitch, _handle, sourceXBytes, sourceY, widthBytes, height);
    }

    /// <summary>
    /// Copies a 2D region from this CUDA array into another CUDA array. 将当前 CUDA array 的二维区域复制到另一个 CUDA array。
    /// </summary>
    public void CopyTo2D(CudaArray destination, int widthBytes, int height)
    {
        CopyTo2D(destination, 0, 0, 0, 0, widthBytes, height);
    }

    /// <summary>
    /// Copies a 2D region between CUDA arrays using explicit source and destination offsets. 使用显式源/目标偏移在 CUDA array 之间复制二维区域。
    /// </summary>
    public void CopyTo2D(CudaArray destination, int destinationXBytes, int destinationY, int sourceXBytes, int sourceY, int widthBytes, int height)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        Validate2DExtent(widthBytes, height);
        NativeCudaApi.Copy2DArrayToArray(destination._handle, destinationXBytes, destinationY, _handle, sourceXBytes, sourceY, widthBytes, height);
    }

    /// <summary>
    /// Copies a 2D pinned host buffer into this CUDA array asynchronously. 异步把二维 pinned host 缓冲区复制到当前 CUDA array。
    /// </summary>
    public void CopyFrom2DAsync(CudaPinnedMemory source, int sourcePitch, int widthBytes, int height, CudaStream stream)
    {
        CopyFrom2DAsync(source, sourcePitch, 0, 0, widthBytes, height, stream);
    }

    /// <summary>
    /// Copies a 2D pinned host buffer into this CUDA array asynchronously at the specified destination offset. 在指定目标偏移处异步复制二维 pinned host 缓冲区。
    /// </summary>
    public void CopyFrom2DAsync(CudaPinnedMemory source, int sourcePitch, int destinationXBytes, int destinationY, int widthBytes, int height, CudaStream stream)
    {
        ValidatePinned2D(source, sourcePitch, widthBytes, height, nameof(source));
        ValidateStream(stream);
        NativeCudaApi.Copy2DToArrayAsync(_handle, destinationXBytes, destinationY, source.Handle, sourcePitch, widthBytes, height, stream.Handle);
    }

    /// <summary>
    /// Copies a 2D region from this CUDA array into pinned host memory asynchronously. 异步把当前 CUDA array 的二维区域复制到 pinned host memory。
    /// </summary>
    public void CopyTo2DAsync(CudaPinnedMemory destination, int destinationPitch, int widthBytes, int height, CudaStream stream)
    {
        CopyTo2DAsync(destination, destinationPitch, 0, 0, widthBytes, height, stream);
    }

    /// <summary>
    /// Copies a 2D region from this CUDA array into pinned host memory asynchronously with an explicit source offset. 使用显式源偏移异步复制二维区域到 pinned host memory。
    /// </summary>
    public void CopyTo2DAsync(CudaPinnedMemory destination, int destinationPitch, int sourceXBytes, int sourceY, int widthBytes, int height, CudaStream stream)
    {
        ValidatePinned2D(destination, destinationPitch, widthBytes, height, nameof(destination));
        ValidateStream(stream);
        NativeCudaApi.Copy2DFromArrayAsync(destination.Handle, destinationPitch, _handle, sourceXBytes, sourceY, widthBytes, height, stream.Handle);
    }

    /// <summary>
    /// Copies a 3D managed byte buffer into this CUDA array. 将三维托管字节缓冲区复制到当前 CUDA array。
    /// </summary>
    public void CopyFrom3D(byte[] source, int sourcePitch, int sourceWidthBytes, int sourceHeight, int widthBytes, int height, int depth)
    {
        CopyFrom3D(source, sourcePitch, sourceWidthBytes, sourceHeight, 0, 0, 0, widthBytes, height, depth);
    }

    /// <summary>
    /// Copies a 3D managed byte buffer into this CUDA array at the specified destination offset. 在指定目标偏移处把三维托管字节缓冲区复制到当前 CUDA array。
    /// </summary>
    public void CopyFrom3D(byte[] source, int sourcePitch, int sourceWidthBytes, int sourceHeight, int destinationXBytes, int destinationY, int destinationZ, int widthBytes, int height, int depth)
    {
        Validate3DRegion(sourcePitch, sourceWidthBytes, sourceHeight, widthBytes, height, depth, nameof(sourcePitch));
        NativeCudaApi.Copy3DToArray(_handle, destinationXBytes, destinationY, destinationZ, source, sourcePitch, sourceWidthBytes, sourceHeight, widthBytes, height, depth);
    }

    /// <summary>
    /// Copies a 3D region from this CUDA array into a managed byte buffer. 将当前 CUDA array 的三维区域复制到托管字节缓冲区。
    /// </summary>
    public void CopyTo3D(byte[] destination, int destinationPitch, int destinationWidthBytes, int destinationHeight, int widthBytes, int height, int depth)
    {
        CopyTo3D(destination, destinationPitch, destinationWidthBytes, destinationHeight, 0, 0, 0, widthBytes, height, depth);
    }

    /// <summary>
    /// Copies a 3D region from this CUDA array into a managed byte buffer using an explicit source offset. 使用显式源偏移将三维区域复制到托管字节缓冲区。
    /// </summary>
    public void CopyTo3D(byte[] destination, int destinationPitch, int destinationWidthBytes, int destinationHeight, int sourceXBytes, int sourceY, int sourceZ, int widthBytes, int height, int depth)
    {
        Validate3DRegion(destinationPitch, destinationWidthBytes, destinationHeight, widthBytes, height, depth, nameof(destinationPitch));
        NativeCudaApi.Copy3DFromArray(destination, destinationPitch, destinationWidthBytes, destinationHeight, _handle, sourceXBytes, sourceY, sourceZ, widthBytes, height, depth);
    }

    /// <summary>
    /// Copies a 3D region from this CUDA array into another CUDA array. 将当前 CUDA array 的三维区域复制到另一个 CUDA array。
    /// </summary>
    public void CopyTo3D(CudaArray destination, int widthBytes, int height, int depth)
    {
        CopyTo3D(destination, 0, 0, 0, 0, 0, 0, widthBytes, height, depth);
    }

    /// <summary>
    /// Copies a 3D region between CUDA arrays using explicit source and destination offsets. 使用显式源/目标偏移在两个 CUDA array 之间复制三维区域。
    /// </summary>
    public void CopyTo3D(CudaArray destination, int destinationXBytes, int destinationY, int destinationZ, int sourceXBytes, int sourceY, int sourceZ, int widthBytes, int height, int depth)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        Validate3DExtent(widthBytes, height, depth);
        NativeCudaApi.Copy3DArrayToArray(destination._handle, destinationXBytes, destinationY, destinationZ, _handle, sourceXBytes, sourceY, sourceZ, widthBytes, height, depth);
    }

    /// <summary>
    /// Copies a 3D pinned host buffer into this CUDA array asynchronously. 异步把三维 pinned host 缓冲区复制到当前 CUDA array。
    /// </summary>
    public void CopyFrom3DAsync(CudaPinnedMemory source, int sourcePitch, int sourceWidthBytes, int sourceHeight, int widthBytes, int height, int depth, CudaStream stream)
    {
        CopyFrom3DAsync(source, sourcePitch, sourceWidthBytes, sourceHeight, 0, 0, 0, widthBytes, height, depth, stream);
    }

    /// <summary>
    /// Copies a 3D pinned host buffer into this CUDA array asynchronously at the specified destination offset. 在指定目标偏移处异步复制三维 pinned host 缓冲区。
    /// </summary>
    public void CopyFrom3DAsync(CudaPinnedMemory source, int sourcePitch, int sourceWidthBytes, int sourceHeight, int destinationXBytes, int destinationY, int destinationZ, int widthBytes, int height, int depth, CudaStream stream)
    {
        ValidatePinned3D(source, sourcePitch, sourceWidthBytes, sourceHeight, widthBytes, height, depth, nameof(source));
        ValidateStream(stream);
        NativeCudaApi.Copy3DToArrayAsync(_handle, destinationXBytes, destinationY, destinationZ, source.Handle, sourcePitch, sourceWidthBytes, sourceHeight, widthBytes, height, depth, stream.Handle);
    }

    /// <summary>
    /// Copies a 3D region from this CUDA array into pinned host memory asynchronously. 异步把当前 CUDA array 的三维区域复制到 pinned host memory。
    /// </summary>
    public void CopyTo3DAsync(CudaPinnedMemory destination, int destinationPitch, int destinationWidthBytes, int destinationHeight, int widthBytes, int height, int depth, CudaStream stream)
    {
        CopyTo3DAsync(destination, destinationPitch, destinationWidthBytes, destinationHeight, 0, 0, 0, widthBytes, height, depth, stream);
    }

    /// <summary>
    /// Copies a 3D region from this CUDA array into pinned host memory asynchronously using an explicit source offset. 使用显式源偏移异步复制三维区域到 pinned host memory。
    /// </summary>
    public void CopyTo3DAsync(CudaPinnedMemory destination, int destinationPitch, int destinationWidthBytes, int destinationHeight, int sourceXBytes, int sourceY, int sourceZ, int widthBytes, int height, int depth, CudaStream stream)
    {
        ValidatePinned3D(destination, destinationPitch, destinationWidthBytes, destinationHeight, widthBytes, height, depth, nameof(destination));
        ValidateStream(stream);
        NativeCudaApi.Copy3DFromArrayAsync(destination.Handle, destinationPitch, destinationWidthBytes, destinationHeight, _handle, sourceXBytes, sourceY, sourceZ, widthBytes, height, depth, stream.Handle);
    }

    /// <summary>
    /// Copies a 3D region from this CUDA array into another CUDA array asynchronously. 异步把当前 CUDA array 的三维区域复制到另一个 CUDA array。
    /// </summary>
    public void CopyTo3DAsync(CudaArray destination, int widthBytes, int height, int depth, CudaStream stream)
    {
        CopyTo3DAsync(destination, 0, 0, 0, 0, 0, 0, widthBytes, height, depth, stream);
    }

    /// <summary>
    /// Copies a 3D region between CUDA arrays asynchronously using explicit source and destination offsets. 使用显式源/目标偏移异步复制三维区域。
    /// </summary>
    public void CopyTo3DAsync(CudaArray destination, int destinationXBytes, int destinationY, int destinationZ, int sourceXBytes, int sourceY, int sourceZ, int widthBytes, int height, int depth, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        ValidateStream(stream);
        Validate3DExtent(widthBytes, height, depth);
        NativeCudaApi.Copy3DArrayToArrayAsync(destination._handle, destinationXBytes, destinationY, destinationZ, _handle, sourceXBytes, sourceY, sourceZ, widthBytes, height, depth, stream.Handle);
    }

    /// <summary>
    /// Materializes a 2D region as a managed byte array. 将二维区域物化为托管字节数组。
    /// </summary>
    public byte[] ToArray2D(int pitch, int widthBytes, int height)
    {
        Validate2DRegion(pitch, widthBytes, height, nameof(pitch));
        byte[] data = new byte[checked(pitch * height)];
        CopyTo2D(data, pitch, widthBytes, height);
        return data;
    }

    /// <summary>
    /// Materializes a 3D region as a managed byte array. 将三维区域物化为托管字节数组。
    /// </summary>
    public byte[] ToArray3D(int pitch, int pitchedWidthBytes, int pitchedHeight, int widthBytes, int height, int depth)
    {
        Validate3DRegion(pitch, pitchedWidthBytes, pitchedHeight, widthBytes, height, depth, nameof(pitch));
        byte[] data = new byte[checked(pitch * pitchedHeight * depth)];
        CopyTo3D(data, pitch, pitchedWidthBytes, pitchedHeight, widthBytes, height, depth);
        return data;
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
