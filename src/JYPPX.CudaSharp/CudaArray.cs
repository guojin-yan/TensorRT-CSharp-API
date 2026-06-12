using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed class CudaArray : IDisposable
{
    private readonly SafeCudaArrayHandle _handle;

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

    public CudaChannelFormatDescriptor Descriptor { get; }
    public CudaArrayExtent Extent { get; }
    public CudaArrayCreationFlags Flags { get; }

    public CudaArrayInfo Info => CudaArrayInfo.FromNative(NativeCudaApi.GetArrayInfo(_handle));

    public CudaChannelFormatDescriptor ChannelDescriptor => CudaChannelFormatDescriptor.FromNative(NativeCudaApi.GetArrayChannelDescriptor(_handle));

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

    public CudaArrayMemoryRequirements GetMemoryRequirements(int device)
    {
        return CudaArrayMemoryRequirements.FromNative(NativeCudaApi.GetArrayMemoryRequirements(_handle, device));
    }

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

    public CudaArraySparseProperties GetSparseProperties()
    {
        return CudaArraySparseProperties.FromNative(NativeCudaApi.GetArraySparseProperties(_handle));
    }

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

    public void CopyFrom(byte[] source)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        CopyFrom(source, source.Length);
    }

    public void CopyFrom(byte[] source, int byteCount)
    {
        CopyFrom(source, 0, 0, byteCount);
    }

    public void CopyFrom(byte[] source, int destinationXBytes, int destinationY, int byteCount)
    {
        ValidateByteCount(byteCount);
        NativeCudaApi.CopyToArray(_handle, destinationXBytes, destinationY, source, byteCount);
    }

    public void CopyTo(byte[] destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        CopyTo(destination, destination.Length);
    }

    public void CopyTo(byte[] destination, int byteCount)
    {
        CopyTo(destination, 0, 0, byteCount);
    }

    public void CopyTo(byte[] destination, int sourceXBytes, int sourceY, int byteCount)
    {
        ValidateByteCount(byteCount);
        NativeCudaApi.CopyFromArray(destination, _handle, sourceXBytes, sourceY, byteCount);
    }

    public void CopyTo(CudaArray destination, int byteCount)
    {
        CopyTo(destination, 0, 0, 0, 0, byteCount);
    }

    public void CopyTo(CudaArray destination, int destinationXBytes, int destinationY, int sourceXBytes, int sourceY, int byteCount)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        ValidateByteCount(byteCount);
        NativeCudaApi.CopyArrayToArray(destination._handle, destinationXBytes, destinationY, _handle, sourceXBytes, sourceY, byteCount);
    }

    public void CopyFromAsync(CudaPinnedMemory source, int byteCount, CudaStream stream)
    {
        CopyFromAsync(source, 0, 0, byteCount, stream);
    }

    public void CopyFromAsync(CudaPinnedMemory source, int destinationXBytes, int destinationY, int byteCount, CudaStream stream)
    {
        ValidatePinnedBuffer(source, byteCount, nameof(source));
        ValidateStream(stream);
        NativeCudaApi.CopyToArrayAsync(_handle, destinationXBytes, destinationY, source.Handle, byteCount, stream.Handle);
    }

    public void CopyToAsync(CudaPinnedMemory destination, int byteCount, CudaStream stream)
    {
        CopyToAsync(destination, 0, 0, byteCount, stream);
    }

    public void CopyToAsync(CudaPinnedMemory destination, int sourceXBytes, int sourceY, int byteCount, CudaStream stream)
    {
        ValidatePinnedBuffer(destination, byteCount, nameof(destination));
        ValidateStream(stream);
        NativeCudaApi.CopyFromArrayAsync(destination.Handle, _handle, sourceXBytes, sourceY, byteCount, stream.Handle);
    }

    public void CopyFrom2D(byte[] source, int sourcePitch, int widthBytes, int height)
    {
        CopyFrom2D(source, sourcePitch, 0, 0, widthBytes, height);
    }

    public void CopyFrom2D(byte[] source, int sourcePitch, int destinationXBytes, int destinationY, int widthBytes, int height)
    {
        Validate2DRegion(sourcePitch, widthBytes, height, nameof(sourcePitch));
        NativeCudaApi.Copy2DToArray(_handle, destinationXBytes, destinationY, source, sourcePitch, widthBytes, height);
    }

    public void CopyTo2D(byte[] destination, int destinationPitch, int widthBytes, int height)
    {
        CopyTo2D(destination, destinationPitch, 0, 0, widthBytes, height);
    }

    public void CopyTo2D(byte[] destination, int destinationPitch, int sourceXBytes, int sourceY, int widthBytes, int height)
    {
        Validate2DRegion(destinationPitch, widthBytes, height, nameof(destinationPitch));
        NativeCudaApi.Copy2DFromArray(destination, destinationPitch, _handle, sourceXBytes, sourceY, widthBytes, height);
    }

    public void CopyTo2D(CudaArray destination, int widthBytes, int height)
    {
        CopyTo2D(destination, 0, 0, 0, 0, widthBytes, height);
    }

    public void CopyTo2D(CudaArray destination, int destinationXBytes, int destinationY, int sourceXBytes, int sourceY, int widthBytes, int height)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        Validate2DExtent(widthBytes, height);
        NativeCudaApi.Copy2DArrayToArray(destination._handle, destinationXBytes, destinationY, _handle, sourceXBytes, sourceY, widthBytes, height);
    }

    public void CopyFrom2DAsync(CudaPinnedMemory source, int sourcePitch, int widthBytes, int height, CudaStream stream)
    {
        CopyFrom2DAsync(source, sourcePitch, 0, 0, widthBytes, height, stream);
    }

    public void CopyFrom2DAsync(CudaPinnedMemory source, int sourcePitch, int destinationXBytes, int destinationY, int widthBytes, int height, CudaStream stream)
    {
        ValidatePinned2D(source, sourcePitch, widthBytes, height, nameof(source));
        ValidateStream(stream);
        NativeCudaApi.Copy2DToArrayAsync(_handle, destinationXBytes, destinationY, source.Handle, sourcePitch, widthBytes, height, stream.Handle);
    }

    public void CopyTo2DAsync(CudaPinnedMemory destination, int destinationPitch, int widthBytes, int height, CudaStream stream)
    {
        CopyTo2DAsync(destination, destinationPitch, 0, 0, widthBytes, height, stream);
    }

    public void CopyTo2DAsync(CudaPinnedMemory destination, int destinationPitch, int sourceXBytes, int sourceY, int widthBytes, int height, CudaStream stream)
    {
        ValidatePinned2D(destination, destinationPitch, widthBytes, height, nameof(destination));
        ValidateStream(stream);
        NativeCudaApi.Copy2DFromArrayAsync(destination.Handle, destinationPitch, _handle, sourceXBytes, sourceY, widthBytes, height, stream.Handle);
    }

    public void CopyFrom3D(byte[] source, int sourcePitch, int sourceWidthBytes, int sourceHeight, int widthBytes, int height, int depth)
    {
        CopyFrom3D(source, sourcePitch, sourceWidthBytes, sourceHeight, 0, 0, 0, widthBytes, height, depth);
    }

    public void CopyFrom3D(byte[] source, int sourcePitch, int sourceWidthBytes, int sourceHeight, int destinationXBytes, int destinationY, int destinationZ, int widthBytes, int height, int depth)
    {
        Validate3DRegion(sourcePitch, sourceWidthBytes, sourceHeight, widthBytes, height, depth, nameof(sourcePitch));
        NativeCudaApi.Copy3DToArray(_handle, destinationXBytes, destinationY, destinationZ, source, sourcePitch, sourceWidthBytes, sourceHeight, widthBytes, height, depth);
    }

    public void CopyTo3D(byte[] destination, int destinationPitch, int destinationWidthBytes, int destinationHeight, int widthBytes, int height, int depth)
    {
        CopyTo3D(destination, destinationPitch, destinationWidthBytes, destinationHeight, 0, 0, 0, widthBytes, height, depth);
    }

    public void CopyTo3D(byte[] destination, int destinationPitch, int destinationWidthBytes, int destinationHeight, int sourceXBytes, int sourceY, int sourceZ, int widthBytes, int height, int depth)
    {
        Validate3DRegion(destinationPitch, destinationWidthBytes, destinationHeight, widthBytes, height, depth, nameof(destinationPitch));
        NativeCudaApi.Copy3DFromArray(destination, destinationPitch, destinationWidthBytes, destinationHeight, _handle, sourceXBytes, sourceY, sourceZ, widthBytes, height, depth);
    }

    public void CopyTo3D(CudaArray destination, int widthBytes, int height, int depth)
    {
        CopyTo3D(destination, 0, 0, 0, 0, 0, 0, widthBytes, height, depth);
    }

    public void CopyTo3D(CudaArray destination, int destinationXBytes, int destinationY, int destinationZ, int sourceXBytes, int sourceY, int sourceZ, int widthBytes, int height, int depth)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        Validate3DExtent(widthBytes, height, depth);
        NativeCudaApi.Copy3DArrayToArray(destination._handle, destinationXBytes, destinationY, destinationZ, _handle, sourceXBytes, sourceY, sourceZ, widthBytes, height, depth);
    }

    public void CopyFrom3DAsync(CudaPinnedMemory source, int sourcePitch, int sourceWidthBytes, int sourceHeight, int widthBytes, int height, int depth, CudaStream stream)
    {
        CopyFrom3DAsync(source, sourcePitch, sourceWidthBytes, sourceHeight, 0, 0, 0, widthBytes, height, depth, stream);
    }

    public void CopyFrom3DAsync(CudaPinnedMemory source, int sourcePitch, int sourceWidthBytes, int sourceHeight, int destinationXBytes, int destinationY, int destinationZ, int widthBytes, int height, int depth, CudaStream stream)
    {
        ValidatePinned3D(source, sourcePitch, sourceWidthBytes, sourceHeight, widthBytes, height, depth, nameof(source));
        ValidateStream(stream);
        NativeCudaApi.Copy3DToArrayAsync(_handle, destinationXBytes, destinationY, destinationZ, source.Handle, sourcePitch, sourceWidthBytes, sourceHeight, widthBytes, height, depth, stream.Handle);
    }

    public void CopyTo3DAsync(CudaPinnedMemory destination, int destinationPitch, int destinationWidthBytes, int destinationHeight, int widthBytes, int height, int depth, CudaStream stream)
    {
        CopyTo3DAsync(destination, destinationPitch, destinationWidthBytes, destinationHeight, 0, 0, 0, widthBytes, height, depth, stream);
    }

    public void CopyTo3DAsync(CudaPinnedMemory destination, int destinationPitch, int destinationWidthBytes, int destinationHeight, int sourceXBytes, int sourceY, int sourceZ, int widthBytes, int height, int depth, CudaStream stream)
    {
        ValidatePinned3D(destination, destinationPitch, destinationWidthBytes, destinationHeight, widthBytes, height, depth, nameof(destination));
        ValidateStream(stream);
        NativeCudaApi.Copy3DFromArrayAsync(destination.Handle, destinationPitch, destinationWidthBytes, destinationHeight, _handle, sourceXBytes, sourceY, sourceZ, widthBytes, height, depth, stream.Handle);
    }

    public void CopyTo3DAsync(CudaArray destination, int widthBytes, int height, int depth, CudaStream stream)
    {
        CopyTo3DAsync(destination, 0, 0, 0, 0, 0, 0, widthBytes, height, depth, stream);
    }

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

    public byte[] ToArray2D(int pitch, int widthBytes, int height)
    {
        Validate2DRegion(pitch, widthBytes, height, nameof(pitch));
        byte[] data = new byte[checked(pitch * height)];
        CopyTo2D(data, pitch, widthBytes, height);
        return data;
    }

    public byte[] ToArray3D(int pitch, int pitchedWidthBytes, int pitchedHeight, int widthBytes, int height, int depth)
    {
        Validate3DRegion(pitch, pitchedWidthBytes, pitchedHeight, widthBytes, height, depth, nameof(pitch));
        byte[] data = new byte[checked(pitch * pitchedHeight * depth)];
        CopyTo3D(data, pitch, pitchedWidthBytes, pitchedHeight, widthBytes, height, depth);
        return data;
    }

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
