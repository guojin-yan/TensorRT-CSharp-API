using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed class CudaMipmappedArray : IDisposable
{
    private readonly SafeCudaMipmappedArrayHandle _handle;

    public CudaMipmappedArray(CudaChannelFormatDescriptor descriptor, CudaArrayExtent extent, uint levelCount, CudaArrayCreationFlags flags = CudaArrayCreationFlags.Default)
    {
        if (extent.Width == 0 || extent.Height == 0 || extent.Depth == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(extent));
        }

        if (levelCount == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(levelCount));
        }

        NativeBridgeLoader.EnsureInitialized();
        _handle = NativeCudaApi.AllocateMipmappedArray(descriptor, extent, levelCount, flags);
        Descriptor = descriptor;
        Extent = extent;
        LevelCount = levelCount;
        Flags = flags;
    }

    internal SafeCudaMipmappedArrayHandle Handle => _handle;

    public CudaChannelFormatDescriptor Descriptor { get; }
    public CudaArrayExtent Extent { get; }
    public uint LevelCount { get; }
    public CudaArrayCreationFlags Flags { get; }

    public CudaArrayInfo GetLevelInfo(uint level)
    {
        if (level >= LevelCount)
        {
            throw new ArgumentOutOfRangeException(nameof(level));
        }

        return CudaArrayInfo.FromNative(NativeCudaApi.GetMipmappedArrayLevelInfo(_handle, level));
    }

    public CudaArrayMemoryRequirements GetMemoryRequirements(int device)
    {
        return CudaArrayMemoryRequirements.FromNative(NativeCudaApi.GetMipmappedArrayMemoryRequirements(_handle, device));
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
        return CudaArraySparseProperties.FromNative(NativeCudaApi.GetMipmappedArraySparseProperties(_handle));
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

    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}
