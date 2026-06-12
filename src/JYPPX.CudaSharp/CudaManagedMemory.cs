using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Managed wrapper around a CUDA unified-memory allocation.
/// </summary>
public sealed class CudaManagedMemory : CudaMemory
{
    public CudaManagedMemory(int sizeInBytes, CudaManagedMemoryAttachmentFlags flags = CudaManagedMemoryAttachmentFlags.Global)
        : base(AllocateManagedMemory(sizeInBytes, flags))
    {
        AttachmentFlags = flags;
    }

    public CudaManagedMemoryAttachmentFlags AttachmentFlags { get; }

    private static SafeCudaMemoryHandle AllocateManagedMemory(int sizeInBytes, CudaManagedMemoryAttachmentFlags flags)
    {
        if (sizeInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInBytes));
        }

        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.AllocateManagedMemory(sizeInBytes, (uint)flags);
    }
}
