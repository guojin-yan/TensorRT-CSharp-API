using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Managed wrapper around a CUDA unified-memory allocation.
/// CUDA unified memory 分配的托管封装。
/// </summary>
public sealed class CudaManagedMemory : CudaMemory
{
    /// <summary>
    /// Allocates CUDA managed memory with the requested attachment flags.
    /// 使用指定的附着标志分配 CUDA managed memory。
    /// </summary>
    /// <param name="sizeInBytes">The allocation size in bytes. 分配大小，单位为字节。</param>
    /// <param name="flags">The managed-memory attachment flags. managed memory 附着标志。</param>
    public CudaManagedMemory(int sizeInBytes, CudaManagedMemoryAttachmentFlags flags = CudaManagedMemoryAttachmentFlags.Global)
        : base(AllocateManagedMemory(sizeInBytes, flags))
    {
        AttachmentFlags = flags;
    }

    /// <summary>
    /// Gets the managed-memory attachment flags used for the allocation.
    /// 获取该分配使用的 managed memory 附着标志。
    /// </summary>
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
