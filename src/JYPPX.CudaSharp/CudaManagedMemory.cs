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

    /// <summary>
    /// Asynchronously prefetches a managed-memory range to a strongly typed CUDA location.
    /// 异步将 managed memory 范围预取到强类型 CUDA 位置。
    /// </summary>
    /// <param name="offset">The byte offset within this allocation. 当前分配内的字节偏移。</param>
    /// <param name="count">The number of bytes to prefetch. 要预取的字节数。</param>
    /// <param name="location">The destination CUDA memory location. 目标 CUDA memory 位置。</param>
    /// <param name="stream">The CUDA stream that orders the prefetch. 用于排序预取操作的 CUDA stream。</param>
    /// <remarks>
    /// Keep this allocation and <paramref name="stream"/> alive until the stream has synchronized.
    /// 在 stream 完成同步前，调用方必须保持此分配和 <paramref name="stream"/> 存活。
    /// </remarks>
    public void PrefetchAsync(int offset, int count, CudaMemoryLocation location, CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        location.Validate(nameof(location));
        ValidateRange(offset, count, nameof(offset), nameof(count));
        NativeCudaApi.PrefetchManagedMemoryLocationRangeAsync(
            Handle,
            offset,
            count,
            (int)location.Kind,
            location.Id,
            stream.Handle);
    }

    /// <summary>
    /// Asynchronously prefetches this entire managed allocation to a strongly typed CUDA location.
    /// 异步将整个 managed memory 分配预取到强类型 CUDA 位置。
    /// </summary>
    public void PrefetchAsync(CudaMemoryLocation location, CudaStream stream)
    {
        PrefetchAsync(0, SizeInBytes, location, stream);
    }

    /// <summary>
    /// Applies CUDA memory advice to a managed-memory range at a strongly typed location.
    /// 对强类型位置上的 managed memory 范围应用 CUDA memory advice。
    /// </summary>
    /// <param name="offset">The byte offset within this allocation. 当前分配内的字节偏移。</param>
    /// <param name="count">The number of bytes covered by the advice. advice 覆盖的字节数。</param>
    /// <param name="advice">The CUDA memory advice. CUDA memory advice。</param>
    /// <param name="location">The CUDA memory location associated with the advice. 与 advice 关联的 CUDA memory 位置。</param>
    public void Advise(int offset, int count, CudaMemoryAdvice advice, CudaMemoryLocation location)
    {
        ValidateRange(offset, count, nameof(offset), nameof(count));
        ValidateMemoryAdvice(advice, nameof(advice));
        location.Validate(nameof(location));
        if ((advice == CudaMemoryAdvice.SetAccessedBy || advice == CudaMemoryAdvice.UnsetAccessedBy) &&
            location.Kind != CudaMemoryLocationKind.Device &&
            location.Kind != CudaMemoryLocationKind.Host)
        {
            throw new ArgumentException("Accessed-by advice accepts only CUDA device or host locations.", nameof(location));
        }

        NativeCudaApi.AdviseManagedMemoryLocationRange(
            Handle,
            offset,
            count,
            (int)advice,
            (int)location.Kind,
            location.Id);
    }

    /// <summary>
    /// Applies CUDA memory advice to this entire managed allocation at a strongly typed location.
    /// 对强类型位置上的整个 managed memory 分配应用 CUDA memory advice。
    /// </summary>
    public void Advise(CudaMemoryAdvice advice, CudaMemoryLocation location)
    {
        Advise(0, SizeInBytes, advice, location);
    }

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
