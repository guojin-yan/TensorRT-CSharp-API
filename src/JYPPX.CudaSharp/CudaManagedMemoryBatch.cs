using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Describes an owner-bound subrange of a CUDA managed-memory allocation.
/// 描述 CUDA managed memory 分配中与 owner 绑定的子范围。
/// </summary>
public readonly struct CudaManagedMemoryRange
{
    /// <summary>
    /// Creates an owner-bound range covering an entire managed-memory allocation.
    /// 创建覆盖整个 managed memory 分配的 owner-bound 范围。
    /// </summary>
    /// <param name="memory">The managed-memory owner. managed memory owner。</param>
    public CudaManagedMemoryRange(CudaManagedMemory memory)
        : this(memory, 0, memory == null ? 0 : memory.SizeInBytes)
    {
    }

    /// <summary>
    /// Creates an owner-bound managed-memory subrange.
    /// 创建与 owner 绑定的 managed memory 子范围。
    /// </summary>
    /// <param name="memory">The managed-memory owner. managed memory owner。</param>
    /// <param name="offset">The byte offset within the allocation. 分配内的字节偏移。</param>
    /// <param name="count">The range length in bytes. 范围长度，单位为字节。</param>
    public CudaManagedMemoryRange(CudaManagedMemory memory, int offset, int count)
    {
        Memory = memory ?? throw new ArgumentNullException(nameof(memory));
        ValidateRange(memory, offset, count);
        Offset = offset;
        Count = count;
    }

    /// <summary>
    /// Gets the managed-memory owner.
    /// 获取 managed memory owner。
    /// </summary>
    public CudaManagedMemory Memory { get; }

    /// <summary>
    /// Gets the byte offset within the allocation.
    /// 获取分配内的字节偏移。
    /// </summary>
    public int Offset { get; }

    /// <summary>
    /// Gets the range length in bytes.
    /// 获取范围长度，单位为字节。
    /// </summary>
    public int Count { get; }

    internal SafeCudaMemoryHandle Handle => Memory.Handle;

    internal static void ValidateRange(CudaManagedMemory memory, int offset, int count)
    {
        if (offset < 0 || offset > memory.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(offset));
        }

        if (count <= 0 || count > memory.SizeInBytes - offset)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }
    }
}

/// <summary>
/// Describes an owner-bound CUDA managed-memory range and its prefetch destination.
/// 描述与 owner 绑定的 CUDA managed memory 范围及其预取目标设备。
/// </summary>
public readonly struct CudaManagedMemoryPrefetchRange
{
    /// <summary>
    /// Creates a prefetch range covering an entire managed-memory allocation.
    /// 创建覆盖整个 managed memory 分配的预取范围。
    /// </summary>
    /// <param name="memory">The managed-memory owner. managed memory owner。</param>
    /// <param name="destinationDevice">The destination CUDA device ordinal. 目标 CUDA 设备序号。</param>
    public CudaManagedMemoryPrefetchRange(CudaManagedMemory memory, int destinationDevice)
        : this(memory, 0, memory == null ? 0 : memory.SizeInBytes, destinationDevice)
    {
    }

    /// <summary>
    /// Creates an owner-bound managed-memory prefetch subrange.
    /// 创建与 owner 绑定的 managed memory 预取子范围。
    /// </summary>
    /// <param name="memory">The managed-memory owner. managed memory owner。</param>
    /// <param name="offset">The byte offset within the allocation. 分配内的字节偏移。</param>
    /// <param name="count">The range length in bytes. 范围长度，单位为字节。</param>
    /// <param name="destinationDevice">The destination CUDA device ordinal. 目标 CUDA 设备序号。</param>
    public CudaManagedMemoryPrefetchRange(CudaManagedMemory memory, int offset, int count, int destinationDevice)
    {
        Memory = memory ?? throw new ArgumentNullException(nameof(memory));
        CudaManagedMemoryRange.ValidateRange(memory, offset, count);
        if (destinationDevice < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(destinationDevice));
        }

        Offset = offset;
        Count = count;
        DestinationDevice = destinationDevice;
    }

    /// <summary>
    /// Gets the managed-memory owner.
    /// 获取 managed memory owner。
    /// </summary>
    public CudaManagedMemory Memory { get; }

    /// <summary>
    /// Gets the byte offset within the allocation.
    /// 获取分配内的字节偏移。
    /// </summary>
    public int Offset { get; }

    /// <summary>
    /// Gets the range length in bytes.
    /// 获取范围长度，单位为字节。
    /// </summary>
    public int Count { get; }

    /// <summary>
    /// Gets the destination CUDA device ordinal.
    /// 获取目标 CUDA 设备序号。
    /// </summary>
    public int DestinationDevice { get; }

    internal SafeCudaMemoryHandle Handle => Memory.Handle;
}

/// <summary>
/// Provides owner-safe CUDA 13 managed-memory batch operations.
/// 提供 owner-safe 的 CUDA 13 managed memory 批量操作。
/// </summary>
public static class CudaManagedMemoryBatch
{
    /// <summary>
    /// Asynchronously prefetches managed-memory ranges in one CUDA batch.
    /// 在一个 CUDA batch 中异步预取多个 managed memory 范围。
    /// </summary>
    /// <remarks>
    /// Keep every range owner and the stream alive until the stream is synchronized.
    /// 在 stream 完成同步前，调用方必须保持所有范围 owner 和 stream 存活。
    /// </remarks>
    public static void PrefetchAsync(IReadOnlyList<CudaManagedMemoryPrefetchRange> ranges, CudaStream stream)
    {
        ValidateStreamAndCount(ranges, stream);
        NativeCudaApi.PrefetchManagedMemoryBatchAsync(CopyPrefetchRanges(ranges), stream.Handle);
    }

    /// <summary>
    /// Asynchronously discards managed-memory ranges in one CUDA batch.
    /// 在一个 CUDA batch 中异步丢弃多个 managed memory 范围的内容。
    /// </summary>
    /// <remarks>
    /// Keep every range owner and the stream alive until the stream is synchronized.
    /// 在 stream 完成同步前，调用方必须保持所有范围 owner 和 stream 存活。
    /// </remarks>
    public static void DiscardAsync(IReadOnlyList<CudaManagedMemoryRange> ranges, CudaStream stream)
    {
        ValidateStreamAndCount(ranges, stream);
        NativeCudaApi.DiscardManagedMemoryBatchAsync(CopyRanges(ranges), stream.Handle);
    }

    /// <summary>
    /// Asynchronously discards and then prefetches managed-memory ranges in one CUDA batch.
    /// 在一个 CUDA batch 中异步丢弃并预取多个 managed memory 范围。
    /// </summary>
    /// <remarks>
    /// Keep every range owner and the stream alive until the stream is synchronized.
    /// 在 stream 完成同步前，调用方必须保持所有范围 owner 和 stream 存活。
    /// </remarks>
    public static void DiscardAndPrefetchAsync(IReadOnlyList<CudaManagedMemoryPrefetchRange> ranges, CudaStream stream)
    {
        ValidateStreamAndCount(ranges, stream);
        NativeCudaApi.DiscardAndPrefetchManagedMemoryBatchAsync(CopyPrefetchRanges(ranges), stream.Handle);
    }

    private static void ValidateStreamAndCount<T>(IReadOnlyList<T> ranges, CudaStream stream)
    {
        if (ranges == null)
        {
            throw new ArgumentNullException(nameof(ranges));
        }

        if (ranges.Count == 0)
        {
            throw new ArgumentException("At least one CUDA managed-memory range is required.", nameof(ranges));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }
    }

    private static CudaManagedMemoryRange[] CopyRanges(IReadOnlyList<CudaManagedMemoryRange> ranges)
    {
        CudaManagedMemoryRange[] copied = new CudaManagedMemoryRange[ranges.Count];
        for (int index = 0; index < copied.Length; ++index)
        {
            CudaManagedMemoryRange range = ranges[index];
            if (range.Memory == null)
            {
                throw new ArgumentException("CUDA managed-memory batch ranges must contain a memory owner.", nameof(ranges));
            }

            CudaManagedMemoryRange.ValidateRange(range.Memory, range.Offset, range.Count);
            copied[index] = range;
        }

        return copied;
    }

    private static CudaManagedMemoryPrefetchRange[] CopyPrefetchRanges(IReadOnlyList<CudaManagedMemoryPrefetchRange> ranges)
    {
        CudaManagedMemoryPrefetchRange[] copied = new CudaManagedMemoryPrefetchRange[ranges.Count];
        for (int index = 0; index < copied.Length; ++index)
        {
            CudaManagedMemoryPrefetchRange range = ranges[index];
            if (range.Memory == null)
            {
                throw new ArgumentException("CUDA managed-memory prefetch ranges must contain a memory owner.", nameof(ranges));
            }

            CudaManagedMemoryRange.ValidateRange(range.Memory, range.Offset, range.Count);
            if (range.DestinationDevice < 0)
            {
                throw new ArgumentException("CUDA managed-memory prefetch destinations must be non-negative device ordinals.", nameof(ranges));
            }

            copied[index] = range;
        }

        return copied;
    }
}
