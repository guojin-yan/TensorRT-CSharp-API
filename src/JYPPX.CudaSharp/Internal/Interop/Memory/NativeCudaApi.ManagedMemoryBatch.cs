using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.Shared.Interop;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static void PrefetchManagedMemoryBatchAsync(
        IReadOnlyList<CudaManagedMemoryPrefetchRange> ranges,
        SafeCudaStreamHandle stream)
    {
        InvokePrefetchBatch(ranges, stream, discardFirst: false);
    }

    public static void DiscardManagedMemoryBatchAsync(
        IReadOnlyList<CudaManagedMemoryRange> ranges,
        SafeCudaStreamHandle stream)
    {
        int count = ranges.Count;
        NativeCudaManagedMemoryBatchRange[] nativeRanges = new NativeCudaManagedMemoryBatchRange[count];
        SafeCudaMemoryHandle[] owners = new SafeCudaMemoryHandle[count];
        bool[] leases = new bool[count];
        GCHandle rangesHandle = default;
        try
        {
            for (int index = 0; index < count; ++index)
            {
                CudaManagedMemoryRange range = ranges[index];
                SafeCudaMemoryHandle owner = range.Handle;
                owner.DangerousAddRef(ref leases[index]);
                owners[index] = owner;
                nativeRanges[index] = new NativeCudaManagedMemoryBatchRange
                {
                    Memory = owner.DangerousGetHandle(),
                    Offset = (UIntPtr)range.Offset,
                    Size = (UIntPtr)range.Count,
                    DestinationDevice = 0
                };
            }

            rangesHandle = GCHandle.Alloc(nativeRanges, GCHandleType.Pinned);
            CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_managed_memory_discard_batch_async_safe(
                rangesHandle.AddrOfPinnedObject(),
                (UIntPtr)count,
                stream));
        }
        finally
        {
            if (rangesHandle.IsAllocated)
            {
                rangesHandle.Free();
            }

            ReleaseLeases(owners, leases);
        }
    }

    public static void DiscardAndPrefetchManagedMemoryBatchAsync(
        IReadOnlyList<CudaManagedMemoryPrefetchRange> ranges,
        SafeCudaStreamHandle stream)
    {
        InvokePrefetchBatch(ranges, stream, discardFirst: true);
    }

    private static void InvokePrefetchBatch(
        IReadOnlyList<CudaManagedMemoryPrefetchRange> ranges,
        SafeCudaStreamHandle stream,
        bool discardFirst)
    {
        int count = ranges.Count;
        NativeCudaManagedMemoryBatchRange[] nativeRanges = new NativeCudaManagedMemoryBatchRange[count];
        SafeCudaMemoryHandle[] owners = new SafeCudaMemoryHandle[count];
        bool[] leases = new bool[count];
        GCHandle rangesHandle = default;
        try
        {
            for (int index = 0; index < count; ++index)
            {
                CudaManagedMemoryPrefetchRange range = ranges[index];
                SafeCudaMemoryHandle owner = range.Handle;
                owner.DangerousAddRef(ref leases[index]);
                owners[index] = owner;
                nativeRanges[index] = new NativeCudaManagedMemoryBatchRange
                {
                    Memory = owner.DangerousGetHandle(),
                    Offset = (UIntPtr)range.Offset,
                    Size = (UIntPtr)range.Count,
                    DestinationDevice = range.DestinationDevice
                };
            }

            rangesHandle = GCHandle.Alloc(nativeRanges, GCHandleType.Pinned);
            BridgeStatusCode status = discardFirst
                ? NativeMethodsCuda.jyppx_cuda_managed_memory_discard_and_prefetch_batch_async_safe(
                    rangesHandle.AddrOfPinnedObject(),
                    (UIntPtr)count,
                    stream)
                : NativeMethodsCuda.jyppx_cuda_managed_memory_prefetch_batch_async_safe(
                    rangesHandle.AddrOfPinnedObject(),
                    (UIntPtr)count,
                    stream);
            CudaNativeStatus.ThrowIfFailed(status);
        }
        finally
        {
            if (rangesHandle.IsAllocated)
            {
                rangesHandle.Free();
            }

            ReleaseLeases(owners, leases);
        }
    }

    private static void ReleaseLeases(SafeCudaMemoryHandle[] owners, bool[] leases)
    {
        for (int index = owners.Length - 1; index >= 0; --index)
        {
            if (leases[index])
            {
                owners[index].DangerousRelease();
            }
        }
    }
}
