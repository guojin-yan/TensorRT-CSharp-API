using System;
using System.Runtime.InteropServices;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.Shared.Interop;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static void PrefetchMemoryRangeAsync(SafeCudaMemoryHandle memory, int offset, int count, int destinationDevice, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_prefetch_range_async(memory, (UIntPtr)offset, (UIntPtr)count, destinationDevice, stream));
    }

    public static void AdviseMemoryRange(SafeCudaMemoryHandle memory, int offset, int count, int advice, int device)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_advise_range(memory, (UIntPtr)offset, (UIntPtr)count, advice, device));
    }

    public static void PrefetchManagedMemoryLocationRangeAsync(
        SafeCudaMemoryHandle memory,
        int offset,
        int count,
        int locationType,
        int locationId,
        SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(
            NativeMethodsCuda.jyppx_cuda_managed_memory_prefetch_location_range_async_safe(
                memory,
                (UIntPtr)offset,
                (UIntPtr)count,
                locationType,
                locationId,
                stream));
    }

    public static void AdviseManagedMemoryLocationRange(
        SafeCudaMemoryHandle memory,
        int offset,
        int count,
        int advice,
        int locationType,
        int locationId)
    {
        CudaNativeStatus.ThrowIfFailed(
            NativeMethodsCuda.jyppx_cuda_managed_memory_advise_location_range_safe(
                memory,
                (UIntPtr)offset,
                (UIntPtr)count,
                advice,
                locationType,
                locationId));
    }

    public static int GetMemoryRangeAttribute(SafeCudaMemoryHandle memory, int offset, int count, int attribute)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_range_get_attribute(memory, (UIntPtr)offset, (UIntPtr)count, attribute, out int value));
        return value;
    }

    public static NativeCudaMemRangeAttributeValue[] GetMemoryRangeAttributes(SafeCudaMemoryHandle memory, int offset, int count, int[] attributes)
    {
        NativeCudaMemRangeAttributeValue[] values = new NativeCudaMemRangeAttributeValue[attributes.Length];
        GCHandle attributesHandle = GCHandle.Alloc(attributes, GCHandleType.Pinned);
        GCHandle valuesHandle = GCHandle.Alloc(values, GCHandleType.Pinned);
        try
        {
            CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_range_get_attributes(
                memory,
                (UIntPtr)offset,
                (UIntPtr)count,
                attributesHandle.AddrOfPinnedObject(),
                (UIntPtr)attributes.Length,
                valuesHandle.AddrOfPinnedObject()));
            return values;
        }
        finally
        {
            valuesHandle.Free();
            attributesHandle.Free();
        }
    }

    public static int GetMemoryRangeAccessedByDeviceCount(SafeCudaMemoryHandle memory, int offset, int count)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_range_get_accessed_by_count(memory, (UIntPtr)offset, (UIntPtr)count, out UIntPtr deviceCount));
        ulong value = deviceCount.ToUInt64();
        if (value > int.MaxValue)
        {
            throw new CudaException(BridgeStatusCode.InvalidState, BridgeErrorCategory.Cuda, "CUDA returned too many AccessedBy devices for a managed array.");
        }

        return (int)value;
    }

    public static int[] GetMemoryRangeAccessedByDevices(SafeCudaMemoryHandle memory, int offset, int count)
    {
        int deviceCount = GetMemoryRangeAccessedByDeviceCount(memory, offset, count);
        if (deviceCount == 0)
        {
            return Array.Empty<int>();
        }

        return CopyMemoryRangeAccessedByDevices(memory, offset, count, deviceCount);
    }

    private static int[] CopyMemoryRangeAccessedByDevices(SafeCudaMemoryHandle memory, int offset, int count, int deviceCount)
    {
        int[] devices = new int[deviceCount];
        while (true)
        {
            GCHandle devicesHandle = GCHandle.Alloc(devices, GCHandleType.Pinned);
            try
            {
                UIntPtr requiredCount;
                BridgeStatusCode status = NativeMethodsCuda.jyppx_cuda_memory_range_copy_accessed_by_devices(
                    memory,
                    (UIntPtr)offset,
                    (UIntPtr)count,
                    devicesHandle.AddrOfPinnedObject(),
                    (UIntPtr)devices.Length,
                    out requiredCount);

                ulong requiredValue = requiredCount.ToUInt64();
                if (status == BridgeStatusCode.BufferTooSmall && requiredValue > (ulong)devices.Length && requiredValue <= (ulong)int.MaxValue)
                {
                    devices = new int[(int)requiredValue];
                    continue;
                }

                CudaNativeStatus.ThrowIfFailed(status);
                if (requiredValue < (ulong)devices.Length)
                {
                    Array.Resize(ref devices, (int)requiredValue);
                }

                return devices;
            }
            finally
            {
                devicesHandle.Free();
            }
        }
    }
}
