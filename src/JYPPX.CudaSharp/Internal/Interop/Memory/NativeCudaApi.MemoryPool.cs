using System;
using JYPPX.CudaSharp.Internal.Handles;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static SafeCudaMemoryHandle AllocateMemoryFromPoolAsync(int sizeInBytes, ulong poolHandle, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_alloc_from_pool_async((UIntPtr)sizeInBytes, poolHandle, stream, out SafeCudaMemoryHandle handle));
        return handle;
    }

    public static void CopyDefault(SafeCudaMemoryHandle destination, SafeCudaMemoryHandle source, int size)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_copy_default(destination, source, (UIntPtr)size));
    }

    public static void CopyDefaultAsync(SafeCudaMemoryHandle destination, SafeCudaMemoryHandle source, int size, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_copy_default_async(destination, source, (UIntPtr)size, stream));
    }

    public static void SetMemoryPoolAccess(ulong poolHandle, int device, uint flags)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_pool_set_access(poolHandle, device, flags));
    }

    public static uint GetMemoryPoolAccess(ulong poolHandle, int device)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_pool_get_access(poolHandle, device, out uint flags));
        return flags;
    }
}
