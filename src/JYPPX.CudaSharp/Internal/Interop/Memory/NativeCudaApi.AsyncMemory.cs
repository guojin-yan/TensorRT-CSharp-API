using System;
using JYPPX.CudaSharp.Internal.Handles;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static SafeCudaMemoryHandle AllocateMemoryAsync(int sizeInBytes, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_alloc_async((UIntPtr)sizeInBytes, stream, out SafeCudaMemoryHandle handle));
        return handle;
    }

    public static void FreeMemoryAsync(SafeCudaMemoryHandle memory, SafeCudaStreamHandle stream)
    {
        if (memory == null)
        {
            throw new ArgumentNullException(nameof(memory));
        }

        if (memory.IsInvalid)
        {
            return;
        }

        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memory_free_async(memory.DangerousGetHandle(), stream));
        memory.SetHandleAsInvalid();
    }

}
