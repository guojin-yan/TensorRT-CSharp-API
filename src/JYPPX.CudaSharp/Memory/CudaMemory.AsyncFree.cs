using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public partial class CudaMemory
{
    /// <summary>
    /// Asynchronously frees this allocation on the supplied CUDA stream and invalidates the managed handle.
    /// 在指定 CUDA stream 上异步释放当前分配，并使托管句柄失效。
    /// </summary>
    /// <param name="stream">The CUDA stream that orders the free operation. 用于排序释放操作的 CUDA stream。</param>
    public void FreeAsync(CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        if (IsIpcImported)
        {
            throw new InvalidOperationException(
                "Imported CUDA IPC memory must be disposed synchronously so cudaIpcCloseMemHandle can release the mapping.");
        }

        NativeCudaApi.FreeMemoryAsync(_handle, stream.Handle);
        _handle.MarkReleased();
    }

}
