using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public partial class CudaMemory
{
    /// <summary>
    /// Asynchronously allocates device memory on a CUDA stream.
    /// 在 CUDA stream 上异步分配设备内存。
    /// </summary>
    /// <param name="sizeInBytes">The allocation size in bytes. 分配大小，单位为字节。</param>
    /// <param name="stream">The CUDA stream that orders the allocation. 用于排序分配操作的 CUDA stream。</param>
    /// <returns>A managed device-memory wrapper. 设备内存的托管封装。</returns>
    public static CudaMemory AllocateAsync(int sizeInBytes, CudaStream stream)
    {
        if (sizeInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInBytes));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeBridgeLoader.EnsureInitialized();
        return new CudaMemory(NativeCudaApi.AllocateMemoryAsync(sizeInBytes, stream.Handle));
    }

    /// <summary>
    /// Asynchronously allocates device memory from a specific CUDA memory pool.
    /// 从指定 CUDA memory pool 中异步分配设备内存。
    /// </summary>
    /// <param name="sizeInBytes">The allocation size in bytes. 分配大小，单位为字节。</param>
    /// <param name="memoryPool">The CUDA memory pool used for allocation. 用于分配的 CUDA memory pool。</param>
    /// <param name="stream">The CUDA stream that orders the allocation. 用于排序分配操作的 CUDA stream。</param>
    /// <returns>A managed CUDA memory wrapper. CUDA 设备内存托管封装。</returns>
    public static CudaMemory AllocateFromPoolAsync(int sizeInBytes, CudaMemoryPool memoryPool, CudaStream stream)
    {
        if (sizeInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInBytes));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeBridgeLoader.EnsureInitialized();
        return new CudaMemory(NativeCudaApi.AllocateMemoryFromPoolAsync(sizeInBytes, memoryPool.Handle, stream.Handle));
    }

}
