using System;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public readonly partial struct CudaMemoryPool
{
    /// <summary>
    /// Asynchronously allocates device memory from this CUDA memory pool.
    /// 从当前 CUDA memory pool 中异步分配设备内存。
    /// </summary>
    /// <param name="sizeInBytes">The allocation size in bytes. 分配大小，单位为字节。</param>
    /// <param name="stream">The CUDA stream that orders the allocation. 用于排序分配操作的 CUDA stream。</param>
    /// <returns>A managed CUDA memory allocation. CUDA 设备内存托管封装。</returns>
    public CudaMemory AllocateAsync(int sizeInBytes, CudaStream stream)
    {
        return CudaMemory.AllocateFromPoolAsync(sizeInBytes, this, stream);
    }

    /// <summary>
    /// Trims unused allocations in the CUDA memory pool.
    /// 裁剪 CUDA 内存池中未使用的缓存分配。
    /// </summary>
    /// <param name="minBytesToKeep">The minimum number of bytes CUDA should keep cached. CUDA 至少应保留的缓存字节数。</param>
    public void TrimTo(ulong minBytesToKeep)
    {
        NativeCudaApi.TrimMemoryPoolTo(_handle, minBytesToKeep);
    }

}
