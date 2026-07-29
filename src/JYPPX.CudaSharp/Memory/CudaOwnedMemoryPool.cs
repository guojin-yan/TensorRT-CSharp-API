using System;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Owns a CUDA memory pool created through the bridge and releases it during disposal.
/// 拥有通过桥接层创建的 CUDA 内存池，并在释放时销毁它。
/// </summary>
public sealed class CudaOwnedMemoryPool : IDisposable
{
    private CudaMemoryPool _pool;
    private bool _disposed;

    internal CudaOwnedMemoryPool(CudaMemoryPool pool)
    {
        _pool = pool;
    }

    /// <summary>
    /// Gets a non-owning memory-pool view for querying attributes or making the pool current.
    /// 获取一个非拥有视图，用于查询属性或将该池设为当前池。
    /// </summary>
    public CudaMemoryPool Pool
    {
        get
        {
            ThrowIfDisposed();
            return _pool;
        }
    }

    /// <summary>
    /// Gets the CUDA device ordinal that owns this pool.
    /// 获取拥有当前内存池的 CUDA 设备序号。
    /// </summary>
    public int DeviceOrdinal => Pool.DeviceOrdinal;

    /// <summary>
    /// Makes this owned pool the current pool for its CUDA device.
    /// 将当前 owned 内存池设置为其 CUDA 设备的当前池。
    /// </summary>
    public void MakeCurrent()
    {
        Pool.MakeCurrent();
    }

    /// <summary>
    /// Asynchronously allocates device memory from this owned CUDA memory pool.
    /// 从当前 owned CUDA memory pool 中异步分配设备内存。
    /// </summary>
    /// <param name="sizeInBytes">The allocation size in bytes. 分配大小，单位为字节。</param>
    /// <param name="stream">The CUDA stream that orders the allocation. 用于排序分配操作的 CUDA stream。</param>
    /// <returns>A managed CUDA memory allocation. CUDA 设备内存托管封装。</returns>
    public CudaMemory AllocateAsync(int sizeInBytes, CudaStream stream)
    {
        return Pool.AllocateAsync(sizeInBytes, stream);
    }

    /// <summary>
    /// Trims unused allocations in this owned CUDA memory pool.
    /// 裁剪当前 owned CUDA 内存池中未使用的缓存分配。
    /// </summary>
    /// <param name="minBytesToKeep">The minimum number of cached bytes to keep. 至少保留的缓存字节数。</param>
    public void TrimTo(ulong minBytesToKeep)
    {
        Pool.TrimTo(minBytesToKeep);
    }

    /// <summary>
    /// Releases the owned CUDA memory pool if it has not already been released.
    /// 如果尚未释放，则销毁当前 owned CUDA 内存池。
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        NativeCudaApi.DestroyMemoryPoolHandle(_pool.Handle);
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CudaOwnedMemoryPool));
        }
    }
}
