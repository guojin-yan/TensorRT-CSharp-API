using System;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Identifies a CUDA device memory pool without exposing the raw native handle to ordinary users.
/// 标识一个 CUDA 设备内存池，同时避免向普通用户暴露原生句柄。
/// </summary>
public readonly struct CudaMemoryPool
{
    private readonly ulong _handle;

    internal CudaMemoryPool(ulong handle, int deviceOrdinal)
    {
        if (handle == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(handle));
        }

        _handle = handle;
        DeviceOrdinal = deviceOrdinal;
    }

    /// <summary>
    /// Gets the CUDA device ordinal that owns this memory pool.
    /// 获取拥有当前内存池的 CUDA 设备序号。
    /// </summary>
    public int DeviceOrdinal { get; }

    internal ulong Handle => _handle;

    /// <summary>
    /// Creates a standalone CUDA memory pool owned by the returned managed object.
    /// 创建一个由返回托管对象拥有并负责释放的独立 CUDA 内存池。
    /// </summary>
    /// <param name="deviceOrdinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>An owned memory-pool wrapper that must be disposed. 需要释放的 owned 内存池封装。</returns>
    public static CudaOwnedMemoryPool Create(int deviceOrdinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        ulong handle = NativeCudaApi.CreateMemoryPoolHandle(deviceOrdinal);
        return new CudaOwnedMemoryPool(new CudaMemoryPool(handle, deviceOrdinal));
    }

    /// <summary>
    /// Gets the default CUDA memory pool for a device.
    /// 获取指定设备的默认 CUDA 内存池。
    /// </summary>
    /// <param name="deviceOrdinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>The default memory pool wrapper. 默认内存池封装。</returns>
    public static CudaMemoryPool GetDefault(int deviceOrdinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        ulong handle = NativeCudaApi.GetDefaultMemoryPoolHandle(deviceOrdinal);
        return new CudaMemoryPool(handle, deviceOrdinal);
    }

    /// <summary>
    /// Gets the current CUDA memory pool for a device.
    /// 获取指定设备当前使用的 CUDA 内存池。
    /// </summary>
    /// <param name="deviceOrdinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>The current memory pool wrapper. 当前内存池封装。</returns>
    public static CudaMemoryPool GetCurrent(int deviceOrdinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        ulong handle = NativeCudaApi.GetCurrentMemoryPoolHandle(deviceOrdinal);
        return new CudaMemoryPool(handle, deviceOrdinal);
    }

    /// <summary>
    /// Makes this memory pool the current pool for its owning device.
    /// 将当前内存池设置为其所属 CUDA 设备的当前内存池。
    /// </summary>
    public void MakeCurrent()
    {
        NativeCudaApi.SetCurrentMemoryPoolHandle(DeviceOrdinal, _handle);
    }

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

    /// <summary>
    /// Sets access permissions for this memory pool from another CUDA device.
    /// 设置另一个 CUDA 设备访问当前 memory pool 的权限。
    /// </summary>
    /// <param name="deviceOrdinal">The device ordinal whose access should be updated. 需要更新访问权限的设备序号。</param>
    /// <param name="flags">The requested access flags. 目标访问权限标志。</param>
    /// <remarks>
    /// This maps to CUDA's <c>cudaMemPoolSetAccess</c> safe subset with a device-location descriptor.
    /// 该方法映射到 CUDA <c>cudaMemPoolSetAccess</c> 的设备位置描述符安全子集。
    /// </remarks>
    public void SetAccess(int deviceOrdinal, CudaMemoryPoolAccessFlags flags)
    {
        NativeCudaApi.SetMemoryPoolAccess(_handle, deviceOrdinal, (uint)flags);
    }

    /// <summary>
    /// Gets access permissions for this memory pool from a CUDA device.
    /// 查询指定 CUDA 设备访问当前 memory pool 的权限。
    /// </summary>
    /// <param name="deviceOrdinal">The device ordinal to query. 要查询的设备序号。</param>
    /// <returns>The access flags reported by CUDA. CUDA 返回的访问权限标志。</returns>
    public CudaMemoryPoolAccessFlags GetAccess(int deviceOrdinal)
    {
        return (CudaMemoryPoolAccessFlags)NativeCudaApi.GetMemoryPoolAccess(_handle, deviceOrdinal);
    }

    /// <summary>
    /// Gets a memory-pool attribute value.
    /// 获取内存池属性值。
    /// </summary>
    /// <param name="attribute">The memory-pool attribute to query. 要查询的内存池属性。</param>
    /// <returns>The attribute value. 属性值。</returns>
    public long GetAttribute(CudaMemoryPoolAttribute attribute)
    {
        return NativeCudaApi.GetMemoryPoolAttribute(_handle, (int)attribute);
    }

    /// <summary>
    /// Sets a memory-pool attribute value.
    /// 设置内存池属性值。
    /// </summary>
    /// <param name="attribute">The memory-pool attribute to update. 要更新的内存池属性。</param>
    /// <param name="value">The new attribute value. 新属性值。</param>
    public void SetAttribute(CudaMemoryPoolAttribute attribute, long value)
    {
        NativeCudaApi.SetMemoryPoolAttribute(_handle, (int)attribute, value);
    }

    /// <summary>
    /// Gets or sets the release threshold in bytes.
    /// 获取或设置内存池释放阈值，单位为字节。
    /// </summary>
    public long ReleaseThresholdBytes
    {
        get => GetAttribute(CudaMemoryPoolAttribute.ReleaseThreshold);
        set => SetAttribute(CudaMemoryPoolAttribute.ReleaseThreshold, value);
    }

    /// <summary>
    /// Gets the current number of reserved bytes reported by CUDA.
    /// 获取 CUDA 报告的当前已保留字节数。
    /// </summary>
    public long ReservedMemoryCurrentBytes => GetAttribute(CudaMemoryPoolAttribute.ReservedMemoryCurrent);

    /// <summary>
    /// Gets the high-water reserved byte count reported by CUDA.
    /// 获取 CUDA 报告的已保留字节数峰值。
    /// </summary>
    public long ReservedMemoryHighBytes => GetAttribute(CudaMemoryPoolAttribute.ReservedMemoryHigh);

    /// <summary>
    /// Gets the current number of used bytes reported by CUDA.
    /// 获取 CUDA 报告的当前已使用字节数。
    /// </summary>
    public long UsedMemoryCurrentBytes => GetAttribute(CudaMemoryPoolAttribute.UsedMemoryCurrent);

    /// <summary>
    /// Gets the high-water used byte count reported by CUDA.
    /// 获取 CUDA 报告的已使用字节数峰值。
    /// </summary>
    public long UsedMemoryHighBytes => GetAttribute(CudaMemoryPoolAttribute.UsedMemoryHigh);

    /// <summary>
    /// Resets CUDA's reserved-memory high-water counter for this pool.
    /// 重置当前内存池的 CUDA 已保留内存峰值计数。
    /// </summary>
    public void ResetReservedMemoryHigh()
    {
        SetAttribute(CudaMemoryPoolAttribute.ReservedMemoryHigh, 0);
    }

    /// <summary>
    /// Resets CUDA's used-memory high-water counter for this pool.
    /// 重置当前内存池的 CUDA 已使用内存峰值计数。
    /// </summary>
    public void ResetUsedMemoryHigh()
    {
        SetAttribute(CudaMemoryPoolAttribute.UsedMemoryHigh, 0);
    }
}

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

/// <summary>
/// Identifies CUDA memory-pool attributes used by deployment memory management.
/// 标识部署内存管理中常用的 CUDA 内存池属性。
/// </summary>
public enum CudaMemoryPoolAttribute
{
    /// <summary>
    /// Enables reuse that follows event dependencies.
    /// 启用遵循事件依赖关系的内存复用。
    /// </summary>
    ReuseFollowEventDependencies = 1,

    /// <summary>
    /// Allows opportunistic allocation reuse.
    /// 允许机会性的分配复用。
    /// </summary>
    ReuseAllowOpportunistic = 2,

    /// <summary>
    /// Allows CUDA to insert internal dependencies for reuse.
    /// 允许 CUDA 为复用插入内部依赖。
    /// </summary>
    ReuseAllowInternalDependencies = 3,

    /// <summary>
    /// Controls the amount of memory retained before release.
    /// 控制释放前保留的缓存内存量。
    /// </summary>
    ReleaseThreshold = 4,

    /// <summary>
    /// Reports current reserved memory.
    /// 报告当前已保留的内存。
    /// </summary>
    ReservedMemoryCurrent = 5,

    /// <summary>
    /// Reports high-water reserved memory.
    /// 报告已保留内存的峰值。
    /// </summary>
    ReservedMemoryHigh = 6,

    /// <summary>
    /// Reports current used memory.
    /// 报告当前已使用的内存。
    /// </summary>
    UsedMemoryCurrent = 7,

    /// <summary>
    /// Reports high-water used memory.
    /// 报告已使用内存的峰值。
    /// </summary>
    UsedMemoryHigh = 8
}

/// <summary>
/// Describes CUDA memory-pool access permissions for a device location.
/// 描述某个 CUDA 设备位置对 memory pool 的访问权限。
/// </summary>
[Flags]
public enum CudaMemoryPoolAccessFlags : uint
{
    /// <summary>
    /// No access is allowed.
    /// 不允许访问。
    /// </summary>
    None = 0,

    /// <summary>
    /// Read-only access is allowed.
    /// 允许只读访问。
    /// </summary>
    Read = 1,

    /// <summary>
    /// Read-write access is allowed.
    /// 允许读写访问。
    /// </summary>
    ReadWrite = 3
}
