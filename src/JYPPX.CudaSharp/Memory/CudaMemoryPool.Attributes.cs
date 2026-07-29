using System;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public readonly partial struct CudaMemoryPool
{
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
