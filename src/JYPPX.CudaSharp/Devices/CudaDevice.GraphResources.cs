using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public static partial class CudaDevice
{
    /// <summary>
    /// Gets a CUDA graph-memory allocator attribute for a device.
    /// 获取指定设备的 CUDA graph memory 分配器属性。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <param name="attribute">The graph-memory attribute to query. 要查询的 graph memory 属性。</param>
    /// <returns>The attribute value in bytes. 以字节表示的属性值。</returns>
    public static ulong GetGraphMemoryAttribute(int ordinal, CudaGraphMemoryAttribute attribute)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetDeviceGraphMemoryAttribute(ordinal, attribute);
    }

    /// <summary>
    /// Gets all CUDA graph-memory allocator counters for a device.
    /// 获取指定设备的全部 CUDA graph memory 分配器计数。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>A graph-memory allocator snapshot. Graph memory 分配器快照。</returns>
    public static CudaDeviceGraphMemoryInfo GetGraphMemoryInfo(int ordinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetDeviceGraphMemoryInfo(ordinal);
    }

    /// <summary>
    /// Gets a pointer-free copied summary of CUDA graph-memory allocator counters for a device.
    /// 获取指定设备 CUDA graph memory 分配器计数的无指针复制型摘要。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>A graph-memory copied readonly summary. Graph memory 复制型只读摘要。</returns>
    /// <remarks>
    /// This is a convenience wrapper over <see cref="GetGraphMemoryInfo(int)"/> and
    /// <see cref="CudaDeviceGraphMemoryInfo.ToSummary"/>. It is diagnostic evidence only and cannot
    /// promote runtime proof.
    /// 这是 <see cref="GetGraphMemoryInfo(int)"/> 与 <see cref="CudaDeviceGraphMemoryInfo.ToSummary"/>
    /// 的便利封装；它只是诊断证据，不能晋级 runtime proof。
    /// </remarks>
    public static CudaDeviceGraphMemorySummary GetGraphMemorySummary(int ordinal)
    {
        return GetGraphMemoryInfo(ordinal).ToSummary();
    }

    /// <summary>
    /// Gets CUDA graph-memory allocator counters for the current device.
    /// 获取当前设备的 CUDA graph memory 分配器计数。
    /// </summary>
    public static CudaDeviceGraphMemoryInfo CurrentGraphMemoryInfo => GetGraphMemoryInfo(Current);

    /// <summary>
    /// Gets a pointer-free copied graph-memory allocator summary for the current device.
    /// 获取当前设备 CUDA graph memory 分配器的无指针复制型摘要。
    /// </summary>
    public static CudaDeviceGraphMemorySummary CurrentGraphMemorySummary => GetGraphMemorySummary(Current);

    /// <summary>
    /// Gets a pointer-free CUDA 13 device-resource snapshot.
    /// 获取 CUDA 13 设备资源的无指针快照。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <param name="resourceType">The resource union variant to query. 要查询的资源 union 变体。</param>
    /// <returns>A copied resource snapshot. 复制后的资源快照。</returns>
    /// <remarks>
    /// This query does not expose the internal resource chain or enable green-context creation.
    /// 此查询不暴露内部资源链，也不启用 green context 创建。
    /// </remarks>
    public static CudaDevResourceSnapshot GetDevResourceSnapshot(int ordinal, CudaDevResourceType resourceType)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetDeviceDevResourceSnapshot(ordinal, resourceType);
    }

    /// <summary>Gets a pointer-free CUDA 13 device-resource snapshot for the current device. 获取当前设备 CUDA 13 设备资源的无指针快照。</summary>
    public static CudaDevResourceSnapshot CurrentDevResourceSnapshot(CudaDevResourceType resourceType) =>
        GetDevResourceSnapshot(Current, resourceType);

    /// <summary>
    /// Trims graph-memory allocations cached by CUDA for a device.
    /// 裁剪 CUDA 为指定设备缓存的 graph memory 分配。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    public static void TrimGraphMemory(int ordinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.TrimDeviceGraphMemory(ordinal);
    }

    /// <summary>
    /// Resets one CUDA graph-memory high-watermark counter to zero.
    /// 将一个 CUDA graph memory 峰值计数器重置为零。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <param name="attribute">The high-watermark attribute to reset. 要重置的峰值属性。</param>
    public static void ResetGraphMemoryHighWatermark(int ordinal, CudaGraphMemoryAttribute attribute)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.ResetDeviceGraphMemoryHighWatermark(ordinal, attribute);
    }

    /// <summary>
    /// Resets both CUDA graph-memory high-watermark counters to zero.
    /// 将两个 CUDA graph memory 峰值计数器都重置为零。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    public static void ResetGraphMemoryHighWatermarks(int ordinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.ResetDeviceGraphMemoryHighWatermarks(ordinal);
    }

}
