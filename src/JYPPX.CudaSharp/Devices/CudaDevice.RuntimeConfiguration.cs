using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public static partial class CudaDevice
{
    /// <summary>
    /// Gets a raw CUDA device attribute value.
    /// 获取原始 CUDA 设备属性值。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <param name="attribute">The device attribute to query. 要查询的设备属性。</param>
    /// <returns>The raw CUDA attribute value. 原始 CUDA 属性值。</returns>
    public static int GetAttribute(int ordinal, CudaDeviceAttribute attribute)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetDeviceAttribute(ordinal, (int)attribute);
    }

    /// <summary>
    /// Gets a CUDA runtime device limit for the current process.
    /// 获取当前进程的 CUDA runtime device limit。
    /// </summary>
    /// <param name="limit">The CUDA device limit to query. 要查询的 CUDA device limit。</param>
    /// <returns>The configured limit value in bytes or CUDA-defined units. 以字节或 CUDA 定义单位表示的 limit 值。</returns>
    public static ulong GetLimit(CudaDeviceLimit limit)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetDeviceLimit((int)limit);
    }

    /// <summary>
    /// Sets a CUDA runtime device limit for the current process.
    /// 设置当前进程的 CUDA runtime device limit。
    /// </summary>
    /// <param name="limit">The CUDA device limit to update. 要更新的 CUDA device limit。</param>
    /// <param name="value">The new limit value in bytes or CUDA-defined units. 以字节或 CUDA 定义单位表示的新 limit 值。</param>
    public static void SetLimit(CudaDeviceLimit limit, ulong value)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.SetDeviceLimit((int)limit, value);
    }

    /// <summary>
    /// Gets the current process-wide CUDA device cache preference.
    /// 获取当前进程级 CUDA device cache 偏好。
    /// </summary>
    public static CudaFunctionCachePreference CacheConfig
    {
        get
        {
            NativeBridgeLoader.EnsureInitialized();
            return (CudaFunctionCachePreference)NativeCudaApi.GetDeviceCacheConfig();
        }

        set
        {
            NativeBridgeLoader.EnsureInitialized();
            NativeCudaApi.SetDeviceCacheConfig((int)value);
        }
    }

    /// <summary>
    /// Gets or sets the CUDA shared-memory bank configuration for compatible devices.
    /// 获取或设置兼容设备上的 CUDA shared-memory bank 配置。
    /// </summary>
    public static CudaSharedMemoryConfig SharedMemoryConfig
    {
        get
        {
            NativeBridgeLoader.EnsureInitialized();
            return (CudaSharedMemoryConfig)NativeCudaApi.GetSharedMemoryConfig();
        }

        set
        {
            NativeBridgeLoader.EnsureInitialized();
            NativeCudaApi.SetSharedMemoryConfig((int)value);
        }
    }

    /// <summary>
    /// Gets or sets CUDA runtime device flags used by the current process.
    /// 获取或设置当前进程使用的 CUDA runtime device flags。
    /// </summary>
    /// <remarks>
    /// Set this before creating a CUDA context when you need mapped host memory or a specific scheduling policy.
    /// 如果需要 mapped host memory 或特定调度策略，应在创建 CUDA context 前设置。
    /// </remarks>
    public static CudaDeviceRuntimeFlags RuntimeFlags
    {
        get
        {
            NativeBridgeLoader.EnsureInitialized();
            return (CudaDeviceRuntimeFlags)NativeCudaApi.GetDeviceFlags();
        }

        set
        {
            NativeBridgeLoader.EnsureInitialized();
            NativeCudaApi.SetDeviceFlags((uint)value);
        }
    }

}
