using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Provides current-device helpers over the CUDA runtime.
/// 提供 CUDA runtime 当前设备相关 helper。
/// </summary>
public static partial class CudaDevice
{
    /// <summary>
    /// Gets the CUDA runtime version reported by the bridge.
    /// 获取桥接层报告的 CUDA runtime 版本。
    /// </summary>
    public static int RuntimeVersion
    {
        get
        {
            NativeBridgeLoader.EnsureInitialized();
            return NativeCudaApi.GetRuntimeVersion();
        }
    }

    /// <summary>
    /// Gets the CUDA driver version reported by the bridge.
    /// 获取桥接层报告的 CUDA driver 版本。
    /// </summary>
    public static int DriverVersion
    {
        get
        {
            NativeBridgeLoader.EnsureInitialized();
            return NativeCudaApi.GetDriverVersion();
        }
    }

    /// <summary>
    /// Gets the number of CUDA devices visible to the current process.
    /// 获取当前进程可见的 CUDA 设备数量。
    /// </summary>
    public static int Count
    {
        get
        {
            NativeBridgeLoader.EnsureInitialized();
            return NativeCudaApi.GetDeviceCount();
        }
    }

    /// <summary>
    /// Gets the current CUDA device ordinal for the process.
    /// 获取当前进程的 CUDA 当前设备序号。
    /// </summary>
    public static int Current
    {
        get
        {
            NativeBridgeLoader.EnsureInitialized();
            return NativeCudaApi.GetCurrentDevice();
        }
    }

    /// <summary>
    /// Sets the current CUDA device for the process.
    /// 为当前进程设置 CUDA 当前设备。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    public static void SetCurrent(int ordinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.SetDevice(ordinal);
    }

    /// <summary>
    /// Temporarily switches the current CUDA device and restores the previous device on dispose.
    /// 临时切换当前 CUDA 设备，并在释放时恢复之前的设备。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal to activate. 要激活的 CUDA 设备序号。</param>
    /// <returns>A disposable scope that restores the previous device. 可恢复之前设备的可释放作用域。</returns>
    public static CudaDeviceScope Use(int ordinal)
    {
        return new CudaDeviceScope(ordinal);
    }

    /// <summary>
    /// Gets a basic CUDA device information snapshot.
    /// 获取基础 CUDA 设备信息快照。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>The mapped CUDA device information. 已映射的 CUDA 设备信息。</returns>
    public static CudaDeviceInfo GetInfo(int ordinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        return Internal.CudaInfoMapper.ToManaged(NativeCudaApi.GetDeviceInfo(ordinal));
    }

    /// <summary>
    /// Gets a deployment-oriented property snapshot for a CUDA device.
    /// 获取面向模型部署的 CUDA 设备属性快照。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>A high-level CUDA device property snapshot. 高层 CUDA 设备属性快照。</returns>
    public static CudaDeviceProperties GetProperties(int ordinal)
    {
        CudaDeviceInfo info = GetInfo(ordinal);
        return new CudaDeviceProperties(
            info,
            new[]
            {
                TryGetAttribute(ordinal, CudaDeviceAttribute.MaxBlockDimX).GetValueOrDefault(),
                TryGetAttribute(ordinal, CudaDeviceAttribute.MaxBlockDimY).GetValueOrDefault(),
                TryGetAttribute(ordinal, CudaDeviceAttribute.MaxBlockDimZ).GetValueOrDefault()
            },
            new[]
            {
                TryGetAttribute(ordinal, CudaDeviceAttribute.MaxGridDimX).GetValueOrDefault(),
                TryGetAttribute(ordinal, CudaDeviceAttribute.MaxGridDimY).GetValueOrDefault(),
                TryGetAttribute(ordinal, CudaDeviceAttribute.MaxGridDimZ).GetValueOrDefault()
            },
            TryGetAttribute(ordinal, CudaDeviceAttribute.ClockRate),
            TryGetAttribute(ordinal, CudaDeviceAttribute.MemoryClockRate),
            TryGetAttribute(ordinal, CudaDeviceAttribute.GlobalMemoryBusWidth),
            TryGetAttribute(ordinal, CudaDeviceAttribute.L2CacheSize),
            TryGetAttribute(ordinal, CudaDeviceAttribute.MaxThreadsPerMultiProcessor),
            TryGetAttribute(ordinal, CudaDeviceAttribute.AsyncEngineCount),
            TryGetBooleanAttribute(ordinal, CudaDeviceAttribute.ConcurrentKernels),
            TryGetBooleanAttribute(ordinal, CudaDeviceAttribute.UnifiedAddressing),
            TryGetBooleanAttribute(ordinal, CudaDeviceAttribute.ManagedMemory),
            TryGetBooleanAttribute(ordinal, CudaDeviceAttribute.ConcurrentManagedAccess),
            TryGetBooleanAttribute(ordinal, CudaDeviceAttribute.StreamPrioritiesSupported),
            TryGetBooleanAttribute(ordinal, CudaDeviceAttribute.ComputePreemptionSupported),
            TryGetBooleanAttribute(ordinal, CudaDeviceAttribute.HostRegisterSupported),
            TryGetBooleanAttribute(ordinal, CudaDeviceAttribute.DirectManagedMemoryAccessFromHost));
    }

    /// <summary>
    /// Gets a deployment-oriented property snapshot for the current CUDA device.
    /// 获取当前 CUDA 设备面向模型部署的属性快照。
    /// </summary>
    public static CudaDeviceProperties CurrentProperties => GetProperties(Current);

}
