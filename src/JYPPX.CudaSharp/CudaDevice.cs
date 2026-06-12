using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Provides current-device helpers over the CUDA runtime.
/// 提供 CUDA runtime 当前设备相关 helper。
/// </summary>
public static class CudaDevice
{
    public static int RuntimeVersion
    {
        get
        {
            NativeBridgeLoader.EnsureInitialized();
            return NativeCudaApi.GetRuntimeVersion();
        }
    }

    public static int DriverVersion
    {
        get
        {
            NativeBridgeLoader.EnsureInitialized();
            return NativeCudaApi.GetDriverVersion();
        }
    }

    public static int Count
    {
        get
        {
            NativeBridgeLoader.EnsureInitialized();
            return NativeCudaApi.GetDeviceCount();
        }
    }

    public static int Current
    {
        get
        {
            NativeBridgeLoader.EnsureInitialized();
            return NativeCudaApi.GetCurrentDevice();
        }
    }

    public static void SetCurrent(int ordinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.SetDevice(ordinal);
    }

    public static CudaDeviceScope Use(int ordinal)
    {
        return new CudaDeviceScope(ordinal);
    }

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
    /// Gets CUDA graph-memory allocator counters for the current device.
    /// 获取当前设备的 CUDA graph memory 分配器计数。
    /// </summary>
    public static CudaDeviceGraphMemoryInfo CurrentGraphMemoryInfo => GetGraphMemoryInfo(Current);

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

    /// <summary>
    /// Gets the PCI bus id string for a CUDA device.
    /// 获取 CUDA 设备的 PCI bus id 字符串。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>The PCI bus id reported by CUDA, such as <c>0000:65:00.0</c>. CUDA 返回的 PCI bus id。</returns>
    public static string GetPciBusId(int ordinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetDevicePciBusId(ordinal);
    }

    /// <summary>
    /// Resolves a CUDA device ordinal from a PCI bus id string.
    /// 根据 PCI bus id 字符串解析 CUDA 设备序号。
    /// </summary>
    /// <param name="pciBusId">The PCI bus id reported by CUDA. CUDA 返回的 PCI bus id。</param>
    /// <returns>The CUDA device ordinal. CUDA 设备序号。</returns>
    public static int GetByPciBusId(string pciBusId)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetDeviceByPciBusId(pciBusId);
    }

    public static bool GetBooleanAttribute(int ordinal, CudaDeviceAttribute attribute)
    {
        return GetAttribute(ordinal, attribute) != 0;
    }

    /// <summary>
    /// Tries to read a CUDA device attribute without failing the whole capability snapshot.
    /// 尝试读取 CUDA device attribute，不会因为单个 attribute 不可用而中断整个能力快照。
    /// </summary>
    public static int? TryGetAttribute(int ordinal, CudaDeviceAttribute attribute)
    {
        try
        {
            return GetAttribute(ordinal, attribute);
        }
        catch (CudaException)
        {
            return null;
        }
    }

    /// <summary>
    /// Tries to read a CUDA device attribute as a boolean capability.
    /// 尝试把 CUDA device attribute 读取为布尔能力。
    /// </summary>
    public static bool? TryGetBooleanAttribute(int ordinal, CudaDeviceAttribute attribute)
    {
        int? value = TryGetAttribute(ordinal, attribute);
        return value.HasValue ? value.Value != 0 : (bool?)null;
    }

    public static bool CanAccessPeer(int ordinal, int peerOrdinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.CanDeviceAccessPeer(ordinal, peerOrdinal);
    }

    /// <summary>
    /// Gets a CUDA peer-to-peer attribute for a pair of devices.
    /// 获取两个 CUDA 设备之间的 peer-to-peer 属性。
    /// </summary>
    /// <param name="attribute">The CUDA P2P attribute to query. 要查询的 CUDA P2P 属性。</param>
    /// <param name="sourceOrdinal">The source CUDA device ordinal. 源 CUDA 设备序号。</param>
    /// <param name="destinationOrdinal">The destination CUDA device ordinal. 目标 CUDA 设备序号。</param>
    /// <returns>The raw CUDA attribute value. CUDA 返回的原始属性值。</returns>
    public static int GetP2PAttribute(CudaDeviceP2PAttribute attribute, int sourceOrdinal, int destinationOrdinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetDeviceP2PAttribute((int)attribute, sourceOrdinal, destinationOrdinal);
    }

    public static void EnablePeerAccess(int peerOrdinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.EnablePeerAccess(peerOrdinal, 0);
    }

    public static void DisablePeerAccess(int peerOrdinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.DisablePeerAccess(peerOrdinal);
    }

    public static CudaMemoryInfo GetMemoryInfo()
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaMemoryInfo info = NativeCudaApi.GetMemoryInfo();
        return new CudaMemoryInfo(info.FreeBytes, info.TotalBytes);
    }

    /// <summary>
    /// Tries to read CUDA memory information without throwing when the runtime is unavailable.
    /// 在 CUDA runtime 不可用时以诊断字符串返回失败，而不是抛出异常。
    /// </summary>
    /// <param name="memoryInfo">The memory information when the query succeeds. 查询成功时返回的显存信息。</param>
    /// <param name="diagnostic">An empty string on success, or the CUDA bridge diagnostic on failure. 成功时为空字符串，失败时为 CUDA 桥接诊断。</param>
    /// <returns><c>true</c> when memory information was read successfully. 成功读取显存信息时返回 <c>true</c>。</returns>
    public static bool TryGetMemoryInfo(out CudaMemoryInfo? memoryInfo, out string diagnostic)
    {
        try
        {
            memoryInfo = GetMemoryInfo();
            diagnostic = string.Empty;
            return true;
        }
        catch (CudaException exception)
        {
            memoryInfo = null;
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Gets a CUDA memory pressure snapshot for the current device.
    /// 获取当前 CUDA 设备的显存压力快照。
    /// </summary>
    /// <returns>A memory pressure snapshot for deployment diagnostics. 用于部署诊断的显存压力快照。</returns>
    public static CudaMemoryPressureSnapshot GetMemoryPressureSnapshot()
    {
        int ordinal = Current;
        return new CudaMemoryPressureSnapshot(ordinal, GetMemoryInfo(), GetProperties(ordinal));
    }

    public static CudaMemoryPool GetDefaultMemoryPool(int ordinal)
    {
        return CudaMemoryPool.GetDefault(ordinal);
    }

    /// <summary>
    /// Gets the current CUDA memory pool for a device.
    /// 获取指定设备当前使用的 CUDA memory pool。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>The current memory pool wrapper. 当前 memory pool 封装。</returns>
    public static CudaMemoryPool GetCurrentMemoryPool(int ordinal)
    {
        return CudaMemoryPool.GetCurrent(ordinal);
    }

    /// <summary>
    /// Sets the current CUDA memory pool for the memory pool's device.
    /// 为该 memory pool 所属设备设置当前 CUDA memory pool。
    /// </summary>
    /// <param name="memoryPool">The memory pool to make current. 要设置为当前池的 memory pool。</param>
    public static void SetCurrentMemoryPool(CudaMemoryPool memoryPool)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.SetCurrentMemoryPoolHandle(memoryPool.DeviceOrdinal, memoryPool.Handle);
    }

    public static ulong GetTexture1DLinearMaxWidth(CudaChannelFormatDescriptor descriptor, int ordinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetTexture1DLinearMaxWidth(descriptor, ordinal);
    }

    public static void ResetPersistingL2Cache()
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.ResetPersistingL2Cache();
    }

    public static void FlushGpuDirectRdmaWrites(CudaGpuDirectRdmaWritesTarget target, CudaGpuDirectRdmaWritesScope scope)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.FlushGpuDirectRdmaWrites(target, scope);
    }

    public static void Synchronize()
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.SynchronizeDevice();
    }

    public static void Reset()
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.ResetDevice();
    }

    public static int GetLastErrorCode()
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetLastErrorCode();
    }

    public static int PeekAtLastErrorCode()
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.PeekAtLastErrorCode();
    }

    public static string GetErrorName(int errorCode)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetErrorName(errorCode);
    }

    public static string GetErrorString(int errorCode)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetErrorString(errorCode);
    }
}
