using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Provides current-device helpers over the CUDA runtime.
/// 提供 CUDA runtime 当前设备相关 helper。
/// </summary>
public static class CudaDevice
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

    /// <summary>
    /// Initializes the primary CUDA context for a device using CUDA 12.0+ <c>cudaInitDevice</c>.
    /// 使用 CUDA 12.0+ <c>cudaInitDevice</c> 初始化指定设备的 primary context。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <param name="deviceFlags">The CUDA runtime device flags to apply during initialization. 初始化时应用的 CUDA runtime device flags。</param>
    /// <param name="flags">Reserved CUDA flags. CUDA 保留 flags，通常为 0。</param>
    /// <remarks>
    /// This wrapper keeps the boundary scalar-only and does not expose CUDA context handles.
    /// 该封装仅暴露标量边界，不向托管层暴露 CUDA context 句柄。
    /// </remarks>
    public static void InitDevice(int ordinal, CudaDeviceRuntimeFlags deviceFlags = CudaDeviceRuntimeFlags.ScheduleAuto, uint flags = 0)
    {
        if (ordinal < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(ordinal), ordinal, "CUDA device ordinal must be greater than or equal to zero.");
        }

        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.InitDevice(ordinal, (uint)deviceFlags, flags);
    }

    /// <summary>
    /// Gets an owner-safe bridge wrapper for a device primary execution context on CUDA 13.0 or later.
    /// 在 CUDA 13.0 或更高版本上获取设备主执行上下文的 owner-safe bridge 包装器。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>A non-destroying primary execution-context wrapper. 不会销毁主上下文的执行上下文包装器。</returns>
    public static CudaPrimaryExecutionContext GetPrimaryExecutionContext(int ordinal)
    {
        if (ordinal < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(ordinal), ordinal, "CUDA device ordinal must be greater than or equal to zero.");
        }

        NativeBridgeLoader.EnsureInitialized();
        return new CudaPrimaryExecutionContext(NativeCudaApi.GetPrimaryExecutionContext(ordinal));
    }

    /// <summary>
    /// Restricts CUDA runtime initialization to a caller-owned list of valid device ordinals using <c>cudaSetValidDevices</c>.
    /// 使用 <c>cudaSetValidDevices</c> 和 caller-owned 的设备序号列表限制 CUDA runtime 可初始化的设备集合。
    /// </summary>
    /// <param name="ordinals">The CUDA device ordinals that may be used by the process. 允许当前进程使用的 CUDA 设备序号。</param>
    /// <remarks>
    /// This method copies the managed list before calling native code. Call it before creating a CUDA context.
    /// 该方法会先复制托管列表再调用 native；应在创建 CUDA context 前调用。
    /// </remarks>
    public static void SetValidDevices(IReadOnlyList<int> ordinals)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.SetValidDevices(CopyDeviceOrdinals(ordinals));
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

    /// <summary>
    /// Gets a CUDA device attribute as a boolean capability.
    /// 将 CUDA 设备属性读取为布尔能力值。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <param name="attribute">The device attribute to query. 要查询的设备属性。</param>
    /// <returns><see langword="true"/> when the attribute is non-zero. 属性非零时返回 <see langword="true"/>。</returns>
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

    /// <summary>
    /// Gets whether one CUDA device can access another through peer access.
    /// 获取一个 CUDA 设备是否可以通过 peer access 访问另一个设备。
    /// </summary>
    /// <param name="ordinal">The source CUDA device ordinal. 源 CUDA 设备序号。</param>
    /// <param name="peerOrdinal">The peer CUDA device ordinal. 对端 CUDA 设备序号。</param>
    /// <returns><see langword="true"/> when peer access is available. 可以进行 peer access 时返回 <see langword="true"/>。</returns>
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

    /// <summary>
    /// Gets native host atomic capabilities for the selected CUDA operations on a device.
    /// 获取指定设备对一组 CUDA atomic operation 的 host atomic 原生能力。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <param name="operations">The CUDA atomic operations to query. 要查询的 CUDA atomic operation 列表。</param>
    /// <returns>One capability bitmask per operation, in the same order. 按输入顺序返回每个 operation 的能力位掩码。</returns>
    public static CudaAtomicCapability[] GetHostAtomicCapabilities(int ordinal, IReadOnlyList<CudaAtomicOperation> operations)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetDeviceHostAtomicCapabilities(ordinal, CopyAtomicOperations(operations));
    }

    /// <summary>
    /// Gets native peer-to-peer atomic capabilities for the selected CUDA operations between two devices.
    /// 获取两个 CUDA 设备之间对一组 CUDA atomic operation 的 P2P atomic 原生能力。
    /// </summary>
    /// <param name="sourceOrdinal">The source CUDA device ordinal. 源 CUDA 设备序号。</param>
    /// <param name="destinationOrdinal">The destination CUDA device ordinal. 目标 CUDA 设备序号。</param>
    /// <param name="operations">The CUDA atomic operations to query. 要查询的 CUDA atomic operation 列表。</param>
    /// <returns>One capability bitmask per operation, in the same order. 按输入顺序返回每个 operation 的能力位掩码。</returns>
    public static CudaAtomicCapability[] GetP2PAtomicCapabilities(int sourceOrdinal, int destinationOrdinal, IReadOnlyList<CudaAtomicOperation> operations)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetDeviceP2PAtomicCapabilities(sourceOrdinal, destinationOrdinal, CopyAtomicOperations(operations));
    }

    /// <summary>
    /// Chooses the CUDA device that best matches a safe managed subset of <c>cudaDeviceProp</c> requirements.
    /// 根据托管层安全表达的 <c>cudaDeviceProp</c> 子集选择最匹配的 CUDA 设备。
    /// </summary>
    /// <param name="requirements">The desired CUDA device requirements. 期望的 CUDA 设备约束。</param>
    /// <returns>The CUDA device ordinal chosen by <c>cudaChooseDevice</c>. <c>cudaChooseDevice</c> 选择的 CUDA 设备序号。</returns>
    /// <remarks>
    /// The wrapper does not expose native <c>cudaDeviceProp</c> pointers. It copies a small caller-owned value structure into native code.
    /// 该封装不会暴露原生 <c>cudaDeviceProp</c> 指针，而是把小型 caller-owned 值结构复制到 native 侧。
    /// </remarks>
    public static int ChooseDevice(CudaDeviceSelectionRequirements requirements)
    {
        if (requirements == null)
        {
            throw new ArgumentNullException(nameof(requirements));
        }

        NativeBridgeLoader.EnsureInitialized();
        NativeCudaDeviceSelectionRequirements nativeRequirements = requirements.ToNative();
        return NativeCudaApi.ChooseDevice(in nativeRequirements);
    }

    /// <summary>
    /// Enables peer access from the current device to the specified peer device.
    /// 启用当前设备到指定 peer 设备的 peer access。
    /// </summary>
    /// <param name="peerOrdinal">The peer CUDA device ordinal. 对端 CUDA 设备序号。</param>
    public static void EnablePeerAccess(int peerOrdinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.EnablePeerAccess(peerOrdinal, 0);
    }

    /// <summary>
    /// Disables peer access from the current device to the specified peer device.
    /// 关闭当前设备到指定 peer 设备的 peer access。
    /// </summary>
    /// <param name="peerOrdinal">The peer CUDA device ordinal. 对端 CUDA 设备序号。</param>
    public static void DisablePeerAccess(int peerOrdinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.DisablePeerAccess(peerOrdinal);
    }

    /// <summary>
    /// Gets free and total memory information for the current CUDA device.
    /// 获取当前 CUDA 设备的空闲与总显存信息。
    /// </summary>
    /// <returns>The current CUDA memory information snapshot. 当前 CUDA 显存信息快照。</returns>
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

    /// <summary>
    /// Gets the default CUDA memory pool for a device.
    /// 获取指定设备的默认 CUDA memory pool。
    /// </summary>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>The default memory pool wrapper. 默认 memory pool 封装。</returns>
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

    /// <summary>
    /// Gets the maximum linear texture width supported for the specified descriptor.
    /// 获取指定通道描述符支持的最大 linear texture 宽度。
    /// </summary>
    /// <param name="descriptor">The CUDA channel descriptor. CUDA 通道描述符。</param>
    /// <param name="ordinal">The CUDA device ordinal. CUDA 设备序号。</param>
    /// <returns>The maximum width in bytes or CUDA-defined units. 以字节或 CUDA 定义单位表示的最大宽度。</returns>
    public static ulong GetTexture1DLinearMaxWidth(CudaChannelFormatDescriptor descriptor, int ordinal)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetTexture1DLinearMaxWidth(descriptor, ordinal);
    }

    /// <summary>
    /// Resets the persisting L2 cache state for the current device.
    /// 重置当前设备的 persisting L2 cache 状态。
    /// </summary>
    public static void ResetPersistingL2Cache()
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.ResetPersistingL2Cache();
    }

    /// <summary>
    /// Flushes GPU Direct RDMA writes for the selected target and scope.
    /// 为指定目标与范围刷新 GPU Direct RDMA 写入。
    /// </summary>
    /// <param name="target">The RDMA target to flush. 要刷新的 RDMA 目标。</param>
    /// <param name="scope">The RDMA visibility scope. RDMA 可见性范围。</param>
    public static void FlushGpuDirectRdmaWrites(CudaGpuDirectRdmaWritesTarget target, CudaGpuDirectRdmaWritesScope scope)
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.FlushGpuDirectRdmaWrites(target, scope);
    }

    /// <summary>
    /// Synchronizes the current CUDA device.
    /// 同步当前 CUDA 设备。
    /// </summary>
    public static void Synchronize()
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.SynchronizeDevice();
    }

    /// <summary>
    /// Resets the current CUDA device.
    /// 重置当前 CUDA 设备。
    /// </summary>
    public static void Reset()
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.ResetDevice();
    }

    /// <summary>
    /// Gets and clears the last CUDA error code.
    /// 获取并清除最近一次 CUDA 错误码。
    /// </summary>
    /// <returns>The last CUDA error code. 最近一次 CUDA 错误码。</returns>
    public static int GetLastErrorCode()
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetLastErrorCode();
    }

    /// <summary>
    /// Gets the last CUDA error code without clearing it.
    /// 获取最近一次 CUDA 错误码，但不清除。
    /// </summary>
    /// <returns>The last CUDA error code. 最近一次 CUDA 错误码。</returns>
    public static int PeekAtLastErrorCode()
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.PeekAtLastErrorCode();
    }

    /// <summary>
    /// Gets the symbolic CUDA error name for an error code.
    /// 获取某个错误码对应的 CUDA 符号名。
    /// </summary>
    /// <param name="errorCode">The CUDA error code. CUDA 错误码。</param>
    /// <returns>The CUDA symbolic error name. CUDA 符号错误名。</returns>
    public static string GetErrorName(int errorCode)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetErrorName(errorCode);
    }

    /// <summary>
    /// Gets the human-readable CUDA error string for an error code.
    /// 获取某个错误码对应的 CUDA 可读错误描述。
    /// </summary>
    /// <param name="errorCode">The CUDA error code. CUDA 错误码。</param>
    /// <returns>The CUDA error description. CUDA 错误描述。</returns>
    public static string GetErrorString(int errorCode)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetErrorString(errorCode);
    }

    private static CudaAtomicOperation[] CopyAtomicOperations(IReadOnlyList<CudaAtomicOperation> operations)
    {
        if (operations == null)
        {
            throw new ArgumentNullException(nameof(operations));
        }

        if (operations.Count == 0)
        {
            throw new ArgumentException("At least one CUDA atomic operation is required.", nameof(operations));
        }

        CudaAtomicOperation[] copy = new CudaAtomicOperation[operations.Count];
        for (int index = 0; index < operations.Count; index++)
        {
            copy[index] = operations[index];
        }

        return copy;
    }

    private static int[] CopyDeviceOrdinals(IReadOnlyList<int> ordinals)
    {
        if (ordinals == null)
        {
            throw new ArgumentNullException(nameof(ordinals));
        }

        if (ordinals.Count == 0)
        {
            throw new ArgumentException("At least one CUDA device ordinal is required.", nameof(ordinals));
        }

        int[] copy = new int[ordinals.Count];
        for (int index = 0; index < ordinals.Count; index++)
        {
            int ordinal = ordinals[index];
            if (ordinal < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(ordinals), ordinal, "CUDA device ordinal must be greater than or equal to zero.");
            }

            copy[index] = ordinal;
        }

        return copy;
    }
}
