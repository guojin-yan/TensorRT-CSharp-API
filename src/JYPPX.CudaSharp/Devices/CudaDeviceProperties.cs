using System.Collections.Generic;

namespace JYPPX.CudaSharp;

/// <summary>
/// Aggregates deployment-relevant CUDA device properties and optional runtime attributes.
/// 聚合模型部署相关的 CUDA 设备属性以及可选 runtime attribute。
/// </summary>
public sealed class CudaDeviceProperties
{
    /// <summary>
    /// Creates a CUDA device property snapshot.
    /// 创建 CUDA 设备属性快照。
    /// </summary>
    public CudaDeviceProperties(
        CudaDeviceInfo info,
        IReadOnlyList<int> maxBlockDimensions,
        IReadOnlyList<int> maxGridDimensions,
        int? clockRateKilohertz,
        int? memoryClockRateKilohertz,
        int? globalMemoryBusWidthBits,
        int? l2CacheSizeBytes,
        int? maxThreadsPerMultiProcessor,
        int? asyncEngineCount,
        bool? concurrentKernels,
        bool? unifiedAddressing,
        bool? managedMemory,
        bool? concurrentManagedAccess,
        bool? streamPrioritiesSupported,
        bool? computePreemptionSupported,
        bool? hostRegisterSupported,
        bool? directManagedMemoryAccessFromHost)
    {
        Info = info;
        MaxBlockDimensions = maxBlockDimensions;
        MaxGridDimensions = maxGridDimensions;
        ClockRateKilohertz = clockRateKilohertz;
        MemoryClockRateKilohertz = memoryClockRateKilohertz;
        GlobalMemoryBusWidthBits = globalMemoryBusWidthBits;
        L2CacheSizeBytes = l2CacheSizeBytes;
        MaxThreadsPerMultiProcessor = maxThreadsPerMultiProcessor;
        AsyncEngineCount = asyncEngineCount;
        ConcurrentKernels = concurrentKernels;
        UnifiedAddressing = unifiedAddressing;
        ManagedMemory = managedMemory;
        ConcurrentManagedAccess = concurrentManagedAccess;
        StreamPrioritiesSupported = streamPrioritiesSupported;
        ComputePreemptionSupported = computePreemptionSupported;
        HostRegisterSupported = hostRegisterSupported;
        DirectManagedMemoryAccessFromHost = directManagedMemoryAccessFromHost;
    }

    /// <summary>
    /// Gets the base CUDA device info returned by the bridge.
    /// 获取桥接层返回的 CUDA 设备基础信息。
    /// </summary>
    public CudaDeviceInfo Info { get; }

    /// <summary>
    /// Gets the maximum CUDA block dimensions as X/Y/Z.
    /// 获取 CUDA block 维度上限，顺序为 X/Y/Z。
    /// </summary>
    public IReadOnlyList<int> MaxBlockDimensions { get; }

    /// <summary>
    /// Gets the maximum CUDA grid dimensions as X/Y/Z.
    /// 获取 CUDA grid 维度上限，顺序为 X/Y/Z。
    /// </summary>
    public IReadOnlyList<int> MaxGridDimensions { get; }

    /// <summary>
    /// Gets the device core clock rate in kHz when available.
    /// 获取可用时的设备核心频率，单位 kHz。
    /// </summary>
    public int? ClockRateKilohertz { get; }

    /// <summary>
    /// Gets the memory clock rate in kHz when available.
    /// 获取可用时的显存频率，单位 kHz。
    /// </summary>
    public int? MemoryClockRateKilohertz { get; }

    /// <summary>
    /// Gets the global memory bus width in bits when available.
    /// 获取可用时的显存总线宽度，单位 bit。
    /// </summary>
    public int? GlobalMemoryBusWidthBits { get; }

    /// <summary>
    /// Gets the L2 cache size in bytes when available.
    /// 获取可用时的 L2 cache 大小，单位字节。
    /// </summary>
    public int? L2CacheSizeBytes { get; }

    /// <summary>
    /// Gets the maximum threads per streaming multiprocessor when available.
    /// 获取可用时每个 SM 支持的最大线程数。
    /// </summary>
    public int? MaxThreadsPerMultiProcessor { get; }

    /// <summary>
    /// Gets the number of async copy engines when available.
    /// 获取可用时异步拷贝引擎数量。
    /// </summary>
    public int? AsyncEngineCount { get; }

    /// <summary>
    /// Gets whether concurrent kernels are supported when available.
    /// 获取可用时设备是否支持并发 kernel。
    /// </summary>
    public bool? ConcurrentKernels { get; }

    /// <summary>
    /// Gets whether unified virtual addressing is supported when available.
    /// 获取可用时设备是否支持统一虚拟地址。
    /// </summary>
    public bool? UnifiedAddressing { get; }

    /// <summary>
    /// Gets whether CUDA managed memory is supported when available.
    /// 获取可用时设备是否支持 CUDA managed memory。
    /// </summary>
    public bool? ManagedMemory { get; }

    /// <summary>
    /// Gets whether concurrent managed-memory access is supported when available.
    /// 获取可用时设备是否支持并发 managed-memory 访问。
    /// </summary>
    public bool? ConcurrentManagedAccess { get; }

    /// <summary>
    /// Gets whether stream priorities are supported when available.
    /// 获取可用时设备是否支持 stream priority。
    /// </summary>
    public bool? StreamPrioritiesSupported { get; }

    /// <summary>
    /// Gets whether compute preemption is supported when available.
    /// 获取可用时设备是否支持计算抢占。
    /// </summary>
    public bool? ComputePreemptionSupported { get; }

    /// <summary>
    /// Gets whether host memory registration is supported when available.
    /// 获取可用时设备是否支持 host memory registration。
    /// </summary>
    public bool? HostRegisterSupported { get; }

    /// <summary>
    /// Gets whether the host can directly access managed memory when available.
    /// 获取可用时主机是否可以直接访问 managed memory。
    /// </summary>
    public bool? DirectManagedMemoryAccessFromHost { get; }

    /// <summary>
    /// Gets a concise device capability label such as "sm_86".
    /// 获取简洁设备能力标识，例如 "sm_86"。
    /// </summary>
    public string ComputeCapabilityLabel => $"sm_{Info.Major}{Info.Minor}";
}
