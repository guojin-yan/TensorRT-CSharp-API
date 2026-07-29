using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Selects the preferred shared-memory/L1 cache split for subsequent CUDA launches.
/// 选择后续 CUDA launch 使用的 shared memory / L1 cache 倾向。
/// </summary>
public enum CudaFunctionCachePreference
{
    /// <summary>
    /// Do not prefer either shared memory or L1 cache.
    /// 不显式偏向 shared memory 或 L1 cache。
    /// </summary>
    PreferNone = 0,

    /// <summary>
    /// Prefer larger shared memory and smaller L1 cache.
    /// 偏向更大的 shared memory 和更小的 L1 cache。
    /// </summary>
    PreferShared = 1,

    /// <summary>
    /// Prefer larger L1 cache and smaller shared memory.
    /// 偏向更大的 L1 cache 和更小的 shared memory。
    /// </summary>
    PreferL1 = 2,

    /// <summary>
    /// Prefer equal shared memory and L1 cache split.
    /// 偏向 shared memory 与 L1 cache 等分。
    /// </summary>
    PreferEqual = 3
}

/// <summary>
/// Selects CUDA shared-memory bank width for compatible devices.
/// 选择兼容设备上的 CUDA shared-memory bank 宽度。
/// </summary>
public enum CudaSharedMemoryConfig
{
    /// <summary>
    /// Use CUDA's default shared-memory bank size.
    /// 使用 CUDA 默认 shared-memory bank 大小。
    /// </summary>
    DefaultBankSize = 0,

    /// <summary>
    /// Use four-byte shared-memory banks.
    /// 使用 4 字节 shared-memory bank。
    /// </summary>
    FourByteBankSize = 1,

    /// <summary>
    /// Use eight-byte shared-memory banks.
    /// 使用 8 字节 shared-memory bank。
    /// </summary>
    EightByteBankSize = 2
}

/// <summary>
/// CUDA runtime device scheduling and mapped-host-memory flags.
/// CUDA runtime 设备调度与 mapped host memory 标志。
/// </summary>
[Flags]
public enum CudaDeviceRuntimeFlags : uint
{
    /// <summary>
    /// Let CUDA choose the device scheduling behavior automatically.
    /// 让 CUDA 自动选择设备调度行为。
    /// </summary>
    ScheduleAuto = 0x00,

    /// <summary>
    /// Spin while waiting for device work to complete.
    /// 等待设备工作完成时使用自旋。
    /// </summary>
    ScheduleSpin = 0x01,

    /// <summary>
    /// Yield the CPU thread while waiting for device work.
    /// 等待设备工作时让出 CPU 线程。
    /// </summary>
    ScheduleYield = 0x02,

    /// <summary>
    /// Use blocking synchronization while waiting for device work.
    /// 等待设备工作时使用阻塞同步。
    /// </summary>
    ScheduleBlockingSync = 0x04,

    /// <summary>
    /// Enable mapped host memory support.
    /// 启用 mapped host memory 支持。
    /// </summary>
    MapHost = 0x08,

    /// <summary>
    /// Keep local memory allocations after launch.
    /// launch 后保留 local memory 分配。
    /// </summary>
    LocalMemoryResizeToMax = 0x10
}

/// <summary>
/// Identifies CUDA runtime device limits that affect device-side execution behavior.
/// 标识影响设备端执行行为的 CUDA runtime device limit。
/// </summary>
public enum CudaDeviceLimit
{
    /// <summary>
    /// Device runtime stack size for each GPU thread.
    /// 每个 GPU 线程的设备端 runtime 栈大小。
    /// </summary>
    StackSize = 0,

    /// <summary>
    /// FIFO size used by device-side printf.
    /// 设备端 printf 使用的 FIFO 大小。
    /// </summary>
    PrintfFifoSize = 1,

    /// <summary>
    /// Device-side malloc heap size.
    /// 设备端 malloc 堆大小。
    /// </summary>
    MallocHeapSize = 2,

    /// <summary>
    /// Device runtime synchronization depth.
    /// 设备端 runtime 同步深度。
    /// </summary>
    DevRuntimeSyncDepth = 3,

    /// <summary>
    /// Maximum pending launch count for device runtime launches.
    /// 设备端 runtime launch 的最大 pending launch 数量。
    /// </summary>
    DevRuntimePendingLaunchCount = 4,

    /// <summary>
    /// Maximum L2 fetch granularity.
    /// 最大 L2 fetch 粒度。
    /// </summary>
    MaxL2FetchGranularity = 5,

    /// <summary>
    /// Persisting L2 cache size.
    /// 持久化 L2 cache 大小。
    /// </summary>
    PersistingL2CacheSize = 6
}

/// <summary>
/// CUDA device attributes exposed through the runtime API.
/// 通过 CUDA runtime API 暴露的设备属性。
/// </summary>
public enum CudaDeviceAttribute
{
    /// <summary>Maximum threads per block. 每个 block 的最大线程数。</summary>
    MaxThreadsPerBlock = 1,
    /// <summary>Maximum X dimension for a block. block 的 X 维上限。</summary>
    MaxBlockDimX = 2,
    /// <summary>Maximum Y dimension for a block. block 的 Y 维上限。</summary>
    MaxBlockDimY = 3,
    /// <summary>Maximum Z dimension for a block. block 的 Z 维上限。</summary>
    MaxBlockDimZ = 4,
    /// <summary>Maximum X dimension for a grid. grid 的 X 维上限。</summary>
    MaxGridDimX = 5,
    /// <summary>Maximum Y dimension for a grid. grid 的 Y 维上限。</summary>
    MaxGridDimY = 6,
    /// <summary>Maximum Z dimension for a grid. grid 的 Z 维上限。</summary>
    MaxGridDimZ = 7,
    /// <summary>Maximum shared memory per block. 每个 block 的最大 shared memory。</summary>
    MaxSharedMemoryPerBlock = 8,
    /// <summary>Total constant memory size. constant memory 总大小。</summary>
    TotalConstantMemory = 9,
    /// <summary>Warp size. warp 大小。</summary>
    WarpSize = 10,
    /// <summary>Maximum registers per block. 每个 block 的最大寄存器数。</summary>
    MaxRegistersPerBlock = 12,
    /// <summary>Core clock rate. 核心时钟频率。</summary>
    ClockRate = 13,
    /// <summary>Streaming multiprocessor count. 流式多处理器数量。</summary>
    MultiProcessorCount = 16,
    /// <summary>Kernel execution timeout support. kernel 执行超时属性。</summary>
    KernelExecTimeout = 17,
    /// <summary>Whether the GPU is integrated. GPU 是否为集成式。</summary>
    Integrated = 18,
    /// <summary>Whether host memory can be mapped. 是否可映射 host memory。</summary>
    CanMapHostMemory = 19,
    /// <summary>Whether concurrent kernels are supported. 是否支持并发 kernel。</summary>
    ConcurrentKernels = 31,
    /// <summary>Memory clock rate. 显存时钟频率。</summary>
    MemoryClockRate = 36,
    /// <summary>Global-memory bus width. 显存总线宽度。</summary>
    GlobalMemoryBusWidth = 37,
    /// <summary>L2 cache size. L2 cache 大小。</summary>
    L2CacheSize = 38,
    /// <summary>Maximum threads per multiprocessor. 每个多处理器的最大线程数。</summary>
    MaxThreadsPerMultiProcessor = 39,
    /// <summary>Async engine count. 异步引擎数量。</summary>
    AsyncEngineCount = 40,
    /// <summary>Unified addressing support. 统一寻址支持。</summary>
    UnifiedAddressing = 41,
    /// <summary>Compute capability major version. 计算能力主版本号。</summary>
    ComputeCapabilityMajor = 75,
    /// <summary>Compute capability minor version. 计算能力次版本号。</summary>
    ComputeCapabilityMinor = 76,
    /// <summary>Stream-priority support. stream 优先级支持。</summary>
    StreamPrioritiesSupported = 78,
    /// <summary>Managed-memory support. managed memory 支持。</summary>
    ManagedMemory = 83,
    /// <summary>Concurrent managed-memory access support. 并发 managed memory 访问支持。</summary>
    ConcurrentManagedAccess = 89,
    /// <summary>Compute preemption support. 计算抢占支持。</summary>
    ComputePreemptionSupported = 90,
    /// <summary>Host-register support. host register 支持。</summary>
    HostRegisterSupported = 99,
    /// <summary>Direct host access to managed memory support. 主机直接访问 managed memory 支持。</summary>
    DirectManagedMemoryAccessFromHost = 101,
    /// <summary>CUDA IPC event support. CUDA IPC event 支持。</summary>
    IpcEventSupport = 125
}
