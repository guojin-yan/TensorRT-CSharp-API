using System;

namespace JYPPX.CudaSharp;

[Flags]
public enum CudaStreamCreationFlags : uint
{
    Default = 0,
    NonBlocking = 1
}

[Flags]
public enum CudaEventCreationFlags : uint
{
    Default = 0,
    BlockingSync = 1,
    DisableTiming = 2,
    Interprocess = 4
}

[Flags]
public enum CudaPinnedMemoryAllocationFlags : uint
{
    Default = 0,
    Portable = 1,
    Mapped = 2,
    WriteCombined = 4
}

[Flags]
public enum CudaHostRegistrationFlags : uint
{
    Default = 0,
    Portable = 1,
    Mapped = 2,
    IoMemory = 4,
    ReadOnly = 8
}

public enum CudaMemoryAdvice
{
    SetReadMostly = 1,
    UnsetReadMostly = 2,
    SetPreferredLocation = 3,
    UnsetPreferredLocation = 4,
    SetAccessedBy = 5,
    UnsetAccessedBy = 6
}

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
/// CUDA peer-to-peer attribute identifiers.
/// CUDA 点对点访问属性标识。
/// </summary>
public enum CudaDeviceP2PAttribute
{
    /// <summary>
    /// Relative performance rank for the link between two devices.
    /// 两个设备之间链路的相对性能等级。
    /// </summary>
    PerformanceRank = 1,

    /// <summary>
    /// Whether peer access is supported for the selected devices.
    /// 指定设备之间是否支持 peer access。
    /// </summary>
    AccessSupported = 2,

    /// <summary>
    /// Whether native atomic operations are supported over the link.
    /// 链路上是否支持 native atomic 操作。
    /// </summary>
    NativeAtomicSupported = 3,

    /// <summary>
    /// Whether CUDA array access is supported over the peer link.
    /// peer 链路上是否支持 CUDA array 访问。
    /// </summary>
    CudaArrayAccessSupported = 4
}

/// <summary>
/// Flags used when recording an event into a stream.
/// 向 stream 记录 event 时使用的标志。
/// </summary>
[Flags]
public enum CudaEventRecordFlags : uint
{
    /// <summary>
    /// Use CUDA's default event-record behavior.
    /// 使用 CUDA 默认 event record 行为。
    /// </summary>
    Default = 0,

    /// <summary>
    /// Capture the event as an external event node during graph capture.
    /// graph capture 期间将 event 捕获为 external event node。
    /// </summary>
    External = 1
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

public enum CudaStreamCaptureMode
{
    Global = 0,
    ThreadLocal = 1,
    Relaxed = 2
}

public enum CudaChannelFormatKind
{
    Signed = 0,
    Unsigned = 1,
    Float = 2,
    None = 3,
    NV12 = 4,
    UnsignedNormalized8X1 = 5,
    UnsignedNormalized8X2 = 6,
    UnsignedNormalized8X4 = 7,
    UnsignedNormalized16X1 = 8,
    UnsignedNormalized16X2 = 9,
    UnsignedNormalized16X4 = 10,
    SignedNormalized8X1 = 11,
    SignedNormalized8X2 = 12,
    SignedNormalized8X4 = 13,
    SignedNormalized16X1 = 14,
    SignedNormalized16X2 = 15,
    SignedNormalized16X4 = 16,
    UnsignedBlockCompressed1 = 17,
    UnsignedBlockCompressed1SRgb = 18,
    UnsignedBlockCompressed2 = 19,
    UnsignedBlockCompressed2SRgb = 20,
    UnsignedBlockCompressed3 = 21,
    UnsignedBlockCompressed3SRgb = 22,
    UnsignedBlockCompressed4 = 23,
    SignedBlockCompressed4 = 24,
    UnsignedBlockCompressed5 = 25,
    SignedBlockCompressed5 = 26,
    UnsignedBlockCompressed6H = 27,
    SignedBlockCompressed6H = 28,
    UnsignedBlockCompressed7 = 29,
    UnsignedBlockCompressed7SRgb = 30,
    UnsignedNormalized1010102 = 31
}

[Flags]
public enum CudaArrayCreationFlags : uint
{
    Default = 0x00,
    Layered = 0x01,
    SurfaceLoadStore = 0x02,
    Cubemap = 0x04,
    TextureGather = 0x08,
    ColorAttachment = 0x20,
    Sparse = 0x40,
    DeferredMapping = 0x80
}

[Flags]
public enum CudaArraySparseFlags : uint
{
    None = 0,
    SingleMipTail = 0x01
}

public enum CudaGpuDirectRdmaWritesTarget
{
    CurrentDevice = 0
}

public enum CudaGpuDirectRdmaWritesScope
{
    ToOwner = 100,
    ToAllDevices = 200
}

/// <summary>
/// Describes the CUDA stream capture state used by CUDA Graph capture.
/// 描述 CUDA Graph 捕获流程中的 CUDA stream 捕获状态。
/// </summary>
public enum CudaStreamCaptureStatus
{
    /// <summary>
    /// The stream is not currently capturing.
    /// 当前 stream 没有处于捕获状态。
    /// </summary>
    None = 0,

    /// <summary>
    /// The stream is actively capturing commands.
    /// 当前 stream 正在捕获命令。
    /// </summary>
    Active = 1,

    /// <summary>
    /// The stream capture was invalidated by an error.
    /// stream 捕获流程已因错误失效。
    /// </summary>
    Invalidated = 2
}

/// <summary>
/// Identifies CUDA graph node kinds reported by the CUDA runtime.
/// 标识 CUDA runtime 报告的 CUDA graph node 类型。
/// </summary>
public enum CudaGraphNodeType
{
    Kernel = 0,
    Memcpy = 1,
    Memset = 2,
    Host = 3,
    Graph = 4,
    Empty = 5,
    WaitEvent = 6,
    EventRecord = 7,
    ExternalSemaphoreSignal = 8,
    ExternalSemaphoreWait = 9,
    MemoryAlloc = 10,
    MemoryFree = 11,
    BatchMemoryOperation = 12,
    Conditional = 13
}

[Flags]
public enum CudaManagedMemoryAttachmentFlags : uint
{
    Global = 1,
    Host = 2,
    Single = 4
}

public enum CudaDeviceAttribute
{
    MaxThreadsPerBlock = 1,
    MaxBlockDimX = 2,
    MaxBlockDimY = 3,
    MaxBlockDimZ = 4,
    MaxGridDimX = 5,
    MaxGridDimY = 6,
    MaxGridDimZ = 7,
    MaxSharedMemoryPerBlock = 8,
    TotalConstantMemory = 9,
    WarpSize = 10,
    MaxRegistersPerBlock = 12,
    ClockRate = 13,
    MultiProcessorCount = 16,
    KernelExecTimeout = 17,
    Integrated = 18,
    CanMapHostMemory = 19,
    ConcurrentKernels = 31,
    MemoryClockRate = 36,
    GlobalMemoryBusWidth = 37,
    L2CacheSize = 38,
    MaxThreadsPerMultiProcessor = 39,
    AsyncEngineCount = 40,
    UnifiedAddressing = 41,
    ComputeCapabilityMajor = 75,
    ComputeCapabilityMinor = 76,
    StreamPrioritiesSupported = 78,
    ManagedMemory = 83,
    ConcurrentManagedAccess = 89,
    ComputePreemptionSupported = 90,
    HostRegisterSupported = 99,
    DirectManagedMemoryAccessFromHost = 101
}
