using System;

namespace JYPPX.CudaSharp;

/// <summary>
/// Flags that control CUDA stream creation behavior.
/// 控制 CUDA stream 创建行为的标志。
/// </summary>
[Flags]
public enum CudaStreamCreationFlags : uint
{
    /// <summary>
    /// Uses CUDA default stream-creation behavior.
    /// 使用 CUDA 默认的 stream 创建行为。
    /// </summary>
    Default = 0,
    /// <summary>
    /// Creates a non-blocking CUDA stream.
    /// 创建一个 non-blocking CUDA stream。
    /// </summary>
    NonBlocking = 1
}

/// <summary>
/// Flags that control CUDA event creation behavior.
/// 控制 CUDA event 创建行为的标志。
/// </summary>
[Flags]
public enum CudaEventCreationFlags : uint
{
    /// <summary>
    /// Uses CUDA default event-creation behavior.
    /// 使用 CUDA 默认的 event 创建行为。
    /// </summary>
    Default = 0,
    /// <summary>
    /// Creates an event that blocks the waiting host thread.
    /// 创建一个会阻塞等待线程的 event。
    /// </summary>
    BlockingSync = 1,
    /// <summary>
    /// Creates an event without timing information.
    /// 创建一个不记录计时信息的 event。
    /// </summary>
    DisableTiming = 2,
    /// <summary>
    /// Creates an event that can be shared across processes.
    /// 创建一个可跨进程共享的 event。
    /// </summary>
    Interprocess = 4
}

/// <summary>
/// Flags that control pinned host-memory allocation behavior.
/// 控制 pinned host memory 分配行为的标志。
/// </summary>
[Flags]
public enum CudaPinnedMemoryAllocationFlags : uint
{
    /// <summary>
    /// Uses CUDA default pinned-memory allocation behavior.
    /// 使用 CUDA 默认的 pinned memory 分配行为。
    /// </summary>
    Default = 0,
    /// <summary>
    /// Makes the allocation portable across CUDA contexts.
    /// 使该分配在多个 CUDA context 之间可移植。
    /// </summary>
    Portable = 1,
    /// <summary>
    /// Maps the pinned allocation into the device address space.
    /// 将该 pinned 分配映射到设备地址空间。
    /// </summary>
    Mapped = 2,
    /// <summary>
    /// Requests write-combined host memory.
    /// 请求 write-combined host memory。
    /// </summary>
    WriteCombined = 4
}

/// <summary>
/// Flags that control registered host-memory behavior.
/// 控制 registered host memory 行为的标志。
/// </summary>
[Flags]
public enum CudaHostRegistrationFlags : uint
{
    /// <summary>
    /// Uses CUDA default host-registration behavior.
    /// 使用 CUDA 默认的 host registration 行为。
    /// </summary>
    Default = 0,
    /// <summary>
    /// Makes the registration portable across CUDA contexts.
    /// 使该注册在多个 CUDA context 之间可移植。
    /// </summary>
    Portable = 1,
    /// <summary>
    /// Maps the registered memory into the device address space.
    /// 将注册的内存映射到设备地址空间。
    /// </summary>
    Mapped = 2,
    /// <summary>
    /// Marks the registration as I/O memory.
    /// 将该注册标记为 I/O memory。
    /// </summary>
    IoMemory = 4,
    /// <summary>
    /// Registers the host memory as read-only from the device perspective.
    /// 从设备视角将 host memory 注册为只读。
    /// </summary>
    ReadOnly = 8
}

/// <summary>
/// Memory-advice commands accepted by CUDA managed memory.
/// CUDA managed memory 接受的内存建议命令。
/// </summary>
public enum CudaMemoryAdvice
{
    /// <summary>
    /// Marks the memory range as read-mostly.
    /// 将该内存范围标记为主要用于读取。
    /// </summary>
    SetReadMostly = 1,
    /// <summary>
    /// Clears the read-mostly advice.
    /// 清除主要读取的建议标记。
    /// </summary>
    UnsetReadMostly = 2,
    /// <summary>
    /// Sets a preferred location for the memory range.
    /// 为该内存范围设置首选位置。
    /// </summary>
    SetPreferredLocation = 3,
    /// <summary>
    /// Clears the preferred location advice.
    /// 清除首选位置建议。
    /// </summary>
    UnsetPreferredLocation = 4,
    /// <summary>
    /// Marks the memory range as accessed by a device.
    /// 将该内存范围标记为会被某个设备访问。
    /// </summary>
    SetAccessedBy = 5,
    /// <summary>
    /// Clears the accessed-by advice.
    /// 清除 accessed-by 建议。
    /// </summary>
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
/// CUDA atomic operations accepted by CUDA 13 host/P2P atomic capability queries.
/// CUDA 13 host/P2P atomic 能力查询支持的 atomic operation。
/// </summary>
public enum CudaAtomicOperation
{
    /// <summary>Integer add. 整数加法。</summary>
    IntegerAdd = 0,
    /// <summary>Integer minimum. 整数最小值。</summary>
    IntegerMin = 1,
    /// <summary>Integer maximum. 整数最大值。</summary>
    IntegerMax = 2,
    /// <summary>Integer increment. 整数递增。</summary>
    IntegerIncrement = 3,
    /// <summary>Integer decrement. 整数递减。</summary>
    IntegerDecrement = 4,
    /// <summary>Bitwise and. 按位与。</summary>
    And = 5,
    /// <summary>Bitwise or. 按位或。</summary>
    Or = 6,
    /// <summary>Bitwise xor. 按位异或。</summary>
    Xor = 7,
    /// <summary>Exchange. 交换。</summary>
    Exchange = 8,
    /// <summary>Compare and swap. 比较并交换。</summary>
    CompareAndSwap = 9,
    /// <summary>Floating-point add. 浮点加法。</summary>
    FloatAdd = 10,
    /// <summary>Floating-point minimum. 浮点最小值。</summary>
    FloatMin = 11,
    /// <summary>Floating-point maximum. 浮点最大值。</summary>
    FloatMax = 12
}

/// <summary>
/// Capability bitmask returned for a CUDA atomic operation.
/// CUDA atomic operation 返回的能力位掩码。
/// </summary>
[Flags]
public enum CudaAtomicCapability : uint
{
    /// <summary>No native capability was reported. 未报告原生能力。</summary>
    None = 0,
    /// <summary>Signed operand support. 支持有符号操作数。</summary>
    Signed = 1u << 0,
    /// <summary>Unsigned operand support. 支持无符号操作数。</summary>
    Unsigned = 1u << 1,
    /// <summary>Reduction support. 支持 reduction。</summary>
    Reduction = 1u << 2,
    /// <summary>32-bit scalar support. 支持 32-bit scalar。</summary>
    Scalar32 = 1u << 3,
    /// <summary>64-bit scalar support. 支持 64-bit scalar。</summary>
    Scalar64 = 1u << 4,
    /// <summary>128-bit scalar support. 支持 128-bit scalar。</summary>
    Scalar128 = 1u << 5,
    /// <summary>Four-lane 32-bit vector support. 支持 Vector32x4。</summary>
    Vector32x4 = 1u << 6
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

/// <summary>
/// Modes that control CUDA stream capture validation.
/// 控制 CUDA stream capture 校验行为的模式。
/// </summary>
public enum CudaStreamCaptureMode
{
    /// <summary>
    /// Uses global capture validation.
    /// 使用全局 capture 校验。
    /// </summary>
    Global = 0,
    /// <summary>
    /// Uses thread-local capture validation.
    /// 使用线程本地 capture 校验。
    /// </summary>
    ThreadLocal = 1,
    /// <summary>
    /// Uses relaxed capture validation.
    /// 使用宽松 capture 校验。
    /// </summary>
    Relaxed = 2
}

/// <summary>
/// Controls whether stream-capture dependencies are added or replaced.
/// 控制 stream capture dependencies 是追加还是替换。
/// </summary>
public enum CudaStreamCaptureDependencyMode
{
    /// <summary>Adds nodes to the current dependency set. 向当前 dependency set 追加节点。</summary>
    Add = 0,
    /// <summary>Replaces the current dependency set. 替换当前 dependency set。</summary>
    Replace = 1
}

/// <summary>
/// Channel-format kinds supported by CUDA arrays and textures.
/// CUDA array 与 texture 支持的通道格式类型。
/// </summary>
public enum CudaChannelFormatKind
{
    /// <summary>
    /// Signed integer channel format.
    /// 有符号整数通道格式。
    /// </summary>
    Signed = 0,
    /// <summary>
    /// Unsigned integer channel format.
    /// 无符号整数通道格式。
    /// </summary>
    Unsigned = 1,
    /// <summary>
    /// Floating-point channel format.
    /// 浮点通道格式。
    /// </summary>
    Float = 2,
    /// <summary>
    /// No channel format is specified.
    /// 未指定通道格式。
    /// </summary>
    None = 3,
    /// <summary>
    /// NV12 planar format.
    /// NV12 平面格式。
    /// </summary>
    NV12 = 4,
    /// <summary>
    /// Unsigned normalized 8-bit single-channel format.
    /// 无符号归一化 8 位单通道格式。
    /// </summary>
    UnsignedNormalized8X1 = 5,
    /// <summary>
    /// Unsigned normalized 8-bit two-channel format.
    /// 无符号归一化 8 位双通道格式。
    /// </summary>
    UnsignedNormalized8X2 = 6,
    /// <summary>
    /// Unsigned normalized 8-bit four-channel format.
    /// 无符号归一化 8 位四通道格式。
    /// </summary>
    UnsignedNormalized8X4 = 7,
    /// <summary>
    /// Unsigned normalized 16-bit single-channel format.
    /// 无符号归一化 16 位单通道格式。
    /// </summary>
    UnsignedNormalized16X1 = 8,
    /// <summary>
    /// Unsigned normalized 16-bit two-channel format.
    /// 无符号归一化 16 位双通道格式。
    /// </summary>
    UnsignedNormalized16X2 = 9,
    /// <summary>
    /// Unsigned normalized 16-bit four-channel format.
    /// 无符号归一化 16 位四通道格式。
    /// </summary>
    UnsignedNormalized16X4 = 10,
    /// <summary>
    /// Signed normalized 8-bit single-channel format.
    /// 有符号归一化 8 位单通道格式。
    /// </summary>
    SignedNormalized8X1 = 11,
    /// <summary>
    /// Signed normalized 8-bit two-channel format.
    /// 有符号归一化 8 位双通道格式。
    /// </summary>
    SignedNormalized8X2 = 12,
    /// <summary>
    /// Signed normalized 8-bit four-channel format.
    /// 有符号归一化 8 位四通道格式。
    /// </summary>
    SignedNormalized8X4 = 13,
    /// <summary>
    /// Signed normalized 16-bit single-channel format.
    /// 有符号归一化 16 位单通道格式。
    /// </summary>
    SignedNormalized16X1 = 14,
    /// <summary>
    /// Signed normalized 16-bit two-channel format.
    /// 有符号归一化 16 位双通道格式。
    /// </summary>
    SignedNormalized16X2 = 15,
    /// <summary>
    /// Signed normalized 16-bit four-channel format.
    /// 有符号归一化 16 位四通道格式。
    /// </summary>
    SignedNormalized16X4 = 16,
    /// <summary>
    /// Unsigned block-compressed format BC1.
    /// 无符号块压缩 BC1 格式。
    /// </summary>
    UnsignedBlockCompressed1 = 17,
    /// <summary>
    /// Unsigned block-compressed sRGB BC1 format.
    /// 无符号 sRGB 块压缩 BC1 格式。
    /// </summary>
    UnsignedBlockCompressed1SRgb = 18,
    /// <summary>
    /// Unsigned block-compressed format BC2.
    /// 无符号块压缩 BC2 格式。
    /// </summary>
    UnsignedBlockCompressed2 = 19,
    /// <summary>
    /// Unsigned block-compressed sRGB BC2 format.
    /// 无符号 sRGB 块压缩 BC2 格式。
    /// </summary>
    UnsignedBlockCompressed2SRgb = 20,
    /// <summary>
    /// Unsigned block-compressed format BC3.
    /// 无符号块压缩 BC3 格式。
    /// </summary>
    UnsignedBlockCompressed3 = 21,
    /// <summary>
    /// Unsigned block-compressed sRGB BC3 format.
    /// 无符号 sRGB 块压缩 BC3 格式。
    /// </summary>
    UnsignedBlockCompressed3SRgb = 22,
    /// <summary>
    /// Unsigned block-compressed format BC4.
    /// 无符号块压缩 BC4 格式。
    /// </summary>
    UnsignedBlockCompressed4 = 23,
    /// <summary>
    /// Signed block-compressed format BC4.
    /// 有符号块压缩 BC4 格式。
    /// </summary>
    SignedBlockCompressed4 = 24,
    /// <summary>
    /// Unsigned block-compressed format BC5.
    /// 无符号块压缩 BC5 格式。
    /// </summary>
    UnsignedBlockCompressed5 = 25,
    /// <summary>
    /// Signed block-compressed format BC5.
    /// 有符号块压缩 BC5 格式。
    /// </summary>
    SignedBlockCompressed5 = 26,
    /// <summary>
    /// Unsigned block-compressed format BC6H.
    /// 无符号块压缩 BC6H 格式。
    /// </summary>
    UnsignedBlockCompressed6H = 27,
    /// <summary>
    /// Signed block-compressed format BC6H.
    /// 有符号块压缩 BC6H 格式。
    /// </summary>
    SignedBlockCompressed6H = 28,
    /// <summary>
    /// Unsigned block-compressed format BC7.
    /// 无符号块压缩 BC7 格式。
    /// </summary>
    UnsignedBlockCompressed7 = 29,
    /// <summary>
    /// Unsigned block-compressed sRGB BC7 format.
    /// 无符号 sRGB 块压缩 BC7 格式。
    /// </summary>
    UnsignedBlockCompressed7SRgb = 30,
    /// <summary>
    /// Unsigned normalized 10:10:10:2 packed format.
    /// 无符号归一化 10:10:10:2 打包格式。
    /// </summary>
    UnsignedNormalized1010102 = 31
}

/// <summary>
/// Flags that control CUDA array creation behavior.
/// 控制 CUDA array 创建行为的标志。
/// </summary>
[Flags]
public enum CudaArrayCreationFlags : uint
{
    /// <summary>
    /// Uses CUDA default array-creation behavior.
    /// 使用 CUDA 默认的 array 创建行为。
    /// </summary>
    Default = 0x00,
    /// <summary>
    /// Creates a layered CUDA array.
    /// 创建 layered CUDA array。
    /// </summary>
    Layered = 0x01,
    /// <summary>
    /// Enables surface load/store support.
    /// 启用 surface load/store 支持。
    /// </summary>
    SurfaceLoadStore = 0x02,
    /// <summary>
    /// Creates a cubemap-compatible CUDA array.
    /// 创建兼容 cubemap 的 CUDA array。
    /// </summary>
    Cubemap = 0x04,
    /// <summary>
    /// Enables texture gather support.
    /// 启用 texture gather 支持。
    /// </summary>
    TextureGather = 0x08,
    /// <summary>
    /// Marks the array as a color attachment.
    /// 将该 array 标记为 color attachment。
    /// </summary>
    ColorAttachment = 0x20,
    /// <summary>
    /// Creates a sparse CUDA array.
    /// 创建 sparse CUDA array。
    /// </summary>
    Sparse = 0x40,
    /// <summary>
    /// Enables deferred mapping for sparse arrays.
    /// 为 sparse array 启用 deferred mapping。
    /// </summary>
    DeferredMapping = 0x80
}

/// <summary>
/// Flags that describe sparse CUDA array behavior.
/// 描述 sparse CUDA array 行为的标志。
/// </summary>
[Flags]
public enum CudaArraySparseFlags : uint
{
    /// <summary>
    /// No sparse-array flags are set.
    /// 不设置任何 sparse array 标志。
    /// </summary>
    None = 0,
    /// <summary>
    /// Uses a single mip tail for the sparse array.
    /// 为 sparse array 使用单个 mip tail。
    /// </summary>
    SingleMipTail = 0x01
}

/// <summary>
/// Selects the target for GPU Direct RDMA write flushing.
/// 选择 GPU Direct RDMA 写入刷新的目标。
/// </summary>
public enum CudaGpuDirectRdmaWritesTarget
{
    /// <summary>
    /// Flushes writes for the current CUDA device.
    /// 为当前 CUDA 设备刷新写入。
    /// </summary>
    CurrentDevice = 0
}

/// <summary>
/// Selects the visibility scope for GPU Direct RDMA write flushing.
/// 选择 GPU Direct RDMA 写入刷新的可见性范围。
/// </summary>
public enum CudaGpuDirectRdmaWritesScope
{
    /// <summary>
    /// Flushes writes so the owner can observe them.
    /// 刷新写入以便拥有者可见。
    /// </summary>
    ToOwner = 100,
    /// <summary>
    /// Flushes writes so all devices can observe them.
    /// 刷新写入以便所有设备都可见。
    /// </summary>
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
    /// <summary>
    /// A kernel-launch node.
    /// kernel 启动节点。
    /// </summary>
    Kernel = 0,
    /// <summary>
    /// A memory-copy node.
    /// 内存复制节点。
    /// </summary>
    Memcpy = 1,
    /// <summary>
    /// A memory-set node.
    /// 内存填充节点。
    /// </summary>
    Memset = 2,
    /// <summary>
    /// A host-callback node.
    /// 主机回调节点。
    /// </summary>
    Host = 3,
    /// <summary>
    /// A child-graph node.
    /// 子 graph 节点。
    /// </summary>
    Graph = 4,
    /// <summary>
    /// An empty synchronization node.
    /// 空同步节点。
    /// </summary>
    Empty = 5,
    /// <summary>
    /// An event-wait node.
    /// event 等待节点。
    /// </summary>
    WaitEvent = 6,
    /// <summary>
    /// An event-record node.
    /// event 记录节点。
    /// </summary>
    EventRecord = 7,
    /// <summary>
    /// An external-semaphore signal node.
    /// 外部 semaphore signal 节点。
    /// </summary>
    ExternalSemaphoreSignal = 8,
    /// <summary>
    /// An external-semaphore wait node.
    /// 外部 semaphore wait 节点。
    /// </summary>
    ExternalSemaphoreWait = 9,
    /// <summary>
    /// A memory-allocation node.
    /// 内存分配节点。
    /// </summary>
    MemoryAlloc = 10,
    /// <summary>
    /// A memory-free node.
    /// 内存释放节点。
    /// </summary>
    MemoryFree = 11,
    /// <summary>
    /// A batch memory-operation node.
    /// 批量内存操作节点。
    /// </summary>
    BatchMemoryOperation = 12,
    /// <summary>
    /// A conditional-execution node.
    /// 条件执行节点。
    /// </summary>
    Conditional = 13
}

/// <summary>
/// Flags that control CUDA managed-memory attachment behavior.
/// 控制 CUDA managed memory 附着行为的标志。
/// </summary>
[Flags]
public enum CudaManagedMemoryAttachmentFlags : uint
{
    /// <summary>
    /// Attaches the allocation globally.
    /// 全局附着该分配。
    /// </summary>
    Global = 1,
    /// <summary>
    /// Attaches the allocation to the host.
    /// 将该分配附着到 host。
    /// </summary>
    Host = 2,
    /// <summary>
    /// Attaches the allocation to a single stream.
    /// 将该分配附着到单个 stream。
    /// </summary>
    Single = 4
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
    DirectManagedMemoryAccessFromHost = 101
}
