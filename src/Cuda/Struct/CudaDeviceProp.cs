using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// CUDA 设备属性结构体
    /// CUDA Device Properties structure
    /// 对应于 C/C++ 中的 cudaDeviceProp
    /// Corresponds to cudaDeviceProp in C/C++
    /// </summary>
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct CudaDeviceProp
    {
        /// <summary>
        /// 标识设备的 ASCII 字符串
        /// ASCII string identifying device
        /// </summary>
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
        public string Name;
        /// <summary>
        /// 16 字节的唯一标识符
        /// 16-byte unique identifier
        /// </summary>
        public CudaUUID Uuid;
        /// <summary>
        /// 8 字节的本地唯一标识符。在 TCC 和非 Windows 平台上值未定义
        /// 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        public byte[] Luid;
        /// <summary>
        /// LUID 设备节点掩码。在 TCC 和非 Windows 平台上值未定义
        /// LUID device node mask. Value is undefined on TCC and non-Windows platforms
        /// </summary>
        public uint LuidDeviceNodeMask;
        /// <summary>
        /// 设备上可用的全局内存（以字节为单位）
        /// Global memory available on device in bytes
        /// </summary>
        public ulong TotalGlobalMem;
        /// <summary>
        /// 每个 Block 可用的共享内存（以字节为单位）
        /// Shared memory available per block in bytes
        /// </summary>
        public ulong SharedMemPerBlock;
        /// <summary>
        /// 每个 Block 可用的 32 位寄存器数量
        /// 32-bit registers available per block
        /// </summary>
        public int RegsPerBlock;
        /// <summary>
        /// Warp 中包含的线程大小
        /// Warp size in threads
        /// </summary>
        public int WarpSize;
        /// <summary>
        /// 内存拷贝允许的最大间距（以字节为单位）
        /// Maximum pitch in bytes allowed by memory copies
        /// </summary>
        public ulong MemPitch;
        /// <summary>
        /// 每个 Block 的最大线程数
        /// Maximum number of threads per block
        /// </summary>
        public int MaxThreadsPerBlock;
        /// <summary>
        /// Block 每个维度的最大大小
        /// Maximum size of each dimension of a block
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] MaxThreadsDim;
        /// <summary>
        /// Grid 每个维度的最大大小
        /// Maximum size of each dimension of a grid
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] MaxGridSize;
        /// <summary>
        /// 已弃用。时钟频率（千赫）
        /// Deprecated. Clock frequency in kilohertz
        /// </summary>
        public int ClockRate;
        /// <summary>
        /// 设备上可用的常量内存（以字节为单位）
        /// Constant memory available on device in bytes
        /// </summary>
        public ulong TotalConstMem;
        /// <summary>
        /// 计算能力的主要版本号
        /// Major compute capability
        /// </summary>
        public int Major;
        /// <summary>
        /// 计算能力的次要版本号
        /// Minor compute capability
        /// </summary>
        public int Minor;
        /// <summary>
        /// 纹理的对齐要求
        /// Alignment requirement for textures
        /// </summary>
        public ulong TextureAlignment;
        /// <summary>
        /// 绑定到 Pitch 内存的纹理引用的间距对齐要求
        /// Pitch alignment requirement for texture references bound to pitched memory
        /// </summary>
        public ulong TexturePitchAlignment;
        /// <summary>
        /// 已弃用。如果设备可以同时执行内核和内存拷贝，则为 1。请改用 asyncEngineCount。
        /// Deprecated. Device can concurrently copy memory and execute a kernel. Use instead asyncEngineCount.
        /// </summary>
        public int DeviceOverlap;
        /// <summary>
        /// 设备上的多处理器数量
        /// Number of multiprocessors on device
        /// </summary>
        public int MultiProcessorCount;
        /// <summary>
        /// 已弃用。指定内核是否有运行时间限制。
        /// Deprecated. Specified whether there is a run time limit on kernels
        /// </summary>
        public int KernelExecTimeoutEnabled;
        /// <summary>
        /// 如果设备是集成的（而非独立的），则为 1
        /// Device is integrated as opposed to discrete
        /// </summary>
        public int Integrated;
        /// <summary>
        /// 设备是否可以通过 cudaHostAlloc/cudaHostGetDevicePointer 映射主机内存
        /// Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
        /// </summary>
        public int CanMapHostMemory;
        /// <summary>
        /// 已弃用。计算模式 (参见 ::cudaComputeMode)
        /// Deprecated. Compute mode (See ::cudaComputeMode)
        /// </summary>
        public int ComputeMode;
        /// <summary>
        /// 最大 1D 纹理大小
        /// Maximum 1D texture size
        /// </summary>
        public int MaxTexture1D;
        /// <summary>
        /// 最大 1D Mipmap 纹理大小
        /// Maximum 1D mipmapped texture size
        /// </summary>
        public int MaxTexture1DMipmap;
        /// <summary>
        /// 已弃用。不要使用。请改用 cudaDeviceGetTexture1DLinearMaxWidth() 或 cuDeviceGetTexture1DLinearMaxWidth()。
        /// Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead.
        /// </summary>
        public int MaxTexture1DLinear;
        /// <summary>
        /// 最大 2D 纹理维度
        /// Maximum 2D texture dimensions
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] MaxTexture2D;
        /// <summary>
        /// 最大 2D Mipmap 纹理维度
        /// Maximum 2D mipmapped texture dimensions
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] MaxTexture2DMipmap;
        /// <summary>
        /// 绑定到 Pitch 内存的 2D 纹理的最大维度（宽、高、间距）
        /// Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] MaxTexture2DLinear;
        /// <summary>
        /// 如果必须执行纹理收集操作时的最大 2D 纹理维度
        /// Maximum 2D texture dimensions if texture gather operations have to be performed
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] MaxTexture2DGather;
        /// <summary>
        /// 最大 3D 纹理维度
        /// Maximum 3D texture dimensions
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] MaxTexture3D;
        /// <summary>
        /// 最大备用 3D 纹理维度
        /// Maximum alternate 3D texture dimensions
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] MaxTexture3DAlt;
        /// <summary>
        /// 最大 Cubemap 纹理维度
        /// Maximum Cubemap texture dimensions
        /// </summary>
        public int MaxTextureCubemap;
        /// <summary>
        /// 最大 1D 分层纹理维度
        /// Maximum 1D layered texture dimensions
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] MaxTexture1DLayered;
        /// <summary>
        /// 最大 2D 分层纹理维度
        /// Maximum 2D layered texture dimensions
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] MaxTexture2DLayered;
        /// <summary>
        /// 最大 Cubemap 分层纹理维度
        /// Maximum Cubemap layered texture dimensions
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] MaxTextureCubemapLayered;
        /// <summary>
        /// 最大 1D Surface 大小
        /// Maximum 1D surface size
        /// </summary>
        public int MaxSurface1D;
        /// <summary>
        /// 最大 2D Surface 维度
        /// Maximum 2D surface dimensions
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] MaxSurface2D;
        /// <summary>
        /// 最大 3D Surface 维度
        /// Maximum 3D surface dimensions
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] MaxSurface3D;
        /// <summary>
        /// 最大 1D 分层 Surface 维度
        /// Maximum 1D layered surface dimensions
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] MaxSurface1DLayered;
        /// <summary>
        /// 最大 2D 分层 Surface 维度
        /// Maximum 2D layered surface dimensions
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] MaxSurface2DLayered;
        /// <summary>
        /// 最大 Cubemap Surface 维度
        /// Maximum Cubemap surface dimensions
        /// </summary>
        public int MaxSurfaceCubemap;
        /// <summary>
        /// 最大 Cubemap 分层 Surface 维度
        /// Maximum Cubemap layered surface dimensions
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] MaxSurfaceCubemapLayered;
        /// <summary>
        /// Surface 的对齐要求
        /// Alignment requirements for surfaces
        /// </summary>
        public ulong SurfaceAlignment;
        /// <summary>
        /// 设备可能同时执行多个内核
        /// Device can possibly execute multiple kernels concurrently
        /// </summary>
        public int ConcurrentKernels;
        /// <summary>
        /// 设备已启用 ECC 支持
        /// Device has ECC support enabled
        /// </summary>
        public int ECCEnabled;
        /// <summary>
        /// 设备的 PCI 总线 ID
        /// PCI bus ID of the device
        /// </summary>
        public int PciBusID;
        /// <summary>
        /// 设备的 PCI 设备 ID
        /// PCI device ID of the device
        /// </summary>
        public int PciDeviceID;
        /// <summary>
        /// 设备的 PCI 域 ID
        /// PCI domain ID of the device
        /// </summary>
        public int PciDomainID;
        /// <summary>
        /// 如果设备是使用 TCC 驱动程序的 Tesla 设备，则为 1，否则为 0
        /// 1 if device is a Tesla device using TCC driver, 0 otherwise
        /// </summary>
        public int TccDriver;
        /// <summary>
        /// 异步引擎的数量
        /// Number of asynchronous engines
        /// </summary>
        public int AsyncEngineCount;
        /// <summary>
        /// 设备与主机共享统一地址空间
        /// Device shares a unified address space with the host
        /// </summary>
        public int UnifiedAddressing;
        /// <summary>
        /// 已弃用。峰值内存时钟频率（千赫）
        /// Deprecated. Peak memory clock frequency in kilohertz
        /// </summary>
        public int MemoryClockRate;
        /// <summary>
        /// 全局内存总线宽度（位）
        /// Global memory bus width in bits
        /// </summary>
        public int MemoryBusWidth;
        /// <summary>
        /// L2 缓存大小（字节）
        /// Size of L2 cache in bytes
        /// </summary>
        public int L2CacheSize;
        /// <summary>
        /// 设备的最大 L2 持久化行容量设置（字节）
        /// Device's maximum l2 persisting lines capacity setting in bytes
        /// </summary>
        public int PersistingL2CacheMaxSize;
        /// <summary>
        /// 每个多处理器的最大驻留线程数
        /// Maximum resident threads per multiprocessor
        /// </summary>
        public int MaxThreadsPerMultiProcessor;
        /// <summary>
        /// 设备支持流优先级
        /// Device supports stream priorities
        /// </summary>
        public int StreamPrioritiesSupported;
        /// <summary>
        /// 设备支持在 L1 中缓存全局变量
        /// Device supports caching globals in L1
        /// </summary>
        public int GlobalL1CacheSupported;
        /// <summary>
        /// 设备支持在 L1 中缓存局部变量
        /// Device supports caching locals in L1
        /// </summary>
        public int LocalL1CacheSupported;
        /// <summary>
        /// 每个多处理器可用的共享内存（以字节为单位）
        /// Shared memory available per multiprocessor in bytes
        /// </summary>
        public ulong SharedMemPerMultiprocessor;
        /// <summary>
        /// 每个多处理器可用的 32 位寄存器数量
        /// 32-bit registers available per multiprocessor
        /// </summary>
        public int RegsPerMultiprocessor;
        /// <summary>
        /// 设备支持在此系统上分配托管内存
        /// Device supports allocating managed memory on this system
        /// </summary>
        public int ManagedMemory;
        /// <summary>
        /// 设备位于多 GPU 板上
        /// Device is on a multi-GPU board
        /// </summary>
        public int IsMultiGpuBoard;
        /// <summary>
        /// 同一多 GPU 板上的一组设备的唯一标识符
        /// Unique identifier for a group of devices on the same multi-GPU board
        /// </summary>
        public int MultiGpuBoardGroupID;
        /// <summary>
        /// 设备与主机之间的链路支持本机原子操作
        /// Link between the device and the host supports native atomic operations
        /// </summary>
        public int HostNativeAtomicSupported;
        /// <summary>
        /// 已弃用。单精度性能（每秒浮点运算次数）与双精度性能的比率
        /// Deprecated. Ratio of single precision performance (in floating-point operations per second) to double precision performance
        /// </summary>
        public int SingleToDoublePrecisionPerfRatio;
        /// <summary>
        /// 设备支持在不调用 cudaHostRegister 的情况下一致地访问可分页内存
        /// Device supports coherently accessing pageable memory without calling cudaHostRegister on it
        /// </summary>
        public int PageableMemoryAccess;
        /// <summary>
        /// 设备可以与 CPU 一致地并发访问托管内存
        /// Device can coherently access managed memory concurrently with the CPU
        /// </summary>
        public int ConcurrentManagedAccess;
        /// <summary>
        /// 设备支持计算抢占
        /// Device supports Compute Preemption
        /// </summary>
        public int ComputePreemptionSupported;
        /// <summary>
        /// 设备可以以与 CPU 相同的虚拟地址访问主机注册内存
        /// Device can access host registered memory at the same virtual address as the CPU
        /// </summary>
        public int CanUseHostPointerForRegisteredMem;
        /// <summary>
        /// 设备支持通过 ::cudaLaunchCooperativeKernel 启动协作内核
        /// Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel
        /// </summary>
        public int CooperativeLaunch;
        /// <summary>
        /// 已弃用。cudaLaunchCooperativeKernelMultiDevice 已弃用。
        /// Deprecated. cudaLaunchCooperativeKernelMultiDevice is deprecated.
        /// </summary>
        public int CooperativeMultiDeviceLaunch;
        /// <summary>
        /// 每个 Block 可通过特殊选择加入使用的最大共享内存
        /// Per device maximum shared memory per block usable by special opt in
        /// </summary>
        public ulong SharedMemPerBlockOptin;
        /// <summary>
        /// 设备通过主机的页表访问可分页内存
        /// Device accesses pageable memory via the host's page tables
        /// </summary>
        public int PageableMemoryAccessUsesHostPageTables;
        /// <summary>
        /// 主机可以直接访问设备上的托管内存而无需迁移
        /// Host can directly access managed memory on the device without migration
        /// </summary>
        public int DirectManagedMemAccessFromHost;
        /// <summary>
        /// 每个多处理器的最大驻留块数
        /// Maximum number of resident blocks per multiprocessor
        /// </summary>
        public int MaxBlocksPerMultiProcessor;
        /// <summary>
        /// ::cudaAccessPolicyWindow::num_bytes 的最大值
        /// The maximum value of ::cudaAccessPolicyWindow::num_bytes
        /// </summary>
        public int AccessPolicyMaxWindowSize;
        /// <summary>
        /// CUDA 驱动程序为每个 Block 保留的共享内存（以字节为单位）
        /// Shared memory reserved by CUDA driver per block in bytes
        /// </summary>
        public ulong ReservedSharedMemPerBlock;
        /// <summary>
        /// 设备支持通过 ::cudaHostRegister 进行主机内存注册
        /// Device supports host memory registration via ::cudaHostRegister
        /// </summary>
        public int HostRegisterSupported;
        /// <summary>
        /// 如果设备支持稀疏 CUDA 数组和稀疏 CUDA Mipmap 数组，则为 1，否则为 0
        /// 1 if the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays, 0 otherwise
        /// </summary>
        public int SparseCudaArraySupported;
        /// <summary>
        /// 设备支持使用 ::cudaHostRegister 标志 cudaHostRegisterReadOnly 来注册必须映射为 GPU 只读的内存
        /// Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU
        /// </summary>
        public int HostRegisterReadOnlySupported;
        /// <summary>
        /// 设备支持外部时间线信号量互操作
        /// External timeline semaphore interop is supported on the device
        /// </summary>
        public int TimelineSemaphoreInteropSupported;
        /// <summary>
        /// 如果设备支持使用 cudaMallocAsync 和 cudaMemPool 系列 API，则为 1，否则为 0
        /// 1 if the device supports using the cudaMallocAsync and cudaMemPool family of APIs, 0 otherwise
        /// </summary>
        public int MemoryPoolsSupported;
        /// <summary>
        /// 如果设备支持 GPUDirect RDMA API，则为 1，否则为 0
        /// 1 if the device supports GPUDirect RDMA APIs, 0 otherwise
        /// </summary>
        public int GpuDirectRDMASupported;
        /// <summary>
        /// 根据 ::cudaFlushGPUDirectRDMAWritesOptions 枚举解释的位掩码
        /// Bitmask to be interpreted according to the ::cudaFlushGPUDirectRDMAWritesOptions enum
        /// </summary>
        public uint GpuDirectRDMAFlushWritesOptions;
        /// <summary>
        /// 有关数值，请参阅 ::cudaGPUDirectRDMAWritesOrdering 枚举
        /// See the ::cudaGPUDirectRDMAWritesOrdering enum for numerical values
        /// </summary>
        public int GpuDirectRDMAWritesOrdering;
        /// <summary>
        /// 支持基于 Mpool IPC 的句柄类型的位掩码
        /// Bitmask of handle types supported with mempool-based IPC
        /// </summary>
        public uint MemoryPoolSupportedHandleTypes;
        /// <summary>
        /// 如果设备支持延迟映射 CUDA 数组和 CUDA Mipmap 数组，则为 1
        /// 1 if the device supports deferred mapping CUDA arrays and CUDA mipmapped arrays
        /// </summary>
        public int DeferredMappingCudaArraySupported;
        /// <summary>
        /// 设备支持 IPC 事件
        /// Device supports IPC Events
        /// </summary>
        public int IpcEventSupported;
        /// <summary>
        /// 指示设备支持集群启动
        /// Indicates device supports cluster launch
        /// </summary>
        public int ClusterLaunch;
        /// <summary>
        /// 指示设备支持统一指针
        /// Indicates device supports unified pointers
        /// </summary>
        public int UnifiedFunctionPointers;
        /// <summary>
        /// 保留供将来使用
        /// Reserved for future use
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] Reserved2;
        /// <summary>
        /// 保留供将来使用
        /// Reserved for future use
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 1)]
        public int[] Reserved1;
        /// <summary>
        /// 保留供将来使用
        /// Reserved for future use
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 60)]
        public int[] Reserved;

        /// <summary>
        /// 初始化结构体内的数组。
        /// 如果结构体是在 C# 中创建并传递给 C++，则应调用此方法。
        /// 通常，该结构体由 cudaGetDeviceProperties 填充。
        /// Initializes the struct arrays.
        /// Should be called if the struct is created in C# and passed to C++.
        /// though usually this struct is populated by cudaGetDeviceProperties.
        /// </summary>
        public void InitializeArrays()
        {
            if (Luid == null) Luid = new byte[8];
            if (MaxThreadsDim == null) MaxThreadsDim = new int[3];
            if (MaxGridSize == null) MaxGridSize = new int[3];
            if (MaxTexture2D == null) MaxTexture2D = new int[2];
            if (MaxTexture2DMipmap == null) MaxTexture2DMipmap = new int[2];
            if (MaxTexture2DLinear == null) MaxTexture2DLinear = new int[3];
            if (MaxTexture2DGather == null) MaxTexture2DGather = new int[2];
            if (MaxTexture3D == null) MaxTexture3D = new int[3];
            if (MaxTexture3DAlt == null) MaxTexture3DAlt = new int[3];
            if (MaxTexture1DLayered == null) MaxTexture1DLayered = new int[2];
            if (MaxTexture2DLayered == null) MaxTexture2DLayered = new int[3];
            if (MaxTextureCubemapLayered == null) MaxTextureCubemapLayered = new int[2];
            if (MaxSurface2D == null) MaxSurface2D = new int[2];
            if (MaxSurface3D == null) MaxSurface3D = new int[3];
            if (MaxSurface1DLayered == null) MaxSurface1DLayered = new int[2];
            if (MaxSurface2DLayered == null) MaxSurface2DLayered = new int[3];
            if (MaxSurfaceCubemapLayered == null) MaxSurfaceCubemapLayered = new int[2];
            if (Reserved2 == null) Reserved2 = new int[2];
            if (Reserved1 == null) Reserved1 = new int[1];
            if (Reserved == null) Reserved = new int[60];
            // Initialize UUID bytes
            if (Uuid.Bytes == null) Uuid = new CudaUUID(true);
        }
    }
}
