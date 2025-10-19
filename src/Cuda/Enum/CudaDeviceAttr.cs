using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /**
     * CUDA device attributes
     */
    public enum CudaDeviceAttr : int
    {
        MaxThreadsPerBlock             = 1,  /**< Maximum number of threads per block */
        MaxBlockDimX                   = 2,  /**< Maximum block dimension X */
        MaxBlockDimY                   = 3,  /**< Maximum block dimension Y */
        MaxBlockDimZ                   = 4,  /**< Maximum block dimension Z */
        MaxGridDimX                    = 5,  /**< Maximum grid dimension X */
        MaxGridDimY                    = 6,  /**< Maximum grid dimension Y */
        MaxGridDimZ                    = 7,  /**< Maximum grid dimension Z */
        MaxSharedMemoryPerBlock        = 8,  /**< Maximum shared memory available per block in bytes */
        TotalConstantMemory            = 9,  /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
        WarpSize                       = 10, /**< Warp size in threads */
        MaxPitch                       = 11, /**< Maximum pitch in bytes allowed by memory copies */
        MaxRegistersPerBlock           = 12, /**< Maximum number of 32-bit registers available per block */
        ClockRate                      = 13, /**< Peak clock frequency in kilohertz */
        TextureAlignment               = 14, /**< Alignment requirement for textures */
        GpuOverlap                     = 15, /**< Device can possibly copy memory and execute a kernel concurrently */
        MultiProcessorCount            = 16, /**< Number of multiprocessors on device */
        KernelExecTimeout              = 17, /**< Specifies whether there is a run time limit on kernels */
        Integrated                     = 18, /**< Device is integrated with host memory */
        CanMapHostMemory               = 19, /**< Device can map host memory into CUDA address space */
        ComputeMode                    = 20, /**< Compute mode (See ::cudaComputeMode for details) */
        MaxTexture1DWidth              = 21, /**< Maximum 1D texture width */
        MaxTexture2DWidth              = 22, /**< Maximum 2D texture width */
        MaxTexture2DHeight             = 23, /**< Maximum 2D texture height */
        MaxTexture3DWidth              = 24, /**< Maximum 3D texture width */
        MaxTexture3DHeight             = 25, /**< Maximum 3D texture height */
        MaxTexture3DDepth              = 26, /**< Maximum 3D texture depth */
        MaxTexture2DLayeredWidth       = 27, /**< Maximum 2D layered texture width */
        MaxTexture2DLayeredHeight      = 28, /**< Maximum 2D layered texture height */
        MaxTexture2DLayeredLayers      = 29, /**< Maximum layers in a 2D layered texture */
        SurfaceAlignment               = 30, /**< Alignment requirement for surfaces */
        ConcurrentKernels              = 31, /**< Device can possibly execute multiple kernels concurrently */
        EccEnabled                     = 32, /**< Device has ECC support enabled */
        PciBusId                       = 33, /**< PCI bus ID of the device */
        PciDeviceId                    = 34, /**< PCI device ID of the device */
        TccDriver                      = 35, /**< Device is using TCC driver model */
        MemoryClockRate                = 36, /**< Peak memory clock frequency in kilohertz */
        GlobalMemoryBusWidth           = 37, /**< Global memory bus width in bits */
        L2CacheSize                    = 38, /**< Size of L2 cache in bytes */
        MaxThreadsPerMultiProcessor    = 39, /**< Maximum resident threads per multiprocessor */
        AsyncEngineCount               = 40, /**< Number of asynchronous engines */
        UnifiedAddressing              = 41, /**< Device shares a unified address space with the host */    
        MaxTexture1DLayeredWidth       = 42, /**< Maximum 1D layered texture width */
        MaxTexture1DLayeredLayers      = 43, /**< Maximum layers in a 1D layered texture */
        MaxTexture2DGatherWidth        = 45, /**< Maximum 2D texture width if cudaArrayTextureGather is set */
        MaxTexture2DGatherHeight       = 46, /**< Maximum 2D texture height if cudaArrayTextureGather is set */
        MaxTexture3DWidthAlt           = 47, /**< Alternate maximum 3D texture width */
        MaxTexture3DHeightAlt          = 48, /**< Alternate maximum 3D texture height */
        MaxTexture3DDepthAlt           = 49, /**< Alternate maximum 3D texture depth */
        PciDomainId                    = 50, /**< PCI domain ID of the device */
        TexturePitchAlignment          = 51, /**< Pitch alignment requirement for textures */
        MaxTextureCubemapWidth         = 52, /**< Maximum cubemap texture width/height */
        MaxTextureCubemapLayeredWidth  = 53, /**< Maximum cubemap layered texture width/height */
        MaxTextureCubemapLayeredLayers = 54, /**< Maximum layers in a cubemap layered texture */
        MaxSurface1DWidth              = 55, /**< Maximum 1D surface width */
        MaxSurface2DWidth              = 56, /**< Maximum 2D surface width */
        MaxSurface2DHeight             = 57, /**< Maximum 2D surface height */
        MaxSurface3DWidth              = 58, /**< Maximum 3D surface width */
        MaxSurface3DHeight             = 59, /**< Maximum 3D surface height */
        MaxSurface3DDepth              = 60, /**< Maximum 3D surface depth */
        MaxSurface1DLayeredWidth       = 61, /**< Maximum 1D layered surface width */
        MaxSurface1DLayeredLayers      = 62, /**< Maximum layers in a 1D layered surface */
        MaxSurface2DLayeredWidth       = 63, /**< Maximum 2D layered surface width */
        MaxSurface2DLayeredHeight      = 64, /**< Maximum 2D layered surface height */
        MaxSurface2DLayeredLayers      = 65, /**< Maximum layers in a 2D layered surface */
        MaxSurfaceCubemapWidth         = 66, /**< Maximum cubemap surface width */
        MaxSurfaceCubemapLayeredWidth  = 67, /**< Maximum cubemap layered surface width */
        MaxSurfaceCubemapLayeredLayers = 68, /**< Maximum layers in a cubemap layered surface */
        MaxTexture1DLinearWidth        = 69, /**< Maximum 1D linear texture width */
        MaxTexture2DLinearWidth        = 70, /**< Maximum 2D linear texture width */
        MaxTexture2DLinearHeight       = 71, /**< Maximum 2D linear texture height */
        MaxTexture2DLinearPitch        = 72, /**< Maximum 2D linear texture pitch in bytes */
        MaxTexture2DMipmappedWidth     = 73, /**< Maximum mipmapped 2D texture width */
        MaxTexture2DMipmappedHeight    = 74, /**< Maximum mipmapped 2D texture height */
        ComputeCapabilityMajor         = 75, /**< Major compute capability version number */ 
        ComputeCapabilityMinor         = 76, /**< Minor compute capability version number */
        MaxTexture1DMipmappedWidth     = 77, /**< Maximum mipmapped 1D texture width */
        StreamPrioritiesSupported      = 78, /**< Device supports stream priorities */
        GlobalL1CacheSupported         = 79, /**< Device supports caching globals in L1 */
        LocalL1CacheSupported          = 80, /**< Device supports caching locals in L1 */
        MaxSharedMemoryPerMultiprocessor = 81, /**< Maximum shared memory available per multiprocessor in bytes */
        MaxRegistersPerMultiprocessor  = 82, /**< Maximum number of 32-bit registers available per multiprocessor */
        ManagedMemory                  = 83, /**< Device can allocate managed memory on this system */
        IsMultiGpuBoard                = 84, /**< Device is on a multi-GPU board */
        MultiGpuBoardGroupID           = 85, /**< Unique identifier for a group of devices on the same multi-GPU board */
        HostNativeAtomicSupported      = 86, /**< Link between the device and the host supports native atomic operations */
        SingleToDoublePrecisionPerfRatio = 87, /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
        PageableMemoryAccess           = 88, /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
        ConcurrentManagedAccess        = 89, /**< Device can coherently access managed memory concurrently with the CPU */
        ComputePreemptionSupported     = 90, /**< Device supports Compute Preemption */
        CanUseHostPointerForRegisteredMem = 91, /**< Device can access host registered memory at the same virtual address as the CPU */
        Reserved92                     = 92,
        Reserved93                     = 93,
        Reserved94                     = 94,
        CooperativeLaunch              = 95, /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel*/
        CooperativeMultiDeviceLaunch   = 96, /**< Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated. */
        MaxSharedMemoryPerBlockOptin   = 97, /**< The maximum optin shared memory per block. This value may vary by chip. See ::cudaFuncSetAttribute */
        CanFlushRemoteWrites           = 98, /**< Device supports flushing of outstanding remote writes. */
        HostRegisterSupported          = 99, /**< Device supports host memory registration via ::cudaHostRegister. */
        PageableMemoryAccessUsesHostPageTables = 100, /**< Device accesses pageable memory via the host's page tables. */
        DirectManagedMemAccessFromHost = 101, /**< Host can directly access managed memory on the device without migration. */
        MaxBlocksPerMultiprocessor     = 106, /**< Maximum number of blocks per multiprocessor */
        MaxPersistingL2CacheSize       = 108, /**< Maximum L2 persisting lines capacity setting in bytes. */
        MaxAccessPolicyWindowSize      = 109, /**< Maximum value of cudaAccessPolicyWindow::num_bytes. */
        ReservedSharedMemoryPerBlock   = 111, /**< Shared memory reserved by CUDA driver per block in bytes */
        SparseCudaArraySupported       = 112, /**< Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays */
        HostRegisterReadOnlySupported  = 113,  /**< Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU */
        MaxTimelineSemaphoreInteropSupported = 114,  /**< External timeline semaphore interop is supported on the device */
        MemoryPoolsSupported           = 115, /**< Device supports using the ::cudaMallocAsync and ::cudaMemPool family of APIs */
        GPUDirectRDMASupported         = 116, /**< Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information) */
        GPUDirectRDMAFlushWritesOptions = 117, /**< The returned attribute shall be interpreted as a bitmask, where the individual bits are listed in the ::cudaFlushGPUDirectRDMAWritesOptions enum */
        GPUDirectRDMAWritesOrdering    = 118, /**< GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::cudaGPUDirectRDMAWritesOrdering for the numerical values returned here. */
        MemoryPoolSupportedHandleTypes = 119, /**< Handle types supported with mempool based IPC */
        Max
    };

}
