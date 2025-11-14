using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{

    /// <summary>
    /// 硬件兼容性级别枚举，描述了与构建引擎的GPU架构之外的其他GPU架构的兼容性要求
    /// Hardware compatibility level enumeration, describing requirements of compatibility with GPU architectures 
    /// other than that of the GPU on which the engine was built
    /// </summary>
    /// <remarks>
    /// \warning 注意，与未来硬件的兼容性取决于CUDA前向兼容性支持。
    /// \warning Note that compatibility with future hardware depends on CUDA forward compatibility support.
    /// </remarks>
    public enum TrtHardwareCompatibilityLevel : int
    {
        /// <summary>
        /// 不需要与构建引擎的GPU架构之外的其他GPU架构的硬件兼容性
        /// Do not require hardware compatibility with GPU architectures other than that of the GPU on which the engine was built
        /// </summary>
        //! Do not require hardware compatibility with GPU architectures other than that of the GPU on which the engine was
        //! built.
        kNONE = 0,

        /// <summary>
        /// 要求引擎与Ampere及更新的GPU兼容。这将限制驱动程序保留和后端内核最大共享内存的组合使用为48KiB，
        /// 可能会减少每层可用的策略数量，并可能阻止某些融合的发生。因此这可能会降低性能，特别是对于tf32模型。
        /// 此选项将禁用cuDNN、cuBLAS和cuBLASLt作为策略来源。
        /// <br/>
        /// Require that the engine is compatible with Ampere and newer GPUs. This will limit the combined usage of driver
        /// reserved and backend kernel max shared memory to 48KiB, may reduce the number of available tactics for each
        /// layer, and may prevent some fusions from occurring. Thus this can decrease the performance, especially for tf32
        /// models. This option will disable cuDNN, cuBLAS, and cuBLASLt as tactic sources.
        /// </summary>
        /// <remarks>
        /// 此选项仅支持在NVIDIA Ampere及更新的GPU上构建的引擎。
        /// <br/>
        /// This option is only supported for engines built on NVIDIA Ampere and later GPUs.
        /// </remarks>
        /// <remarks>
        /// 驱动程序保留的共享内存可以通过cuDeviceGetAttribute(&reservedShmem, 
        /// CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK)查询。
        /// <br/>
        /// The driver reserved shared memory can be queried from cuDeviceGetAttribute(&reservedShmem,
        /// CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK).
        /// </remarks>
        kAMPERE_PLUS = 1,

        /// <summary>
        /// 要求引擎与其构建时所用的GPU具有相同的计算能力
        /// (https://developer.nvidia.com/cuda-gpus)的GPU兼容。与无兼容性的引擎相比，这可能会降低性能。
        /// 此选项将禁用cuDNN、cuBLAS和cuBLASLt作为策略来源。
        /// <br/>
        /// Require that the engine is compatible with GPUs that have the same Compute Capability
        /// (https://developer.nvidia.com/cuda-gpus) as the one it was built on. This may decrease the performance compared
        /// to an engine with no compatibility. This option will disable cuDNN, cuBLAS, and cuBLASLt as tactic sources.
        /// </summary>
        /// <remarks>
        /// 此选项仅支持在NVIDIA Turing及更新的GPU上构建的引擎。
        /// <br/>
        /// This option is only supported for engines built on NVIDIA Turing and later GPUs.
        /// </remarks>
        kSAME_COMPUTE_CAPABILITY = 2,
    };

}
