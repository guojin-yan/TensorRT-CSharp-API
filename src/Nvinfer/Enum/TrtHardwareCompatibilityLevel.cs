using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{

    //!
    //! \enum HardwareCompatibilityLevel
    //!
    //! \brief Describes requirements of compatibility with GPU architectures other than that of the GPU on which the engine
    //! was built.
    //!
    //! \warning Note that compatibility with future hardware depends on CUDA forward compatibility support.
    //!
    public enum TrtHardwareCompatibilityLevel : int
    {
        //! Do not require hardware compatibility with GPU architectures other than that of the GPU on which the engine was
        //! built.
        kNONE = 0,

        //! Require that the engine is compatible with Ampere and newer GPUs. This will limit the combined usage of driver
        //! reserved and backend kernel max shared memory to 48KiB, may reduce the number of available tactics for each
        //! layer, and may prevent some fusions from occurring. Thus this can decrease the performance, especially for tf32
        //! models.
        //! This option will disable cuDNN, cuBLAS, and cuBLASLt as tactic sources.
        //!
        //! This option is only supported for engines built on NVIDIA Ampere and later GPUs.
        //!
        //! The driver reserved shared memory can be queried from cuDeviceGetAttribute(&reservedShmem,
        //! CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK).
        //!
        kAMPERE_PLUS = 1,

        //! Require that the engine is compatible with GPUs that have the same Compute Capability
        //! (https://developer.nvidia.com/cuda-gpus) as the one it was built on. This may decrease the performance compared
        //! to an engine with no compatibility.
        //!
        //! This option will disable cuDNN, cuBLAS, and cuBLASLt as tactic sources.
        //!
        //! This option is only supported for engines built on NVIDIA Turing and later GPUs.
        //!
        kSAME_COMPUTE_CAPABILITY = 2,
    };
}
