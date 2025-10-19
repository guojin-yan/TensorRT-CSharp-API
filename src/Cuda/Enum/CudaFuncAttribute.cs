using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /**
     * CUDA function attributes that can be set using ::cudaFuncSetAttribute
     */
    public enum CudaFuncAttribute : int
    {
        MaxDynamicSharedMemorySize = 8, /**< Maximum dynamic shared memory size */
        PreferredSharedMemoryCarveout = 9, /**< Preferred shared memory-L1 cache split */
        Max
    };
}

