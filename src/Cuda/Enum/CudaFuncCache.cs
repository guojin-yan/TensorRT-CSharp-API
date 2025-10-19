using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /**
     * CUDA function cache configurations
     */
    public enum CudaFuncCache : int
    {
        PreferNone = 0,    /**< Default function cache configuration, no preference */
        PreferShared = 1,    /**< Prefer larger shared memory and smaller L1 cache  */
        PreferL1 = 2,    /**< Prefer larger L1 cache and smaller shared memory */
        PreferEqual = 3     /**< Prefer equal size L1 cache and shared memory */
    };


}
