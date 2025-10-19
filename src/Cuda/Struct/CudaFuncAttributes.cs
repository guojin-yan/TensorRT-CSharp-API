using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{

    /**
     * CUDA function attributes
     */
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaFuncAttributes
    {
        /**
         * The size in bytes of statically-allocated shared memory per block
         * required by this function. This does not include dynamically-allocated
         * shared memory requested by the user at runtime.
         */
        long sharedSizeBytes;

        /**
         * The size in bytes of user-allocated constant memory required by this
         * function.
         */
        long constSizeBytes;

        /**
         * The size in bytes of local memory used by each thread of this function.
         */
        long localSizeBytes;

        /**
         * The maximum number of threads per block, beyond which a launch of the
         * function would fail. This number depends on both the function and the
         * device on which the function is currently loaded.
         */
        int maxThreadsPerBlock;

        /**
         * The number of registers used by each thread of this function.
         */
        int numRegs;

        /**
         * The PTX virtual architecture version for which the function was
         * compiled. This value is the major PTX version * 10 + the minor PTX
         * version, so a PTX version 1.3 function would return the value 13.
         */
        int ptxVersion;

        /**
         * The binary architecture version for which the function was compiled.
         * This value is the major binary version * 10 + the minor binary version,
         * so a binary version 1.3 function would return the value 13.
         */
        int binaryVersion;

        /**
         * The attribute to indicate whether the function has been compiled with 
         * user specified option "-Xptxas --dlcm=ca" set.
         */
        int cacheModeCA;

        /**
         * The maximum size in bytes of dynamic shared memory per block for 
         * this function. Any launch must have a dynamic shared memory size
         * smaller than this value.
         */
        int maxDynamicSharedSizeBytes;

        /**
         * On devices where the L1 cache and shared memory use the same hardware resources, 
         * this sets the shared memory carveout preference, in percent of the maximum shared memory. 
         * Refer to ::cudaDevAttrMaxSharedMemoryPerMultiprocessor.
         * This is only a hint, and the driver can choose a different ratio if required to execute the function.
         * See ::cudaFuncSetAttribute
         */
        int preferredShmemCarveout;
    };

}
