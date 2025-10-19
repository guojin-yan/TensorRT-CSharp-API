using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /**
     * CUDA Limits
     */
    public enum CudaLimit : int
    {
        StackSize                    = 0x00, /**< GPU thread stack size */
        PrintfFifoSize               = 0x01, /**< GPU printf FIFO size */
        MallocHeapSize               = 0x02, /**< GPU malloc heap size */
        DevRuntimeSyncDepth          = 0x03, /**< GPU device runtime synchronize depth */
        DevRuntimePendingLaunchCount = 0x04, /**< GPU device runtime pending launch count */
        MaxL2FetchGranularity        = 0x05, /**< A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint */
        PersistingL2CacheSize        = 0x06  /**< A size in bytes for L2 persisting lines cache size */
    };
}
