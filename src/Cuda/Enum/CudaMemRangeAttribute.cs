using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda.Enum
{
    /// <summary>
    /// 内存范围属性
    /// </summary>
    public enum CudaMemRangeAttribute
    {
        ReadMostly = 1, /**< Whether the range will mostly be read and only occassionally be written to */
        PreferredLocation = 2, /**< The preferred location of the range */
        AccessedBy = 3, /**< Memory range has ::cudaMemAdviseSetAccessedBy set for specified device */
        LastPrefetchLocation = 4  /**< The last location to which the range was prefetched */
    }
}
