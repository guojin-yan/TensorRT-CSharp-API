using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /**
     * Channel format kind
     */
    public enum CudaChannelFormatKind
    {
        Signed = 0,      /**< Signed channel format */
        Unsigned = 1,      /**< Unsigned channel format */
        Float = 2,      /**< Float channel format */
        None = 3,      /**< No channel format */
        NV12 = 4
    };
}
