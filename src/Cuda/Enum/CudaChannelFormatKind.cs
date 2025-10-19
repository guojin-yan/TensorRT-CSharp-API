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
        cudaChannelFormatKindSigned = 0,      /**< Signed channel format */
        cudaChannelFormatKindUnsigned = 1,      /**< Unsigned channel format */
        cudaChannelFormatKindFloat = 2,      /**< Float channel format */
        cudaChannelFormatKindNone = 3,      /**< No channel format */
        cudaChannelFormatKindNV12 = 4
    };
}
