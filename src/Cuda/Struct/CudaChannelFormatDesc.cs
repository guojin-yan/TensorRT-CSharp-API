using JYPPX.TensorRtSharp.Cuda.Enum;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda.Struct
{

    public struct CudaChannelFormatDesc
    {
        int x; /**< x */
        int y; /**< y */
        int z; /**< z */
        int w; /**< w */
        CudaChannelFormatKind f; /**< Channel format kind */
    };

}
