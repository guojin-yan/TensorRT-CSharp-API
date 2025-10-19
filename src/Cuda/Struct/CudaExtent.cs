using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /**
     * CUDA extent
     *
     * \sa ::make_cudaExtent
     */
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaExtent
    {
        public ulong width;     /**< Width in elements when referring to array memory, in bytes when referring to linear memory */
        public ulong height;    /**< Height in elements */
        public ulong depth;     /**< Depth in elements */
    };

}
