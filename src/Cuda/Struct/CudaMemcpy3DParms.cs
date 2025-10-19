using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /**
     * CUDA 3D memory copying parameters
     */
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaMemcpy3DParms
    {
        public IntPtr srcArray;  /**< Source memory address */
        public CudaPos srcPos;    /**< Source position offset */
        public CudaPitchedPtr srcPtr;    /**< Pitched source memory address */

        public IntPtr dstArray;  /**< Destination memory address */
        public CudaPos dstPos;    /**< Destination position offset */
        public CudaPitchedPtr dstPtr;    /**< Pitched destination memory address */

        public CudaExtent extent;    /**< Requested memory copy size */
        public CudaMemcpyKind kind;      /**< Type of transfer */
    };


}
