using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /**
     * CUDA Pitched memory pointer
     *
     * \sa ::make_cudaPitchedPtr
     */
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaPitchedPtr
    {
        public IntPtr ptr;      /**< Pointer to allocated memory */
        public ulong pitch;    /**< Pitch of allocated memory in bytes */
        public ulong xsize;    /**< Logical width of allocation in elements */
        public ulong ysize;    /**< Logical height of allocation in elements */
    };
}
