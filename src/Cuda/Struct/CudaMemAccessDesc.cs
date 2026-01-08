using JYPPX.TensorRtSharp.Cuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{ 
    /// <summary>
    /// 内存访问描述符。
    /// Memory access descriptor.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaMemAccessDesc
    {
        /// <summary>
        /// 请求更改其可访问性的位置。
        /// Location on which the request is to change it's accessibility.
        /// </summary>
        public CudaMemLocation location;
        /// <summary>
        /// 要在请求上设置的 ::CUmemProt 可访问性标志。
        /// ::CUmemProt accessibility flags to set on the request.
        /// </summary>
        public CudaMemAccessFlags flags;
    }
}
