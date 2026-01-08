using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// 指定映射的内存保护标志。
    /// Specifies the memory protection flags for mapping.
    /// </summary>
    public enum CudaMemAccessFlags : int
    {
        /// <summary>
        /// 默认值，使地址范围不可访问。
        /// Default, make the address range not accessible.
        /// </summary>
        ProtNone = 0,
        /// <summary>
        /// 使地址范围可读。
        /// Make the address range read accessible.
        /// </summary>
        ProtRead = 1,
        /// <summary>
        /// 使地址范围可读写。
        /// Make the address range read-write accessible.
        /// </summary>
        ProtReadWrite = 3
    }
}
