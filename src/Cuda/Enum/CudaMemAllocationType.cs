using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    
     /// <summary>
     /// 定义可用的分配类型。
     /// Defines the allocation types available.
     /// </summary>
    public enum CudaMemAllocationType : int
    {
        /// <summary>
        /// 无效的分配类型。
        /// Invalid allocation type.
        /// </summary>
        Invalid = 0x0,
        /// <summary>
        /// 此分配类型是“固定”的，即在应用程序主动使用期间无法从其当前位置迁移。
        /// This allocation type is 'pinned', i.e. cannot migrate from its current
        /// location while the application is actively using it.
        /// </summary>
        Pinned = 0x1,
        /// <summary>
        /// 最大枚举值（用于边界检查等）。
        /// Max enum value.
        /// </summary>
        Max = 0x7FFFFFFF
    }
}
