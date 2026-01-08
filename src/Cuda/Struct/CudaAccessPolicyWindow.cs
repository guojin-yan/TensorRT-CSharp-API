using System;
using System.Collections.Generic;
using System.Text;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// Specifies an access policy for a window, a contiguous extent of memory
    /// beginning at base_ptr and ending at base_ptr + num_bytes.
    /// 为窗口指定访问策略，这是一个从 base_ptr 开始到 base_ptr + num_bytes 结束的连续内存范围。
    /// </summary>
    public struct CudaAccessPolicyWindow
    {
        /// <summary>
        /// Starting address of the access policy window. CUDA driver may align it.
        /// 访问策略窗口的起始地址。CUDA 驱动程序可能会对其进行对齐处理。
        /// </summary>
        public IntPtr BasePtr;
        /// <summary>
        /// Size in bytes of the window policy. CUDA driver may restrict the maximum size and alignment.
        /// 窗口策略的大小（字节）。CUDA 驱动程序可能会限制最大大小和对齐方式。
        /// </summary>
        public long NumBytes;
        /// <summary>
        /// hitRatio specifies percentage of lines assigned hitProp, rest are assigned missProp.
        /// hitRatio 指定分配给 hitProp 的行百分比，其余分配给 missProp。
        /// </summary>
        public float HitRatio;
        /// <summary>
        /// ::CUaccessProperty set for hit.
        /// 为命中 设置的 CUaccessProperty。
        /// </summary>
        public CudaAccessProperty HitProp;
        /// <summary>
        /// ::CUaccessProperty set for miss. Must be either NORMAL or STREAMING.
        /// 为未命中 设置的 CUaccessProperty。必须是 NORMAL 或 STREAMING。
        /// </summary>
        public CudaAccessProperty MissProp;
    }
}
