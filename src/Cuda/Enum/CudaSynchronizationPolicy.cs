using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// Possible stream capture statuses returned by ::cudaStreamIsCapturing (Note: Original comment seems mismatched in source, but mapped here correctly).
    /// 同步策略枚举。
    /// </summary>
    public enum CudaSynchronizationPolicy
    {
        /// <summary>
        /// Default automatic policy
        /// 默认自动策略
        /// </summary>
        Auto = 1,
        /// <summary>
        /// Spin (busy-wait) policy
        /// 自旋（忙等待）策略
        /// </summary>
        Spin = 2,
        /// <summary>
        /// Yield (context switch) policy
        /// 让步（上下文切换）策略
        /// </summary>
        Yield = 3,
        /// <summary>
        /// Blocking sync policy
        /// 阻塞同步策略
        /// </summary>
        BlockingSync = 4
    }
   
}
