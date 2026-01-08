using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// Stream attributes union used with ::cudaStreamSetAttribute/::cudaStreamGetAttribute
    /// 与 ::cudaStreamSetAttribute/::cudaStreamGetAttribute 一起使用的流属性联合体
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CudaStreamAttrValue
    {
        /// <summary>
        /// Access Policy Window value
        /// 访问策略窗口的值
        /// </summary>
        [FieldOffset(0)]
        public CudaAccessPolicyWindow AccessPolicyWindow;
        /// <summary>
        /// Synchronization Policy value
        /// 同步策略的值
        /// </summary>
        [FieldOffset(0)]
        public CudaSynchronizationPolicy SyncPolicy;
    }
   
}
