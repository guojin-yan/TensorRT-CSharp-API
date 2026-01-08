using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// Stream Attributes
    /// 流属性标识符
    /// </summary>
    public enum CudaStreamAttrID
    {
        /// <summary>
        /// Identifier for ::cudaStreamAttrValue::accessPolicyWindow.
        /// ::cudaStreamAttrValue::accessPolicyWindow 的标识符。
        /// </summary>
        AccessPolicyWindow = 1,
        /// <summary>
        /// ::cudaSynchronizationPolicy for work queued up in this stream
        /// 此流中排队的 ::cudaSynchronizationPolicy 工作策略
        /// </summary>
        SynchronizationPolicy = 3
    }
   
}
