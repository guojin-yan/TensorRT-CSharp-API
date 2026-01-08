using System;
using System.Collections.Generic;
using System.Text;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// Specifies performance hint with ::cudaAccessPolicyWindow for hitProp and missProp members.
    /// 为 ::cudaAccessPolicyWindow 的 hitProp 和 missProp 成员指定性能提示。
    /// </summary>
    public enum CudaAccessProperty
    {
        /// <summary>
        /// Normal cache persistence.
        /// 普通的缓存持久性。
        /// </summary>
        Normal = 0,

        /// <summary>
        /// Streaming access is less likely to persit from cache.
        /// 流式访问不太可能从缓存中持久保留（即更可能被驱逐）。
        /// </summary>
        Streaming = 1,

        /// <summary>
        /// Persisting access is more likely to persist in cache.
        /// 持久化访问更有可能在缓存中保留。
        /// </summary>
        Persisting = 2
    }

}
