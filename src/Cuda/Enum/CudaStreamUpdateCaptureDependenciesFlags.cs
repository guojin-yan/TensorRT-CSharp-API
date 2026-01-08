using System;
using System.Collections.Generic;
using System.Text;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// Flags for ::cudaStreamUpdateCaptureDependencies
    /// ::cudaStreamUpdateCaptureDependencies 的标志位
    /// </summary>
    [Flags]
    public enum CudaStreamUpdateCaptureDependenciesFlags
    {
        /// <summary>
        /// Add new nodes to the dependency set
        /// 将新节点添加到依赖集中
        /// </summary>
        AddCaptureDependencies = 0x0,
        /// <summary>
        /// Replace the dependency set with the new nodes
        /// 用新节点替换依赖集
        /// </summary>
        SetCaptureDependencies = 0x1
    }
}
