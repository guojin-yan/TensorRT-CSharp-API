using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// 定义内存位置类型。
    /// Defines the memory location type.
    /// </summary>
    public enum CudaMemLocationType : int
    {
        /// <summary>
        /// 无效位置。
        /// Invalid location.
        /// </summary>
        Invalid = 0,
        /// <summary>
        /// 位置为 GPU 设备。
        /// Location is a GPU device.
        /// </summary>
        Device = 1,
        /// <summary>
        /// 位置为 CPU NUMA 节点。
        /// Location is a CPU NUMA node.
        /// </summary>
        HostNuma = 2
    }
}
