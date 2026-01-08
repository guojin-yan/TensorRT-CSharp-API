using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// 指定一个内存位置。
    /// Specifies a memory location.
    /// 
    /// 要指定 GPU，请将 type 设置为 ::cudaMemLocationTypeDevice 并将 id 设置为 GPU 的设备序号。
    /// To specify a gpu, set type = ::cudaMemLocationTypeDevice and set id = the gpu's device ordinal.
    /// 
    /// 要指定 CPU NUMA 节点，请将 type 设置为 ::cudaMemLocationTypeHostNuma 并将 id 设置为主机 NUMA 节点 ID。
    /// To specify a cpu NUMA node, set type = ::cudaMemLocationTypeHostNuma and set id = host NUMA node id.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaMemLocation
    {
        /// <summary>
        /// 指定位置类型，该类型将修改 id 的含义。
        /// Specifies the location type, which modifies the meaning of id.
        /// </summary>
        public CudaMemLocationType type;
        /// <summary>
        /// 给定位置的标识符。
        /// identifier for a given this location's ::cudaMemLocationType.
        /// </summary>
        public int id;
    }
}
