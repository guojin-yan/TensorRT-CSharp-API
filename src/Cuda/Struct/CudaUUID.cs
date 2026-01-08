using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// CUDA UUID 结构体 (16 字节)
    /// CUDA UUID structure (16 bytes)
    /// 对应于 C/C++ 中的 cudaUUID_t
    /// Corresponds to cudaUUID_t in C/C++
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaUUID
    {
        /// <summary>
        /// 16 字节的唯一标识符数据
        /// 16-byte unique identifier data
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
        public byte[] Bytes;
        /// <summary>
        /// 构造函数，初始化字节数组
        /// Constructor to initialize the byte array
        /// </summary>
        /// <param name="init">是否初始化 (Whether to initialize)</param>
        public CudaUUID(bool init = true)
        {
            Bytes = new byte[16];
        }
    }
}
