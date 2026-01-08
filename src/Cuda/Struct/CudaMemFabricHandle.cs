using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// CUDA Mem Fabric 句柄。
    /// Represents an opaque handle for MemFabric operations.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaMemFabricHandle
    {
        private const int CUDA_IPC_HANDLE_SIZE = 64;
        /// <summary>
        /// 保留的内部数据，不透明缓冲区。
        /// Reserved internal data, opaque buffer.
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = CUDA_IPC_HANDLE_SIZE)]
        private byte[] reserved;
        public CudaMemFabricHandle(byte[] data)
        {
            this.reserved = new byte[CUDA_IPC_HANDLE_SIZE];
            if (data != null && data.Length == CUDA_IPC_HANDLE_SIZE)
            {
                Buffer.BlockCopy(data, 0, this.reserved, 0, CUDA_IPC_HANDLE_SIZE);
            }
        }
        public byte[] ToByteArray()
        {
            return (byte[])this.reserved.Clone();
        }
    }
}
