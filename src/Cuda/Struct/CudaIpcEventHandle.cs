using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// CUDA IPC 事件句柄。
    /// Represents an opaque handle that can be used to open an event in another process.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaIpcEventHandle
    {
        private const int CUDA_IPC_HANDLE_SIZE = 64;
        /// <summary>
        /// 保留的内部数据，不透明缓冲区。
        /// Reserved internal data, opaque buffer.
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = CUDA_IPC_HANDLE_SIZE)]
        private byte[] reserved;
        /// <summary>
        /// 初始化句柄。
        /// Initializes the handle.
        /// </summary>
        public CudaIpcEventHandle(byte[] data)
        {
            this.reserved = new byte[CUDA_IPC_HANDLE_SIZE];
            if (data != null && data.Length == CUDA_IPC_HANDLE_SIZE)
            {
                Buffer.BlockCopy(data, 0, this.reserved, 0, CUDA_IPC_HANDLE_SIZE);
            }
        }
        /// <summary>
        /// 将句柄转换为字节数组以便传输。
        /// Converts the handle to a byte array for transport.
        /// </summary>
        public byte[] ToByteArray()
        {
            return (byte[])this.reserved.Clone();
        }
    }
}
