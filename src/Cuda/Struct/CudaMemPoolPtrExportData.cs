using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{

    /// <summary>
    /// 用于导出内存池分配的不透明数据。
    /// Opaque data for exporting a pool allocation.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaMemPoolPtrExportData
    {
        /// <summary>
        /// 保留字段，用于存储导出数据的具体内容。
        /// Reserved field for storing the specific content of the export data.
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 64)]
        public byte[] reserved;
        /// <summary>
        /// 初始化结构体，分配内存给保留字段。
        /// Initializes the struct, allocating memory for the reserved field.
        /// </summary>
        public CudaMemPoolPtrExportData(bool initReserved = true)
        {
            if (initReserved)
            {
                // 必须初始化数组，否则在进行 Marshal 操作时会抛出异常
                // Must initialize the array, otherwise Marshal operations will throw an exception
                reserved = new byte[64];
            }
            else
            {
                reserved = null;
            }
        }
    }
}
