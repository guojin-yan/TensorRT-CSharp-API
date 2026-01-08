using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// 用于指定特定句柄类型的标志。
    /// Flags for specifying particular handle types.
    /// </summary>
    public enum CudaMemAllocationHandleType : int
    {
        /// <summary>
        /// 不允许任何导出机制。
        /// Does not allow any export mechanism.
        /// </summary>
        None = 0x0,
        /// <summary>
        /// 允许使用文件描述符进行导出。仅在 POSIX 系统上允许。
        /// Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int)
        /// </summary>
        PosixFileDescriptor = 0x1,
        /// <summary>
        /// 允许使用 Win32 NT 句柄进行导出。
        /// Allows a Win32 NT handle to be used for exporting. (HANDLE)
        /// </summary>
        Win32 = 0x2,
        /// <summary>
        /// 允许使用 Win32 KMT 句柄进行导出。
        /// Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE)
        /// </summary>
        Win32Kmt = 0x4
    }
}
