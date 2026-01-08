using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// 表示 CUDA 内存池的句柄。
    /// </summary>
    public readonly struct CudaMemPoolStr
    {
        public readonly IntPtr Handle;

        public CudaMemPoolStr(IntPtr handle)
        {
            Handle = handle;
        }
    }
}
