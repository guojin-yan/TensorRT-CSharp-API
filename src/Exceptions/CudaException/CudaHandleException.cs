using JYPPX.TensorRtSharp;
using JYPPX.TensorRtSharp.ExternalInterface;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Exceptions
{
    /// <summary>
    /// OpenVINO C API Return value anomaly detection handle
    /// </summary>
    static class CudaHandleException
    {
        /// <summary>
        /// Check if there are any abnormalities in the return value, and if so, return the 
        /// corresponding exceptions according to the abnormal value
        /// </summary>
        /// <param name="status"></param>
        public static void handler(CudaExceptionStatus status)
        {
            if (CudaExceptionStatus.CudaSuccess == status)
            {
                return;
            }
            else
            {
                throw new CudaException(status, Marshal.PtrToStringAnsi(NativeMethods.GetLastErrMsg())); ;
            }
        }
    }
}
