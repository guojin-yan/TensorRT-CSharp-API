using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using TensorRtSharp.Internal.PInvoke;

namespace TensorRtSharp.Internal
{
    /// <summary>
    /// TensorRT C API Return value anomaly detection handle
    /// </summary>
    static class HandleException
    {
        /// <summary>
        /// Check if there are any abnormalities in the return value, and if so, return the 
        /// corresponding exceptions according to the abnormal value
        /// </summary>
        /// <param name="status"></param>
        public static void Handler(ExceptionStatus status)
        {
            if (ExceptionStatus.NotOccurred == status)
            {
                return;
            }
            else if (ExceptionStatus.Occurred == status)
            {
                general_exception();
            }
            else if (ExceptionStatus.OccurredTRT == status)
            {
                tensorrt_exception();
            }
            else if (ExceptionStatus.OccurredCuda == status)
            {
                cuda_exception();
            }
      

        }
        /// <summary>
        /// Throw general_exception TRTException.
        /// </summary>
        /// <exception cref="OVException">general_exception!</exception>
        private static void general_exception()
        {
            throw new TRTException(ExceptionStatus.Occurred, Marshal.PtrToStringAnsi(NativeMethods.trt_get_last_err_msg()));
        }
        /// <summary>
        /// Throw tensorrt_exception TRTException.
        /// </summary>
        /// <exception cref="OVException">tensorrt_exception!</exception>
        private static void tensorrt_exception()
        {
            throw new TRTException(ExceptionStatus.OccurredTRT, Marshal.PtrToStringAnsi(NativeMethods.trt_get_last_err_msg()));
        }

        /// <summary>
        /// Throw cuda_exception TRTException.
        /// </summary>
        /// <exception cref="OVException">cuda_exception!</exception>
        private static void cuda_exception()
        {
            throw new TRTException(ExceptionStatus.OccurredCuda, Marshal.PtrToStringAnsi(NativeMethods.trt_get_last_err_msg()));
        }
    }
}
