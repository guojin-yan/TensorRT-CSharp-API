using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    public static class CudaRuntime
    {
        public static int getRuntimeVersion()
        {
            CudaHandleException.handler(NativeMethods.cudaRuntime_cudaRuntimeGetVersion(out int version));

            return version;
        }

        public static int getDriverVersion() {
            CudaHandleException.handler(NativeMethods.cudaRuntime_cudaDriverGetVersion(out int version));
            return version;
        }
    }
}
