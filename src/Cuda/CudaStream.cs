using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    public class CudaStream : DisposableTrtObject
    {


        /// <summary>
        /// Creates Build
        /// </summary>
        public CudaStream()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamCreate(out ptr));
        }


        /// <summary>
        /// Creates Build
        /// </summary>
        public CudaStream(uint flags)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamCreateWithFlags(out ptr, flags));
        }
        public CudaStream(uint flags, int priority)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamCreateWithPriority(out ptr, flags, priority));
        }
        /// <summary>
        /// Creates from native  pointer
        /// </summary>
        /// <param name="ptr"></param>
        internal CudaStream(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                throw new CudaException("Native object address is NULL");
            this.ptr = ptr;
        }
        /// <summary>
        /// Releases the resources
        /// </summary>
        public void Release()
        {
            Dispose();
        }

        /// <inheritdoc />
        /// <summary>
        /// Releases unmanaged resources
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            if (ptr != IntPtr.Zero && IsEnabledDispose)
                CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamDestroy(ptr));
            base.DisposeUnmanaged();
        }

    }
}
