using JYPPX.TensorRtSharp.Cuda.Enum;
using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    public class CudaManagedMemory : DisposableTrtObject
    {


        public CudaManagedMemory(ulong size, CudaMemAttach flags)
        {
            this.size = size;
            CudaHandleException.handler(NativeMethods.cudaRuntime_cudaMallocManaged(out ptr, size, flags));
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
                NativeMethods.cudaRuntime_cudaFree(ptr));
            base.DisposeUnmanaged();
        }

        public void prefetchAsync(int device_id, CudaStream stream)
        {
            CudaHandleException.handler(
                    NativeMethods.cudaRuntime_cudaMemPrefetchAsync(ptr, size * sizeof(ulong), device_id, stream.TrtPtr));
        }
        public void advise(CudaMemoryAdvise advice, int device = 0)
        {
            CudaHandleException.handler(
                    NativeMethods.cudaRuntime_cudaMemAdvise(ptr, size * sizeof(ulong), advice, device));
        }

        private ulong size;
    }
}
