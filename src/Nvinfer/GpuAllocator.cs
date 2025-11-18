using JYPPX.TensorRtSharp.Cuda;
using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    public class GpuAllocator : DisposableTrtObject
    {


        /// <summary>
        /// Creates empty GpuAllocator
        /// </summary>
        public GpuAllocator()
        {
        }


        /// <summary>
        /// Creates from native  pointer
        /// </summary>
        /// <param name="ptr"></param>
        internal GpuAllocator(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                throw new TrtException("Native object address is NULL");
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
                NativeMethods.trtGpuAllocator_free(ptr);
            base.DisposeUnmanaged();
        }


        public void allocateAsync(ulong size, ulong alignment, uint flags, CudaStream stream, out IntPtr memoryPtr)
        {
            TrtHandleException.handler(
                NativeMethods.trtGpuAllocator_allocateAsync(
                    ptr,
                    size,
                    alignment,
                    flags,
                    stream.TrtPtr,
                    out memoryPtr));
        }
        

        public bool deallocateAsync(IntPtr memory, CudaStream stream)
        {
            TrtHandleException.handler(
                NativeMethods.trtGpuAllocator_deallocateAsync(
                    ptr,
                    memory,
                    stream.TrtPtr,
                    out int successStatus));
            return successStatus != 0;
        }
       
        public void reallocate(IntPtr baseAddr, ulong alignment, ulong newSize, out IntPtr memoryPtr)
        {
            TrtHandleException.handler(
                NativeMethods.trtGpuAllocator_reallocate(
                    ptr,
                    baseAddr,
                    alignment,
                    newSize,
                    out memoryPtr));
        }
        

        public InterfaceInfo getInterfaceInfo()
        {
            TrtHandleException.handler(
                NativeMethods.trtGpuAllocator_getInterfaceInfo(
                    ptr,
                    out InterfaceInfo info));
            return info;
        }

    }
}
