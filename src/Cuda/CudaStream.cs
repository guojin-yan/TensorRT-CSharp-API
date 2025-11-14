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
    /// <summary>
    /// 表示一个CUDA流，用于在GPU上并发执行操作。继承自DisposableTrtObject。
    /// Represents a CUDA stream for concurrent execution of operations on a GPU. Inherits from DisposableTrtObject.
    /// </summary>
    public class CudaStream : DisposableTrtObject
    {


        /// <summary>
        /// 创建一个默认的CUDA流。
        /// Creates a default CUDA stream.
        /// </summary>
        public CudaStream()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamCreate(out ptr));
        }


        /// <summary>
        /// 使用指定的标志创建一个CUDA流。
        /// Creates a CUDA stream with the specified flags.
        /// </summary>
        /// <param name="flags">流的标志，用于控制流的行为（例如，非阻塞流使用 cudaStreamNonBlocking）。/ Flags for the stream to control its behavior (e.g., use cudaStreamNonBlocking for a non-blocking stream).</param>
        public CudaStream(uint flags)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamCreateWithFlags(out ptr, flags));
        }

        /// <summary>
        /// 使用指定的标志和优先级创建一个CUDA流。
        /// Creates a CUDA stream with the specified flags and priority.
        /// </summary>
        /// <param name="flags">流的标志。/ Flags for the stream.</param>
        /// <param name="priority">流的优先级。较低的数值代表较高的优先级。/ The priority of the stream. Lower numbers represent higher priorities.</param>
        public CudaStream(uint flags, int priority)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamCreateWithPriority(out ptr, flags, priority));
        }

        /// <summary>
        /// 使用一个原生指针来初始化 CudaStream 实例。主要用于内部封装。
        /// Initializes a CudaStream instance from a native pointer. Primarily used for internal wrapping.
        /// </summary>
        /// <param name="ptr">指向原生对象的非托管指针。/ The unmanaged pointer to the native object.</param>
        internal CudaStream(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                throw new CudaException("Native object address is NULL");
            this.ptr = ptr;
        }

        /// <summary>
        /// 释放当前对象持有的所有资源。此方法为 Dispose 的显式别名。
        /// Releases all resources held by the current object. This method is an explicit alias for Dispose.
        /// </summary>
        public void Release()
        {
            Dispose();
        }

        /// <inheritdoc />
        /// <summary>
        /// 释放所有非托管资源，即销毁CUDA流。此方法由 Dispose 模式调用，不应直接调用。
        /// Releases all unmanaged resources, i.e., destroys the CUDA stream. This method is called by the Dispose pattern and should not be called directly.
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            if (ptr != IntPtr.Zero && IsEnabledDispose)
                CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamDestroy(ptr));
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// 阻塞CPU线程，直到此流中的所有任务都完成执行。
        /// Blocks the CPU thread until all tasks in this stream have completed execution.
        /// </summary>
        public void Synchronize()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamSynchronize(ptr));
        }

    }

}
