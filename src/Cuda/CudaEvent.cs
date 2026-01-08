using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Runtime.InteropServices;


namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// 表示一个 CUDA 事件，用于 CUDA 流中的操作同步和性能监控。继承自 DisposableTrtObject。
    /// Represents a CUDA event for operation synchronization and performance monitoring in CUDA streams. Inherits from DisposableTrtObject.
    /// </summary>
    public class CudaEvent : DisposableTrtObject
    {
        /// <summary>
        /// 获取此事件对象的非托管指针。
        /// Gets the unmanaged pointer of this event object.
        /// </summary>
        public IntPtr NativePtr => ptr;

        /// <summary>
        /// 创建一个默认的 CUDA 事件。
        /// Creates a default CUDA event.
        /// </summary>
        public CudaEvent()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaEventCreate(out ptr));
        }

        /// <summary>
        /// 使用指定的标志创建一个 CUDA 事件。
        /// Creates a CUDA event with the specified flags.
        /// </summary>
        /// <param name="flags">事件标志，用于控制事件的行为（例如，禁用计时或启用阻塞同步）。/ Event flags to control the behavior (e.g., disable timing or enable blocking sync).</param>
        public CudaEvent(uint flags)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaEventCreateWithFlags(out ptr, flags));
        }

        /// <summary>
        /// 使用一个原生指针来初始化 CudaEvent 实例。主要用于内部封装。
        /// Initializes a CudaEvent instance from a native pointer. Primarily used for internal wrapping.
        /// </summary>
        /// <param name="ptr">指向原生对象的非托管指针。/ The unmanaged pointer to the native object.</param>
        internal CudaEvent(IntPtr ptr)
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
        /// 释放所有非托管资源，即销毁CUDA事件。此方法由 Dispose 模式调用，不应直接调用。
        /// Releases all unmanaged resources, i.e., destroys the CUDA event. This method is called by the Dispose pattern and should not be called directly.
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            if (ptr != IntPtr.Zero && IsEnabledDispose)
                CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaEventDestroy(ptr));
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// 将事件记录到指定的流中。如果流为 null，则记录到默认流中。
        /// Records the event into the specified stream. If the stream is null, records into the default stream.
        /// </summary>
        /// <param name="stream">要记录事件的流。/ The stream to record the event into.</param>
        public void Record(CudaStream stream = null)
        {
            IntPtr streamPtr = stream != null ? stream.NativePtr : IntPtr.Zero;
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaEventRecord(ptr, streamPtr));
        }

        /// <summary>
        /// 使用指定的标志将事件记录到指定的流中。
        /// Records the event into the specified stream with the specified flags.
        /// </summary>
        /// <param name="stream">要记录事件的流。/ The stream to record the event into.</param>
        /// <param name="flags">记录标志。/ The record flags.</param>
        public void RecordWithFlags(CudaStream stream, uint flags)
        {
            IntPtr streamPtr = stream != null ? stream.NativePtr : IntPtr.Zero;
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaEventRecordWithFlags(ptr, streamPtr, flags));
        }

        /// <summary>
        /// 查询事件是否已记录。
        /// Queries if the event has been recorded.
        /// </summary>
        /// <returns>如果事件已记录返回 true，如果事件未记录返回 false。/ Returns true if the event has been recorded, false if not.</returns>
        public bool Query()
        {
            var status = NativeMethods.cudaRuntime_cudaEventQuery(ptr);
            if (status == CudaExceptionStatus.CudaErrorNotReady)
            {
                return false;
            }
            CudaHandleException.handler(status);
            return true;
        }

        /// <summary>
        /// 阻塞 CPU 线程，直到事件完成。
        /// Blocks the CPU thread until the event completes.
        /// </summary>
        public void Synchronize()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaEventSynchronize(ptr));
        }

        /// <summary>
        /// 计算两个事件之间经过的时间（以毫秒为单位）。
        /// Computes the elapsed time between two events (in milliseconds).
        /// </summary>
        /// <param name="start">起始事件。/ The start event.</param>
        /// <param name="end">结束事件。/ The end event.</param>
        /// <returns>经过的时间（毫秒）。/ The elapsed time in milliseconds.</returns>
        public static float ElapsedTime(CudaEvent start, CudaEvent end)
        {
            if (start == null) throw new ArgumentNullException(nameof(start));
            if (end == null) throw new ArgumentNullException(nameof(end));

            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaEventElapsedTime(out float ms, start.NativePtr, end.NativePtr));
            return ms;
        }

        /// <summary>
        /// 获取可用于在另一个进程中打开此事件的 IPC 事件句柄。
        /// Gets an IPC event handle that can be used to open this event in another process.
        /// </summary>
        /// <returns>IPC 事件句柄。/ The IPC event handle.</returns>
        public CudaIpcEventHandle GetIpcHandle()
        {
            CudaIpcEventHandle handle = new CudaIpcEventHandle();
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaIpcGetEventHandle(out handle, ptr));
            return handle;
        }

        /// <summary>
        /// 使用 IPC 事件句柄打开一个事件。
        /// Opens an event using an IPC event handle.
        /// </summary>
        /// <param name="handle">IPC 事件句柄。/ The IPC event handle.</param>
        /// <returns>一个新的 CudaEvent 实例。/ A new CudaEvent instance.</returns>
        public static CudaEvent OpenIpcEventHandle(CudaIpcEventHandle handle)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaIpcOpenEventHandle(out IntPtr eventPtr, handle));
            return new CudaEvent(eventPtr);
        }

    }
}

