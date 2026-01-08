using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;

namespace JYPPX.TensorRtSharp.Cuda
{


    /// <summary>
    /// 定义 Stream 回调函数的委托，以匹配 C 语言中的函数指针。
    /// </summary>
    /// <param name="stream">触发回调的 Stream。</param>
    /// <param name="status">操作完成的状态（成功或错误）。</param>
    /// <param name="userData">用户通过 `userData` 参数传递的数据。</param>
    public delegate void CudaStreamCallback(IntPtr stream, CudaExceptionStatus status, IntPtr userData);

    /// <summary>
    /// 表示一个CUDA流，用于在GPU上并发执行操作。继承自DisposableTrtObject。
    /// Represents a CUDA stream for concurrent execution of operations on a GPU. Inherits from DisposableTrtObject.
    /// </summary>
    public class CudaStream : DisposableTrtObject
    {
        /// <summary>
        /// 获取此流对象的非托管指针。
        /// Gets the unmanaged pointer of this stream object.
        /// </summary>
        public IntPtr NativePtr => ptr;

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

        /// <summary>
        /// 获取流的优先级。
        /// Gets the priority of the stream.
        /// </summary>
        /// <returns>流的优先级。/ The priority of the stream.</returns>
        public int GetPriority()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamGetPriority(ptr, out int priority));
            return priority;
        }

        /// <summary>
        /// 获取用于创建流的标志。
        /// Gets the flags used to create the stream.
        /// </summary>
        /// <returns>流的标志。/ The flags of the stream.</returns>
        public uint GetFlags()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamGetFlags(ptr, out uint flags));
            return flags;
        }

        /// <summary>
        /// 将源流的属性复制到此流。
        /// Copies the attributes of the source stream to this stream.
        /// </summary>
        /// <param name="source">要复制属性的源流。/ The source stream to copy attributes from.</param>
        public void CopyAttributes(CudaStream source)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamCopyAttributes(ptr, source.NativePtr));
        }

        /// <summary>
        /// 获取流的属性。
        /// Gets an attribute of the stream.
        /// </summary>
        /// <param name="attr">要查询的属性ID。/ The attribute ID to query.</param>
        /// <param name="value">接收属性值的结构体。/ The structure to receive the attribute value.</param>
        public void GetAttribute(CudaStreamAttrID attr, ref CudaStreamAttrValue value)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamGetAttribute(ptr, attr, ref value));
        }

        /// <summary>
        /// 设置流的属性。
        /// Sets an attribute of the stream.
        /// </summary>
        /// <param name="attr">要设置的属性ID。/ The attribute ID to set.</param>
        /// <param name="value">包含属性值的结构体。/ The structure containing the attribute value.</param>
        public void SetAttribute(CudaStreamAttrID attr, ref CudaStreamAttrValue value)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamSetAttribute(ptr, attr, ref value));
        }

        /// <summary>
        /// 使此流等待事件完成。
        /// Makes this stream wait for an event to complete.
        /// </summary>
        /// <param name="cudaEvent">要等待的事件句柄。/ The event handle to wait for.</param>
        /// <param name="flags">保留供将来使用，必须为0。/ Reserved for future use, must be 0.</param>
        public void WaitEvent(CudaEvent  cudaEvent, uint flags = 0)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamWaitEvent(ptr, cudaEvent.TrtPtr, flags));
        }

        /// <summary>
        /// 向流添加一个回调函数。
        /// Adds a callback function to the stream.
        /// </summary>
        /// <param name="callback">回调函数委托。/ The callback function delegate.</param>
        /// <param name="userData">传递给回调的用户数据。/ User data to pass to the callback.</param>
        /// <param name="flags">保留供将来使用，必须为0。/ Reserved for future use, must be 0.</param>
        public void AddCallback(CudaStreamCallback callback, IntPtr userData, uint flags = 0)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamAddCallback(ptr, callback, userData, flags));
        }

        /// <summary>
        /// 查询流中所有操作是否已完成。
        /// Queries if all operations in the stream have completed.
        /// </summary>
        /// <returns>如果流中所有操作都已完成返回 true，如果操作未完成返回 false。/ Returns true if all operations in the stream are completed, false if operations are not complete.</returns>
        public bool Query()
        {
            var status = NativeMethods.cudaRuntime_cudaStreamQuery(ptr);
            if (status == CudaExceptionStatus.CudaErrorNotReady)
            {
                return false;
            }
            CudaHandleException.handler(status);
            return true;
        }

        /// <summary>
        /// 将内存范围异步附加到流上。
        /// Asynchronously attaches a memory range to the stream.
        /// </summary>
        /// <param name="devPtr">设备内存指针。/ Device memory pointer.</param>
        /// <param name="length">内存大小（字节）。/ Size of memory in bytes.</param>
        /// <param name="flags">标志。/ Flags.</param>
        public void AttachMemAsync(IntPtr devPtr, ulong length, uint flags)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamAttachMemAsync(ptr, devPtr, length, flags));
        }

        /// <summary>
        /// 开始捕获流到 CUDA 图。
        /// Begins capturing the stream into a CUDA graph.
        /// </summary>
        /// <param name="mode">捕获模式。/ Capture mode.</param>
        public void BeginCapture(CudaStreamCaptureMode mode)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamBeginCapture(ptr, mode));
        }

        /// <summary>
        /// 交换当前线程的流捕获模式。
        /// Exchanges the stream capture mode for the current thread.
        /// </summary>
        /// <param name="mode">要交换的模式。/ The mode to exchange.</param>
        /// <returns>之前的捕获模式。/ The previous capture mode.</returns>
        public static CudaStreamCaptureMode ThreadExchangeStreamCaptureMode(CudaStreamCaptureMode mode)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaThreadExchangeStreamCaptureMode(out CudaStreamCaptureMode outMode));
            return outMode;
        }

        /// <summary>
        /// 结束捕获流并返回捕获的图。
        /// Ends capture for the stream and returns the captured graph.
        /// </summary>
        /// <returns>捕获的 CUDA 图句柄。/ The captured CUDA graph handle.</returns>
        public CudaGraph_t EndCapture()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamEndCapture(ptr, out CudaGraph_t pGraph));
            return pGraph;
        }

        /// <summary>
        /// 查询流是否处于捕获模式。
        /// Queries if the stream is in capture mode.
        /// </summary>
        /// <returns>捕获状态。/ Capture status.</returns>
        public CudaStreamCaptureStatus IsCapturing()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamIsCapturing(ptr, out CudaStreamCaptureStatus pCaptureStatus));
            return pCaptureStatus;
        }

        /// <summary>
        /// 获取流的捕获信息。
        /// Gets the capture information of the stream.
        /// </summary>
        /// <param name="captureStatus">输出捕获状态。/ Output capture status.</param>
        /// <param name="id">输出捕获ID。/ Output capture ID.</param>
        public void GetCaptureInfo(out CudaStreamCaptureStatus captureStatus, out ulong id)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamGetCaptureInfo(ptr, out captureStatus, out id));
        }

        /// <summary>
        /// 获取流的捕获信息（版本2，支持依赖项）。
        /// Gets the capture information of the stream (Version 2, supporting dependencies).
        /// </summary>
        /// <param name="captureStatus">输出捕获状态。/ Output capture status.</param>
        /// <param name="id">输出捕获ID。/ Output capture ID.</param>
        /// <param name="graph">输出图句柄。/ Output graph handle.</param>
        /// <param name="dependencies">输出依赖项节点数组。/ Output dependency node array.</param>
        /// <param name="numDependencies">输出依赖项数量。/ Output number of dependencies.</param>
        public void GetCaptureInfo_v2(out CudaStreamCaptureStatus captureStatus, out ulong id, out CudaGraph_t graph, IntPtr[] dependencies, out ulong numDependencies)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamGetCaptureInfo_v2(ptr, out captureStatus, out id, out graph, dependencies, out numDependencies));
        }

        /// <summary>
        /// 更新流的捕获依赖项。
        /// Updates the capture dependencies of the stream.
        /// </summary>
        /// <param name="dependencies">依赖项节点数组。/ Dependency node array.</param>
        /// <param name="flags">保留供将来使用。/ Reserved for future use.</param>
        public void UpdateCaptureDependencies(IntPtr[] dependencies, uint flags = 0)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaStreamUpdateCaptureDependencies(ptr, dependencies, (ulong)(dependencies?.Length ?? 0), flags));
        }

    }
}

