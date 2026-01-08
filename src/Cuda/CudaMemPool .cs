using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// 表示一个CUDA异步内存池，用于高效分配和跨进程共享显存。继承自DisposableTrtObject。
    /// Represents a CUDA async memory pool for efficient allocation and cross-process sharing of device memory. Inherits from DisposableTrtObject.
    /// </summary>
    public class CudaMemPool : DisposableTrtObject
    {
        /// <summary>
        /// 获取此CUDA异步内存池对象的非托管指针。
        /// Gets the unmanaged pointer of this stream object.
        /// </summary>
        public IntPtr NativePtr => ptr;
        /// <summary>
        /// 使用指定的属性创建一个CUDA内存池。
        /// Creates a CUDA memory pool with the specified properties.
        /// </summary>
        /// <param name="props">内存池的属性。/ Properties of the memory pool.</param>
        public CudaMemPool(ref CudaMemPoolProps props)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemPoolCreate(out ptr, ref props));
        }

        /// <summary>
        /// 使用一个原生指针来初始化 CudaMemPool 实例。主要用于内部封装。
        /// Initializes a CudaMemPool instance from a native pointer. Primarily used for internal wrapping.
        /// </summary>
        /// <param name="ptr">指向原生对象的非托管指针。/ The unmanaged pointer to the native object.</param>
        internal CudaMemPool(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                throw new CudaException("Native object address is NULL");
            this.ptr = ptr;
        }

        /// <summary>
        /// 从可共享句柄导入内存池。这是创建跨进程内存池对象的工厂方法。
        /// Imports a memory pool from a shareable handle. This is a factory method to create a cross-process memory pool object.
        /// </summary>
        /// <param name="handle">共享句柄。/ The shareable handle.</param>
        /// <param name="handleType">句柄类型。/ The type of the handle.</param>
        /// <param name="flags">保留标志，必须为0。/ Reserved flags, must be 0.</param>
        /// <returns>CudaMemPool 实例。/ An instance of CudaMemPool.</returns>
        public static CudaMemPool ImportFromShareableHandle(IntPtr handle, CudaMemAllocationHandleType handleType, int flags)
        {
            IntPtr poolPtr = IntPtr.Zero;
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemPoolImportFromShareableHandle(out poolPtr, handle, handleType, (uint)flags));
            return new CudaMemPool(poolPtr);
        }



        /// <summary>
        /// 从内存池中异步分配内存（另一种API形式）。
        /// Allocates memory asynchronously from the pool (alternative API form).
        /// </summary>
        /// <param name="size">要分配的字节数。/ Number of bytes to allocate.</param>
        /// <param name="stream">要在其中执行分配的流。/ The stream on which to perform the allocation.</param>
        /// <returns>指向分配的内存的设备指针。/ Device pointer to the allocated memory.</returns>
        public IntPtr MallocFromPoolAsync(ulong size, CudaStream stream)
        {
            IntPtr dptr = IntPtr.Zero;
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMallocFromPoolAsync(out dptr, size, NativePtr, stream.NativePtr));
            return dptr;
        }

        /// <summary>
        /// 异步释放内存。
        /// Frees memory asynchronously.
        /// </summary>
        /// <param name="dptr">要释放的设备指针。/ Device pointer to free.</param>
        /// <param name="stream">要在其中执行释放的流。/ The stream on which to perform the free.</param>
        public void FreeAsync(IntPtr dptr, CudaStream stream)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaFreeAsync(dptr, stream.NativePtr));
        }

        /// <summary>
        /// 获取内存池的属性。
        /// Gets an attribute of the memory pool.
        /// </summary>
        /// <param name="attr">要查询的属性。/ The attribute to query.</param>
        /// <returns>属性的值。/ The value of the attribute.</returns>
        public IntPtr GetAttribute(CudaMemPoolAttr attr)
        {
            IntPtr value = IntPtr.Zero;
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemPoolGetAttribute(NativePtr, attr,  value));
            return value;
        }

        /// <summary>
        /// 设置内存池的属性。
        /// Sets an attribute of the memory pool.
        /// </summary>
        /// <param name="attr">要设置的属性。/ The attribute to set.</param>
        /// <param name="value">属性的值。/ The value of the attribute.</param>
        public void SetAttribute(CudaMemPoolAttr attr, IntPtr value)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemPoolSetAttribute(NativePtr, attr, value));
        }

        /// <summary>
        /// 将内存池修剪到指定大小。
        /// Trims the memory pool to a specified size.
        /// </summary>
        /// <param name="minBytesToKeep">保留的最小字节数。/ The minimum number of bytes to keep.</param>
        public void TrimTo(ulong minBytesToKeep)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemPoolTrimTo(NativePtr, minBytesToKeep));
        }

        /// <summary>
        /// 设置内存池对特定内存位置的访问权限。
        /// Sets access to the memory pool for a specific memory location.
        /// </summary>
        /// <param name="descArray">访问描述符数组。/ Array of access descriptors.</param>
        /// <param name="count">数组中描述符的数量。/ Number of descriptors in the array.</param>
        public void SetAccess(CudaMemAccessDesc[] descArray, uint count)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemPoolSetAccess(NativePtr, ref descArray[0], count));
        }

        /// <summary>
        /// 获取内存池对特定内存位置的访问权限。
        /// Gets access flags for the memory pool for a specific memory location.
        /// </summary>
        /// <param name="location">内存位置。/ The memory location.</param>
        /// <returns>访问权限标志。/ The access flags.</returns>
        public CudaMemAccessFlags GetAccess(CudaMemLocation location)
        {
            CudaMemAccessFlags flags = 0;
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemPoolGetAccess(out flags, NativePtr, ref location));
            return flags;
        }

        /// <summary>
        /// 将内存池导出为可共享句柄。
        /// Exports the memory pool to a shareable handle.
        /// </summary>
        /// <param name="handleType">导出的句柄类型。/ The type of handle to export.</param>
        /// <param name="flags">保留标志，必须为0。/ Reserved flags, must be 0.</param>
        /// <returns>共享句柄。/ The shared handle.</returns>
        public IntPtr ExportToShareableHandle(CudaMemAllocationHandleType handleType, int flags)
        {
            IntPtr handle = IntPtr.Zero;
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemPoolExportToShareableHandle(handle, ptr, handleType, (uint)flags));
            return handle;
        }

        /// <summary>
        /// 导出指针数据。
        /// Exports pointer data.
        /// </summary>
        /// <param name="dptr">要导出的设备指针。/ The device pointer to export.</param>
        /// <param name="dataBuffer">用于存储导出数据的缓冲区（由 cudaMemPoolPtrExportData 定义）。/ Buffer to store the exported data (defined by cudaMemPoolPtrExportData).</param>
        public CudaMemPoolPtrExportData ExportPointer(IntPtr dptr)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemPoolExportPointer(out CudaMemPoolPtrExportData dataBuffer, dptr));
            return dataBuffer;
        }

        /// <summary>
        /// 导入指针数据。
        /// Imports pointer data.
        /// </summary>
        /// <param name="dataBuffer">包含导出数据的缓冲区。/ Buffer containing the exported data.</param>
        /// <returns>导入的设备指针。/ The imported device pointer.</returns>
        public IntPtr ImportPointer(CudaMemPoolPtrExportData dataBuffer)
        {
            IntPtr dptr = IntPtr.Zero;
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemPoolImportPointer(out dptr, NativePtr, ref dataBuffer));
            return dptr;
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
        /// 释放所有非托管资源，即销毁CUDA内存池。此方法由 Dispose 模式调用，不应直接调用。
        /// Releases all unmanaged resources, i.e., destroys the CUDA memory pool. This method is called by the Dispose pattern and should not be called directly.
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            if (ptr != IntPtr.Zero && IsEnabledDispose)
                CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemPoolDestroy(ptr));
            base.DisposeUnmanaged();
        }

    }
}
