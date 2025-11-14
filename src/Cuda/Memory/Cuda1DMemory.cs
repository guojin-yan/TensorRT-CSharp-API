using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda.Memory
{

    /// <summary>
    /// 代表一维托管类型PropertyChangedEventArgs数组在设备上的内存。继承自DisposableTrtObject。
    /// Represents a one-dimensional managed type T array in device memory. Inherits from DisposableTrtObject.
    /// </summary>
    /// <typeparam name="T">内存中元素的类型，必须是一个结构体。/ The type of elements in the memory, must be a struct.</typeparam>
    public class Cuda1DMemory<T> : DisposableTrtObject where T : struct
    {
        ulong length = 0;

        /// <summary>
        /// 初始化一个空的 Cuda1DMemory 实例。
        /// Initializes an empty Cuda1DMemory instance.
        /// </summary>
        public Cuda1DMemory() { }

        /// <summary>
        /// 在设备上分配指定数量的元素的内存。
        /// Allocates memory for a specified number of elements on the device.
        /// </summary>
        /// <param name="num_elements">要分配的元素数量。/ The number of elements to allocate.</param>
        public Cuda1DMemory(ulong num_elements)
        {
            length = num_elements;
            CudaHandleException.handler(NativeMethods.cudaRuntime_cudaMalloc(
               out ptr, length * (ulong)Marshal.SizeOf(typeof(T))));
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
        /// 释放所有非托管资源，即释放设备内存。此方法由 Dispose 模式调用，不应直接调用。
        /// Releases all unmanaged resources, i.e., frees the device memory. This method is called by the Dispose pattern and should not be called directly.
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            if (ptr != IntPtr.Zero && IsEnabledDispose)
                CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaFree(ptr));
            base.DisposeUnmanaged();
        }

        // --- 数据操作接口 ---

        /// <summary>
        /// 从主机内存复制数据到此设备内存。这是一个同步操作。
        /// Copies data from host memory to this device memory. This is a synchronous operation.
        /// </summary>
        /// <param name="host_ptr">源主机内存数组。/ The source host memory array.</param>
        public void copyFromHost(T[] host_ptr)
        {
            CudaHandleException.handler(
            NativeMethods.cudaRuntime_cudaMemcpy(ptr,
            Marshal.UnsafeAddrOfPinnedArrayElement(host_ptr, 0),
            length * (ulong)Marshal.SizeOf(typeof(T)), CudaMemcpyKind.HostToDevice));
        }

        /// <summary>
        ///从此设备内存复制数据到主机内存。这是一个同步操作。
        /// Copies data from this device memory to host memory. This is a synchronous operation.
        /// </summary>
        /// <param name="host_ptr">目标主机内存数组。/ The destination host memory array.</param>
        public void copyToHost(T[] host_ptr)
        {
            CudaHandleException.handler(
            NativeMethods.cudaRuntime_cudaMemcpy(
                Marshal.UnsafeAddrOfPinnedArrayElement(host_ptr, 0),
                ptr,
                length * (ulong)Marshal.SizeOf(typeof(T)),
                CudaMemcpyKind.DeviceToHost));
        }

        /// <summary>
        /// 从主机内存异步复制数据到此设备内存。
        /// Asynchronously copies data from host memory to this device memory.
        /// </summary>
        /// <param name="host_ptr">源主机内存数组。/ The source host memory array.</param>
        /// <param name="stream">用于执行复制操作的CUDA流。/ The CUDA stream to perform the copy operation.</param>
        public void copyFromHostAsync(T[] host_ptr, CudaStream stream)
        {
            CudaHandleException.handler(
            NativeMethods.cudaRuntime_cudaMemcpyAsync(
                ptr,
                Marshal.UnsafeAddrOfPinnedArrayElement(host_ptr, 0),
                length * (ulong)Marshal.SizeOf(typeof(T)),
                CudaMemcpyKind.HostToDevice,
                stream.TrtPtr));
        }

        /// <summary>
        ///从此设备内存异步复制数据到主机内存。
        /// Asynchronously copies data from this device memory to host memory.
        /// </summary>
        /// <param name="host_ptr">目标主机内存数组。/ The destination host memory array.</param>
        /// <param name="stream">用于执行复制操作的CUDA流。/ The CUDA stream to perform the copy operation.</param>
        public void copyToHostAsync(T[] host_ptr, CudaStream stream)
        {
            CudaHandleException.handler(
            NativeMethods.cudaRuntime_cudaMemcpyAsync(
                Marshal.UnsafeAddrOfPinnedArrayElement(host_ptr, 0),
                ptr,
                length * (ulong)Marshal.SizeOf(typeof(T)),
                CudaMemcpyKind.DeviceToHost,
            stream.TrtPtr));
        }

        /// <summary>
        /// 将此设备内存设置为指定的字节值。这是一个同步操作。
        /// Sets the device memory to the specified byte value. This is a synchronous operation.
        /// </summary>
        /// <param name="value">用于设置内存的字节值。/ The byte value to set the memory to.</param>
        public void memset(int value)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemset(ptr,
                value,
                length * (ulong)Marshal.SizeOf(typeof(T))));
        }

        /// <summary>
        /// 异步地将此设备内存设置为指定的字节值。
        /// Asynchronously sets the device memory to the specified byte value.
        /// </summary>
        /// <param name="value">用于设置内存的字节值。/ The byte value to set the memory to.</param>
        /// <param name="stream">用于执行内存设置操作的CUDA流。/ The CUDA stream to perform the memory set operation.</param>
        public void memsetAsync(int value, CudaStream stream)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemsetAsync(
                    ptr,
                    value,
                    length * (ulong)Marshal.SizeOf(typeof(T)),
                    stream.TrtPtr));
        }

        // --- 高级管理接口 ---

        /// <summary>
        /// 异步地将数据预取到指定设备。
        /// Asynchronously prefetches data to the specified device.
        /// </summary>
        /// <param name="device_id">目标设备的ID。/ The ID of the target device.</param>
        /// <param name="stream">用于执行预取操作的CUDA流。/ The CUDA stream to perform the prefetch.</param>
        public void prefetchAsync(int device_id, CudaStream stream)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemPrefetchAsync(
                    ptr,
                    length * (ulong)Marshal.SizeOf(typeof(T)),
                    device_id,
                    stream.TrtPtr));
        }

        /// <summary>
        /// 为此内存范围提供建议，以指导数据管理。
        /// Provides advice for this memory range to guide data management.
        /// </summary>
        /// <param name="advice">建议的类型（例如，最常用位置为CPU）。/ The type of advice (e.g., most commonly located on CPU).</param>
        public void advise(CudaMemoryAdvise advice)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemAdvise(
                    ptr,
                    length * (ulong)Marshal.SizeOf(typeof(T)),
                    advice, 0));
        }

        // --- 访问器 ---

        /// <summary>
        /// 获取指向设备内存的原始指针。
        /// Gets the raw pointer to the device memory.
        /// </summary>
        /// <returns>设备内存的指针。/ The pointer to the device memory.</returns>
        public IntPtr get() { return ptr; }

        /// <summary>
        /// 获取内存中的元素数量。
        /// Gets the number of elements in the memory.
        /// </summary>
        /// <returns>元素的数量。/ The number of elements.</returns>
        public ulong size() { return length; }

        /// <summary>
        /// 获取内存的总大小（以字节为单位）。
        /// Gets the total size of the memory in bytes.
        /// </summary>
        /// <returns>内存的字节大小。/ The size of the memory in bytes.</returns>
        public ulong sizeBytes() { return length * (ulong)Marshal.SizeOf(typeof(T)); }

        /// <summary>
        /// 从另一个GPU设备（对等设备）异步复制数据到此内存。
        /// Asynchronously copies data to this memory from another GPU device (peer device).
        /// </summary>
        /// <param name="src">源设备内存。/ The source device memory.</param>
        /// <param name="src_device_id">源设备所在的GPU ID。/ The GPU ID where the source device is located.</param>
        /// <param name="stream">用于执行复制操作的CUDA流。/ The CUDA stream to perform the copy operation.</param>
        public void copyFromPeerAsync(Cuda1DMemory<T> src, int src_device_id, CudaStream stream)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaGetDevice(out int device));
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemcpyPeerAsync(
                ptr, device, src.get(), src_device_id, sizeBytes(), stream.TrtPtr));
        }

        /// <summary>
        /// 异步复制数据从此内存到另一个GPU设备（对等设备）。
        /// Asynchronously copies data from this memory to another GPU device (peer device).
        /// </summary>
        /// <param name="dst">目标设备内存。/ The destination device memory.</param>
        /// <param name="dst_device_id">目标设备所在的GPU ID。/ The GPU ID where the destination device is located.</param>
        /// <param name="stream">用于执行复制操作的CUDA流。/ The CUDA stream to perform the copy operation.</param>
        public void copyToPeerAsync(Cuda1DMemory<T> dst, int dst_device_id, CudaStream stream)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaGetDevice(out int device));
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemcpyPeerAsync(
                    dst.get(), dst_device_id, ptr, device, sizeBytes(), stream.TrtPtr));
        }

    }

}
