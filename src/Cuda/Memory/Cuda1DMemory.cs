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
    
    public class Cuda1DMemory<T> : DisposableTrtObject where T : struct
    {


        ulong length = 0; 
        bool from_pool_ = false;

        // 默认构造函数
        public Cuda1DMemory() { }
        // 构造函数：从标准设备内存分配
        public Cuda1DMemory(ulong num_elements)
        {
            length = num_elements;
            CudaHandleException.handler(NativeMethods.cudaRuntime_cudaMalloc(
               out ptr, length * (ulong)Marshal.SizeOf(typeof(T))));
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
        // --- 数据操作接口 ---
        public void copyFromHost(T[] host_ptr)
        {
            CudaHandleException.handler(
            NativeMethods.cudaRuntime_cudaMemcpy(ptr, 
            Marshal.UnsafeAddrOfPinnedArrayElement(host_ptr, 0),
            length * (ulong)Marshal.SizeOf(typeof(T)), CudaMemcpyKind.HostToDevice));
        }
        public void copyToHost(T[] host_ptr)
        {
            CudaHandleException.handler(
            NativeMethods.cudaRuntime_cudaMemcpy(
                Marshal.UnsafeAddrOfPinnedArrayElement(host_ptr, 0),
                ptr,
                length * (ulong)Marshal.SizeOf(typeof(T)), 
                CudaMemcpyKind.DeviceToHost));
        }
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
        public void memset(int value)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemset(ptr,
                value,
                length * (ulong)Marshal.SizeOf(typeof(T))));
        }
        public void memsetAsync(int value, CudaStream stream)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemsetAsync(
                    ptr,
                    value,
                    length * (ulong)Marshal.SizeOf(typeof(T)),
                    stream.TrtPtr));
        }
        // --- 高级管理接口 (通用) ---
        public void prefetchAsync(int device_id, CudaStream stream)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemPrefetchAsync(
                    ptr,
                    length * (ulong)Marshal.SizeOf(typeof(T)),
                    device_id,
                    stream.TrtPtr));
        }
        public void advise(CudaMemoryAdvise advice)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemAdvise(
                    ptr,
                    length * (ulong)Marshal.SizeOf(typeof(T)), 
                    advice, 0));
        }

        // --- 访问器 ---
        public IntPtr get() { return ptr; }
        public ulong size() { return length; }
        public ulong sizeBytes() { return length * (ulong)Marshal.SizeOf(typeof(T)); }

        public void copyFromPeerAsync(Cuda1DMemory<T> src, int src_device_id, CudaStream stream)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaGetDevice(out int device));
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemcpyPeerAsync(
                ptr, device, src.get(), src_device_id, sizeBytes(), stream.TrtPtr));
        }
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
