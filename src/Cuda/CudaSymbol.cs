using JYPPX.TensorRtSharp.Cuda;
using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// 符号与常量模块。
    /// 提供操作常量内存符号的方法，例如将数据复制到设备符号或从设备符号复制数据。
    /// Symbol and Constants Module.
    /// Provides methods to operate on constant memory symbols, such as copying data to/from device symbols.
    /// </summary>
    public static class CudaSymbol
    {
        /// <summary>
        /// 将数据从主机内存复制到设备符号（同步）。
        /// Copies data from host memory to a device symbol (Synchronous).
        /// </summary>
        /// <param name="symbol">设备符号地址。/ Device symbol address.</param>
        /// <param name="src">主机内存指针。/ Source pointer in host memory.</param>
        /// <param name="byteCount">要复制的字节数。/ Number of bytes to copy.</param>
        /// <param name="offset">从符号起始位置的偏移量（字节）。/ Offset from the start of the symbol in bytes.</param>
        /// <param name="kind">复制方向。/ Copy direction.</param>
        public static void CopyToSymbol(IntPtr symbol, IntPtr src, ulong byteCount, ulong offset, CudaMemcpyKind kind)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemcpyToSymbol(symbol, src, byteCount, offset, kind));
        }
        /// <summary>
        /// 将数据从设备符号复制到主机内存（同步）。
        /// Copies data from a device symbol to host memory (Synchronous).
        /// </summary>
        /// <param name="symbol">设备符号地址。/ Device symbol address.</param>
        /// <param name="dst">主机内存指针。/ Destination pointer in host memory.</param>
        /// <param name="byteCount">要复制的字节数。/ Number of bytes to copy.</param>
        /// <param name="offset">从符号起始位置的偏移量（字节）。/ Offset from the start of the symbol in bytes.</param>
        /// <param name="kind">复制方向。/ Copy direction.</param>
        public static void CopyFromSymbol(IntPtr symbol, IntPtr dst, ulong byteCount, ulong offset, CudaMemcpyKind kind)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemcpyFromSymbol(symbol, dst, byteCount, offset, kind));
        }
        /// <summary>
        /// 将数据从主机内存复制到设备符号（异步）。
        /// Copies data from host memory to a device symbol (Asynchronous).
        /// </summary>
        /// <param name="symbol">设备符号地址。/ Device symbol address.</param>
        /// <param name="src">主机内存指针。/ Source pointer in host memory.</param>
        /// <param name="byteCount">要复制的字节数。/ Number of bytes to copy.</param>
        /// <param name="offset">从符号起始位置的偏移量（字节）。/ Offset from the start of the symbol in bytes.</param>
        /// <param name="kind">复制方向。/ Copy direction.</param>
        /// <param name="stream">执行操作所在的流。/ The stream to perform the operation on.</param>
        public static void CopyToSymbolAsync(IntPtr symbol, IntPtr src, ulong byteCount, ulong offset, CudaMemcpyKind kind, CudaStream stream)
        {
            IntPtr streamPtr = stream != null ? stream.NativePtr : IntPtr.Zero;
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemcpyToSymbolAsync(symbol, src, byteCount, offset, kind, streamPtr));
        }
        /// <summary>
        /// 将数据从设备符号复制到主机内存（异步）。
        /// Copies data from a device symbol to host memory (Asynchronous).
        /// </summary>
        /// <param name="symbol">设备符号地址。/ Device symbol address.</param>
        /// <param name="dst">主机内存指针。/ Destination pointer in host memory.</param>
        /// <param name="byteCount">要复制的字节数。/ Number of bytes to copy.</param>
        /// <param name="offset">从符号起始位置的偏移量（字节）。/ Offset from the start of the symbol in bytes.</param>
        /// <param name="kind">复制方向。/ Copy direction.</param>
        /// <param name="stream">执行操作所在的流。/ The stream to perform the operation on.</param>
        public static void CopyFromSymbolAsync(IntPtr symbol, IntPtr dst, ulong byteCount, ulong offset, CudaMemcpyKind kind, CudaStream stream)
        {
            IntPtr streamPtr = stream != null ? stream.NativePtr : IntPtr.Zero;
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaMemcpyFromSymbolAsync(symbol, dst, byteCount, offset, kind, streamPtr));
        }
        /// <summary>
        /// 获取设备符号的地址。
        /// Gets the address of a device symbol.
        /// </summary>
        /// <param name="symbol">设备符号地址。/ Device symbol address.</param>
        /// <returns>符号在设备内存中的地址。/ The address of the symbol in device memory.</returns>
        public static IntPtr GetSymbolAddress(IntPtr symbol)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaGetSymbolAddress(out IntPtr devPtr, symbol));
            return devPtr;
        }
        /// <summary>
        /// 获取设备符号的大小（字节）。
        /// Gets the size of a device symbol (in bytes).
        /// </summary>
        /// <param name="symbol">设备符号地址。/ Device symbol address.</param>
        /// <returns>符号的大小（字节）。/ The size of the symbol in bytes.</returns>
        public static ulong GetSymbolSize(IntPtr symbol)
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaGetSymbolSize(out ulong size, symbol));
            return size;
        }
    }

}
