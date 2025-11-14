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

    /// <summary>
    /// 异步流读取器类，用于从文件中异步读取数据
    /// Async stream reader class for reading data asynchronously from a file
    /// </summary>
    /// <remarks>
    /// 继承自DisposableTrtObject，实现了IDisposable接口，可以安全地释放资源
    /// Inherits from DisposableTrtObject and implements IDisposable interface for safe resource disposal
    /// </remarks>
    public class AsyncStreamReader : DisposableTrtObject
    {
        /// <summary>
        /// 创建空的文件流读取器
        /// Creates empty FileStreamReader
        /// </summary>
        public AsyncStreamReader()
        {
            NativeMethods.trtAsyncStreamReader_createAsyncStreamReader(out ptr);
        }

        /// <summary>
        /// 从原生指针创建异步流读取器
        /// Creates from native pointer
        /// </summary>
        /// <param name="ptr">原生对象指针，Native object pointer</param>
        internal AsyncStreamReader(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                throw new TrtException("Native object address is NULL");
            this.ptr = ptr;
        }

        /// <summary>
        /// 释放资源
        /// Releases the resources
        /// </summary>
        public void Release()
        {
            Dispose();
        }

        /// <summary>
        /// 释放非托管资源
        /// Releases unmanaged resources
        /// </summary>
        /// <inheritdoc />
        protected override void DisposeUnmanaged()
        {
            if (ptr != IntPtr.Zero && IsEnabledDispose)
                NativeMethods.trtAsyncStreamReader_free(ptr);
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// 打开指定路径的文件
        /// Opens a file at the specified path
        /// </summary>
        /// <param name="filepath">文件路径，File path to open</param>
        public void open(string filepath)
        {
            TrtHandleException.handler(NativeMethods.trtAsyncStreamReader_open(
                ptr, filepath));
        }

        /// <summary>
        /// 关闭当前打开的文件
        /// Closes the currently open file
        /// </summary>
        public void close()
        {
            TrtHandleException.handler(NativeMethods.trtAsyncStreamReader_close(
                ptr));
        }

        /// <summary>
        /// 从流中读取数据到指定缓冲区
        /// Reads data from stream into the specified buffer
        /// </summary>
        /// <param name="dest">目标缓冲区指针，Pointer to destination buffer</param>
        /// <param name="bytes">要读取的字节数，Number of bytes to read</param>
        /// <param name="stream">CUDA流对象指针，Pointer to CUDA stream object</param>
        /// <returns>实际读取的字节数，Number of bytes actually read</returns>
        public long read(IntPtr dest, long bytes, IntPtr stream)
        {
            TrtHandleException.handler(NativeMethods.trtAsyncStreamReader_read(
                ptr, dest, bytes, stream, out long bytesRead));
            return bytesRead;
        }

        /// <summary>
        /// 设置文件指针位置
        /// Sets the file pointer position
        /// </summary>
        /// <param name="offset">偏移量，Offset value</param>
        /// <param name="where">查找位置，Seek position reference</param>
        public void seek(long offset, TrtSeekPosition where)
        {
            TrtHandleException.handler(NativeMethods.trtAsyncStreamReader_seek(
                ptr, offset, where, out int success));
        }

        /// <summary>
        /// 检查文件是否已打开
        /// Checks if the file is open
        /// </summary>
        /// <returns>如果文件已打开则返回true，否则返回false，Returns true if file is open, false otherwise</returns>
        public bool isOpen()
        {
            TrtHandleException.handler(NativeMethods.trtAsyncStreamReader_isOpen(
                ptr, out int isOpen));
            return isOpen != 0;
        }
    }

}
