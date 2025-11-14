using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    using System;

    /// <summary>
    ///一个用于从文件流中读取数据的封装类，继承自DisposableTrtObject。
    ///A wrapper class for reading data from a file stream, inherits from DisposableTrtObject.
    /// </summary>
    public class FileStreamReader : DisposableTrtObject
    {

        /// <summary>
        /// 创建一个空的 FileStreamReader 实例。
        /// Creates an empty FileStreamReader instance.
        /// </summary>
        public FileStreamReader()
        {
            NativeMethods.trtFileStreamReader_createFileStreamReader(out ptr);
        }

        /// <summary>
        /// 使用一个原生指针来初始化 FileStreamReader 实例。主要用于内部封装。
        /// Initializes a FileStreamReader instance from a native pointer. Primarily used for internal wrapping.
        /// </summary>
        /// <param name="ptr">指向原生对象的非托管指针。/ The unmanaged pointer to the native object.</param>
        internal FileStreamReader(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                throw new TrtException("Native object address is NULL");
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
        /// 释放所有非托管资源。此方法由 Dispose 模式调用，不应直接调用。
        /// Releases all unmanaged resources. This method is called by the Dispose pattern and should not be called directly.
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            if (ptr != IntPtr.Zero && IsEnabledDispose)
                NativeMethods.trtFileStreamReader_free(ptr);
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// 打开指定路径的文件以供读取。
        /// Opens the file at the specified path for reading.
        /// </summary>
        /// <param name="filepath">要打开的文件的完整路径。/ The full path of the file to open.</param>
        /// <exception cref="TrtException">如果文件打开失败。/ If the file fails to open.</exception>
        public void open(string filepath)
        {
            TrtHandleException.handler(NativeMethods.trtFileStreamReader_open(
                ptr, filepath));
        }

        /// <summary>
        /// 关闭当前已打开的文件流。
        /// Closes the currently opened file stream.
        /// </summary>
        /// <exception cref="TrtException">如果关闭文件时发生错误。/ If an error occurs while closing the file.</exception>
        public void close()
        {
            TrtHandleException.handler(NativeMethods.trtFileStreamReader_close(
                ptr));
        }

        /// <summary>
        /// 从文件流中读取指定字节数的数据到目标缓冲区。
        /// Reads a specified number of bytes from the file stream into the destination buffer.
        /// </summary>
        /// <param name="dest">指向目标内存缓冲区的指针，用于存储读取的数据。/ A pointer to the destination memory buffer where the read data will be stored.</param>
        /// <param name="bytes">要尝试读取的最大字节数。/ The maximum number of bytes to attempt to read.</param>
        /// <returns>实际成功读取的字节数。/ The number of bytes actually read successfully.</returns>
        /// <exception cref="TrtException">如果读取过程中发生错误。/ If an error occurs during the read operation.</exception>
        public long read(IntPtr dest, long bytes)
        {
            TrtHandleException.handler(NativeMethods.trtFileStreamReader_read(
                ptr, dest, bytes, out long bytesRead));
            return bytesRead;
        }

        /// <summary>
        /// 将文件流的读取位置重置到文件开头。
        /// Resets the read position of the file stream to the beginning of the file.
        /// </summary>
        /// <exception cref="TrtException">如果重置操作失败。/ If the reset operation fails.</exception>
        public void reset()
        {
            TrtHandleException.handler(NativeMethods.trtFileStreamReader_reset(
                ptr));
        }

        /// <summary>
        /// 检查文件流是否处于打开状态。
        /// Checks whether the file stream is currently open.
        /// </summary>
        /// <returns>如果文件流已打开，则为 true；否则为 false。/ True if the file stream is open; otherwise, false.</returns>
        /// <exception cref="TrtException">如果检查状态时发生错误。/ If an error occurs while checking the status.</exception>
        public bool isOpen()
        {
            TrtHandleException.handler(NativeMethods.trtFileStreamReader_isOpen(
                ptr, out int isOpen));
            return isOpen != 0;
        }
    }

}
