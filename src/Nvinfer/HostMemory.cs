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
    /// <summary>
    /// 一个用于封装主机内存（CPU内存）的类，继承自DisposableTrtObject。
    /// A class that encapsulates host memory (CPU memory), inherits from DisposableTrtObject.
    /// </summary>
    public class HostMemory : DisposableTrtObject
    {
        /// <summary>
        /// 创建一个空的 HostMemory 实例。
        /// Creates an empty HostMemory instance.
        /// </summary>
        public HostMemory()
        {
        }
        /// <summary>
        /// 使用一个原生指针来初始化 HostMemory 实例。主要用于内部封装。
        /// Initializes a HostMemory instance from a native pointer. Primarily used for internal wrapping.
        /// </summary>
        /// <param name="ptr">指向原生对象的非托管指针。/ The unmanaged pointer to the native object.</param>
        internal HostMemory(IntPtr ptr)
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
                NativeMethods.trtHostMemory_free(ptr);
            base.DisposeUnmanaged();
        }
        /// <summary>
        /// 获取指向主机内存块的原始数据指针。
        /// Gets the raw data pointer to the host memory block.
        /// </summary>
        /// <returns>指向数据的非托管内存指针。/ An unmanaged memory pointer to the data.</returns>
        /// <exception cref="TrtException">如果访问原生指针时发生错误。/ If an error occurs while accessing the native pointer.</exception>
        public IntPtr Data
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtHostMemory_data(ptr, out IntPtr data));
                return data;
            }
        }
        /// <summary>
        /// 获取内存块的大小（以字节为单位）。
        /// Gets the size of the memory block in bytes.
        /// </summary>
        /// <returns>内存块的大小。/ The size of the memory block.</returns>
        /// <exception cref="TrtException">如果获取大小时发生错误。/ If an error occurs while getting the size.</exception>
        public long Size
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtHostMemory_size(ptr, out long size));
                return size;
            }
        }
        /// <summary>
        /// 获取内存块中存储的数据类型。
        /// Gets the data type of the data stored in the memory block.
        /// </summary>
        /// <returns>表示数据类型的 TrtDataType 枚举值。/ A TrtDataType enumeration value representing the data type.</returns>
        /// <exception cref="TrtException">如果获取数据类型时发生错误。/ If an error occurs while getting the data type.</exception>
        public TrtDataType DataType
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtHostMemory_type(ptr, out TrtDataType dataType));
                return dataType;
            }
        }
        /// <summary>
        /// 将原生内存中的数据复制到一个新的字节数组中。
        /// Copies the data from the native memory into a new byte array.
        /// </summary>
        /// <returns>一个包含内存块数据的字节数组。如果内存无效，则返回空数组。/ A byte array containing the memory block's data. Returns an empty array if the memory is invalid.</returns>
        public byte[] getByteData()
        {
            if (Data == IntPtr.Zero || Size <= 0)
                return Array.Empty<byte>();
            byte[] data = new byte[Size];
            Marshal.Copy(Data, data, 0, (int)Size);
            return data;
        }
    }
}
