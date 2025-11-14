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
    /// 表示一个权重（Weights）对象，用于存储神经网络的权值和偏置等参数。继承自DisposableTrtObject。
    /// Represents a Weights object, used to store parameters like weights and biases for a neural network. Inherits from DisposableTrtObject.
    /// </summary>
    public class Weights : DisposableTrtObject
    {

        /// <summary>
        /// 创建一个空的、未初始化的 Weights 对象。
        /// Creates an empty, uninitialized Weights object.
        /// 注意：许多属性仅在指针已设置时才有效。/ Note: Many properties are only valid when the pointer has been set.
        public Weights()
        {
        }

        /// <summary>
        /// 使用一个原生指针来初始化 Weights 实例。主要用于内部封装。
        /// Initializes a Weights instance from a native pointer. Primarily used for internal wrapping.
        /// </summary>
        /// <param name="ptr">指向原生对象的非托管指针。/ The unmanaged pointer to the native object.</param>
        internal Weights(IntPtr ptr)
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
                NativeMethods.trtWeight_free(ptr);
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// 获取权重元素的数据类型。
        /// Gets the data type of the weight elements.
        /// </summary>
        /// <returns>一个 <see cref="TrtDataType"/> 枚举值，表示权重中元素的数据类型。/ A <see cref="TrtDataType"/> enumeration value representing the data type of the elements in the weights.</returns>
        public TrtDataType DataType
        {
            get => NativeMethods.trtWeight_getDataType(ptr);
        }

        /// <summary>
        /// 获取权重元素的数量。
        /// Gets the number of weight elements.
        /// </summary>
        /// <returns>权重中的元素总数。/ The total number of elements in the weights.</returns>
        public long Count
        {
            get => NativeMethods.trtWeight_getCount(ptr);
        }

        /// <summary>
        /// 获取指向权重值数据的指针。
        /// Gets a pointer to the weight values data.
        /// </summary>
        /// <returns>一个指向权重数据内存的非托管指针。/ An unmanaged pointer to the memory of the weight data.</returns>
        public IntPtr Values
        {
            get => NativeMethods.trtWeight_getValues(ptr);
        }

    }

}
