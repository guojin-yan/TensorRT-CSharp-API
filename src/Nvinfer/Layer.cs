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
    /// 表示一个网络层的基础类，继承自DisposableTrtObject，用于管理网络层的生命周期和操作。<br/>
    /// Represents a base class for a network layer, inheriting from DisposableTrtObject, used to manage the lifecycle and operations of a network layer.
    /// </summary>
    public class Layer : DisposableTrtObject
    {

        /// <summary>
        /// 创建一个空的 Layer 实例。<br/>
        /// Creates an empty Layer instance.
        /// </summary>
        public Layer()
        {
        }

        /// <summary>
        /// 从一个本地（非托管）指针创建 Layer 实例。<br/>
        /// Creates a Layer instance from a native (unmanaged) pointer.
        /// </summary>
        /// <param name="ptr">指向非托管 Layer 对象的指针。<br/>A pointer to the unmanaged Layer object.</param>
        internal Layer(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                throw new TrtException("Native object address is NULL");
            this.ptr = ptr;
        }

        /// <summary>
        /// 释放该 Layer 占用的资源。<br/>
        /// Releases the resources occupied by this Layer.
        /// </summary>
        public void Release()
        {
            Dispose();
        }

        /// <inheritdoc />
        /// <summary>
        /// 释放非托管资源。<br/>
        /// Releases the unmanaged resources.
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            if (ptr != IntPtr.Zero && IsEnabledDispose)
                NativeMethods.trtLayer_free(ptr);
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// 获取网络的类型。<br/>
        /// Gets the type of the layer.
        /// </summary>
        public TrtLayerType Type
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtLayer_getType(ptr, out TrtLayerType type));
                return type;
            }
        }

        /// <summary>
        /// 获取或设置该层的名称。<br/>
        /// Gets or sets the name of this layer.
        /// </summary>
        public string Name
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtLayer_getName(ptr, out IntPtr namePtr));
                return Marshal.PtrToStringUni(namePtr);
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtLayer_setName(ptr, value));
            }
        }

        /// <summary>
        /// 获取该层的输入张量数量。<br/>
        /// Gets the number of input tensors for this layer.
        /// </summary>
        public int NbInputs
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtLayer_getNbInputs(ptr, out int nbInputs));
                return nbInputs;
            }
        }

        /// <summary>
        /// 获取指定索引的输入张量。<br/>
        /// Gets the input tensor at the specified index.
        /// </summary>
        /// <param name="index">输入张量的索引，从0开始。<br/>The zero-based index of the input tensor.</param>
        /// <returns>对应的输入张量对象。<br/>The corresponding input tensor object.</returns>
        public Tensor getInput(int index)
        {
            TrtHandleException.handler(NativeMethods.trtLayer_getInput(ptr, index, out IntPtr inputPtr));
            return new Tensor(inputPtr);
        }

        /// <summary>
        /// 获取该层的输出张量数量。<br/>
        /// Gets the number of output tensors for this layer.
        /// </summary>
        public int NbOutputs
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtLayer_getNbOutputs(ptr, out int nbOutputs));
                return nbOutputs;
            }
        }

        /// <summary>
        /// 获取指定索引的输出张量。<br/>
        /// Gets the output tensor at the specified index.
        /// </summary>
        /// <param name="index">输出张量的索引，从0开始。<br/>The zero-based index of the output tensor.</param>
        /// <returns>对应的输出张量对象。<br/>The corresponding output tensor object.</returns>
        public Tensor getOutput(int index)
        {
            TrtHandleException.handler(NativeMethods.trtLayer_getOutput(ptr, index, out IntPtr outputPtr));
            return new Tensor(outputPtr);
        }

        /// <summary>
        /// 获取或设置该层的计算精度（数据类型）。<br/>
        /// Gets or sets the computation precision (data type) for this layer.
        /// </summary>
        public TrtDataType Precision
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtLayer_getPrecision(ptr, out TrtDataType precision));
                return precision;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtLayer_setPrecision(ptr, value));
            }
        }

        /// <summary>
        /// 获取一个值，指示是否已为该层显式设置了精度。<br/>
        /// Gets a value indicating whether the precision for this layer has been explicitly set.
        /// </summary>
        public bool IsPrecisionSet
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtLayer_precisionIsSet(ptr, out int isSet));
                return isSet != 0;
            }
        }

        /// <summary>
        /// 获取指定输出索引处张量的数据类型。<br/>
        /// Gets the data type of the tensor at the specified output index.
        /// </summary>
        /// <param name="index">输出的索引，从0开始。<br/>The zero-based index of the output.</param>
        /// <returns>输出张量的数据类型。<br/>The data type of the output tensor.</returns>
        public TrtDataType getOutputType(int index)
        {
            TrtHandleException.handler(NativeMethods.trtLayer_getOutputType(ptr, index, out TrtDataType outputType));
            return outputType;
        }

        /// <summary>
        /// 检查是否已为指定索引的输出设置了数据类型。<br/>
        /// Checks if the data type has been set for the output at the specified index.
        /// </summary>
        /// <param name="index">输出的索引，从0开始。<br/>The zero-based index of the output.</param>
        /// <returns>如果已设置，则为 true；否则为 false。<br/>True if it is set, otherwise false.</returns>
        public bool isOutputTypeSet(int index)
        {
            TrtHandleException.handler(NativeMethods.trtLayer_outputTypeIsSet(ptr, index, out int isSet));
            return isSet != 0;
        }

        /// <summary>
        /// 获取或设置与该层关联的元数据。<br/>
        /// Gets or sets the metadata associated with this layer.
        /// </summary>
        public string Metadata
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtLayer_getMetadata(ptr, out IntPtr metadataPtr));
                return Marshal.PtrToStringUni(metadataPtr);
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtLayer_setMetadata(ptr, value));
            }
        }


        /// <summary>
        /// 为指定索引的输入设置张量。<br/>
        /// Sets the tensor for the input at the specified index.
        /// </summary>
        /// <param name="index">输入的索引，从0开始。<br/>The zero-based index of the input.</param>
        /// <param name="tensor">要设置的输入张量。<br/>The input tensor to set.</param>
        public void setInput(int index, Tensor tensor)
        {
            TrtHandleException.handler(NativeMethods.trtLayer_setInput(ptr, index, tensor.TrtPtr));
        }

        /// <summary>
        /// 重置该层的计算精度，使其由网络或构建器自动确定。<br/>
        /// Resets the computation precision for this layer, allowing it to be determined automatically by the network or builder.
        /// </summary>
        public void resetPrecision()
        {
            TrtHandleException.handler(NativeMethods.trtLayer_resetPrecision(ptr));
        }


        /// <summary>
        /// 为指定索引的输出设置数据类型。<br/>
        /// Sets the data type for the output at the specified index.
        /// </summary>
        /// <param name="index">输出的索引，从0开始。<br/>The zero-based index of the output.</param>
        /// <param name="dataType">要设置的数据类型。<br/>The data type to set.</param>
        public void setOutputType(int index, TrtDataType dataType)
        {
            TrtHandleException.handler(NativeMethods.trtLayer_setOutputType(ptr, index, dataType));
        }


        /// <summary>
        /// 重置指定索引输出的数据类型，使其由网络或构建器自动确定。<br/>
        /// Resets the data type for the output at the specified index, allowing it to be determined automatically by the network or builder.
        /// </summary>
        /// <param name="index">输出的索引，从0开始。<br/>The zero-based index of the output.</param>
        public void resetOutputType(int index)
        {
            TrtHandleException.handler(NativeMethods.trtLayer_resetOutputType(ptr, index));
        }

    }

}
