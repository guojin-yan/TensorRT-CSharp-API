using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{

    /// <summary>
    /// 代表一个神经网络定义，是构建网络模型的顶级容器。<br/>
    /// Represents a neural network definition, which is the top-level container for building a network model.
    /// </summary>
    public class NetworkDefinition : DisposableTrtObject
    {

        /// <summary>
        /// 创建一个空的 NetworkDefinition 实例。<br/>
        /// Creates an empty NetworkDefinition instance.
        /// </summary>
        public NetworkDefinition()
        {
            //InitHandleException.handler(
            //    NativeMethods.trtBuild_createInferBuilder(out ptr));
        }

        /// <summary>
        /// 从一个本地（非托管）指针创建 NetworkDefinition 实例。<br/>
        /// Creates a NetworkDefinition instance from a native (unmanaged) pointer.
        /// </summary>
        /// <param name="ptr">指向非托管 NetworkDefinition 对象的指针。<br/>A pointer to the unmanaged NetworkDefinition object.</param>
        internal NetworkDefinition(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                throw new TrtException("Native object address is NULL");
            this.ptr = ptr;
        }

        /// <summary>
        /// 释放该 NetworkDefinition 占用的资源。<br/>
        /// Releases the resources occupied by this NetworkDefinition.
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
                NativeMethods.trtNetworkDefinition_free(ptr);
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// 向网络中添加一个输入张量。<br/>
        /// Adds an input tensor to the network.
        /// </summary>
        /// <param name="name">输入张量的名称。<br/>The name of the input tensor.</param>
        /// <param name="type">输入张量的数据类型。<br/>The data type of the input tensor.</param>
        /// <param name="dimensions">输入张量的维度。<br/>The dimensions of the input tensor.</param>
        /// <returns>新创建的输入张量对象。<br/>The newly created input tensor object.</returns>
        public Tensor addInput(string name, TrtDataType type, Dims dimensions)
        {
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_addInput(
                ptr, name, type, ref dimensions, out IntPtr tensorPtr));
            return new Tensor(tensorPtr);
        }

        /// <summary>
        /// 将一个张量标记为网络的输出。<br/>
        /// Marks a tensor as a network output.
        /// </summary>
        /// <param name="tensor">要标记为输出的张量。<br/>The tensor to mark as an output.</param>
        public void markOutput(Tensor tensor)
        {
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_markOutput(
                ptr, tensor.TrtPtr));
        }

        /// <summary>
        /// 取消标记一个张量作为网络的输出。<br/>
        /// Unmarks a tensor as a network output.
        /// </summary>
        /// <param name="tensor">要取消标记的张量。<br/>The tensor to unmark.</param>
        public void unmarkOutput(Tensor tensor)
        {
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_unmarkOutput(
                ptr, tensor.TrtPtr));
        }


        /// <summary>
        /// 将一个张量标记为形状计算图的输出，用于I/O形状优化。<br/>
        /// Marks a tensor as an output for the shape calculation graph, used for I/O shape optimization.
        /// </summary>
        /// <param name="tensor">要标记的张量。<br/>The tensor to mark.</param>
        /// <returns>如果张量在此之前未被标记，则返回 true；否则返回 false。<br/>Returns true if the tensor was not previously marked; otherwise, false.</returns>
        public bool markOutputForShapes(Tensor tensor)
        {
            int wasMarked = 0;
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_markOutputForShapes(
                ptr, tensor.TrtPtr, ref wasMarked));
            return wasMarked != 0;
        }

        /// <summary>
        /// 取消标记一个张量作为形状计算图的输出。<br/>
        /// Unmarks a tensor as an output for the shape calculation graph.
        /// </summary>
        /// <param name="tensor">要取消标记的张量。<br/>The tensor to unmark.</param>
        /// <returns>如果张量在此之前被标记了，则返回 true；否则返回 false。<br/>Returns true if the tensor was previously marked; otherwise, false.</returns>
        public bool unmarkOutputForShapes(Tensor tensor)
        {
            int wasUnmarked = 0;
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_unmarkOutputForShapes(
                ptr, tensor.TrtPtr, ref wasUnmarked));
            return wasUnmarked != 0;
        }


        /// <summary>
        /// 将一个张量标记为用于调试，可用于同步张量。<br/>
        /// Marks a tensor for debugging, which can be used for a synchronization tensor.
        /// </summary>
        /// <param name="tensor">要调试的张量。<br/>The tensor to debug.</param>
        /// <returns>如果张量在此之前未被标记为调试，则返回 true；否则返回 false。<br/>Returns true if the tensor was not previously marked for debugging; otherwise, false.</returns>
        public bool markDebug(Tensor tensor)
        {
            int wasUnmarked = 0;
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_markDebug(
                ptr, tensor.TrtPtr, ref wasUnmarked));
            return wasUnmarked != 0;
        }


        /// <summary>
        /// 取消标记一个张量用于调试。<br/>
        /// Unmarks a tensor for debugging.
        /// </summary>
        /// <param name="tensor">要取消调试标记的张量。<br/>The tensor to unmark for debugging.</param>
        /// <returns>如果张量在此之前被标记为调试，则返回 true；否则返回 false。<br/>Returns true if the tensor was previously marked for debugging; otherwise, false.</returns>
        public bool unmarkDebug(Tensor tensor)
        {
            int wasUnmarked = 0;
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_unmarkDebug(
                ptr, tensor.TrtPtr, ref wasUnmarked));
            return wasUnmarked != 0;
        }


        /// <summary>
        /// 检查一个张量是否被标记用于调试。<br/>
        /// Checks if a tensor is marked for debugging.
        /// </summary>
        /// <param name="tensor">要检查的张量。<br/>The tensor to check.</param>
        /// <returns>如果张量被标记用于调试，则返回 true；否则返回 false。<br/>Returns true if the tensor is marked for debugging; otherwise, false.</returns>
        public bool isDebugTensor(Tensor tensor)
        {
            int isDebug = 0;
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_isDebugTensor(
                ptr, tensor.TrtPtr, ref isDebug));
            return isDebug != 0;
        }


        /// <summary>
        /// 从网络中移除一个张量。<br/>
        /// Removes a tensor from the network.
        /// </summary>
        /// <param name="tensor">要移除的张量。<br/>The tensor to remove.</param>
        public void removeTensor(Tensor tensor)
        {
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_removeTensor(
                ptr, tensor.TrtPtr));
        }

        /// <summary>
        /// 获取网络中的层数。<br/>
        /// Gets the number of layers in the network.
        /// </summary>
        /// <returns>网络中的层数。<br/>The number of layers in the network.</returns>
        public int getNbLayers()
        {
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_getNbLayers(
                ptr, out int nbLayers));
            return nbLayers;
        }

        /// <summary>
        /// 获取指定索引的网络层。<br/>
        /// Gets the network layer at the specified index.
        /// </summary>
        /// <param name="index">层的索引，从0开始。<br/>The zero-based index of the layer.</param>
        /// <returns>指定索引处的网络层对象。<br/>The network layer object at the specified index.</returns>
        public Layer getLayer(int index)
        {
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_getLayer(
                ptr, index, out IntPtr layer));
            return new Layer(layer);
        }

        // 这是一个外部方法声明，不应添加Doxygen注释，否则文档生成器会尝试为它生成文档。
        // This is an external method declaration and should not have Doxygen comments, otherwise the documentation generator will try to generate documentation for it.
        public extern static TrtExceptionStatus trtNetworkDefinition_getLayer(
            IntPtr network,
            int index,
            out IntPtr layer);

        /// <summary>
        /// 获取网络中输入张量的数量。<br/>
        /// Gets the number of input tensors in the network.
        /// </summary>
        /// <returns>输入张量的数量。<br/>The number of input tensors.</returns>
        public int getNbInputs()
        {
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_getNbInputs(
                ptr, out int nbInputs));
            return nbInputs;
        }

        /// <summary>
        /// 获取指定索引的输入张量。<br/>
        /// Gets the input tensor at the specified index.
        /// </summary>
        /// <param name="index">输入张量的索引，从0开始。<br/>The zero-based index of the input tensor.</param>
        /// <returns>指定索引处的输入张量对象。<br/>The input tensor object at the specified index.</returns>
        public Tensor getInput(int index)
        {
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_getInput(
                ptr, index, out IntPtr input));
            return new Tensor(input);
        }

        /// <summary>
        /// 获取网络中输出张量的数量。<br/>
        /// Gets the number of output tensors in the network.
        /// </summary>
        /// <returns>输出张量的数量。<br/>The number of output tensors.</returns>
        public int getNbOutputs()
        {
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_getNbOutputs(
                ptr, out int nbOutputs));
            return nbOutputs;
        }

        /// <summary>
        /// 获取指定索引的输出张量。<br/>
        /// Gets the output tensor at the specified index.
        /// </summary>
        /// <param name="index">输出张量的索引，从0开始。<br/>The zero-based index of the output tensor.</param>
        /// <returns>指定索引处的输出张量对象。<br/>The output tensor object at the specified index.</returns>
        public Tensor getOutput(int index)
        {
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_getOutput(
                ptr, index, out IntPtr output));
            return new Tensor(output);
        }

        /// <summary>
        /// 设置网络的名称。<br/>
        /// Sets the name of the network.
        /// </summary>
        /// <param name="name">网络的名称。<br/>The name of the network.</param>
        public void setName(string name)
        {
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_setName(
                ptr, name));
        }


        /// <summary>
        /// 获取网络的名称。<br/>
        /// Gets the name of the network.
        /// </summary>
        /// <returns>网络的名称。<br/>The name of the network.</returns>
        public string getName()
        {
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_getName(
                ptr, out IntPtr namePtr));
            return Marshal.PtrToStringAnsi(namePtr);
        }

        /// <summary>
        /// 获取用于创建网络的标志位。<br/>
        /// Gets the flags used to create the network.
        /// </summary>
        /// <returns>表示网络定义标志的位掩码。<br/>A bitmask representing the network definition flags.</returns>
        public uint getFlags()
        {
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_getFlags(
                ptr, out uint flags));
            return flags;
        }

        /// <summary>
        /// 查询在创建网络时是否设置了特定的标志。<br/>
        /// Queries whether a specific flag was set when creating the network.
        /// </summary>
        /// <param name="flag">要查询的标志。<br/>The flag to query.</param>
        /// <returns>如果设置了该标志，则为 true；否则为 false。<br/>True if the flag is set, otherwise false.</returns>
        public bool getFlag(TrtNetworkDefinitionCreationFlag flag)
        {
            TrtHandleException.handler(NativeMethods.trtNetworkDefinition_getFlag(
                ptr, flag, out int isSet));
            return isSet != 0;
        }
    }

}
