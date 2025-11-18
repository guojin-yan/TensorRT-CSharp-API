using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 提供对 TensorRT 引擎内部结构进行检查的功能。<br/>
    /// Provides functionality to inspect the internal structure of a TensorRT engine.
    /// </summary>
    public class EngineInspector : DisposableTrtObject
    {
        /// <summary>
        /// 初始化一个新的、空的 <see cref="EngineInspector"/> 实例。<br/>
        /// Initializes a new, empty instance of the <see cref="EngineInspector"/> class.
        /// </summary>
        public EngineInspector()
        {

        }

        /// <summary>
        /// 使用一个原生（非托管）指针初始化 <see cref="EngineInspector"/> 类的新实例。<br/>
        /// Initializes a new instance of the <see cref="EngineInspector"/> class from a native (unmanaged) pointer.
        /// </summary>
        /// <param name="ptr">
        /// 指向原生 <c>TrtEngineInspector</c> 对象的指针。<br/>
        /// A pointer to the native <c>TrtEngineInspector</c> object.
        /// </param>
        /// <exception cref="TrtException">
        /// 如果 <paramref name="ptr"/> 为 <see cref="IntPtr.Zero"/>，则抛出此异常。<br/>
        /// Thrown if <paramref name="ptr"/> is <see cref="IntPtr.Zero"/>.
        /// </exception>
        internal EngineInspector(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                throw new TrtException("Native object address is NULL");
            this.ptr = ptr;
        }

        /// <summary>
        /// 释放由 <see cref="EngineInspector"/> 使用的所有资源。<br/>
        /// Releases all resources used by the <see cref="EngineInspector"/>.
        /// </summary>
        public void Release()
        {
            Dispose();
        }

        /// <inheritdoc />
        /// <summary>
        /// 释放非托管资源。<br/>
        /// Releases unmanaged resources.
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            if (ptr != IntPtr.Zero && IsEnabledDispose)
                NativeMethods.trtEngineInspector_free(ptr);
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// 设置用于检查引擎的执行上下文。<br/>
        /// Sets the execution context to be used for inspecting the engine.
        /// </summary>
        /// <param name="context">
        /// 要设置的 <see cref="ExecutionContext"/> 实例。<br/>
        /// The <see cref="ExecutionContext"/> instance to set.
        /// </param>
        /// <returns>
        /// 如果操作成功，则为 <c>true</c>；否则为 <c>false</c>。<br/>
        /// <c>true</c> if the operation was successful; otherwise, <c>false</c>.
        /// </returns>
        public bool setExecutionContext(ExecutionContext context)
        {
            TrtHandleException.handler(NativeMethods.trtEngineInspector_setExecutionContext(
                ptr, context.TrtPtr, out int success));
            return success != 0;
        }

        /// <summary>
        /// 获取当前与引擎检查器关联的执行上下文。<br/>
        /// Gets the execution context currently associated with the engine inspector.
        /// </summary>
        /// <returns>
        /// 当前关联的 <see cref="ExecutionContext"/> 实例。<br/>
        /// The currently associated <see cref="ExecutionContext"/> instance.
        /// </returns>
        public ExecutionContext getExecutionContext()
        {
            TrtHandleException.handler(NativeMethods.trtEngineInspector_getExecutionContext(
                ptr, out IntPtr contextPtr));
            return new ExecutionContext(contextPtr);
        }

        /// <summary>
        /// 获取指定索引处的层的详细信息。<br/>
        /// Gets detailed information for the layer at the specified index.
        /// </summary>
        /// <param name="layerIndex">
        /// 要查询的层的索引，从 0 开始。<br/>
        /// The index of the layer to query, starting from 0.
        /// </param>
        /// <param name="format">
        /// 用于指定返回信息格式的 <see cref="TrtLayerInformationFormat"/>。<br/>
        /// The <see cref="TrtLayerInformationFormat"/> that specifies the format of the returned information.
        /// </param>
        /// <returns>
        /// 一个包含指定层信息的字符串。<br/>
        /// A string containing the information for the specified layer.
        /// </returns>
        public string getLayerInformation(int layerIndex, TrtLayerInformationFormat format)
        {
            TrtHandleException.handler(NativeMethods.trtEngineInspector_getLayerInformationByLayerIndex(
                ptr, layerIndex, format, out IntPtr infoPtr));
            return Marshal.PtrToStringAnsi(infoPtr);
        }

        /// <summary>
        /// 获取整个引擎的信息。<br/>
        /// Gets the information for the entire engine.
        /// </summary>
        /// <param name="format">
        /// 用于指定返回信息格式的 <see cref="TrtLayerInformationFormat"/>。<br/>
        /// The <see cref="TrtLayerInformationFormat"/> that specifies the format of the returned information.
        /// </param>
        /// <returns>
        /// 一个包含整个引擎信息的字符串。<br/>
        /// A string containing the information for the entire engine.
        /// </returns>
        public string getEngineInformation(TrtLayerInformationFormat format)
        {
            TrtHandleException.handler(NativeMethods.trtEngineInspector_getEngineInformation(
                ptr, format, out IntPtr infoPtr));
            return Marshal.PtrToStringAnsi(infoPtr);
        }

    }

}
