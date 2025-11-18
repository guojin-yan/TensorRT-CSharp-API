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
    /// 定义用于配置 TensorRT 运行时行为的设置。<br/>
    /// Defines settings for configuring the behavior of the TensorRT runtime.
    /// </summary>
    public class RuntimeConfig : DisposableTrtObject
    {
        /// <summary>
        /// 初始化一个新的、空的 <see cref="RuntimeConfig"/> 实例。<br/>
        /// Initializes a new, empty instance of the <see cref="RuntimeConfig"/> class.
        /// </summary>
        public RuntimeConfig()
        {
        }

        /// <summary>
        /// 使用一个原生（非托管）指针初始化 <see cref="RuntimeConfig"/> 类的新实例。<br/>
        /// Initializes a new instance of the <see cref="RuntimeConfig"/> class from a native (unmanaged) pointer.
        /// </summary>
        /// <param name="ptr">
        /// 指向原生 <c>TrtRuntimeConfig</c> 对象的指针。<br/>
        /// A pointer to the native <c>TrtRuntimeConfig</c> object.
        /// </param>
        /// <exception cref="TrtException">
        /// 如果 <paramref name="ptr"/> 为 <see cref="IntPtr.Zero"/>，则抛出此异常。<br/>
        /// Thrown if <paramref name="ptr"/> is <see cref="IntPtr.Zero"/>.
        /// </exception>
        internal RuntimeConfig(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                throw new TrtException("Native object address is NULL");
            this.ptr = ptr;
        }

        /// <summary>
        /// 释放由 <see cref="RuntimeConfig"/> 使用的所有资源。<br/>
        /// Releases all resources used by the <see cref="RuntimeConfig"/>.
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
                NativeMethods.trtRuntimeConfig_free(ptr);
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// 获取或设置执行上下文的分配策略。<br/>
        /// Gets or sets the allocation strategy for the execution context.
        /// </summary>
        /// <value>
        /// 用于配置引擎如何管理执行上下文的 <see cref="TrtExecutionContextAllocationStrategy"/> 枚举值。<br/>
        /// The <see cref="TrtExecutionContextAllocationStrategy"/> enum value that configures how the engine manages execution contexts.
        /// </value>
        /// <exception cref="TrtException">
        /// 在获取或设置值时，如果底层调用失败，则可能抛出此异常。<br/>
        /// This exception may be thrown if the underlying call fails when getting or setting the value.
        /// </exception>
        public TrtExecutionContextAllocationStrategy ExecutionContextAllocationStrategy
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtRuntimeConfig_getExecutionContextAllocationStrategy(
                    ptr,
                    out TrtExecutionContextAllocationStrategy strategy));
                return strategy;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtRuntimeConfig_setExecutionContextAllocationStrategy(
                    ptr,
                    value));
            }
        }
    }

}
