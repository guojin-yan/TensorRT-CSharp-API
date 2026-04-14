using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using System;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 一元操作层 - 对输入应用逐元素一元操作
    /// Unary layer - applies a per-element unary operation to its input
    /// </summary>
    public class UnaryLayer : Layer
    {
        internal UnaryLayer(IntPtr ptr) : base(ptr) { }

        /// <summary>
        /// 获取或设置一元操作类型
        /// Gets or sets the unary operation type
        /// </summary>
        public TrtUnaryOperation Operation
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtUnaryLayer_getOperation(ptr, out TrtUnaryOperation op));
                return op;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtUnaryLayer_setOperation(ptr, value));
            }
        }
    }
}
