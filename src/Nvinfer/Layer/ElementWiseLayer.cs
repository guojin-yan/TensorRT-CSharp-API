using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using System;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 逐元素操作层 - 对两个输入执行逐元素二元操作<br/>
    /// Element-wise layer - performs a per-element binary operation on two inputs
    /// </summary>
    public class ElementWiseLayer : Layer
    {
        internal ElementWiseLayer(IntPtr ptr) : base(ptr) { }

        /// <summary>
        /// 获取或设置逐元素操作类型<br/>
        /// Gets or sets the element-wise operation type
        /// </summary>
        public TrtElementWiseOperation Operation
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtElementWiseLayer_getOperation(ptr, out TrtElementWiseOperation op));
                return op;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtElementWiseLayer_setOperation(ptr, value));
            }
        }
    }
}
