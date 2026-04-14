using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using System;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 拼接层 - 沿指定轴拼接多个输入张量
    /// Concatenation layer - concatenates multiple input tensors along a specified axis
    /// </summary>
    public class ConcatenationLayer : Layer
    {
        internal ConcatenationLayer(IntPtr ptr) : base(ptr) { }

        /// <summary>
        /// 获取或设置拼接的轴
        /// Gets or sets the axis along which concatenation occurs
        /// </summary>
        public int Axis
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtConcatenationLayer_getAxis(ptr, out int axis));
                return axis;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtConcatenationLayer_setAxis(ptr, value));
            }
        }
    }
}
