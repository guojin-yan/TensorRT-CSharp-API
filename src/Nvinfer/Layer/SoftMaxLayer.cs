using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using System;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// SoftMax层 - 对输入应用SoftMax归一化
    /// SoftMax layer - applies SoftMax normalization to its input
    /// </summary>
    public class SoftMaxLayer : Layer
    {
        internal SoftMaxLayer(IntPtr ptr) : base(ptr) { }

        /// <summary>
        /// 获取或设置SoftMax计算的轴位掩码
        /// Gets or sets the axis mask for the SoftMax computation
        /// </summary>
        public uint Axes
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtSoftMaxLayer_getAxes(ptr, out uint axes));
                return axes;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtSoftMaxLayer_setAxes(ptr, value));
            }
        }
    }
}
