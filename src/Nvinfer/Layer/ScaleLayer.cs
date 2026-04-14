using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using System;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 缩放层 - 对输入应用逐元素缩放、位移和幂运算
    /// Scale layer - applies per-element scale, shift, and power operations to its input
    /// </summary>
    public class ScaleLayer : Layer
    {
        internal ScaleLayer(IntPtr ptr) : base(ptr) { }

        /// <summary>
        /// 获取或设置缩放模式，控制如何应用缩放、位移和幂运算系数
        /// Gets or sets the scale mode controlling how shift, scale and power coefficients are applied
        /// </summary>
        public TrtScaleMode Mode
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtScaleLayer_getMode(ptr, out TrtScaleMode mode));
                return mode;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtScaleLayer_setMode(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置位移值（shift values）
        /// Gets or sets the shift values
        /// </summary>
        public Weights Shift
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtScaleLayer_getShift(ptr, out TrtWeights weights));
                return new Weights(weights);
            }
            set
            {
                TrtWeights weights = value.NativeWeights;
                TrtHandleException.handler(NativeMethods.trtScaleLayer_setShift(ptr, ref weights));
            }
        }

        /// <summary>
        /// 获取或设置缩放值（scale values）
        /// Gets or sets the scale values
        /// </summary>
        public Weights Scale
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtScaleLayer_getScale(ptr, out TrtWeights weights));
                return new Weights(weights);
            }
            set
            {
                TrtWeights weights = value.NativeWeights;
                TrtHandleException.handler(NativeMethods.trtScaleLayer_setScale(ptr, ref weights));
            }
        }

        /// <summary>
        /// 获取或设置幂运算值（power values）
        /// Gets or sets the power values
        /// </summary>
        public Weights Power
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtScaleLayer_getPower(ptr, out TrtWeights weights));
                return new Weights(weights);
            }
            set
            {
                TrtWeights weights = value.NativeWeights;
                TrtHandleException.handler(NativeMethods.trtScaleLayer_setPower(ptr, ref weights));
            }
        }

        /// <summary>
        /// 获取或设置通道轴（channel axis），用于 kCHANNEL 模式
        /// Gets or sets the channel axis, used for kCHANNEL mode
        /// </summary>
        public int ChannelAxis
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtScaleLayer_getChannelAxis(ptr, out int axis));
                return axis;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtScaleLayer_setChannelAxis(ptr, value));
            }
        }
    }
}
