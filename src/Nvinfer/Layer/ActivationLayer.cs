using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using System;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 激活层 - 对输入应用逐元素激活函数
    /// Activation layer - applies a per-element activation function to its input
    /// </summary>
    public class ActivationLayer : Layer
    {
        internal ActivationLayer(IntPtr ptr) : base(ptr) { }

        /// <summary>
        /// 获取或设置激活类型
        /// Gets or sets the activation type
        /// </summary>
        public TrtActivationType ActivationType
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtActivationLayer_getActivationType(ptr, out TrtActivationType type));
                return type;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtActivationLayer_setActivationType(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置激活函数的 alpha 参数
        /// Gets or sets the alpha parameter for the activation function
        /// </summary>
        public float Alpha
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtActivationLayer_getAlpha(ptr, out float alpha));
                return alpha;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtActivationLayer_setAlpha(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置激活函数的 beta 参数
        /// Gets or sets the beta parameter for the activation function
        /// </summary>
        public float Beta
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtActivationLayer_getBeta(ptr, out float beta));
                return beta;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtActivationLayer_setBeta(ptr, value));
            }
        }
    }
}
