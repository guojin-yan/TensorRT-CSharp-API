using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using System;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 池化层 - 对输入应用池化操作
    /// Pooling layer - applies a pooling operation to its input
    /// </summary>
    public class PoolingLayer : Layer
    {
        internal PoolingLayer(IntPtr ptr) : base(ptr) { }

        /// <summary>
        /// 获取或设置池化类型
        /// Gets or sets the pooling type
        /// </summary>
        public TrtPoolingType PoolingType
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtPoolingLayer_getPoolingType(ptr, out TrtPoolingType poolingType));
                return poolingType;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtPoolingLayer_setPoolingType(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置窗口大小（支持多维）
        /// Gets or sets the window size for pooling (N-dimensional)
        /// </summary>
        public Dims WindowSizeNd
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtPoolingLayer_getWindowSizeNd(ptr, out Dims windowSize));
                return windowSize;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtPoolingLayer_setWindowSizeNd(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置步幅（支持多维）
        /// Gets or sets the stride for pooling (N-dimensional)
        /// </summary>
        public Dims StrideNd
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtPoolingLayer_getStrideNd(ptr, out Dims stride));
                return stride;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtPoolingLayer_setStrideNd(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置填充（支持多维）
        /// Gets or sets the padding for pooling (N-dimensional)
        /// </summary>
        public Dims PaddingNd
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtPoolingLayer_getPaddingNd(ptr, out Dims padding));
                return padding;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtPoolingLayer_setPaddingNd(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置前置填充
        /// Gets or sets the pre-padding
        /// </summary>
        public Dims PrePadding
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtPoolingLayer_getPrePadding(ptr, out Dims padding));
                return padding;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtPoolingLayer_setPrePadding(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置后置填充
        /// Gets or sets the post-padding
        /// </summary>
        public Dims PostPadding
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtPoolingLayer_getPostPadding(ptr, out Dims padding));
                return padding;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtPoolingLayer_setPostPadding(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置填充模式
        /// Gets or sets the padding mode
        /// </summary>
        public TrtPaddingMode PaddingMode
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtPoolingLayer_getPaddingMode(ptr, out TrtPaddingMode paddingMode));
                return paddingMode;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtPoolingLayer_setPaddingMode(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置混合因子（用于 MAX_AVERAGE_BLEND 池化类型）
        /// Gets or sets the blend factor (used for MAX_AVERAGE_BLEND pooling type)
        /// </summary>
        /// <remarks>
        /// 混合因子用于 MAX_AVERAGE_BLEND 池化类型，计算方式为: (1 - blendFactor) * maxPool + blendFactor * avgPool
        /// The blend factor is used for MAX_AVERAGE_BLEND pooling type, computed as: (1 - blendFactor) * maxPool + blendFactor * avgPool
        /// </remarks>
        public float BlendFactor
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtPoolingLayer_getBlendFactor(ptr, out float blendFactor));
                return blendFactor;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtPoolingLayer_setBlendFactor(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置平均池化时是否排除填充值
        /// Gets or sets whether average pooling excludes padding when computing the average
        /// </summary>
        public bool AverageCountExcludesPadding
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtPoolingLayer_getAverageCountExcludesPadding(ptr, out int averageCountExcludesPadding));
                return averageCountExcludesPadding != 0;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtPoolingLayer_setAverageCountExcludesPadding(ptr, value ? 1 : 0));
            }
        }
    }
}
