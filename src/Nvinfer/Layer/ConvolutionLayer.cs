using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using System;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 卷积层 - 对输入执行卷积操作<br/>
    /// Convolution layer - performs a convolution operation on its input
    /// </summary>
    public class ConvolutionLayer : Layer
    {
        internal ConvolutionLayer(IntPtr ptr) : base(ptr) { }

        /// <summary>
        /// 获取或设置输出特征图的数量。<br/>
        /// Gets or sets the number of output feature maps (output channels).
        /// </summary>
        public long NbOutputMaps
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_getNbOutputMaps(ptr, out long nbOutputMaps));
                return nbOutputMaps;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_setNbOutputMaps(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置卷积组数。<br/>
        /// Gets or sets the number of groups for the convolution.
        /// </summary>
        public long NbGroups
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_getNbGroups(ptr, out long nbGroups));
                return nbGroups;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_setNbGroups(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置卷积核权重。<br/>
        /// Gets or sets the kernel weights for the convolution.
        /// </summary>
        public TrtWeights KernelWeights
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_getKernelWeights(ptr, out TrtWeights weights));
                return weights;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_setKernelWeights(ptr, ref value));
            }
        }

        /// <summary>
        /// 获取或设置偏置权重。<br/>
        /// Gets or sets the bias weights for the convolution.
        /// </summary>
        public TrtWeights BiasWeights
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_getBiasWeights(ptr, out TrtWeights weights));
                return weights;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_setBiasWeights(ptr, ref value));
            }
        }

        /// <summary>
        /// 获取或设置前置填充。<br/>
        /// Gets or sets the pre-padding.
        /// </summary>
        public Dims PrePadding
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_getPrePadding(ptr, out Dims padding));
                return padding;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_setPrePadding(ptr, ref value));
            }
        }

        /// <summary>
        /// 获取或设置后置填充。<br/>
        /// Gets or sets the post-padding.
        /// </summary>
        public Dims PostPadding
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_getPostPadding(ptr, out Dims padding));
                return padding;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_setPostPadding(ptr, ref value));
            }
        }

        /// <summary>
        /// 获取或设置填充模式。<br/>
        /// Gets or sets the padding mode.
        /// </summary>
        public TrtPaddingMode PaddingMode
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_getPaddingMode(ptr, out TrtPaddingMode paddingMode));
                return paddingMode;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_setPaddingMode(ptr, value));
            }
        }

        /// <summary>
        /// 获取或设置 N 维卷积核大小。<br/>
        /// Gets or sets the multi-dimension kernel size.
        /// </summary>
        public Dims KernelSizeNd
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_getKernelSizeNd(ptr, out Dims kernelSize));
                return kernelSize;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_setKernelSizeNd(ptr, ref value));
            }
        }

        /// <summary>
        /// 获取或设置 N 维步幅。<br/>
        /// Gets or sets the multi-dimension stride.
        /// </summary>
        public Dims StrideNd
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_getStrideNd(ptr, out Dims stride));
                return stride;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_setStrideNd(ptr, ref value));
            }
        }

        /// <summary>
        /// 获取或设置 N 维填充。<br/>
        /// Gets or sets the multi-dimension padding.
        /// </summary>
        public Dims PaddingNd
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_getPaddingNd(ptr, out Dims padding));
                return padding;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_setPaddingNd(ptr, ref value));
            }
        }

        /// <summary>
        /// 获取或设置 N 维空洞率（膨胀系数）。<br/>
        /// Gets or sets the multi-dimension dilation.
        /// </summary>
        public Dims DilationNd
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_getDilationNd(ptr, out Dims dilation));
                return dilation;
            }
            set
            {
                TrtHandleException.handler(NativeMethods.trtConvolutionLayer_setDilationNd(ptr, ref value));
            }
        }
    }
}
