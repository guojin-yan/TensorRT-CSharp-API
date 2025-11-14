using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 权重角色枚举，定义了特定权重的使用方式
    /// How a layer uses particular Weights
    /// </summary>
    /// <remarks>
    /// IScaleLayer的幂权重被省略，不支持重新拟合这些权重
    /// The power weights of an IScaleLayer are omitted.  Refitting those is not supported.
    /// </remarks>
    public enum TrtWeightsRole : int
    {
        /// <summary>
        /// 用于IConvolutionLayer或IDeconvolutionLayer的卷积核
        /// kernel for IConvolutionLayer or IDeconvolutionLayer
        /// </summary>
        kKERNEL = 0,

        /// <summary>
        /// 用于IConvolutionLayer或IDeconvolutionLayer的偏置项
        /// bias for IConvolutionLayer or IDeconvolutionLayer
        /// </summary>
        kBIAS = 1,

        /// <summary>
        /// IScaleLayer的移位部分
        /// shift part of IScaleLayer
        /// </summary>
        kSHIFT = 2,

        /// <summary>
        /// IScaleLayer的缩放部分
        /// scale part of IScaleLayer
        /// </summary>
        kSCALE = 3,

        /// <summary>
        /// IConstantLayer的权重
        /// weights for IConstantLayer
        /// </summary>
        kCONSTANT = 4,

        /// <summary>
        /// 任何其他权重角色
        /// Any other weights role
        /// </summary>
        kANY = 5,
    };

}
