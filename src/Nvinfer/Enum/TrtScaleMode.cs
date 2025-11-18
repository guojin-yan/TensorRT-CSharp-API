using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 控制 Scale 层如何应用位移、缩放和幂运算系数。<br/>
    /// Controls how shift, scale and power are applied in a Scale layer.
    /// </summary>
    /// <seealso cref="IScaleLayer"/>
    public enum TrtScaleMode : int
    {
        /// <summary>
        /// 均一模式。对张量中的所有元素应用相同的缩放、位移和幂系数。<br/>
        /// Identical coefficients across all elements of the tensor.
        /// </summary>
        kUNIFORM = 0,

        /// <summary>
        /// 逐通道模式。对张量的每个通道应用独立的缩放、位移和幂系数。<br/>
        /// Per-channel coefficients.
        /// </summary>
        kCHANNEL = 1,

        /// <summary>
        /// 逐元素模式。对张量中的每一个元素应用独立的缩放、位移和幂系数。<br/>
        /// Elementwise coefficients.
        /// </summary>
        kELEMENTWISE = 2
    };

}
