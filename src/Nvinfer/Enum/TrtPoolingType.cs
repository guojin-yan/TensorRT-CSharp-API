using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{

    /// <summary>
    /// 指定池化层中要执行的池化类型。<br/>
    /// The type of pooling to perform in a pooling layer.
    /// </summary>
    public enum TrtPoolingType : int
    {
        /// <summary>
        /// 最大池化。取输入窗口内所有元素的最大值。<br/>
        /// Maximum over elements.
        /// </summary>
        kMAX = 0,

        /// <summary>
        /// 平均池化。取输入窗口内所有元素的平均值。如果输入张量进行了填充，则计算平均值时包含填充值。<br/>
        /// Average over elements. If the tensor is padded, the count includes the padding.
        /// </summary>
        kAVERAGE = 1,

        /// <summary>
        /// 最大池化与平均池化的混合。其计算方式为：(1 - blendFactor) * maxPool + blendFactor * avgPool。<br/>
        /// Blending between max and average pooling: (1 - blendFactor) * maxPool + blendFactor * avgPool.
        /// </summary>
        /// <remarks>
        /// 其中 <c>blendFactor</c> 是一个指定的混合因子，用于控制最大池化和平均池化的权重。<br/>
        /// Where <c>blendFactor</c> is a specified factor controlling the weight of max and average pooling.
        /// </remarks>
        kMAX_AVERAGE_BLEND = 2
    };

}
