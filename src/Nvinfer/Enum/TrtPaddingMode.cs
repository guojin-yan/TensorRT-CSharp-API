using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 卷积和池化层的填充模式。<br/>
    /// Padding mode for convolution and pooling layers.
    /// </summary>
    public enum TrtPaddingMode : int
    {
        /// <summary>
        /// 使用显式填充，输出大小向下取整。<br/>
        /// Use explicit padding, rounding output size down.
        /// </summary>
        kEXPLICIT_ROUND_DOWN = 0,

        /// <summary>
        /// 使用显式填充，输出大小向上取整。<br/>
        /// Use explicit padding, rounding output size up.
        /// </summary>
        kEXPLICIT_ROUND_UP = 1,

        /// <summary>
        /// 使用 SAME 填充，前置填充小于等于后置填充。<br/>
        /// Use SAME padding, with prePadding &lt;= postPadding.
        /// </summary>
        kSAME_UPPER = 2,

        /// <summary>
        /// 使用 SAME 填充，前置填充大于等于后置填充。<br/>
        /// Use SAME padding, with prePadding &gt;= postPadding.
        /// </summary>
        kSAME_LOWER = 3
    }
}
