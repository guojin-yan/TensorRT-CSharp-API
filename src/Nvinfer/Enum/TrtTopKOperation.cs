using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 枚举了 TopK 层可以执行的操作。<br/>
    /// Enumerates the operations that may be performed by a TopK layer.
    /// </summary>
    public enum TrtTopKOperation : int
    {
        /// <summary>
        /// 求元素的 K 个最大值。<br/>
        /// Get the K maximum elements.
        /// </summary>
        kMAX = 0,

        /// <summary>
        /// 求元素的 K 个最小值。<br/>
        /// Get the K minimum elements.
        /// </summary>
        kMIN = 1
    };

}
