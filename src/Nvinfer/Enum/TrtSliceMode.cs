using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 切片模式 - 控制 ISliceLayer 的越界索引处理行为。<br/>
    /// Controls how ISliceLayer handles out-of-bounds indices.
    /// </summary>
    /// <seealso cref="ISliceLayer"/>
    public enum TrtSliceMode : int
    {
        /// <summary>
        /// 使用默认的切片行为。对于越界索引，行为是未定义的。<br/>
        /// Use default slice behavior. Behavior is undefined for out-of-bounds indices.
        /// </summary>
        kDEFAULT = 0,

        /// <summary>
        /// 使用环绕（wrap）模式处理越界索引。例如，在维度大小为5的轴上，索引-1变为4，索引5变为0。<br/>
        /// Use wrap mode for out-of-bounds indices. For example, on an axis with size 5, index -1 becomes 4 and index 5 becomes 0.
        /// </summary>
        kWRAP = 1,

        /// <summary>
        /// 使用钳制（clamp）模式处理越界索引。索引被钳制到有效范围内，例如负索引变为0，超出范围的索引变为最大值。<br/>
        /// Use clamp mode for out-of-bounds indices. Indices are clamped to the valid range, so negative indices become 0 and out-of-range indices become the maximum value.
        /// </summary>
        kCLAMP = 2,

        /// <summary>
        /// 使用填充（fill）模式处理越界索引。越界元素被填充为指定的填充值。<br/>
        /// Use fill mode for out-of-bounds indices. Out-of-bounds elements are filled with a specified fill value.
        /// </summary>
        kFILL = 3
    };
}
