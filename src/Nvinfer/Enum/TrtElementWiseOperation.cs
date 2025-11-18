using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 定义了逐元素层（ElementWise Layer）可以执行的二元操作。<br/>
    /// Enumerates the binary operations that may be performed by an ElementWise layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// 操作 <c>kAND</c>, <c>kOR</c>, 和 <c>kXOR</c> 的输入数据类型必须为 <c>DataType::kBOOL</c>。<br/>
    /// Operations <c>kAND</c>, <c>kOR</c>, and <c>kXOR</c> must have inputs of DataType::kBOOL.
    /// </para>
    /// <para>
    /// 操作 <c>kPOW</c> 的输入必须为浮点类型或 <c>DataType::kINT8</c>。<br/>
    /// Operation <c>kPOW</c> must have inputs of floating-point type or DataType::kINT8.
    /// </para>
    /// <para>
    /// 所有其他操作的输入必须为浮点类型、<c>DataType::kINT8</c>、<c>DataType::kINT32</c> 或 <c>DataType::kINT64</c>。<br/>
    /// All other operations must have inputs of floating-point type, DataType::kINT8, DataType::kINT32, or DataType::kINT64.
    /// </para>
    /// </remarks>
    /// <seealso cref="TrtElementWiseLayer"/>
    public enum TrtElementWiseOperation : int
    {
        /// <summary>
        /// 两个元素相加。<br/>
        /// Sum of the two elements.
        /// </summary>
        kSUM = 0,

        /// <summary>
        /// 两个元素相乘。<br/>
        /// Product of the two elements.
        /// </summary>
        kPROD = 1,

        /// <summary>
        /// 取两个元素中的最大值。<br/>
        /// Maximum of the two elements.
        /// </summary>
        kMAX = 2,

        /// <summary>
        /// 取两个元素中的最小值。<br/>
        /// Minimum of the two elements.
        /// </summary>
        kMIN = 3,

        /// <summary>
        /// 第一个元素减去第二个元素。<br/>
        /// Subtract the second element from the first.
        /// </summary>
        kSUB = 4,

        /// <summary>
        /// 第一个元素除以第二个元素。<br/>
        /// Divide the first element by the second.
        /// </summary>
        kDIV = 5,

        /// <summary>
        /// 计算第一个元素的第二个元素次幂。<br/>
        /// The first element to the power of the second element.
        /// </summary>
        kPOW = 6,

        /// <summary>
        /// 对第一个元素除以第二个元素的结果向下取整。<br/>
        /// Floor division of the first element by the second.
        /// </summary>
        kFLOOR_DIV = 7,

        /// <summary>
        /// 对两个元素进行逻辑与操作。<br/>
        /// Logical AND of two elements.
        /// </summary>
        kAND = 8,

        /// <summary>
        /// 对两个元素进行逻辑或操作。<br/>
        /// Logical OR of two elements.
        /// </summary>
        kOR = 9,

        /// <summary>
        /// 对两个元素进行逻辑异或操作。<br/>
        /// Logical XOR of two elements.
        /// </summary>
        kXOR = 10,

        /// <summary>
        /// 检查两个元素是否相等，返回布尔值。<br/>
        /// Check if two elements are equal, returns a boolean value.
        /// </summary>
        kEQUAL = 11,

        /// <summary>
        /// 检查第一个张量中的元素是否大于第二个张量中对应的元素。<br/>
        /// Check if element in first tensor is greater than corresponding element in second tensor.
        /// </summary>
        kGREATER = 12,

        /// <summary>
        /// 检查第一个张量中的元素是否小于第二个张量中对应的元素。<br/>
        /// Check if element in first tensor is less than corresponding element in second tensor.
        /// </summary>
        kLESS = 13
    };

}
