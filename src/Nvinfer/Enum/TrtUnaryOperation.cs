using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 枚举了 Unary 层可以执行的单元操作。<br/>
    /// Enumerates the unary operations that may be performed by a Unary layer.
    /// </summary>
    /// <remarks>
    /// <para><b>操作的数据类型约束 / Operation Data Type Constraints:</b></para>
    /// <list type="bullet">
    /// <item><description>
    /// <c>kNOT</c> 操作的输入数据类型必须为 <c>DataType::kBOOL</c>。<br/>
    /// The <c>kNOT</c> operation must have inputs of type <c>DataType::kBOOL</c>.
    /// </description></item>
    /// <item><description>
    /// <c>kSIGN</c> 和 <c>kABS</c> 操作的输入可以是浮点类型、<c>DataType::kINT8</c>、<c>DataType::kINT32</c> 或 <c>DataType::kINT64</c>。<br/>
    /// The <c>kSIGN</c> and <c>kABS</c> operations must have inputs of floating-point type, <c>DataType::kINT8</c>, <c>DataType::kINT32</c> or <c>DataType::kINT64</c>.
    /// </description></item>
    /// <item><description>
    /// <c>kISINF</c> 和 <c>kISNAN</c> 操作的输入必须是浮点类型。<br/>
    /// The <c>kISINF</c> and <c>kISNAN</c> operations must have inputs of floating-point type.
    /// </description></item>
    /// <item><description>
    /// 除上述操作外的所有其他操作的输入都必须是浮点类型。<br/>
    /// All other operations must have inputs of floating-point type.
    /// </description></item>
    /// </list>
    /// </remarks>
    /// <seealso cref="IUnaryLayer"/>
    public enum TrtUnaryOperation : int
    {
        /// <summary>
        /// 指数运算。<br/>
        /// Exponentiation.
        /// </summary>
        kEXP = 0,

        /// <summary>
        /// 自然对数 (以 e 为底)。<br/>
        /// Logarithm (base e).
        /// </summary>
        kLOG = 1,

        /// <summary>
        /// 平方根。<br/>
        /// Square root.
        /// </summary>
        kSQRT = 2,

        /// <summary>
        /// 倒数。<br/>
        /// Reciprocal.
        /// </summary>
        kRECIP = 3,

        /// <summary>
        /// 绝对值。<br/>
        /// Absolute value.
        /// </summary>
        kABS = 4,

        /// <summary>
        /// 取负。<br/>
        /// Negation.
        /// </summary>
        kNEG = 5,

        /// <summary>
        /// 正弦。<br/>
        /// Sine.
        /// </summary>
        kSIN = 6,

        /// <summary>
        /// 余弦。<br/>
        /// Cosine.
        /// </summary>
        kCOS = 7,

        /// <summary>
        /// 正切。<br/>
        /// Tangent.
        /// </summary>
        kTAN = 8,

        /// <summary>
        /// 双曲正弦。<br/>
        /// Hyperbolic sine.
        /// </summary>
        kSINH = 9,

        /// <summary>
        /// 双曲余弦。<br/>
        /// Hyperbolic cosine.
        /// </summary>
        kCOSH = 10,

        /// <summary>
        /// 反正弦。<br/>
        /// Inverse sine.
        /// </summary>
        kASIN = 11,

        /// <summary>
        /// 反余弦。<br/>
        /// Inverse cosine.
        /// </summary>
        kACOS = 12,

        /// <summary>
        /// 反正切。<br/>
        /// Inverse tangent.
        /// </summary>
        kATAN = 13,

        /// <summary>
        /// 反双曲正弦。<br/>
        /// Inverse hyperbolic sine.
        /// </summary>
        kASINH = 14,

        /// <summary>
        /// 反双曲余弦。<br/>
        /// Inverse hyperbolic cosine.
        /// </summary>
        kACOSH = 15,

        /// <summary>
        /// 反双曲正切。<br/>
        /// Inverse hyperbolic tangent.
        /// </summary>
        kATANH = 16,

        /// <summary>
        /// 向上取整。<br/>
        /// Ceiling.
        /// </summary>
        kCEIL = 17,

        /// <summary>
        /// 向下取整。<br/>
        /// Floor.
        /// </summary>
        kFLOOR = 18,

        /// <summary>
        /// 高斯误差函数。<br/>
        /// Gauss error function.
        /// </summary>
        kERF = 19,

        /// <summary>
        /// 逻辑非。<br/>
        /// Logical NOT.
        /// </summary>
        kNOT = 20,

        /// <summary>
        /// 求符号。如果输入 > 0，输出 1；如果输入 < 0，输出 -1；如果输入 == 0，输出 0。<br/>
        /// Sign. If input > 0, output 1; if input < 0, output -1; if input == 0, output 0.
        /// </summary>
        kSIGN = 21,

        /// <summary>
        /// 对浮点类型数据四舍五入到最近的偶数。<br/>
        /// Round to nearest even for floating-point data type.
        /// </summary>
        kROUND = 22,

        /// <summary>
        /// 对于浮点数据类型，如果输入值为正/负无穷大，则返回 true。<br/>
        /// Return true if input value equals +/- infinity for floating-point data type.
        /// </summary>
        kISINF = 23,

        /// <summary>
        /// 对于浮点数据类型，如果输入值为 NaN (非数字)，则返回 true。<br/>
        /// Return true if input value is a NaN for floating-point data type.
        /// </summary>
        kISNAN = 24
    };


}
