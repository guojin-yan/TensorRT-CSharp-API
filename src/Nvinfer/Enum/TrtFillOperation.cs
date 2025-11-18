using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 定义了填充层（Fill Layer）可以执行的张量填充操作。<br/>
    /// Enumerates the tensor fill operations that may be performed by a fill layer.
    /// </summary>
    /// <seealso cref="TrtFillLayer"/>
    public enum TrtFillOperation : int
    {
        /// <summary>
        /// 通过其索引的仿射函数计算每个元素的值。<br/>
        /// Compute each value via an affine function of its indices.
        /// </summary>
        /// <remarks>
        /// <para>
        /// 例如，假设填充层的参数为：
        /// <br/>
        /// For example, suppose the parameters for the IFillLayer are:
        /// </para>
        /// <list type="bullet">
        /// <item><description>维度 = [3,4] (Dimensions = [3,4])</description></item>
        /// <item><description>Alpha = 1</description></item>
        /// <item><description>Beta = [100,10]</description></item>
        /// </list>
        /// <para>
        /// 输出张量的元素 [i,j] 的计算公式为：<c>Alpha + Beta[0]*i + Beta[1]*j</c>。
        /// <br/>
        /// Element [i,j] of the output is: <c>Alpha + Beta[0]*i + Beta[1]*j</c>.
        /// </para>
        /// <para>
        /// 因此，输出矩阵为：
        /// <br/>
        /// Thus the output matrix is:
        /// </para>
        /// <code>
        ///      1  11  21  31
        ///    101 111 121 131
        ///    201 211 221 231
        /// </code>
        /// <para>
        /// 一个静态的 beta 值 b 会被隐式地转换为一维张量，即 Beta = [b]。
        /// <br/>
        /// A static beta b is implicitly a 1D tensor, i.e. Beta = [b].
        /// </para>
        /// </remarks>
        kLINSPACE = 0,

        /// <summary>
        /// 从均匀分布中随机抽取值。<br/>
        /// Randomly draw values from a uniform distribution.
        /// </summary>
        kRANDOM_UNIFORM = 1,

        /// <summary>
        /// 从正态分布中随机抽取值。<br/>
        /// Randomly draw values from a normal distribution.
        /// </summary>
        kRANDOM_NORMAL = 2
    };

}
