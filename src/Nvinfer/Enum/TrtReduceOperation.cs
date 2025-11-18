using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{

    /// <summary>
    /// 枚举了 Reduce（归约）层可以执行的归约操作。<br/>
    /// Enumerates the reduce operations that may be performed by a Reduce layer.
    /// </summary>
    /// <remarks>
    /// 以下表格展示了在给定数据类型上对空张量执行归约操作的结果：<br/>
    /// The table shows the result of reducing across an empty volume of a given type:
    /// <list type="table">
    /// <listheader>
    /// <term>Operation</term>
    /// <term>kFLOAT and kHALF</term>
    /// <term>kINT32</term>
    /// <term>kINT8</term>
    /// </listheader>
    /// <item>
    /// <description><c>kSUM</c></description>
    /// <description>0</description>
    /// <description>0</description>
    /// <description>0</description>
    /// </item>
    /// <item>
    /// <description><c>kPROD</c></description>
    /// <description>1</description>
    /// <description>1</description>
    /// <description>1</description>
    /// </item>
    /// <item>
    /// <description><c>kMAX</c></description>
    /// <description>negative infinity</description>
    /// <description>INT_MIN</description>
    /// <description>-128</description>
    /// </item>
    /// <item>
    /// <description><c>kMIN</c></description>
    /// <description>positive infinity</description>
    /// <description>INT_MAX</description>
    /// <description>127</description>
    /// </item>
    /// <item>
    /// <description><c>kAVG</c></description>
    /// <description>NaN</description>
    /// <description>0</description>
    /// <description>-128</description>
    /// </item>
    /// </list>
    /// <para>
    /// 当前版本的 TensorRT 通常通过 kFLOAT 或 kHALF 类型来执行 kINT8 的归约操作。上述 kINT8 列中的值展示了其对应浮点值的量化表示。<br/>
    /// The current version of TensorRT usually performs reduction for kINT8 via kFLOAT or kHALF. The kINT8 values show the quantized representations of the floating-point values.
    /// </para>
    /// </remarks>
    public enum TrtReduceOperation : int
    {
        /// <summary>
        /// 求和。<br/>
        /// Sum of elements.
        /// </summary>
        kSUM = 0,

        /// <summary>
        /// 求积。<br/>
        /// Product of elements.
        /// </summary>
        kPROD = 1,

        /// <summary>
        /// 求最大值。<br/>
        /// Maximum of elements.
        /// </summary>
        kMAX = 2,

        /// <summary>
        /// 求最小值。<br/>
        /// Minimum of elements.
        /// </summary>
        kMIN = 3,

        /// <summary>
        /// 求平均值。<br/>
        /// Average of elements.
        /// </summary>
        kAVG = 4
    };

}
