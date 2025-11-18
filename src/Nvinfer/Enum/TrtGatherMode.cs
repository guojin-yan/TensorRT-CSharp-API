using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 控制收集层的行为模式。<br/>
    /// Controls the form of the IGatherLayer.
    /// </summary>
    /// <seealso cref="TrtGatherLayer"/>
    public enum TrtGatherMode : int
    {
        /// <summary>
        /// 默认模式。<br/>
        /// Default mode.
        /// <remarks>
        /// 功能类似于 ONNX 的 Gather 算子。<br/>
        /// Similar to the ONNX Gather operator.
        /// </remarks>
        /// </summary>
        kDEFAULT = 0,

        /// <summary>
        /// 元素模式。<br/>
        /// Element mode.
        /// <remarks>
        /// 功能类似于 ONNX 的 GatherElements 算子。<br/>
        /// Similar to the ONNX GatherElements operator.
        /// </remarks>
        /// </summary>
        kELEMENT = 1,

        /// <summary>
        /// N维模式。<br/>
        /// N-Dimensional mode.
        /// <remarks>
        /// 功能类似于 ONNX 的 GatherND 算子。<br/>
        /// Similar to the ONNX GatherND operator.
        /// </remarks>
        /// </summary>
        kND = 2
    };


}
