using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 控制 IScatterLayer 的具体操作形式。<br/>
    /// Controls the form of the IScatterLayer.
    /// </summary>
    /// <seealso cref="IScatterLayer"/>
    public enum TrtScatterMode : int
    {
        /// <summary>
        /// 元素级散射。其行为类似于 ONNX 中的 <c>ScatterElements</c> 操作。<br/>
        /// The scatter operation is elementwise. This is similar to the <c>ScatterElements</c> operation in ONNX.
        /// </summary>
        kELEMENT = 0,

        /// <summary>
        /// N维散射。其行为类似于 ONNX 中的 <c>ScatterND</c> 操作，其中 "ND" 代表 "N-Dimensional"。<br/>
        /// The scatter operation is N-Dimensional. This is similar to the <c>ScatterND</c> operation in ONNX, where "ND" stands for "N-Dimensional".
        /// </summary>
        kND = 1
    };

}
