using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 张量IO模式枚举，定义了张量的输入输出模式
    /// Definition of tensor IO Mode
    /// </summary>
    public enum TrtTensorIOMode : int
    {
        /// <summary>
        /// 张量既不是输入也不是输出
        /// Tensor is not an input or output
        /// </summary>
        kNONE = 0,

        /// <summary>
        /// 张量是引擎的输入
        /// Tensor is input to the engine
        /// </summary>
        kINPUT = 1,

        /// <summary>
        /// 张量是引擎的输出
        /// Tensor is output by the engine
        /// </summary>
        kOUTPUT = 2
    };


}
