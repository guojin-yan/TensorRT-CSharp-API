using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \brief Control form of IGatherLayer
    //!
    //! \see IGatherLayer
    //!
    public enum TrtGatherMode : int
    {
        kDEFAULT = 0, //!< Similar to ONNX Gather
        kELEMENT = 1, //!< Similar to ONNX GatherElements
        kND = 2       //!< Similar to ONNX GatherND
    };

}
