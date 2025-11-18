using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum ScatterMode
    //!
    //! \brief Control form of IScatterLayer
    //!
    //! \see IScatterLayer
    //!
    public enum TrtScatterMode : int
    {
        kELEMENT = 0, //!< Similar to ONNX ScatterElements
        kND = 1,      //!< Similar to ONNX ScatterND
    };
}
