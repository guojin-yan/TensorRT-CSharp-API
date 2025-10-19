using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum TensorIOMode
    //!
    //! \brief Definition of tensor IO Mode.
    //!
   public enum TrtTensorIOMode : int
    {
        //! Tensor is not an input or output.
        kNONE = 0,

        //! Tensor is input to the engine.
        kINPUT = 1,

        //! Tensor is output by the engine.
        kOUTPUT = 2
    };

}
