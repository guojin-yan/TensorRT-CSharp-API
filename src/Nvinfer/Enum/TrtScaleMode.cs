using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \brief Controls how shift, scale and power are applied in a Scale layer.
    //!
    //! \see IScaleLayer
    //!
    public enum TrtScaleMode : int
    {
        kUNIFORM = 0,    //!< Identical coefficients across all elements of the tensor.
        kCHANNEL = 1,    //!< Per-channel coefficients.
        kELEMENTWISE = 2 //!< Elementwise coefficients.
    };
}
