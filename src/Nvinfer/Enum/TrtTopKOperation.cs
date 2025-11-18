using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum TopKOperation
    //!
    //! \brief Enumerates the operations that may be performed by a TopK layer.
    //!
    public enum TrtTopKOperation : int
    {
        kMAX = 0, //!< Maximum of the elements.
        kMIN = 1, //!< Minimum of the elements.
    };
}
