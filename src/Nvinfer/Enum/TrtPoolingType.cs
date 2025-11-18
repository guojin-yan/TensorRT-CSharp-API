using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{

    //!
    //! \enum PoolingType
    //!
    //! \brief The type of pooling to perform in a pooling layer.
    //!
    public enum TrtPoolingType : int
    {
        kMAX = 0,              //!< Maximum over elements
        kAVERAGE = 1,          //!< Average over elements. If the tensor is padded, the count includes the padding
        kMAX_AVERAGE_BLEND = 2 //!< Blending between max and average pooling: (1-blendFactor)*maxPool + blendFactor*avgPool
    };
}
