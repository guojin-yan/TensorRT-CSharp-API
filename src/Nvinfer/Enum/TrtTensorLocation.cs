using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum TensorLocation
    //!
    //! \brief The location for tensor data storage, device or host.
    //!
    public enum TrtTensorLocation : int
    {
        kDEVICE = 0, //!< Data stored on device.
        kHOST = 1,   //!< Data stored on host.
    };
}
