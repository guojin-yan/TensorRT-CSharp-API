using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum DeviceType
    //! \brief The device that this layer/network will execute on.
    //!
    //!
    public enum TrtDeviceType : int
    {

        kGPU = 0, //!< GPU Device
        kDLA = 1, //!< DLA Core
    }
}
