using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum LayerInformationFormat
    //!
    //! \brief The format in which the IEngineInspector prints the layer information.
    //!
    //! \see IEngineInspector::getLayerInformation(), IEngineInspector::getEngineInformation()
    //!
    public enum TrtLayerInformationFormat : int
    {
        kONELINE = 0, //!< Print layer information in one line per layer.
        kJSON = 1,    //!< Print layer information in JSON format.
    };
}
