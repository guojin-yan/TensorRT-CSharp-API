using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum ProfilingVerbosity
    //!
    //! \brief List of verbosity levels of layer information exposed in NVTX annotations and in IEngineInspector.
    //!
    //! \see IBuilderConfig::setProfilingVerbosity(),
    //!      IBuilderConfig::getProfilingVerbosity(),
    //!      IEngineInspector
    //!
    public enum TrtProfilingVerbosity : int
    {
        kLAYER_NAMES_ONLY = 0, //!< Print only the layer names. This is the default setting.
        kNONE = 1,             //!< Do not print any layer information.
        kDETAILED = 2,         //!< Print detailed layer information including layer names and layer parameters.
 
    }
}
