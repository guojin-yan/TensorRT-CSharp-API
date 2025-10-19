using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    public enum LoggerSeverity : int
    {
        //! An internal error has occurred. Execution is unrecoverable.
        kINTERNAL_ERROR = 0,
        //! An application error has occurred.
        kERROR = 1,
        //! An application error has been discovered, but TensorRT has recovered or fallen back to a default.
        kWARNING = 2,
        //!  Informational messages with instructional information.
        kINFO = 3,
        //!  Verbose messages with debugging information.
        kVERBOSE = 4,
    };
}
