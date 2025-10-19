using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Exceptions
{
    /// This enum contains codes for all possible return values of the interface functions
    /// </summary>
    public enum InitExceptionStatus : int
    {
        NotOccurred = 0,
        NvinferInitFail = 1,
        NvinferPluginInitFail = 2,
        NvonnxparserInitFail = 3

    }
}
