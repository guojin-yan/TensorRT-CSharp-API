using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorRtSharp.Internal
{
    /// <summary>
    /// Whether native methods for P/Invoke raises an exception
    /// </summary>
    public enum ExceptionStatus
    {
        #pragma warning disable 1591
        NotOccurred = 0,
        Occurred = 1,
        OccurredTRT = 2,
        OccurredCuda = 3
    }
}
