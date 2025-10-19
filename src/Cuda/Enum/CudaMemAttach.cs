using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda.Enum
{
    public enum CudaMemAttach : UInt64
    {
        /// <summary>
        /// Memory can be accessed by any stream on any device
        /// </summary>
        Global = 0x01,
        /// <summary>
        /// Memory cannot be accessed by any stream on any device
        /// </summary>
        Host = 0x02,
        /// <summary>
        /// Memory can only be accessed by a single stream on the associated device
        /// </summary>
        Single = 0x04, 
    }
}
