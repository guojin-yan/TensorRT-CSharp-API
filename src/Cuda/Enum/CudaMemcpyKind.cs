using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /**
     * CUDA memory copy types
     */
    public enum CudaMemcpyKind : int
    {
        HostToHost = 0,      /**< Host   -> Host */
        HostToDevice = 1,      /**< Host   -> Device */
        DeviceToHost = 2,      /**< Device -> Host */
        DeviceToDevice = 3,      /**< Device -> Device */
        Default = 4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
    };
}
