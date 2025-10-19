using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /**
     * CUDA shared memory configuration
     */

    public enum CudaSharedMemConfig : int
    {
        BankSizeDefault = 0,
        BankSizeFourByte = 1,
        BankSizeEightByte = 2
    };

}
