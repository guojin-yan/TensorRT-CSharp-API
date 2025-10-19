using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{

    /**
     * CUDA Memory Advise values
     */
    public enum CudaMemoryAdvise
    {
        SetReadMostly = 1, /**< Data will mostly be read and only occassionally be written to */
        UnsetReadMostly = 2, /**< Undo the effect of ::cudaMemAdviseSetReadMostly */
        SetPreferredLocation = 3, /**< Set the preferred location for the data as the specified device */
        UnsetPreferredLocation = 4, /**< Clear the preferred location for the data */
        SetAccessedBy = 5, /**< Data will be accessed by the specified device, so prevent page faults as much as possible */
        UnsetAccessedBy = 6  /**< Let the Unified Memory subsystem decide on the page faulting policy for the specified device */
    };
}
