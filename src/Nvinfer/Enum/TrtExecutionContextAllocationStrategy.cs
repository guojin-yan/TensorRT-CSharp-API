using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum ExecutionContextAllocationStrategy
    //!
    //! \brief Different memory allocation behaviors for IExecutionContext.
    //!
    //! IExecutionContext requires a block of device memory for internal activation tensors during inference. The user can
    //! either let the execution context manage the memory in various ways or allocate the memory themselves.
    //!
    //! \see ICudaEngine::createExecutionContext()
    //! \see IExecutionContext::setDeviceMemory()
    //!
    public enum TrtExecutionContextAllocationStrategy:int
    {
        kSTATIC = 0,            //!< Default static allocation with the maximum size across all profiles.
        kON_PROFILE_CHANGE = 1, //!< Reallocate for a profile when it's selected.
        kUSER_MANAGED = 2,      //!< The user supplies custom allocation to the execution context.
    }
}
