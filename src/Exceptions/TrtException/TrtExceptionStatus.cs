using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Exceptions
{
    /// This enum contains codes for all possible return values of the interface functions
    /// </summary>
    public enum TrtExceptionStatus : int
    {
        //!
        //! Execution completed successfully.
        //!
        TENSORRT_SUCCESS = 0,

        //!
        //! An error that does not fall into any other category. This error is included for forward compatibility.
        //!
        TENSORRT_UNSPECIFIED_ERROR = 1,

        //!
        //! A non-recoverable TensorRT error occurred. TensorRT is in an invalid internal state when this error is
        //! emitted and any further calls to TensorRT will result in undefined behavior.
        //!
        TENSORRT_INTERNAL_ERROR = 2,

        //!
        //! An argument passed to the function is invalid in isolation.
        //! This is a violation of the API contract.
        //!
        TENSORRT_INVALID_ARGUMENT = 3,

        //!
        //! An error occurred when comparing the state of an argument relative to other arguments. For example, the
        //! dimensions for concat differ between two tensors outside of the channel dimension. This error is triggered
        //! when an argument is correct in isolation, but not relative to other arguments. This is to help to distinguish
        //! from the simple errors from the more complex errors.
        //! This is a violation of the API contract.
        //!
        TENSORRT_INVALID_CONFIG = 4,

        //!
        //! An error occurred when performing an allocation of memory on the host or the device.
        //! A memory allocation error is normally fatal, but in the case where the application provided its own memory
        //! allocation routine, it is possible to increase the pool of available memory and resume execution.
        //!
        TENSORRT_FAILED_ALLOCATION = 5,

        //!
        //! One, or more, of the components that TensorRT relies on did not initialize correctly.
        //! This is a system setup issue.
        //!
        TENSORRT_FAILED_INITIALIZATION = 6,

        //!
        //! An error occurred during execution that caused TensorRT to end prematurely, either an asynchronous error,
        //! user cancellation, or other execution errors reported by CUDA/DLA. In a dynamic system, the
        //! data can be thrown away and the next frame can be processed or execution can be retried.
        //! This is either an execution error or a memory error.
        //!
        TENSORRT_FAILED_EXECUTION = 7,

        //!
        //! An error occurred during execution that caused the data to become corrupted, but execution finished. Examples
        //! of this error are NaN squashing or integer overflow. In a dynamic system, the data can be thrown away and the
        //! next frame can be processed or execution can be retried.
        //! This is either a data corruption error, an input error, or a range error.
        //! This is not used in safety but may be used in standard.
        //!
        TENSORRT_FAILED_COMPUTATION = 8,

        //!
        //! TensorRT was put into a bad state by incorrect sequence of function calls. An example of an invalid state is
        //! specifying a layer to be DLA only without GPU fallback, and that layer is not supported by DLA. This can occur
        //! in situations where a service is optimistically executing networks for multiple different configurations
        //! without checking proper error configurations, and instead throwing away bad configurations caught by TensorRT.
        //! This is a violation of the API contract, but can be recoverable.
        //!
        //! Example of a recovery:
        //! GPU fallback is disabled and conv layer with large filter(63x63) is specified to run on DLA. This will fail due
        //! to DLA not supporting the large TENSORRT_ernel size. This can be recovered by either turning on GPU fallback
        //! or setting the layer to run on the GPU.
        //!
        TENSORRT_INVALID_STATE = 9,

        //!
        //! An error occurred due to the network not being supported on the device due to constraints of the hardware or
        //! system. An example is running an unsafe layer in a safety certified context, or a resource requirement for the
        //! current network is greater than the capabilities of the target device. The network is otherwise correct, but
        //! the network and hardware combination is problematic. This can be recoverable.
        //! Examples:
        //!  * Scratch space requests larger than available device memory and can be recovered by increasing allowed
        //!    workspace size.
        //!  * Tensor size exceeds the maximum element count and can be recovered by reducing the maximum batch size.
        //!
        TENSORRT_UNSUPPORTED_STATE = 10,


        // 自定义错误类型，从100开始
        // C接口传入参数错误
        INVALID_C_PARAM = 100,

    }
}
