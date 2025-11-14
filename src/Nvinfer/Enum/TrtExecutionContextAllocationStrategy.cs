using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 执行上下文内存分配策略枚举，定义了IExecutionContext的不同内存分配行为
    /// Execution context memory allocation strategy enumeration, defining different memory allocation behaviors for IExecutionContext
    /// </summary>
    /// <remarks>
    /// IExecutionContext在推理期间需要一块设备内存用于内部激活张量。用户可以让执行上下文以各种方式管理内存，或者自己分配内存。
    /// <br/>
    /// IExecutionContext requires a block of device memory for internal activation tensors during inference. The user can 
    /// either let the execution context manage the memory in various ways or allocate the memory themselves.
    /// </remarks>
    /// <seealso cref="ICudaEngine.CreateExecutionContext()"/>
    /// <seealso cref="IExecutionContext.SetDeviceMemory()"/>
    //! \enum ExecutionContextAllocationStrategy
    //! \brief Different memory allocation behaviors for IExecutionContext.
    //!
    //! IExecutionContext requires a block of device memory for internal activation tensors during inference. The user can
    //! either let the execution context manage the memory in various ways or allocate the memory themselves.
    //!
    //! \see ICudaEngine::createExecutionContext()
    //! \see IExecutionContext::setDeviceMemory()
    public enum TrtExecutionContextAllocationStrategy : int
    {
        /// <summary>
        /// 默认静态分配，使用所有配置文件中的最大尺寸
        /// Default static allocation with the maximum size across all profiles
        /// </summary>
        kSTATIC = 0,            //!< Default static allocation with the maximum size across all profiles.

        /// <summary>
        /// 当选择某个配置文件时，为该配置文件重新分配内存
        /// Reallocate for a profile when it's selected
        /// </summary>
        kON_PROFILE_CHANGE = 1, //!< Reallocate for a profile when it's selected.

        /// <summary>
        /// 用户向执行上下文提供自定义分配的内存
        /// The user supplies custom allocation to the execution context
        /// </summary>
        kUSER_MANAGED = 2,      //!< The user supplies custom allocation to the execution context.
    }

}
