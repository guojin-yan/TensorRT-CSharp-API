using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// TensorRT错误代码枚举
    /// TensorRT error code enumeration
    /// </summary>
    /// <remarks>
    /// 对应TensorRT C++ API中的nvinfer1::ErrorCode枚举
    /// Corresponds to nvinfer1::ErrorCode enum in TensorRT C++ API
    /// </remarks>
    public enum TrtErrorCode : int
    {
        /// <summary>
        /// 执行成功完成
        /// Execution completed successfully
        /// </summary>
        kSUCCESS = 0,

        /// <summary>
        /// 不属于任何其他类别的错误。此错误用于向前兼容性。
        /// An error that does not fall into any other category. This error is included for forward compatibility.
        /// </summary>
        kUNSPECIFIED_ERROR = 1,

        /// <summary>
        /// 发生了不可恢复的TensorRT错误。发出此错误时，TensorRT处于无效的内部状态，
        /// 对TensorRT的任何进一步调用都将导致未定义的行为。
        /// A non-recoverable TensorRT error occurred. TensorRT is in an invalid internal state when this error is
        /// emitted and any further calls to TensorRT will result in undefined behavior.
        /// </summary>
        kINTERNAL_ERROR = 2,

        /// <summary>
        /// 传递给函数的参数在单独情况下无效。这违反了API契约。
        /// An argument passed to the function is invalid in isolation.
        /// This is a violation of the API contract.
        /// </summary>
        kINVALID_ARGUMENT = 3,

        /// <summary>
        /// 在将参数状态与其他参数进行比较时发生错误。例如，两个张量在通道维度之外的
        /// concat维度不同。当参数单独正确但相对于其他参数不正确时，会触发此错误。
        /// 这有助于区分简单错误和复杂错误。这违反了API契约。
        /// An error occurred when comparing the state of an argument relative to other arguments.
        /// This is a violation of the API contract.
        /// </summary>
        kINVALID_CONFIG = 4,

        /// <summary>
        /// 在主机或设备上执行内存分配时发生错误。内存分配错误通常是致命的，但在应用程序
        /// 提供自己的内存分配例程的情况下，可以增加可用内存池并恢复执行。
        /// An error occurred when performing an allocation of memory on the host or the device.
        /// </summary>
        kFAILED_ALLOCATION = 5,

        /// <summary>
        /// TensorRT依赖的一个或多个组件未正确初始化。这是一个系统设置问题。
        /// One, or more, of the components that TensorRT relies on did not initialize correctly.
        /// This is a system setup issue.
        /// </summary>
        kFAILED_INITIALIZATION = 6,

        /// <summary>
        /// 执行期间发生错误，导致TensorRT过早结束，可能是异步错误、用户取消或CUDA/DLA报告的
        /// 其他执行错误。在动态系统中，可以丢弃数据并处理下一帧，或者可以重试执行。
        /// 这是执行错误或内存错误。
        /// An error occurred during execution that caused TensorRT to end prematurely.
        /// This is either an execution error or a memory error.
        /// </summary>
        kFAILED_EXECUTION = 7,

        /// <summary>
        /// 执行期间发生错误，导致数据损坏，但执行已完成。此错误的示例包括NaN抑制或整数溢出。
        /// 在动态系统中，可以丢弃数据并处理下一帧，或者可以重试执行。
        /// 这是数据损坏错误、输入错误或范围错误。在安全环境中不使用此错误，但可能在标准环境中使用。
        /// An error occurred during execution that caused the data to become corrupted, but execution finished.
        /// This is either a data corruption error, an input error, or a range error.
        /// This is not used in safety but may be used in standard.
        /// </summary>
        kFAILED_COMPUTATION = 8,

        /// <summary>
        /// TensorRT因函数调用序列不正确而进入不良状态。无效状态的示例是指定层仅在DLA上运行
        /// 而不使用GPU回退，但该层不受DLA支持。这可能发生在服务乐观地为多个不同配置执行网络
        /// 而不检查正确错误配置的情况下，而是丢弃被TensorRT捕获的不良配置。
        /// 这违反了API契约，但可能是可恢复的。
        /// TensorRT was put into a bad state by incorrect sequence of function calls.
        /// This is a violation of the API contract, but can be recoverable.
        /// </summary>
        kINVALID_STATE = 9,

        /// <summary>
        /// 由于硬件或系统限制，设备不支持网络而发生错误。例如，在安全认证环境中运行不安全层，
        /// 或者当前网络的资源要求大于目标设备的能力。网络在其他方面是正确的，但网络和硬件组合
        /// 存在问题。这可能是可恢复的。
        /// An error occurred due to the network not being supported on the device due to constraints of the hardware or system.
        /// This can be recoverable.
        /// </summary>
        kUNSUPPORTED_STATE = 10,
    }
}
