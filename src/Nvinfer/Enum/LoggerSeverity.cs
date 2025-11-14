using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 日志严重性枚举，定义了不同级别的日志信息
    /// Enumeration for logging severity levels, defining different levels of log information
    /// </summary>
    /// <remarks>
    /// 枚举值从低到高表示日志严重程度从高到低
    /// The enumeration values from low to high represent the log severity from high to low
    /// </remarks>
    public enum LoggerSeverity : int
    {
        /// <summary>
        /// 内部错误，程序无法继续执行
        /// Internal error, program cannot continue execution
        /// </summary>
        kINTERNAL_ERROR = 0,

        /// <summary>
        /// 应用程序错误
        /// Application error
        /// </summary>
        kERROR = 1,

        /// <summary>
        /// 应用程序错误已被发现，但TensorRT已恢复或回退到默认设置
        /// Application error has been discovered, but TensorRT has recovered or fallen back to a default
        /// </summary>
        kWARNING = 2,

        /// <summary>
        /// 包含指导性信息的信息性消息
        /// Informational messages with instructional information
        /// </summary>
        kINFO = 3,

        /// <summary>
        /// 包含调试信息的详细消息
        /// Verbose messages with debugging information
        /// </summary>
        kVERBOSE = 4,
    };

}
