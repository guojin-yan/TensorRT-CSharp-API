using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using JYPPX.TensorRtSharp.ExternalInterface;
namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 日志回调函数的委托类型，用于接收格式化后的日志消息。
    /// The delegate type for the log callback function, used to receive formatted log messages.
    /// </summary>
    /// <param name="formattedMsg">从原生库传递过来的已格式化的日志字符串。/ The formatted log message string passed from the native library.</param>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void LogCallbackFunction([MarshalAs(UnmanagedType.LPStr)] string formattedMsg);
    /// <summary>
    /// 一个单例模式的日志记录器，用于与TensorRT的原生日志系统交互。
    /// A singleton logger for interacting with the native TensorRT logging system.
    /// </summary>
    public class Logger
    {
        private IntPtr ptr;
        // 私有静态实例（确保唯一）
        // Private static instance (to ensure uniqueness).
        private static Logger _instance;
        // 私有构造函数（防止外部实例化）
        // Private constructor (to prevent external instantiation).
        private Logger()
        {
            InitHandleException.handler(
                NativeMethods.trtLogger_getTrtLogger(out ptr));
        }
        // 公共静态方法，获取唯一实例
        // Public static property to get the unique instance.
        /// <summary>
        /// 获取 Logger 类的全局唯一实例。
        /// Gets the global unique instance of the Logger class.
        /// </summary>
        /// <returns>Logger 的单例实例。/ The singleton instance of Logger.</returns>
        public static Logger Instance
        {
            get
            {
                if (_instance == null)
                {
                    _instance = new Logger();
                }
                return _instance;
            }
        }
        /// <summary>
        /// 设置记录日志的最低严重性级别。低于此级别的日志将被忽略。
        /// Sets the minimum severity level for logging. Logs below this level will be ignored.
        /// </summary>
        /// <param name="threshold">日志的最低严重性级别。/ The minimum severity level for logging.</param>
        public void SetThreshold(LoggerSeverity threshold)
        {
            NativeMethods.trtLogger_setThreshold(threshold);
        }
        /// <summary>
        /// 设置一个自定义的回调函数来处理日志消息。
        /// Sets a custom callback function to handle log messages.
        /// </summary>
        /// <param name="callback">符合 LogCallbackFunction 委托的回调函数。/ The callback function that matches the LogCallbackFunction delegate.</param>
        public void SetCallback(LogCallbackFunction callback)
        {
            NativeMethods.trtLogger_setCallback(callback);
        }
        /// <summary>
        /// 记录一条指定严重性级别的消息。
        /// Logs a message with a specified severity level.
        /// </summary>
        /// <param name="level">日志的严重性级别。/ The severity level of the log.</param>
        /// <param name="msg">要记录的消息内容。/ The content of the message to log.</param>
        public void Log(LoggerSeverity level, string msg)
        {
            NativeMethods.trtLogger_log(level, msg);
        }
        /// <summary>
        /// 记录一条 VERBOSE（详细）级别的消息。
        /// Logs a VERBOSE level message.
        /// </summary>
        /// <param name="message">一个复合格式字符串，包含要记录的文本。/ A composite format string that contains the text to log.</param>
        /// <param name="args">一个对象数组，包含零个或多个要格式化的对象。/ An object array that contains zero or more objects to format.</param>
        public void VERBOSE(string message, params object[] args)
        {
            string formattedMsg = args.Length > 0
                ? string.Format(message, args)  // 格式化 / Format
                : message;  // 无参数时直接输出 / Output directly when there are no arguments.
            Log(LoggerSeverity.kVERBOSE, formattedMsg);
        }
        /// <summary>
        /// 记录一条 INFO（信息）级别的消息。
        /// Logs an INFO level message.
        /// </summary>
        /// <param name="message">一个复合格式字符串，包含要记录的文本。/ A composite format string that contains the text to log.</param>
        /// <param name="args">一个对象数组，包含零个或多个要格式化的对象。/ An object array that contains zero or more objects to format.</param>
        public void INFO(string message, params object[] args)
        {
            string formattedMsg = args.Length > 0
                ? string.Format(message, args)  // 格式化 / Format
                : message;  // 无参数时直接输出 / Output directly when there are no arguments.
            Log(LoggerSeverity.kINFO, formattedMsg);
        }
        /// <summary>
        /// 记录一条 WARNING（警告）级别的消息。
        /// Logs a WARNING level message.
        /// </summary>
        /// <param name="message">一个复合格式字符串，包含要记录的文本。/ A composite format string that contains the text to log.</param>
        /// <param name="args">一个对象数组，包含零个或多个要格式化的对象。/ An object array that contains zero or more objects to format.</param>
        public void WARNING(string message, params object[] args)
        {
            string formattedMsg = args.Length > 0
                ? string.Format(message, args)  // 格式化 / Format
                : message;  // 无参数时直接输出 / Output directly when there are no arguments.
            Log(LoggerSeverity.kWARNING, formattedMsg);
        }
        /// <summary>
        /// 记录一条 ERROR（错误）级别的消息。
        /// Logs an ERROR level message.
        /// </summary>
        /// <param name="message">一个复合格式字符串，包含要记录的文本。/ A composite format string that contains the text to log.</param>
        /// <param name="args">一个对象数组，包含零个或多个要格式化的对象。/ An object array that contains zero or more objects to format.</param>
        public void ERROR(string message, params object[] args)
        {
            string formattedMsg = args.Length > 0
                ? string.Format(message, args)  // 格式化 / Format
                : message;  // 无参数时直接输出 / Output directly when there are no arguments.
            Log(LoggerSeverity.kERROR, formattedMsg);
        }
        /// <summary>
        /// 记录一条 INTERNAL_ERROR（内部错误）级别的消息。
        /// Logs an INTERNAL_ERROR level message.
        /// </summary>
        /// <param name="message">一个复合格式字符串，包含要记录的文本。/ A composite format string that contains the text to log.</param>
        /// <param name="args">一个对象数组，包含零个或多个要格式化的对象。/ An object array that contains zero or more objects to format.</param>
        public void INTERNAL_ERROR(string message, params object[] args)
        {
            string formattedMsg = args.Length > 0
                ? string.Format(message, args)  // 格式化 / Format
                : message;  // 无参数时直接输出 / Output directly when there are no arguments.
            Log(LoggerSeverity.kINTERNAL_ERROR, formattedMsg);
        }
    }
}
