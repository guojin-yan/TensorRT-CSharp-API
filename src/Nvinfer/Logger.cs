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
    // Logger callback delegate type
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void LogCallbackFunction([MarshalAs(UnmanagedType.LPStr)] string formattedMsg);

    public class Logger
    {
        private IntPtr ptr;
        // 私有静态实例（确保唯一）
        private static Logger _instance;
        // 私有构造函数（防止外部实例化）
        private Logger()
        {
            InitHandleException.handler(
                NativeMethods.trtLogger_getTrtLogger(out ptr));
        }
        // 公共静态方法，获取唯一实例
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

        public void SetThreshold(LoggerSeverity threshold)
        {
            NativeMethods.trtLogger_setThreshold(threshold);
        }

        public void SetCallback(LogCallbackFunction callback)
        {
            NativeMethods.trtLogger_setCallback(callback);
        }

        public void Log(LoggerSeverity level, string msg)
        {
            NativeMethods.trtLogger_log(level, msg);
        }

        public void VERBOSE(string message, params object[] args)
        {
            string formattedMsg = args.Length > 0
                ? string.Format(message, args)  // 格式化
                : message;  // 无参数时直接输出
            Log(LoggerSeverity.kVERBOSE, formattedMsg);
        }


        public void INFO(string message, params object[] args)
        {
            string formattedMsg = args.Length > 0
                ? string.Format(message, args)  // 格式化
                : message;  // 无参数时直接输出
            Log(LoggerSeverity.kINFO, formattedMsg);
        }

        public void WARNING(string message, params object[] args)
        {
            string formattedMsg = args.Length > 0
                ? string.Format(message, args)  // 格式化
                : message;  // 无参数时直接输出
            Log(LoggerSeverity.kWARNING, formattedMsg);
        }


        public void ERROR(string message, params object[] args)
        {
            string formattedMsg = args.Length > 0
                ? string.Format(message, args)  // 格式化
                : message;  // 无参数时直接输出
            Log(LoggerSeverity.kERROR, formattedMsg);

        }

        public void INTERNAL_ERROR(string message, params object[] args)
        {
            string formattedMsg = args.Length > 0
                ? string.Format(message, args)  // 格式化
                : message;  // 无参数时直接输出
            Log(LoggerSeverity.kINTERNAL_ERROR, formattedMsg);
        }
    }
}
