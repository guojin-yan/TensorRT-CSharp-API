using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;


namespace JYPPX.TensorRtSharp.ExternalInterface
{
    /// <summary>
    /// ��־�ص������������Ͷ��壬���� severity �Ͳ�����Ϣ
    /// Log callback delegate type with severity and raw message
    /// </summary>
    /// <param name="severity">��־�ȼ� / Log severity level (0=INTERNAL_ERROR, 1=ERROR, 2=WARNING, 3=INFO, 4=VERBOSE)</param>
    /// <param name="msg">��־��Ϣ / Log message</param>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void LogCallbackFunctionV2(int severity, IntPtr msg);

    /// <summary>
    /// ��־�ص����������ͣ����ڽ��ո�ʽ����־��Ϣ
    /// Log callback delegate type for receiving formatted log messages
    /// </summary>
    /// <param name="formattedMsg">��ʽ����־��Ϣ / Formatted log message</param>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void LogCallbackFunction([MarshalAs(UnmanagedType.LPStr)] string formattedMsg);

    public static partial class NativeMethods
    {
        // Logger Management
        [Pure, DllImport(dllExtern, EntryPoint = "trtLogger_getTrtLogger",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static InitExceptionStatus trtLogger_getTrtLogger(
            out IntPtr logger);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLogger_setThreshold",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtLogger_setThreshold(
            LoggerSeverity threshold);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLogger_getThreshold",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtLogger_getThreshold(
            out LoggerSeverity threshold);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLogger_setCallback",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtLogger_setCallback(
            LogCallbackFunction callback);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLogger_setCallbackV2",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtLogger_setCallbackV2(
            LogCallbackFunctionV2 callback);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLogger_log",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtLogger_log(
            LoggerSeverity level,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string msg);
    }
}
