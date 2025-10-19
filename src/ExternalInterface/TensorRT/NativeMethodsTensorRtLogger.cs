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

        [Pure, DllImport(dllExtern, EntryPoint = "trtLogger_setCallback",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtLogger_setCallback(
            LogCallbackFunction callback);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLogger_log",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtLogger_log(
            LoggerSeverity level,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string msg);
    }
}