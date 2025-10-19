using JYPPX.TensorRtSharp.Exceptions;
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
        // Error Recorder Management
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_getTrtErrorRecorder",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static InitExceptionStatus trtErrorRecorder_getTrtErrorRecorder(
            out IntPtr errorRecorder);
        // Error Information Access
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_getNbErrors",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static int trtErrorRecorder_getNbErrors();
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_getErrorCode",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtErrorRecorder_getErrorCode(
            int errorIdx);
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_getErrorDesc",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: MarshalAs(StringUnmanagedTypeNotWindows)]
        public extern static string trtErrorRecorder_getErrorDesc(
            int errorIdx);
        // Status Checks
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_hasOverflowed",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static int trtErrorRecorder_hasOverflowed();
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_empty",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static int trtErrorRecorder_empty();
        // Maintenance Operations
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_clear",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtErrorRecorder_clear();
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_reportError",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static int trtErrorRecorder_reportError(
            TrtExceptionStatus val,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string desc);
        // Reference Counting
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_incRefCount",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static int trtErrorRecorder_incRefCount();
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_decRefCount",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static int trtErrorRecorder_decRefCount();
    }
}