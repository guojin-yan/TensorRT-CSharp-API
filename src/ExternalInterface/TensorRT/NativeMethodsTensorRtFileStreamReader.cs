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
        // Lifecycle Management
        [Pure, DllImport(dllExtern, EntryPoint = "trtFileStreamReader_createFileStreamReader",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtFileStreamReader_createFileStreamReader(
            out IntPtr reader);
        [Pure, DllImport(dllExtern, EntryPoint = "trtFileStreamReader_free",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtFileStreamReader_free(
            IntPtr reader);
        // File Operations
        [Pure, DllImport(dllExtern, EntryPoint = "trtFileStreamReader_open",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtFileStreamReader_open(
            IntPtr reader,
            [MarshalAs(UnmanagedType.LPStr)] string filepath);
        [Pure, DllImport(dllExtern, EntryPoint = "trtFileStreamReader_close",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtFileStreamReader_close(
            IntPtr reader);
        // Data Reading
        [Pure, DllImport(dllExtern, EntryPoint = "trtFileStreamReader_read",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtFileStreamReader_read(
            IntPtr reader,
            IntPtr dest,
            long bytes,
            out long bytesRead);
        // Reader Control
        [Pure, DllImport(dllExtern, EntryPoint = "trtFileStreamReader_reset",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtFileStreamReader_reset(
            IntPtr reader);
        [Pure, DllImport(dllExtern, EntryPoint = "trtFileStreamReader_isOpen",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtFileStreamReader_isOpen(
            IntPtr reader,
            out int isOpen);
    }
}
