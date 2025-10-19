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
        // Lifecycle Management
        [DllImport(dllExtern, EntryPoint = "trtAsyncStreamReader_createAsyncStreamReader",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtAsyncStreamReader_createAsyncStreamReader(
            out IntPtr reader);
        [DllImport(dllExtern, EntryPoint = "trtAsyncStreamReader_free",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtAsyncStreamReader_free(
            IntPtr reader);
        // File Operations
        [DllImport(dllExtern, EntryPoint = "trtAsyncStreamReader_open",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtAsyncStreamReader_open(
            IntPtr reader,
            [MarshalAs(UnmanagedType.LPStr)] string filepath);
        [DllImport(dllExtern, EntryPoint = "trtAsyncStreamReader_close",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtAsyncStreamReader_close(
            IntPtr reader);
        // Async Read Operations
        [DllImport(dllExtern, EntryPoint = "trtAsyncStreamReader_read",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtAsyncStreamReader_read(
            IntPtr reader,
            IntPtr dest,
            long bytes,
            IntPtr stream,  // cudaStream_t is typically represented as IntPtr
            out long bytesRead);
        // Seeking Operations
        [DllImport(dllExtern, EntryPoint = "trtAsyncStreamReader_seek",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtAsyncStreamReader_seek(
            IntPtr reader,
            long offset,
            TrtSeekPosition where,
            out int success);
        // Status Query
        [DllImport(dllExtern, EntryPoint = "trtAsyncStreamReader_isOpen",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtAsyncStreamReader_isOpen(
            IntPtr reader,
            out int isOpen);
    }
}


