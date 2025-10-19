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
        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_createInferRuntime",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static InitExceptionStatus trtRuntime_createInferRuntime(out IntPtr runtime);

        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_free",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtRuntime_free(IntPtr runtime);

        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_setDLACore",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntime_setDLACore(
            IntPtr runtime,
            int dlaCore);

        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_getDLACore",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntime_getDLACore(
            IntPtr runtime,
            out int coreNum);

        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_getNbDLACores",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntime_getNbDLACores(
            IntPtr runtime,
            out int coresNum);


        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_setGpuAllocator",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntime_setGpuAllocator(
            IntPtr runtime,
            IntPtr allocator);

        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_deserializeCudaEngineByBlob",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntime_deserializeCudaEngineByBlob(
            IntPtr runtime,
            ref byte blob,
            ulong size,
            out IntPtr cudaEngine);

        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_deserializeCudaEngineByFileStreamReader",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntime_deserializeCudaEngineByFileStreamReader(
            IntPtr runtime,
            IntPtr streamReader,
            out IntPtr cudaEngine);

        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_deserializeCudaEngineByAsyncStreamReader",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntime_deserializeCudaEngineByAsyncStreamReader(
            IntPtr runtime,
            IntPtr streamReader,
            out IntPtr cudaEngine);

        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_setMaxThreads",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntime_setMaxThreads(
            IntPtr runtime,
            int maxThreads);

        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_getMaxThreads",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntime_getMaxThreads(
            IntPtr runtime,
            out int maxThreads);

        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_setTemporaryDirectory",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntime_setTemporaryDirectory(
            IntPtr runtime,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string path);

        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_getTemporaryDirectory",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntime_getTemporaryDirectory(
            IntPtr runtime,
            out IntPtr path);
    }
}
