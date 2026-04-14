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

        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_setTempfileControlFlags",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntime_setTempfileControlFlags(
            IntPtr runtime,
            TrtTempfileControlFlag tempfileControlFlag);

        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_getTempfileControlFlags",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntime_getTempfileControlFlags(
            IntPtr runtime,
            out TrtTempfileControlFlag tempfileControlFlag);


        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_getPluginRegistry",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntime_getPluginRegistry(
            IntPtr runtime,
            out IntPtr pluginRegistry);

        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_loadRuntime",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntime_loadRuntime(
            IntPtr runtime,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string path,
            out IntPtr re_runtime);

        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_setEngineHostCodeAllowed",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntime_setEngineHostCodeAllowed(
            IntPtr runtime,
            int allowed);


        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_getEngineHostCodeAllowed",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntime_getEngineHostCodeAllowed(
            IntPtr runtime,
            out int allowed);

        /// <summary>
        /// ��ȡ Runtime ���� Logger
        /// Gets the logger associated with the runtime
        /// </summary>
        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntime_getLogger",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntime_getLogger(
            IntPtr runtime,
            out IntPtr logger);
    }
}
