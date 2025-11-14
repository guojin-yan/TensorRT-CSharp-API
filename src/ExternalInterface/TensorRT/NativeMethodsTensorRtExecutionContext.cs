using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Diagnostics.Contracts;
using System.IO;
using System.Runtime.InteropServices;


namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        [Pure, DllImport(dllExtern, EntryPoint = "trtExecutionContext_free",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_free(
            IntPtr context);


        [Pure, DllImport(dllExtern, EntryPoint = "trtExecutionContext_getEngine",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getEngine(
            IntPtr context,
            out IntPtr engine);

        [Pure, DllImport(dllExtern, EntryPoint = "trtExecutionContext_setName",
              CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setName(
            IntPtr context,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string name);

        [Pure, DllImport(dllExtern, EntryPoint = "trtExecutionContext_getName",
              CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getName(
            IntPtr context,
            out IntPtr name);
        [Pure, DllImport(dllExtern, EntryPoint = "trtExecutionContext_setDeviceMemory",
              CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setDeviceMemory(
            IntPtr context,
            IntPtr memory);

        [Pure, DllImport(dllExtern, EntryPoint = "trtExecutionContext_setDeviceMemoryV2",
              CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setDeviceMemoryV2(
            IntPtr context,
            IntPtr memory,
            long size);
        
        [Pure, DllImport(dllExtern, EntryPoint = "trtExecutionContext_getTensorStrides",
              CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getTensorStrides(
            IntPtr context,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string name,
            out Dims dims);

        [Pure, DllImport(dllExtern, EntryPoint = "trtExecutionContext_getOptimizationProfile",
              CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getOptimizationProfile(
            IntPtr context,
            out int profile);

        [Pure, DllImport(dllExtern, EntryPoint = "trtExecutionContext_setInputShape",
              CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setInputShape(
            IntPtr context,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string name,
            Dims dims);


        [Pure, DllImport(dllExtern, EntryPoint = "trtExecutionContext_getTensorShape",
              CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getTensorShape(
            IntPtr context,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string name,
            out Dims dims);
        [Pure, DllImport(dllExtern, EntryPoint = "trtExecutionContext_executeV2",
             CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_executeV2(
            IntPtr context,
            ref IntPtr bindings);

        [Pure, DllImport(dllExtern, EntryPoint = "trtExecutionContext_setTensorAddress",
              CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setTensorAddress(
            IntPtr context,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string tensorName,
            IntPtr tensorAddress);

        [Pure, DllImport(dllExtern, EntryPoint = "trtExecutionContext_getTensorAddress",
              CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getTensorAddress(
            IntPtr context,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string tensorName,
            out IntPtr tensorAddress);

        [Pure, DllImport(dllExtern, EntryPoint = "trtExecutionContext_setOutputTensorAddress",
              CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setOutputTensorAddress(
            IntPtr context,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string tensorName,
            IntPtr tensorAddress);
        [Pure, DllImport(dllExtern, EntryPoint = "trtExecutionContext_setInputTensorAddress",
           CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setInputTensorAddress(
         IntPtr context,
         [MarshalAs(StringUnmanagedTypeNotWindows)] string tensorName,
         IntPtr tensorAddress);

        [Pure, DllImport(dllExtern, EntryPoint = "trtExecutionContext_getOutputTensorAddress",
              CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getOutputTensorAddress(
            IntPtr context,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string tensorName,
            out IntPtr tensorAddress);
        [Pure, DllImport(dllExtern, EntryPoint = "trtExecutionContext_getMaxOutputSize",
              CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getMaxOutputSize(
            IntPtr context,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string tensorName,
            out long maxOutputSize);



        [Pure, DllImport(dllExtern, EntryPoint = "trtExecutionContext_executeV3",
              CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_executeV3(
            IntPtr context,
            IntPtr stream);


        // ===================================================================
        // IntPtr API P/Invoke Signatures
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_setDebugSync",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setDebugSync(IntPtr context, int sync);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_getDebugSync",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getDebugSync(IntPtr context, out int sync);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_setProfiler",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setProfiler(IntPtr context, IntPtr profiler);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_getProfiler",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getProfiler(IntPtr context, out IntPtr profiler);

        // --- Input Shape and Profiling State ---

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_allInputDimensionsSpecified",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_allInputDimensionsSpecified(IntPtr context, out int specified);

        // TRT_DEPRECATED: C# 没有直接的 P/Invoke 特性来标记弃用，但可以在 XML 文档中说明。
        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_allInputShapesSpecified",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_allInputShapesSpecified(IntPtr context, out int specified);

        // --- Asynchronous and Optimized Profiling ---

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_setOptimizationProfileAsync",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setOptimizationProfileAsync(
            IntPtr context,
            int profileIndex,
            IntPtr stream,
            out int success);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_setEnqueueEmitsProfile",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setEnqueueEmitsProfile(IntPtr context, int enqueueEmitsProfile);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_getEnqueueEmitsProfile",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getEnqueueEmitsProfile(IntPtr context, out int enqueueEmitsProfile);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_reportToProfiler",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_reportToProfiler(IntPtr context, out int success);

        // --- Shape Inference and Memory Management ---

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_inferShapes",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_inferShapes(
            IntPtr context,
            int nbMaxNames,
            [In, MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] tensorNames,
            out int result);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_updateDeviceMemorySizeForShapes",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_updateDeviceMemorySizeForShapes(IntPtr context, out ulong size);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_setInputConsumedEvent",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setInputConsumedEvent(IntPtr context, IntPtr cudaEvent, out int success);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_getInputConsumedEvent",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getInputConsumedEvent(IntPtr context, out IntPtr cudaEvent);

        // --- Custom Allocators ---

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_setOutputAllocator",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setOutputAllocator(
            IntPtr context,
            string tensorName,
            IntPtr outputAllocator,
            out int success);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_getOutputAllocator",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getOutputAllocator(
            IntPtr context,
            string tensorName,
            out IntPtr outputAllocator);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_setTemporaryStorageAllocator",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setTemporaryStorageAllocator(
            IntPtr context,
            IntPtr allocator,
            out int success);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_getTemporaryStorageAllocator",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getTemporaryStorageAllocator(
            IntPtr context,
            out IntPtr allocator);

        // --- Cache and NVTX ---

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_setPersistentCacheLimit",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setPersistentCacheLimit(IntPtr context, ulong size);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_getPersistentCacheLimit",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getPersistentCacheLimit(IntPtr context, out ulong size);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_setNvtxVerbosity",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setNvtxVerbosity(
            IntPtr context,
            TrtProfilingVerbosity verbosity,
            out int success);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_getNvtxVerbosity",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getNvtxVerbosity(
            IntPtr context,
            out TrtProfilingVerbosity verbosity);

        // --- Auxiliary Streams ---

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_setAuxStreams",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setAuxStreams(
            IntPtr context,
            IntPtr auxStreams,  // cudaStream_t*
            int nbStreams);

        // --- Debugging and Runtime Configuration ---

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_setDebugListener",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setDebugListener(
            IntPtr context,
            IntPtr listener, // 使用 delegate 替代原始 TRT_DebugListener*
            out int success);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_getDebugListener",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getDebugListener(
            IntPtr context,
            out IntPtr listener);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_setTensorDebugState",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setTensorDebugState(
            IntPtr context,
            string name,
            int flag,
            out int success);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_getDebugState",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getDebugState(
            IntPtr context,
            string name,
            out int state);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_getRuntimeConfig",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_getRuntimeConfig(
            IntPtr context,
            out IntPtr config);

        [DllImport(dllExtern, EntryPoint = "trtExecutionContext_setAllTensorsDebugState",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtExecutionContext_setAllTensorsDebugState(
            IntPtr context,
            int flag,
            out int success);

    }
}
