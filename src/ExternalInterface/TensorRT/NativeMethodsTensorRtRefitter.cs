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
        // Refitter Lifetime Management
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_createInferRefitter",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static InitExceptionStatus trtRefitter_createInferRefitter(
            IntPtr cudaEngine,
            out IntPtr refitter);
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_free",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtRefitter_free(
            IntPtr refitter);
        // Core Refitter Operations
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_setWeights",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_setWeights(
            IntPtr refitter,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string layerName,
            TrtWeightsRole role,
            TrtWeights weights,
            out int success);
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_refitCudaEngine",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_refitCudaEngine(
            IntPtr refitter,
            out int success);
        // Missing/All Weights/Layers Queries
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_getMissing",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_getMissing(
            IntPtr refitter,
            int size,
            [MarshalAs(UnmanagedType.LPArray, ArraySubType = StringUnmanagedTypeNotWindows)] string[] layerNames,
            [In, Out] TrtWeightsRole[] roles,
            out int count);
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_getAll",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_getAll(
            IntPtr refitter,
            int size,
            [MarshalAs(UnmanagedType.LPArray, ArraySubType = StringUnmanagedTypeNotWindows)] string[] layerNames,
            [In, Out] TrtWeightsRole[] roles,
            out int count);
        // Dynamic Range Operations (Deprecated)
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_setDynamicRange",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_setDynamicRange(
            IntPtr refitter,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string tensorName,
            float min,
            float max,
            out int success);
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_getDynamicRangeMin",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_getDynamicRangeMin(
            IntPtr refitter,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string tensorName,
            out float min);
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_getDynamicRangeMax",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_getDynamicRangeMax(
            IntPtr refitter,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string tensorName,
            out float max);
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_getTensorsWithDynamicRange",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_getTensorsWithDynamicRange(
            IntPtr refitter,
            int size,
            [MarshalAs(UnmanagedType.LPArray, ArraySubType = StringUnmanagedTypeNotWindows)] string[] tensorNames,
            out int count);
        // Error Recorder Handling
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_setErrorRecorder",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_setErrorRecorder(
            IntPtr refitter,
            IntPtr recorder);
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_getErrorRecorder",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_getErrorRecorder(
            IntPtr refitter,
            out IntPtr recorder);
        // Named Weights Operations
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_setNamedWeights",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_setNamedWeights(
            IntPtr refitter,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string name,
            TrtWeights weights,
            out int success);
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_setNamedWeightsWithLocation",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_setNamedWeightsWithLocation(
            IntPtr refitter,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string name,
            TrtWeights weights,
            TrtTensorLocation location,
            out int success);
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_getMissingWeights",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_getMissingWeights(
            IntPtr refitter,
            int size,
            [MarshalAs(UnmanagedType.LPArray, ArraySubType = StringUnmanagedTypeNotWindows)] string[] weightsNames,
            out int count);
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_getAllWeights",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_getAllWeights(
            IntPtr refitter,
            int size,
            [MarshalAs(UnmanagedType.LPArray, ArraySubType = StringUnmanagedTypeNotWindows)] string[] weightsNames,
            out int count);
        // Logger and Threading
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_getLogger",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_getLogger(
            IntPtr refitter,
            out IntPtr logger);
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_setMaxThreads",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_setMaxThreads(
            IntPtr refitter,
            int maxThreads,
            out int success);
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_getMaxThreads",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_getMaxThreads(
            IntPtr refitter,
            out int maxThreads);
        // Weight Management Operations
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_getNamedWeights",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_getNamedWeights(
            IntPtr refitter,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string weightsName,
            out TrtWeights weights);
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_getWeightsLocation",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_getWeightsLocation(
            IntPtr refitter,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string weightsName,
            out TrtTensorLocation location);
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_unsetNamedWeights",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_unsetNamedWeights(
            IntPtr refitter,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string weightsName,
            out int success);
        // Validation Controls
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_setWeightsValidation",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_setWeightsValidation(
            IntPtr refitter,
            int weightsValidation);
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_getWeightsValidation",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_getWeightsValidation(
            IntPtr refitter,
            out int weightsValidation);
        // Async Operations
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_refitCudaEngineAsync",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_refitCudaEngineAsync(
            IntPtr refitter,
            IntPtr stream,
            out int success);
        [Pure, DllImport(dllExtern, EntryPoint = "trtRefitter_getWeightsPrototype",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRefitter_getWeightsPrototype(
            IntPtr refitter,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string weightsName,
            out TrtWeights weights);
    }
}
