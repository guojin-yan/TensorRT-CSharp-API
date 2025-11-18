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
        //[Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_free",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static TrtExceptionStatus trtCudaEngine_free(
        //    IntPtr engine);


        //[Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_createExecutionContext",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static TrtExceptionStatus trtCudaEngine_createExecutionContext(
        //    IntPtr engine,
        //    TrtExecutionContextAllocationStrategy strategy,
        //    out IntPtr context);


        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_free",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static InitExceptionStatus trtCudaEngine_free(IntPtr engine);
        // --- Tensor Properties ---
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getTensorShape",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getTensorShape(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string tensorName,
            out Dims shape);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getTensorDataType",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getTensorDataType(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string tensorName,
            out TrtDataType dataType);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getTensorLocation",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getTensorLocation(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string tensorName,
            out TrtTensorLocation location);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_isShapeInferenceIO",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_isShapeInferenceIO(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string tensorName,
            out int isShapeIO);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getTensorIOMode",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getTensorIOMode(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string tensorName,
            out TrtTensorIOMode ioMode);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getTensorBytesPerComponent",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getTensorBytesPerComponent(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string tensorName,
            out int bytesPerComponent);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getTensorBytesPerComponent_ForProfile",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getTensorBytesPerComponent_ForProfile(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string tensorName,
            int profileIndex,
            out int bytesPerComponent);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getTensorComponentsPerElement",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getTensorComponentsPerElement(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string tensorName,
            out int componentsPerElement);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getTensorComponentsPerElement_ForProfile",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getTensorComponentsPerElement_ForProfile(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string tensorName,
            int profileIndex,
            out int componentsPerElement);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getTensorFormat",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getTensorFormat(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string tensorName,
            out TrtTensorFormat format);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getTensorFormat_ForProfile",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getTensorFormat_ForProfile(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string tensorName,
            int profileIndex,
            out TrtTensorFormat format);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getTensorFormatDesc",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getTensorFormatDesc(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string tensorName,
            out IntPtr tensorFormatDesc); // const char**
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getTensorFormatDescByProfileIndex",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getTensorFormatDescByProfileIndex(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string tensorName,
            int profileIndex,
            out IntPtr tensorFormatDesc); // const char**
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getTensorVectorizedDim",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getTensorVectorizedDim(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string tensorName,
            out int vectorizedDim);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getTensorVectorizedDim_ForProfile",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getTensorVectorizedDim_ForProfile(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string tensorName,
            int profileIndex,
            out int vectorizedDim);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_isDebugTensor",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_isDebugTensor(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string name,
            out int isDebug);
        // --- Engine Properties ---
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getNbLayers",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getNbLayers(
            IntPtr engine,
            out int nbLayers);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getName",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getName(
            IntPtr engine,
            out IntPtr networkName); // const char**
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getNbOptimizationProfiles",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getNbOptimizationProfiles(
            IntPtr engine,
            out int nbProfiles);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getEngineCapability",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getEngineCapability(
            IntPtr engine,
            out TrtEngineCapability capability);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_isRefittable",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_isRefittable(
            IntPtr engine,
            out int refittable);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getNbIOTensors",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getNbIOTensors(
            IntPtr engine,
            out int nbIOTensors);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getIOTensorName",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getIOTensorName(
            IntPtr engine,
            int index,
            out IntPtr tensorName); // const char**
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getHardwareCompatibilityLevel",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getHardwareCompatibilityLevel(
            IntPtr engine,
            out TrtHardwareCompatibilityLevel level);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getNbAuxStreams",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getNbAuxStreams(
            IntPtr engine,
            out int nbAuxStreams);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getTacticSources",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getTacticSources(
            IntPtr engine,
            out TrtTacticSource sources);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getProfilingVerbosity",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getProfilingVerbosity(
            IntPtr engine,
            out TrtProfilingVerbosity verbosity);

        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_serialize",
    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_serialize(
    IntPtr engine,
    out IntPtr serializedEngine); // For TrtHostMemory**
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_createSerializationConfig",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_createSerializationConfig(
            IntPtr engine,
            out IntPtr config); // For TrtSerializationConfig**
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_serializeWithConfig",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_serializeWithConfig(
            IntPtr engine,
            IntPtr config, // TrtSerializationConfig*
            out IntPtr serializedEngine); // For TrtHostMemory**
                                          // --- Execution Context ---
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_createExecutionContext",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_createExecutionContext(
            IntPtr engine,
            out IntPtr context); // For TrtExecutionContext**
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_createExecutionContextByStrategy",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_createExecutionContextByStrategy(
            IntPtr engine,
            TrtExecutionContextAllocationStrategy strategy,
            out IntPtr context); // For TrtExecutionContext**
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_createExecutionContextWithoutDeviceMemory",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_createExecutionContextWithoutDeviceMemory(
            IntPtr engine,
            out IntPtr context); // For TrtExecutionContext**
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_createExecutionContextByRuntimeConfig",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_createExecutionContextByRuntimeConfig(
            IntPtr engine,
            IntPtr runtimeConfig, // TrtRuntimeConfig*
            out IntPtr context); // For TrtExecutionContext**
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_createRuntimeConfig",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_createRuntimeConfig(
            IntPtr engine,
            out IntPtr config); // For TrtRuntimeConfig**
                                // --- Memory ---
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getDeviceMemorySize",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getDeviceMemorySize(
            IntPtr engine,
            out long size); // C size_t -> C# ulong
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getDeviceMemorySizeForProfile",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getDeviceMemorySizeForProfile(
            IntPtr engine,
            int profileIndex,
            out ulong size); // C size_t -> C# ulong
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getDeviceMemorySizeV2",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getDeviceMemorySizeV2(
            IntPtr engine,
            out long size);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getDeviceMemorySizeForProfileV2",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getDeviceMemorySizeForProfileV2(
            IntPtr engine,
            int profileIndex,
            out long size);
        // --- Weight Streaming ---
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_setWeightStreamingBudget",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_setWeightStreamingBudget(
            IntPtr engine,
            long gpuMemoryBudget,
            out int success);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getWeightStreamingBudget",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getWeightStreamingBudget(
            IntPtr engine,
            out long budget);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getMinimumWeightStreamingBudget",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getMinimumWeightStreamingBudget(
            IntPtr engine,
            out long budget);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getStreamableWeightsSize",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getStreamableWeightsSize(
            IntPtr engine,
            out long size);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_setWeightStreamingBudgetV2",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_setWeightStreamingBudgetV2(
            IntPtr engine,
            long gpuMemoryBudget,
            out int success);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getWeightStreamingBudgetV2",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getWeightStreamingBudgetV2(
            IntPtr engine,
            out long budget);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getWeightStreamingAutomaticBudget",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getWeightStreamingAutomaticBudget(
            IntPtr engine,
            out long budget);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getWeightStreamingScratchMemorySize",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getWeightStreamingScratchMemorySize(
            IntPtr engine,
            out long size);
        // --- Profiles and Shapes ---
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getProfileShape",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getProfileShape(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string tensorName,
            int profileIndex,
            TrtOptProfileSelector select,
            out Dims shape);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getProfileTensorValues",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getProfileTensorValues(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string tensorName,
            int profileIndex,
            TrtOptProfileSelector select,
            out IntPtr profileTensorValues); // For int32_t**
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getProfileTensorValuesV2",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getProfileTensorValuesV2(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string tensorName,
            int profileIndex,
            TrtOptProfileSelector select,
            out IntPtr profileTensorValuesV2); // For int64_t**
                                               // --- Misc ---
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_setErrorRecorder",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_setErrorRecorder(
            IntPtr engine,
            IntPtr recorder); // TrtErrorRecorder*
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_getErrorRecorder",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_getErrorRecorder(
            IntPtr engine,
            out IntPtr recorder); // For TrtErrorRecorder**
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_hasImplicitBatchDimension",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_hasImplicitBatchDimension(
            IntPtr engine,
            out int hasImplicitBatch);
        [Pure, DllImport(dllExtern, EntryPoint = "trtCudaEngine_createEngineInspector",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtCudaEngine_createEngineInspector(
            IntPtr engine,
            out IntPtr inspector);
    }
}

