using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuild_createInferBuilder",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static InitExceptionStatus trtBuild_createInferBuilder(out IntPtr build);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuild_free",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtBuild_free(IntPtr build);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuild_platformHasFastFp16",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuild_platformHasFastFp16(IntPtr build, out int flag);


        [Pure, DllImport(dllExtern, EntryPoint = "trtBuild_platformHasFastInt8",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuild_platformHasFastInt8(IntPtr build, out int flag);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuild_getMaxDLABatchSize",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuild_getMaxDLABatchSize(IntPtr build, out int maxBatchSize);


        [Pure, DllImport(dllExtern, EntryPoint = "trtBuild_getNbDLACores",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuild_getNbDLACores(IntPtr build, out int coreCount);


        [Pure, DllImport(dllExtern, EntryPoint = "trtBuild_setGpuAllocator",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuild_setGpuAllocator(IntPtr build, IntPtr allocator);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuild_createBuilderConfig",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuild_createBuilderConfig(IntPtr build, out IntPtr builderConfig);


        [Pure, DllImport(dllExtern, EntryPoint = "trtBuild_createNetworkV2",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuild_createNetworkV2(IntPtr build,
            TrtNetworkDefinitionCreationFlag flags, out IntPtr networkDefinition);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuild_createOptimizationProfile",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuild_createOptimizationProfile(IntPtr build,
            out IntPtr optimizationProfile);


        [Pure, DllImport(dllExtern, EntryPoint = "trtBuild_reset",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuild_reset(IntPtr build);


        [Pure, DllImport(dllExtern, EntryPoint = "trtBuild_buildSerializedNetwork",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuild_buildSerializedNetwork(IntPtr build,
            IntPtr network, IntPtr config, out IntPtr hostMemory);


        [Pure, DllImport(dllExtern, EntryPoint = "trtBuild_buildSerializedNetworkToStream",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuild_buildSerializedNetworkToStream(IntPtr build,
            IntPtr network, IntPtr config, IntPtr writer, out int flag);


        [Pure, DllImport(dllExtern, EntryPoint = "trtBuild_buildEngineWithConfig",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuild_buildEngineWithConfig(IntPtr build,
            IntPtr network, IntPtr config, out IntPtr cudaEngine);


        [Pure, DllImport(dllExtern, EntryPoint = "trtBuild_isNetworkSupported",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuild_isNetworkSupported(IntPtr build,
            IntPtr network, IntPtr config, out int flag);


    }
}
