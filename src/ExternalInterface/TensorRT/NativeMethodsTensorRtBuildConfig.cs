using JYPPX.TensorRtSharp.Nvinfer;
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
        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_free",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtBuilderConfig_free(IntPtr builderConfig);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_setAvgTimingIterations",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuildertrtBuilderConfig_setAvgTimingIterationsConfig(
            IntPtr builderConfig,
            int avgTiming);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_getAvgTimingIteration",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_getAvgTimingIteration(
            IntPtr builderConfig,
            out int avgTiming);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_setEngineCapability",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_setEngineCapability(
            IntPtr builderConfig,
            TrtEngineCapability capability);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_getEngineCapability",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_getEngineCapability(
            IntPtr builderConfig,
            out TrtEngineCapability capability);


        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_setFlags",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_setFlags(
            IntPtr builderConfig,
            uint builderFlags);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_getFlags",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_getFlags(
            IntPtr builderConfig,
            out uint builderFlags);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_clearFlag",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_clearFlag(
            IntPtr builderConfig,
            TrtBuilderFlag builderFlags);


        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_setFlag",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_setFlag(
            IntPtr builderConfig,
            TrtBuilderFlag builderFlag);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_getFlag",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_getFlag(
            IntPtr builderConfig,
            out uint builderFlag);


        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_setLayerDeviceType",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_setLayerDeviceType(
            IntPtr builderConfig,
            IntPtr layer,
            TrtDeviceType deviceType);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_getLayerDeviceType",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_getLayerDeviceType(
            IntPtr builderConfig,
            IntPtr layer,
            out TrtDeviceType deviceType);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_isDeviceTypeSet",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_isDeviceTypeSet(
            IntPtr builderConfig,
            IntPtr layer,
            out int outState);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_resetLayerDeviceType",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_resetLayerDeviceType(
            IntPtr builderConfig,
            IntPtr layer);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_canRunOnDLA",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_canRunOnDLA(
            IntPtr builderConfig,
            IntPtr layer,
            out int outState);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_setDLACore",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_setDLACore(
            IntPtr builderConfig,
            int dlaCore);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_getDLACore",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_getDLACore(
            IntPtr builderConfig,
            out int dlaCore);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_setDefaultDeviceType",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_setDefaultDeviceType(
            IntPtr builderConfig,
            TrtDeviceType deviceType);


        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_getDefaultDeviceType",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_getDefaultDeviceType(
            IntPtr builderConfig,
            out TrtDeviceType deviceType);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_reset",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_reset(
            IntPtr builderConfig);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_setProfileStream",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_setProfileStream(
            IntPtr builderConfig,
            IntPtr stream);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_getProfileStream",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_getProfileStream(
            IntPtr builderConfig,
            out IntPtr stream);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_addOptimizationProfile",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_addOptimizationProfile(
            IntPtr builderConfig,
            IntPtr profile,
            out int outIndex);


        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_getNbOptimizationProfiles",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_getNbOptimizationProfiles(
            IntPtr builderConfig,
            out int outCount);

        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_setProfilingVerbosity",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_setProfilingVerbosity(
            IntPtr builderConfig,
            TrtProfilingVerbosity verbosity);
        [Pure, DllImport(dllExtern, EntryPoint = "trtBuilderConfig_getProfilingVerbosity",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtBuilderConfig_getProfilingVerbosity(
            IntPtr builderConfig,
            out TrtProfilingVerbosity verbosity);


    }

}
