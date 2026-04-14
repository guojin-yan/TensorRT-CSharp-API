using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_setNbOutputMaps",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_setNbOutputMaps(
            IntPtr layer,
            long nbOutputMaps);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_getNbOutputMaps",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_getNbOutputMaps(
            IntPtr layer,
            out long nbOutputMaps);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_setNbGroups",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_setNbGroups(
            IntPtr layer,
            long nbGroups);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_getNbGroups",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_getNbGroups(
            IntPtr layer,
            out long nbGroups);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_setKernelWeights",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_setKernelWeights(
            IntPtr layer,
            ref TrtWeights weights);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_getKernelWeights",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_getKernelWeights(
            IntPtr layer,
            out TrtWeights weights);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_setBiasWeights",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_setBiasWeights(
            IntPtr layer,
            ref TrtWeights weights);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_getBiasWeights",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_getBiasWeights(
            IntPtr layer,
            out TrtWeights weights);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_setPrePadding",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_setPrePadding(
            IntPtr layer,
            ref Dims padding);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_getPrePadding",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_getPrePadding(
            IntPtr layer,
            out Dims padding);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_setPostPadding",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_setPostPadding(
            IntPtr layer,
            ref Dims padding);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_getPostPadding",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_getPostPadding(
            IntPtr layer,
            out Dims padding);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_setPaddingMode",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_setPaddingMode(
            IntPtr layer,
            TrtPaddingMode paddingMode);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_getPaddingMode",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_getPaddingMode(
            IntPtr layer,
            out TrtPaddingMode paddingMode);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_setKernelSizeNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_setKernelSizeNd(
            IntPtr layer,
            ref Dims kernelSize);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_getKernelSizeNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_getKernelSizeNd(
            IntPtr layer,
            out Dims kernelSize);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_setStrideNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_setStrideNd(
            IntPtr layer,
            ref Dims stride);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_getStrideNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_getStrideNd(
            IntPtr layer,
            out Dims stride);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_setPaddingNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_setPaddingNd(
            IntPtr layer,
            ref Dims padding);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_getPaddingNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_getPaddingNd(
            IntPtr layer,
            out Dims padding);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_setDilationNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_setDilationNd(
            IntPtr layer,
            ref Dims dilation);

        [Pure, DllImport(dllExtern, EntryPoint = "trtConvolutionLayer_getDilationNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtConvolutionLayer_getDilationNd(
            IntPtr layer,
            out Dims dilation);
    }
}
