using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        [Pure, DllImport(dllExtern, EntryPoint = "trtPoolingLayer_setPoolingType",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPoolingLayer_setPoolingType(
            IntPtr layer,
            TrtPoolingType poolingType);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPoolingLayer_getPoolingType",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPoolingLayer_getPoolingType(
            IntPtr layer,
            out TrtPoolingType poolingType);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPoolingLayer_setWindowSizeNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPoolingLayer_setWindowSizeNd(
            IntPtr layer,
            Dims windowSize);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPoolingLayer_getWindowSizeNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPoolingLayer_getWindowSizeNd(
            IntPtr layer,
            out Dims windowSize);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPoolingLayer_setStrideNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPoolingLayer_setStrideNd(
            IntPtr layer,
            Dims stride);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPoolingLayer_getStrideNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPoolingLayer_getStrideNd(
            IntPtr layer,
            out Dims stride);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPoolingLayer_setPaddingNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPoolingLayer_setPaddingNd(
            IntPtr layer,
            Dims padding);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPoolingLayer_getPaddingNd",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPoolingLayer_getPaddingNd(
            IntPtr layer,
            out Dims padding);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPoolingLayer_setPrePadding",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPoolingLayer_setPrePadding(
            IntPtr layer,
            Dims padding);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPoolingLayer_getPrePadding",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPoolingLayer_getPrePadding(
            IntPtr layer,
            out Dims padding);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPoolingLayer_setPostPadding",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPoolingLayer_setPostPadding(
            IntPtr layer,
            Dims padding);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPoolingLayer_getPostPadding",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPoolingLayer_getPostPadding(
            IntPtr layer,
            out Dims padding);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPoolingLayer_setPaddingMode",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPoolingLayer_setPaddingMode(
            IntPtr layer,
            TrtPaddingMode paddingMode);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPoolingLayer_getPaddingMode",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPoolingLayer_getPaddingMode(
            IntPtr layer,
            out TrtPaddingMode paddingMode);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPoolingLayer_setBlendFactor",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPoolingLayer_setBlendFactor(
            IntPtr layer,
            float blendFactor);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPoolingLayer_getBlendFactor",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPoolingLayer_getBlendFactor(
            IntPtr layer,
            out float blendFactor);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPoolingLayer_setAverageCountExcludesPadding",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPoolingLayer_setAverageCountExcludesPadding(
            IntPtr layer,
            int averageCountExcludesPadding);

        [Pure, DllImport(dllExtern, EntryPoint = "trtPoolingLayer_getAverageCountExcludesPadding",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtPoolingLayer_getAverageCountExcludesPadding(
            IntPtr layer,
            out int averageCountExcludesPadding);
    }
}
