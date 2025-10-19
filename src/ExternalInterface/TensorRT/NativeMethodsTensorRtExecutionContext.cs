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
    }
}
