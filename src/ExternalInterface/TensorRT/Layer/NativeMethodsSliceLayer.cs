using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        [Pure, DllImport(dllExtern, EntryPoint = "trtSliceLayer_setStart",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtSliceLayer_setStart(
            IntPtr layer,
            Dims start);

        [Pure, DllImport(dllExtern, EntryPoint = "trtSliceLayer_getStart",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtSliceLayer_getStart(
            IntPtr layer,
            out Dims start);

        [Pure, DllImport(dllExtern, EntryPoint = "trtSliceLayer_setSize",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtSliceLayer_setSize(
            IntPtr layer,
            Dims size);

        [Pure, DllImport(dllExtern, EntryPoint = "trtSliceLayer_getSize",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtSliceLayer_getSize(
            IntPtr layer,
            out Dims size);

        [Pure, DllImport(dllExtern, EntryPoint = "trtSliceLayer_setStride",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtSliceLayer_setStride(
            IntPtr layer,
            Dims stride);

        [Pure, DllImport(dllExtern, EntryPoint = "trtSliceLayer_getStride",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtSliceLayer_getStride(
            IntPtr layer,
            out Dims stride);

        [Pure, DllImport(dllExtern, EntryPoint = "trtSliceLayer_setMode",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtSliceLayer_setMode(
            IntPtr layer,
            TrtSliceMode mode);

        [Pure, DllImport(dllExtern, EntryPoint = "trtSliceLayer_getMode",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtSliceLayer_getMode(
            IntPtr layer,
            out TrtSliceMode mode);
    }
}
