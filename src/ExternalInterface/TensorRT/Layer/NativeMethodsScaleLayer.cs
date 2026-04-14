using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        [Pure, DllImport(dllExtern, EntryPoint = "trtScaleLayer_setMode",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtScaleLayer_setMode(
            IntPtr layer,
            TrtScaleMode mode);

        [Pure, DllImport(dllExtern, EntryPoint = "trtScaleLayer_getMode",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtScaleLayer_getMode(
            IntPtr layer,
            out TrtScaleMode mode);

        [Pure, DllImport(dllExtern, EntryPoint = "trtScaleLayer_setShift",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtScaleLayer_setShift(
            IntPtr layer,
            ref TrtWeights weights);

        [Pure, DllImport(dllExtern, EntryPoint = "trtScaleLayer_getShift",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtScaleLayer_getShift(
            IntPtr layer,
            out TrtWeights weights);

        [Pure, DllImport(dllExtern, EntryPoint = "trtScaleLayer_setScale",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtScaleLayer_setScale(
            IntPtr layer,
            ref TrtWeights weights);

        [Pure, DllImport(dllExtern, EntryPoint = "trtScaleLayer_getScale",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtScaleLayer_getScale(
            IntPtr layer,
            out TrtWeights weights);

        [Pure, DllImport(dllExtern, EntryPoint = "trtScaleLayer_setPower",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtScaleLayer_setPower(
            IntPtr layer,
            ref TrtWeights weights);

        [Pure, DllImport(dllExtern, EntryPoint = "trtScaleLayer_getPower",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtScaleLayer_getPower(
            IntPtr layer,
            out TrtWeights weights);

        [Pure, DllImport(dllExtern, EntryPoint = "trtScaleLayer_setChannelAxis",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtScaleLayer_setChannelAxis(
            IntPtr layer,
            int axis);

        [Pure, DllImport(dllExtern, EntryPoint = "trtScaleLayer_getChannelAxis",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtScaleLayer_getChannelAxis(
            IntPtr layer,
            out int axis);
    }
}
