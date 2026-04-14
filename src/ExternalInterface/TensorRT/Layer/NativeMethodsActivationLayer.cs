using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        [Pure, DllImport(dllExtern, EntryPoint = "trtActivationLayer_setActivationType",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtActivationLayer_setActivationType(
            IntPtr layer,
            TrtActivationType type);

        [Pure, DllImport(dllExtern, EntryPoint = "trtActivationLayer_getActivationType",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtActivationLayer_getActivationType(
            IntPtr layer,
            out TrtActivationType type);

        [Pure, DllImport(dllExtern, EntryPoint = "trtActivationLayer_setAlpha",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtActivationLayer_setAlpha(
            IntPtr layer,
            float alpha);

        [Pure, DllImport(dllExtern, EntryPoint = "trtActivationLayer_getAlpha",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtActivationLayer_getAlpha(
            IntPtr layer,
            out float alpha);

        [Pure, DllImport(dllExtern, EntryPoint = "trtActivationLayer_setBeta",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtActivationLayer_setBeta(
            IntPtr layer,
            float beta);

        [Pure, DllImport(dllExtern, EntryPoint = "trtActivationLayer_getBeta",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtActivationLayer_getBeta(
            IntPtr layer,
            out float beta);
    }
}
