using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        [Pure, DllImport(dllExtern, EntryPoint = "trtTopKLayer_setOperation",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTopKLayer_setOperation(
            IntPtr layer,
            TrtTopKOperation op);

        [Pure, DllImport(dllExtern, EntryPoint = "trtTopKLayer_getOperation",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTopKLayer_getOperation(
            IntPtr layer,
            out TrtTopKOperation op);

        [Pure, DllImport(dllExtern, EntryPoint = "trtTopKLayer_setK",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTopKLayer_setK(
            IntPtr layer,
            int k);

        [Pure, DllImport(dllExtern, EntryPoint = "trtTopKLayer_getK",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTopKLayer_getK(
            IntPtr layer,
            out int k);

        [Pure, DllImport(dllExtern, EntryPoint = "trtTopKLayer_setAxes",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTopKLayer_setAxes(
            IntPtr layer,
            uint axes);

        [Pure, DllImport(dllExtern, EntryPoint = "trtTopKLayer_getAxes",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTopKLayer_getAxes(
            IntPtr layer,
            out uint axes);
    }
}
