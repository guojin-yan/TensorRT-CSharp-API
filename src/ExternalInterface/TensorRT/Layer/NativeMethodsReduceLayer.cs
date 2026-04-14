using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        [Pure, DllImport(dllExtern, EntryPoint = "trtReduceLayer_setOperation",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtReduceLayer_setOperation(
            IntPtr layer,
            TrtReduceOperation op);

        [Pure, DllImport(dllExtern, EntryPoint = "trtReduceLayer_getOperation",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtReduceLayer_getOperation(
            IntPtr layer,
            out TrtReduceOperation op);

        [Pure, DllImport(dllExtern, EntryPoint = "trtReduceLayer_setAxes",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtReduceLayer_setAxes(
            IntPtr layer,
            uint axes);

        [Pure, DllImport(dllExtern, EntryPoint = "trtReduceLayer_getAxes",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtReduceLayer_getAxes(
            IntPtr layer,
            out uint axes);

        [Pure, DllImport(dllExtern, EntryPoint = "trtReduceLayer_setKeepDimensions",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtReduceLayer_setKeepDimensions(
            IntPtr layer,
            bool keep);

        [Pure, DllImport(dllExtern, EntryPoint = "trtReduceLayer_getKeepDimensions",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtReduceLayer_getKeepDimensions(
            IntPtr layer,
            out bool keep);
    }
}
