using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        [Pure, DllImport(dllExtern, EntryPoint = "trtUnaryLayer_setOperation",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtUnaryLayer_setOperation(
            IntPtr layer,
            TrtUnaryOperation op);

        [Pure, DllImport(dllExtern, EntryPoint = "trtUnaryLayer_getOperation",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtUnaryLayer_getOperation(
            IntPtr layer,
            out TrtUnaryOperation op);
    }
}
