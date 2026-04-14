using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        [Pure, DllImport(dllExtern, EntryPoint = "trtElementWiseLayer_setOperation",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtElementWiseLayer_setOperation(
            IntPtr layer,
            TrtElementWiseOperation op);

        [Pure, DllImport(dllExtern, EntryPoint = "trtElementWiseLayer_getOperation",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtElementWiseLayer_getOperation(
            IntPtr layer,
            out TrtElementWiseOperation op);
    }
}
