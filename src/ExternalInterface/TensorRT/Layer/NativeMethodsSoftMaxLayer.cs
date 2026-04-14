using JYPPX.TensorRtSharp.Exceptions;
using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        [Pure, DllImport(dllExtern, EntryPoint = "trtSoftMaxLayer_setAxes",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtSoftMaxLayer_setAxes(
            IntPtr layer,
            uint axes);

        [Pure, DllImport(dllExtern, EntryPoint = "trtSoftMaxLayer_getAxes",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtSoftMaxLayer_getAxes(
            IntPtr layer,
            out uint axes);
    }
}
