using JYPPX.TensorRtSharp.Exceptions;
using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;


namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        /// <summary>
        /// Get the last error msg.
        /// </summary>
        /// <returns>The last error msg.</returns>
        [Pure, DllImport(dllExtern, EntryPoint = "InitNvinfer",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static InitExceptionStatus InitNvinfer();
    }
}
