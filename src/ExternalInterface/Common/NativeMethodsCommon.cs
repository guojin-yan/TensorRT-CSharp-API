using System;
using System.Runtime.InteropServices;
using System.Diagnostics.Contracts;


namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        /// <summary>
        /// Get the last error msg.
        /// </summary>
        /// <returns>The last error msg.</returns>
        [Pure, DllImport(dllExtern, EntryPoint = "GetLastErrMsg",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static IntPtr GetLastErrMsg();
    }
}
