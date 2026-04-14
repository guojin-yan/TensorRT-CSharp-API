using JYPPX.TensorRtSharp.Exceptions;
using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        /// <summary>
        /// Initialize and register all the existing TensorRT plugins to the Plugin Registry with an optional namespace.
        /// </summary>
        /// <param name="logger">Logger object to print plugin registration information. Can be IntPtr.Zero.</param>
        /// <param name="libNamespace">Namespace used to register all the plugins in this library. Can be null.</param>
        /// <returns>true if initialization succeeded, false otherwise.</returns>
        [Pure, DllImport(dllExtern, EntryPoint = "trtPlugin_initLibNvInferPlugins",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: MarshalAs(UnmanagedType.U1)]
        public extern static bool trtPlugin_initLibNvInferPlugins(
            IntPtr logger,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string libNamespace);
    }
}
