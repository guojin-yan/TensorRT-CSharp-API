using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;


namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntimeConfig_free",
        CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static InitExceptionStatus trtRuntimeConfig_free(
        IntPtr config);
        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntimeConfig_setExecutionContextAllocationStrategy", // Note: typo in original C function name 'Confi'
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntimeConfig_setExecutionContextAllocationStrategy(
            IntPtr config,
            TrtExecutionContextAllocationStrategy strategy);
        [Pure, DllImport(dllExtern, EntryPoint = "trtRuntimeConfig_getExecutionContextAllocationStrategy",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtRuntimeConfig_getExecutionContextAllocationStrategy(
            IntPtr config,
            out TrtExecutionContextAllocationStrategy strategy);
    }
}
