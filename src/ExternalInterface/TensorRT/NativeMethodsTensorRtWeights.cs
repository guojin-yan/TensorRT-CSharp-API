using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        [Pure, DllImport(dllExtern, EntryPoint = "trtWeight_free",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtWeight_free(
        IntPtr weights);
        [Pure, DllImport(dllExtern, EntryPoint = "trtWeight_getDataType",
                CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtDataType trtWeight_getDataType(
                IntPtr weights);
        [Pure, DllImport(dllExtern, EntryPoint = "trtWeight_getCount",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static long trtWeight_getCount(
            IntPtr weights);
        [Pure, DllImport(dllExtern, EntryPoint = "trtWeight_getValues",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static IntPtr trtWeight_getValues(
            IntPtr weights);
    }
}
