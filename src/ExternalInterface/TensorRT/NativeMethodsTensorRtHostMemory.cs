using JYPPX.TensorRtSharp.Exceptions;
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

        [Pure, DllImport(dllExtern, EntryPoint = "trtHostMemory_free",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtHostMemory_free(IntPtr hostMemory);


        [Pure, DllImport(dllExtern, EntryPoint = "trtHostMemory_data",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtHostMemory_data(IntPtr hostMemory, out IntPtr data);


        [Pure, DllImport(dllExtern, EntryPoint = "trtHostMemory_size",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtHostMemory_size(IntPtr hostMemory, out long size);

        [Pure, DllImport(dllExtern, EntryPoint = "trtHostMemory_type",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtHostMemory_type(IntPtr hostMemory, out TrtDataType dataType);

    }
}