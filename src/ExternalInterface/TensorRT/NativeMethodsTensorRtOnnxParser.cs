using JYPPX.TensorRtSharp.Exceptions;
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
        [Pure, DllImport(dllExtern, EntryPoint = "trtOnnxParser_createOnnxParser",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static InitExceptionStatus trtOnnxParser_createOnnxParser(IntPtr networkDefinition, out IntPtr onnxParse);



        [Pure, DllImport(dllExtern, EntryPoint = "trtONNXParser_parse",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtONNXParser_parse(IntPtr onnxParse,
            IntPtr serialized_onnx_model,
            long serialized_onnx_model_size,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string model_path,
            out int flag);


        [Pure, DllImport(dllExtern, EntryPoint = "trtONNXParser_parseFromFile",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtONNXParser_parseFromFile(IntPtr onnxParse,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string onnxModelFile,
            int verbosity,
            out int flag);
    }
}