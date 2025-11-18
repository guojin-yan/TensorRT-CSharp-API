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


        [DllImport(dllExtern, EntryPoint = "trtParser_supportsOperator",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtParser_supportsOperator(
            IntPtr parser,
            string op_name,
            out int supported);
        // --- Error Handling ---
        [DllImport(dllExtern, EntryPoint = "trtParser_getNbErrors",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtParser_getNbErrors(
            IntPtr parser,
            out int nbErrors);
        // 重要提示：IParserError** 返回指针的指针。
        // C# 通常通过返回一个 IntPtr 来返回指针的地址。
        [DllImport(dllExtern, EntryPoint = "trtParser_getError",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtParser_getError(
            IntPtr parser,
            int index,
            out IntPtr error); // 返回 nvonnxparser::IParserError* 的地址
        [DllImport(dllExtern, EntryPoint = "trtParser_clearErrors",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtParser_clearErrors(IntPtr parser);
        // --- Plugin and Parsing Flags ---
        // 重要提示：const char const** pluginLibraries 返回字符串数组的地址。
        [DllImport(dllExtern, EntryPoint = "trtParser_getUsedVCPluginLibraries",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtParser_getUsedVCPluginLibraries(
            IntPtr parser,
            out long nbPluginLibs,
            out IntPtr pluginLibraries); // 返回 const char* const* 的地址
        [DllImport(dllExtern, EntryPoint = "trtParser_setFlags",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtParser_setFlags(IntPtr parser, uint onnxParserFlags);
        [DllImport(dllExtern, EntryPoint = "trtParser_getFlags",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtParser_getFlags(IntPtr parser, out uint onnxParserFlags);
        [DllImport(dllExtern, EntryPoint = "trtParser_clearFlag",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtParser_clearFlag(
            IntPtr parser,
            TrtOnnxParserFlag onnxParserFlag);
        [DllImport(dllExtern, EntryPoint = "trtParser_setFlag",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtParser_setFlag(
            IntPtr parser,
            TrtOnnxParserFlag onnxParserFlag);
        [DllImport(dllExtern, EntryPoint = "trtParser_getFlag",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtParser_getFlag(
            IntPtr parser,
            TrtOnnxParserFlag onnxParserFlag,
            out int isSet);
        // --- Intermediate Tensor and Subgraph Querying ---
        [DllImport(dllExtern, EntryPoint = "trtParser_getLayerOutputTensor",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtParser_getLayerOutputTensor(
            IntPtr parser,
            string name,
            long i,
            out IntPtr tensor);
        [DllImport(dllExtern, EntryPoint = "trtParser_supportsModelV2",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtParser_supportsModelV2(
            IntPtr parser,
            IntPtr serializedOnnxModel,
            ulong serializedOnnxModelSize,
            string modelPath,
            out int success);
        [DllImport(dllExtern, EntryPoint = "trtParser_getNbSubgraphs",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtParser_getNbSubgraphs(
            IntPtr parser,
            out long nbSubgraphs);
        [DllImport(dllExtern, EntryPoint = "trtParser_isSubgraphSupported",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtParser_isSubgraphSupported(
            IntPtr parser,
            long index,
            out int isSupported);
        // 重要提示：int64_t** subgraphNodes 返回指针的地址。
        [DllImport(dllExtern, EntryPoint = "trtParser_getSubgraphNodes",
            CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtParser_getSubgraphNodes(
            IntPtr parser,
            long index,
            out IntPtr subgraphNodes,   // 返回 int64_t* (long*) 的地址
            out long subgraphLength);
    }
}