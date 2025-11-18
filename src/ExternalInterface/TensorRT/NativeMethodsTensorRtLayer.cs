using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;


namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {

        // 释放层
        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_free",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtLayer_free(IntPtr layer);

        // --- 访问器 ---

        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_getType",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtLayer_getType(
            IntPtr layer,
            out TrtLayerType type);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_getName",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtLayer_getName(
            IntPtr layer,
            out IntPtr name);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_getNbInputs",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtLayer_getNbInputs(
            IntPtr layer,
            out int nbInputs);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_getInput",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtLayer_getInput(
            IntPtr layer,
            int index,
            out IntPtr input);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_getNbOutputs",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtLayer_getNbOutputs(
            IntPtr layer,
            out int nbOutputs);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_getOutput",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtLayer_getOutput(
            IntPtr layer,
            int index,
            out IntPtr output);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_getPrecision",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtLayer_getPrecision(
            IntPtr layer,
            out TrtDataType precision);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_precisionIsSet",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtLayer_precisionIsSet(
            IntPtr layer,
            out int isSet);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_getOutputType",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtLayer_getOutputType(
            IntPtr layer,
            int index,
            out TrtDataType outputType);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_outputTypeIsSet",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtLayer_outputTypeIsSet(
            IntPtr layer,
            int index,
            out int isSet);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_getMetadata",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtLayer_getMetadata(
            IntPtr layer,
            out IntPtr metadata);

        // --- 修改器 ---

        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_setName",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtLayer_setName(
            IntPtr layer,
            string name);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_setInput",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtLayer_setInput(
            IntPtr layer,
            int index,
            IntPtr tensor);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_setPrecision",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtLayer_setPrecision(
            IntPtr layer,
            TrtDataType dataType);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_resetPrecision",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtLayer_resetPrecision(
            IntPtr layer);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_setOutputType",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtLayer_setOutputType(
            IntPtr layer,
            int index,
            TrtDataType dataType);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_resetOutputType",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtLayer_resetOutputType(
            IntPtr layer,
            int index);

        [Pure, DllImport(dllExtern, EntryPoint = "trtLayer_setMetadata",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtLayer_setMetadata(
            IntPtr layer,
            string metadata);

    }
}
