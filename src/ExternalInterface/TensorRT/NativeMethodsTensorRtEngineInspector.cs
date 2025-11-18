using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        [DllImport(dllExtern, EntryPoint = "trtEngineInspector_free",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtEngineInspector_free(IntPtr inspector);

        [DllImport(dllExtern, EntryPoint = "trtEngineInspector_setExecutionContext",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtEngineInspector_setExecutionContext(
            IntPtr inspector,
            IntPtr context,
            out int success);

        [DllImport(dllExtern, EntryPoint = "trtEngineInspector_getExecutionContext",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtEngineInspector_getExecutionContext(
            IntPtr inspector,
            out IntPtr context);

        [DllImport(dllExtern, EntryPoint = "trtEngineInspector_getLayerInformationByLayerIndex",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtEngineInspector_getLayerInformationByLayerIndex(
            IntPtr inspector,
            int layerIndex,
            TrtLayerInformationFormat format,
            out IntPtr info);

        [DllImport(dllExtern, EntryPoint = "trtEngineInspector_getEngineInformation",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtEngineInspector_getEngineInformation(
            IntPtr inspector,
            TrtLayerInformationFormat format,
            out IntPtr info);

    }
}
