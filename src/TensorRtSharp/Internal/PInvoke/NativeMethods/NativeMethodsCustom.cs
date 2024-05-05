using System;
using System.Runtime.InteropServices;

namespace TensorRtSharp.Internal
{
    public static partial class NativeMethods
    {
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus onnxToEngine(ref sbyte onnxFile, int memorySize);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]

        public extern static ExceptionStatus onnxToEngineDynamicShape(ref sbyte onnxFile, int memorySize, 
            ref sbyte nodeName, ref int minShapes, ref int optShapes, ref int maxShapes);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus nvinferInit(ref sbyte engineFile, out IntPtr returnNvinferPtr);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus nvinferInitDynamicShape(ref sbyte engineFile, int maxBatahSize, out IntPtr ptr);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus copyFloatHostToDeviceByName(IntPtr nvinferPtr, ref sbyte nodeName, ref float dataArray);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus copyFloatHostToDeviceByIndex(IntPtr nvinferPtr, int nodeIndex, ref float dataArray);

        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus setBindingDimensionsByName(IntPtr ptr, ref sbyte nodeName, int nbDims, ref int dims);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus setBindingDimensionsByIndex(IntPtr ptr, int nodeIndex, int nbDims, ref int dims);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus tensorRtInfer(IntPtr nvinferPtr);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus copyFloatDeviceToHostByName(IntPtr nvinferPtr, ref sbyte nodeName, ref float returnDataArray);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus copyFloatDeviceToHostByIndex(IntPtr nvinferPtr, int nodeIndex, ref float rereturnDataArrayturnData);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus nvinferDelete(IntPtr nvinferPtr);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus getBindingDimensionsByName(IntPtr nvinferPtr, ref sbyte nodeName, out int dimLength, ref int dims);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus getBindingDimensionsByIndex(IntPtr nvinferPtr, int index, out int dimLength, ref int dims);
    }
}
