using System.Runtime.InteropServices;

namespace TensorRtSharp.Internal
{
    public static partial class NativeMethods
    {
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus onnxToEngine(ref sbyte onnxFile);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus nvinferInit(ref sbyte engineFile, out IntPtr returnNvinferPtr);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus copyFloatHostToDeviceByName(IntPtr nvinferPtr, ref sbyte nodeName, ref float dataArray);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus copyFloatHostToDeviceByIndex(IntPtr nvinferPtr, int nodeIndex, ref float dataArray);
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
