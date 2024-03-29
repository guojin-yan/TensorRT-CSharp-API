using System.Runtime.InteropServices;

namespace TensorRtSharp
{
    public class NativeMethods
    {
        private const string tensorrt_dll_path = @"E:\GitSpace\TensorRT-CSharp-API\x64\Release\TensorRtExtern.dll";

        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static void onnx_to_engine(ref sbyte onnx_file_path);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr nvinfer_init(ref sbyte engine_filename, out IntPtr return_nvinfer_ptr);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr copy_float_host_to_device_byname(IntPtr nvinfer_ptr, ref sbyte node_name, ref float data_array);

        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr infer(IntPtr nvinfer_ptr);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static void copy_float_device_to_host_byname(IntPtr nvinfer_ptr, ref sbyte node_name_wchar, out float return_data);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static void nvinfer_delete(IntPtr nvinfer_ptr);
    }
}
