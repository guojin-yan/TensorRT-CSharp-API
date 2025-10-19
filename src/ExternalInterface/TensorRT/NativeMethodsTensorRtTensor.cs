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

        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_free",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static InitExceptionStatus trtTensor_free(IntPtr tensor);


        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_setName",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_setName(IntPtr tensor, [MarshalAs(StringUnmanagedTypeNotWindows)] string name);

        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_getName",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_getName(
            IntPtr tensor,
            out IntPtr name);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_setDimensions",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_setDimensions(
            IntPtr tensor,
            Dims dimensions);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_getDimensions",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_getDimensions(
            IntPtr tensor,
            out Dims dimensions);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_setType",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_setType(
            IntPtr tensor,
            TrtDataType type);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_getType",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_getType(
            IntPtr tensor,
            out TrtDataType type);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_setDynamicRange",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_setDynamicRange(
            IntPtr tensor,
            float min,
            float max,
            out int success);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_isNetworkInput",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_isNetworkInput(
            IntPtr tensor,
            out int isInput);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_isNetworkOutput",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_isNetworkOutput(
            IntPtr tensor,
            out int isOutput);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_setBroadcastAcrossBatch",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_setBroadcastAcrossBatch(
            IntPtr tensor,
            int broadcast);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_getBroadcastAcrossBatch",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_getBroadcastAcrossBatch(
            IntPtr tensor,
            out int broadcast);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_getLocation",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_getLocation(
            IntPtr tensor,
            out TrtTensorLocation location);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_setLocation",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_setLocation(
            IntPtr tensor,
            TrtTensorLocation location);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_dynamicRangeIsSet",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_dynamicRangeIsSet(
            IntPtr tensor,
            out int isSet);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_resetDynamicRange",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_resetDynamicRange(
            IntPtr tensor);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_getDynamicRangeMin",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_getDynamicRangeMin(
            IntPtr tensor,
            out float min);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_getDynamicRangeMax",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_getDynamicRangeMax(
            IntPtr tensor,
            out float max);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_getAllowedFormats",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_getAllowedFormats(
            IntPtr tensor,
            out TrtTensorFormat tensorFormat);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_isShapeTensor",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_isShapeTensor(
            IntPtr tensor,
            out int flagShapeTensor);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_isExecutionTensor",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_isExecutionTensor(
            IntPtr tensor,
            out int executionStatus);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_setDimensionName",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_setDimensionName(
            IntPtr tensor,
            int index,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string name);
        [Pure, DllImport(dllExtern, EntryPoint = "trtTensor_getDimensionName",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtTensor_getDimensionName(
            IntPtr tensor,
            int index,
            out IntPtr name);
    }
}
