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
        [Pure, DllImport(dllExtern, EntryPoint = "trtOptimizationProfile_setDimensions",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtOptimizationProfile_setDimensions(
            IntPtr profile,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string inputName,
            TrtOptProfileSelector select,
            Dims dims,
            out int result);

        [Pure, DllImport(dllExtern, EntryPoint = "trtOptimizationProfile_getDimensions",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtOptimizationProfile_getDimensions(
            IntPtr profile,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string inputName,
            TrtOptProfileSelector select,
            out Dims dims);

        [Pure, DllImport(dllExtern, EntryPoint = "trtOptimizationProfile_setShapeValues",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtOptimizationProfile_setShapeValues(
            IntPtr profile,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string inputName,
            TrtOptProfileSelector select,
            int[] values,
            int nbValues,
            out int result);

        [Pure, DllImport(dllExtern, EntryPoint = "trtOptimizationProfile_getNbShapeValues",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtOptimizationProfile_getNbShapeValues(
            IntPtr profile,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string inputName,
            out int nbValues);

        [Pure, DllImport(dllExtern, EntryPoint = "trtOptimizationProfile_getShapeValues",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtOptimizationProfile_getShapeValues(
            IntPtr profile,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string inputName,
            TrtOptProfileSelector select,
            out IntPtr values);

        [Pure, DllImport(dllExtern, EntryPoint = "trtOptimizationProfile_setExtraMemoryTarget",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtOptimizationProfile_setExtraMemoryTarget(
            IntPtr profile,
            float target,
            out int result);

        [Pure, DllImport(dllExtern, EntryPoint = "trtOptimizationProfile_getExtraMemoryTarget",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtOptimizationProfile_getExtraMemoryTarget(
            IntPtr profile,
            out float target);

        [Pure, DllImport(dllExtern, EntryPoint = "trtOptimizationProfile_isValid",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtOptimizationProfile_isValid(
            IntPtr profile,
            out int result);

        [Pure, DllImport(dllExtern, EntryPoint = "trtOptimizationProfile_setShapeValuesV2",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtOptimizationProfile_setShapeValuesV2(
            IntPtr profile,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string inputName,
            TrtOptProfileSelector select,
            long[] values,
            int nbValues,
            out int result);

        [Pure, DllImport(dllExtern, EntryPoint = "trtOptimizationProfile_getShapeValuesV2",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtOptimizationProfile_getShapeValuesV2(
            IntPtr profile,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string inputName,
            TrtOptProfileSelector select,
            out IntPtr values);
    }
}
