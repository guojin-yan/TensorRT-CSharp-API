using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        [Pure, DllImport(dllExtern, EntryPoint = "trtShuffleLayer_setFirstTranspose",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtShuffleLayer_setFirstTranspose(
            IntPtr layer,
            Dims permutation);

        [Pure, DllImport(dllExtern, EntryPoint = "trtShuffleLayer_getFirstTranspose",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtShuffleLayer_getFirstTranspose(
            IntPtr layer,
            out Dims permutation);

        [Pure, DllImport(dllExtern, EntryPoint = "trtShuffleLayer_setReshapeDimensions",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtShuffleLayer_setReshapeDimensions(
            IntPtr layer,
            Dims dimensions);

        [Pure, DllImport(dllExtern, EntryPoint = "trtShuffleLayer_getReshapeDimensions",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtShuffleLayer_getReshapeDimensions(
            IntPtr layer,
            out Dims dimensions);

        [Pure, DllImport(dllExtern, EntryPoint = "trtShuffleLayer_setSecondTranspose",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtShuffleLayer_setSecondTranspose(
            IntPtr layer,
            Dims permutation);

        [Pure, DllImport(dllExtern, EntryPoint = "trtShuffleLayer_getSecondTranspose",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtShuffleLayer_getSecondTranspose(
            IntPtr layer,
            out Dims permutation);

        [Pure, DllImport(dllExtern, EntryPoint = "trtShuffleLayer_setZeroIsPlaceholder",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtShuffleLayer_setZeroIsPlaceholder(
            IntPtr layer,
            int zeroIsPlaceholder);

        [Pure, DllImport(dllExtern, EntryPoint = "trtShuffleLayer_getZeroIsPlaceholder",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtShuffleLayer_getZeroIsPlaceholder(
            IntPtr layer,
            out int zeroIsPlaceholder);
    }
}
