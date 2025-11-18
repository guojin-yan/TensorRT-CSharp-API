using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;


namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {

        [DllImport(dllExtern, EntryPoint = "trtGpuAllocator_free",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtGpuAllocator_free(
            IntPtr allocator);

        [DllImport(dllExtern, EntryPoint = "trtGpuAllocator_allocateAsync",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtGpuAllocator_allocateAsync(
            IntPtr allocator,
            ulong size,
            ulong alignment,
            uint flags,
            IntPtr stream,
            out IntPtr memoryPtr);

        [DllImport(dllExtern, EntryPoint = "trtGpuAllocator_deallocateAsync",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtGpuAllocator_deallocateAsync(
            IntPtr allocator,
            IntPtr memory,
            IntPtr stream,
            out int successStatus);

        [DllImport(dllExtern, EntryPoint = "trtGpuAllocator_reallocate",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtGpuAllocator_reallocate(
            IntPtr allocator,
            IntPtr baseAddr,
            ulong alignment,
            ulong newSize,
            out IntPtr memoryPtr);

        [DllImport(dllExtern, EntryPoint = "trtGpuAllocator_getInterfaceInfo",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtGpuAllocator_getInterfaceInfo(
            IntPtr allocator,
            out InterfaceInfo info);

    }
}
