using System;
using JYPPX.CudaSharp.Internal.Handles;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static SafeCudaKernelLibraryHandle LoadKernelLibrary(byte[] code)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_kernel_library_load_data_copy_safe(
            code,
            new UIntPtr((uint)code.Length),
            out SafeCudaKernelLibraryHandle library));
        return library;
    }

    public static SafeCudaKernelLibraryHandle LoadKernelLibrary(string path)
    {
        using Utf8Interop.Utf8StringScope pathUtf8 = Utf8Interop.ToNativeString(path);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_kernel_library_load_file_safe(
            pathUtf8.Pointer,
            out SafeCudaKernelLibraryHandle library));
        return library;
    }

    public static uint GetKernelLibraryCount(SafeCudaKernelLibraryHandle library)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_kernel_library_get_count_safe(library, out uint count));
        return count;
    }

    public static NativeCudaKernelLibraryInventory GetKernelLibraryInventory(SafeCudaKernelLibraryHandle library)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_kernel_library_get_inventory_safe(
            library,
            out NativeCudaKernelLibraryInventory inventory));
        return inventory;
    }

    public static bool KernelLibraryContains(SafeCudaKernelLibraryHandle library, string name)
    {
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(name);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_kernel_library_contains_kernel_safe(
            library,
            nameUtf8.Pointer,
            out int exists));
        return exists != 0;
    }
}
