using System;
using System.Runtime.InteropServices;
using System.Text;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    private const ulong MaximumDriverCopiedOutputSize = 64UL * 1024UL;
    private delegate BridgeStatusCode DriverUtf8BufferGetter(byte[] outputBuffer, UIntPtr outputBufferSize, out UIntPtr requiredSize);

    public static NativeCudaDriverCapabilityInfo QueryDriverCapability()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_driver_query_capability_safe(out NativeCudaDriverCapabilityInfo info));
        return info;
    }

    public static string GetDriverLoadedLibraryName()
    {
        return ReadDriverUtf8Buffer(NativeMethodsCuda.jyppx_cuda_driver_get_loaded_library_name_safe);
    }

    public static string GetDriverDependencyDiagnostic()
    {
        return ReadDriverUtf8Buffer(NativeMethodsCuda.jyppx_cuda_driver_get_dependency_diagnostic_safe);
    }

    public static SafeCudaDriverModuleHandle LoadDriverModule(byte[] code, int deviceOrdinal)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_driver_module_load_data_copy_safe(
            code,
            new UIntPtr(checked((uint)code.Length)),
            deviceOrdinal,
            out SafeCudaDriverModuleHandle module));
        return module;
    }

    public static SafeCudaDriverKernelLaunchHandle LaunchDriverModule(
        SafeCudaDriverModuleHandle module,
        string kernelName,
        CudaKernelLaunchConfiguration configuration,
        NativeCudaKernelArgumentDescriptor[] arguments,
        byte[] scalarData,
        SafeCudaStreamHandle stream)
    {
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(kernelName);
        GCHandle argumentPin = default;
        bool argumentPinAllocated = false;
        try
        {
            IntPtr argumentPointer = IntPtr.Zero;
            if (arguments.Length != 0)
            {
                argumentPin = GCHandle.Alloc(arguments, GCHandleType.Pinned);
                argumentPinAllocated = true;
                argumentPointer = argumentPin.AddrOfPinnedObject();
            }

            CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_driver_module_launch_typed_safe(
                module,
                nameUtf8.Pointer,
                configuration.GridDimensions.ToNative(),
                configuration.BlockDimensions.ToNative(),
                argumentPointer,
                new UIntPtr(checked((uint)arguments.Length)),
                scalarData,
                new UIntPtr(checked((uint)scalarData.Length)),
                new UIntPtr(checked((uint)configuration.DynamicSharedMemoryBytes)),
                stream,
                out SafeCudaDriverKernelLaunchHandle launch));
            return launch;
        }
        finally
        {
            if (argumentPinAllocated)
            {
                argumentPin.Free();
            }
        }
    }

    public static bool QueryDriverKernelLaunch(SafeCudaDriverKernelLaunchHandle launch)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_driver_kernel_launch_query_safe(launch, out int completed));
        return completed != 0;
    }

    public static void SynchronizeDriverKernelLaunch(SafeCudaDriverKernelLaunchHandle launch)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_driver_kernel_launch_synchronize_safe(launch));
    }

    private static string ReadDriverUtf8Buffer(DriverUtf8BufferGetter getter)
    {
        CudaNativeStatus.ThrowIfFailed(getter(Array.Empty<byte>(), UIntPtr.Zero, out UIntPtr requiredSize));
        ulong size = requiredSize.ToUInt64();
        if (size == 0 || size > MaximumDriverCopiedOutputSize || size > int.MaxValue)
        {
            throw new InvalidOperationException($"CUDA Driver diagnostic reported an invalid copied size: {size}.");
        }

        byte[] buffer = new byte[checked((int)size)];
        CudaNativeStatus.ThrowIfFailed(getter(buffer, new UIntPtr(checked((uint)buffer.Length)), out UIntPtr copiedSize));
        if (copiedSize.ToUInt64() != (ulong)buffer.Length)
        {
            throw new InvalidOperationException("CUDA Driver diagnostic size changed between query and copy.");
        }

        int terminator = Array.IndexOf(buffer, (byte)0);
        int length = terminator >= 0 ? terminator : buffer.Length;
        return length == 0 ? string.Empty : Encoding.UTF8.GetString(buffer, 0, length);
    }
}
