using System;
using System.Runtime.InteropServices;
using System.Text;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.Shared.Interop;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    private delegate BridgeStatusCode CudaUtf8BufferGetter(byte[] outputBuffer, UIntPtr outputBufferSize, out UIntPtr requiredSize);

    public static int GetLastErrorCode()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_get_last_error(out int errorCode));
        return errorCode;
    }

    public static int PeekAtLastErrorCode()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_peek_at_last_error(out int errorCode));
        return errorCode;
    }

    public static string GetErrorName(int errorCode)
    {
        return ReadCudaUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsCuda.jyppx_cuda_get_error_name(errorCode, buffer, size, out required),
            "CUDA error name is too large for the managed buffer.");
    }

    public static string GetErrorString(int errorCode)
    {
        return ReadCudaUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsCuda.jyppx_cuda_get_error_string(errorCode, buffer, size, out required),
            "CUDA error string is too large for the managed buffer.");
    }

    public static string GetDevicePciBusId(int device)
    {
        return ReadCudaUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsCuda.jyppx_cuda_device_get_pci_bus_id(device, buffer, size, out required),
            "CUDA PCI bus id is too large for the managed buffer.");
    }

    public static int GetDeviceByPciBusId(string pciBusId)
    {
        if (string.IsNullOrWhiteSpace(pciBusId))
        {
            throw new ArgumentException("PCI bus id must not be null or empty.", nameof(pciBusId));
        }

        byte[] utf8 = Encoding.UTF8.GetBytes(pciBusId + "\0");
        GCHandle handle = GCHandle.Alloc(utf8, GCHandleType.Pinned);
        try
        {
            CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_get_by_pci_bus_id(handle.AddrOfPinnedObject(), out int device));
            return device;
        }
        finally
        {
            handle.Free();
        }
    }

    public static SafeCudaStreamHandle CreateStream(CudaStreamCreationFlags flags)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_create((uint)flags, out SafeCudaStreamHandle handle));
        return handle;
    }

    public static CudaStreamCreationFlags GetStreamFlags(SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_get_flags(stream, out uint flags));
        return (CudaStreamCreationFlags)flags;
    }

    public static SafeCudaEventHandle CreateEvent(CudaEventCreationFlags flags)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_event_create((uint)flags, out SafeCudaEventHandle handle));
        return handle;
    }

    public static CudaEventCreationFlags GetEventFlags(SafeCudaEventHandle eventHandle)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_event_get_flags(eventHandle, out uint flags));
        return (CudaEventCreationFlags)flags;
    }

    public static SafeCudaPinnedMemoryHandle AllocatePinnedMemory(int sizeInBytes, CudaPinnedMemoryAllocationFlags flags)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pinned_memory_alloc_with_flags((UIntPtr)sizeInBytes, (uint)flags, out SafeCudaPinnedMemoryHandle handle));
        return handle;
    }

    public static SafeCudaPinnedMemoryHandle RegisterPinnedMemory(IntPtr pointer, int sizeInBytes, CudaHostRegistrationFlags flags)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pinned_memory_register(pointer, (UIntPtr)sizeInBytes, (uint)flags, out SafeCudaPinnedMemoryHandle handle));
        return handle;
    }

    public static CudaPinnedMemoryAllocationFlags GetPinnedMemoryFlags(SafeCudaPinnedMemoryHandle memory)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pinned_memory_get_flags(memory, out uint flags));
        return (CudaPinnedMemoryAllocationFlags)flags;
    }

    private static string ReadCudaUtf8Buffer(CudaUtf8BufferGetter getter, string tooLargeMessage)
    {
        BridgeStatusCode status = getter(Array.Empty<byte>(), UIntPtr.Zero, out UIntPtr requiredSize);
        CudaNativeStatus.ThrowIfFailed(status);
        ulong required = requiredSize.ToUInt64();
        if (required == 0)
        {
            return string.Empty;
        }

        if (required > int.MaxValue)
        {
            throw new CudaException(BridgeStatusCode.BufferTooSmall, BridgeErrorCategory.Cuda, tooLargeMessage);
        }

        byte[] buffer = new byte[checked((int)required)];
        status = getter(buffer, requiredSize, out _);
        CudaNativeStatus.ThrowIfFailed(status);
        int terminator = Array.IndexOf(buffer, (byte)0);
        int length = terminator >= 0 ? terminator : buffer.Length;
        return length == 0 ? string.Empty : Encoding.UTF8.GetString(buffer, 0, length);
    }

}
