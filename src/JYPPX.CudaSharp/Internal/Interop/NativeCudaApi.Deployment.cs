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

    public static CudaAtomicCapability[] GetDeviceHostAtomicCapabilities(int device, CudaAtomicOperation[] operations)
    {
        return GetDeviceAtomicCapabilities(
            operations,
            (IntPtr capabilities, IntPtr pinnedOperations, uint count) =>
                NativeMethodsCuda.jyppx_cuda_device_get_host_atomic_capabilities(capabilities, pinnedOperations, count, device));
    }

    public static CudaAtomicCapability[] GetDeviceP2PAtomicCapabilities(int sourceDevice, int destinationDevice, CudaAtomicOperation[] operations)
    {
        return GetDeviceAtomicCapabilities(
            operations,
            (IntPtr capabilities, IntPtr pinnedOperations, uint count) =>
                NativeMethodsCuda.jyppx_cuda_device_get_p2p_atomic_capabilities(capabilities, pinnedOperations, count, sourceDevice, destinationDevice));
    }

    public static int ChooseDevice(in NativeCudaDeviceSelectionRequirements requirements)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_choose_device(in requirements, out int device));
        return device;
    }

    public static void InitDevice(int device, uint deviceFlags, uint flags)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_init_device(device, deviceFlags, flags));
    }

    public static void SetValidDevices(int[] ordinals)
    {
        if (ordinals.Length == 0)
        {
            throw new ArgumentException("At least one CUDA device ordinal is required.", nameof(ordinals));
        }

        GCHandle ordinalsHandle = GCHandle.Alloc(ordinals, GCHandleType.Pinned);
        try
        {
            CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_set_valid_devices(ordinalsHandle.AddrOfPinnedObject(), (uint)ordinals.Length));
        }
        finally
        {
            ordinalsHandle.Free();
        }
    }

    private static CudaAtomicCapability[] GetDeviceAtomicCapabilities(
        CudaAtomicOperation[] operations,
        Func<IntPtr, IntPtr, uint, BridgeStatusCode> query)
    {
        if (operations.Length == 0)
        {
            throw new ArgumentException("At least one CUDA atomic operation is required.", nameof(operations));
        }

        int[] operationValues = new int[operations.Length];
        uint[] capabilityValues = new uint[operations.Length];
        for (int index = 0; index < operations.Length; index++)
        {
            operationValues[index] = (int)operations[index];
        }

        GCHandle operationsHandle = GCHandle.Alloc(operationValues, GCHandleType.Pinned);
        GCHandle capabilitiesHandle = GCHandle.Alloc(capabilityValues, GCHandleType.Pinned);
        try
        {
            CudaNativeStatus.ThrowIfFailed(query(capabilitiesHandle.AddrOfPinnedObject(), operationsHandle.AddrOfPinnedObject(), (uint)operations.Length));
        }
        finally
        {
            capabilitiesHandle.Free();
            operationsHandle.Free();
        }

        CudaAtomicCapability[] capabilities = new CudaAtomicCapability[capabilityValues.Length];
        for (int index = 0; index < capabilityValues.Length; index++)
        {
            capabilities[index] = (CudaAtomicCapability)capabilityValues[index];
        }

        return capabilities;
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
