using System;
using JYPPX.CudaSharp.Internal.Handles;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static SafeCudaPitchedMemoryHandle AllocatePitchedMemory(int widthInBytes, int height)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_alloc((UIntPtr)widthInBytes, (UIntPtr)height, out SafeCudaPitchedMemoryHandle handle));
        return handle;
    }

    public static NativeCudaPitchedMemoryInfo GetPitchedMemoryInfo(SafeCudaPitchedMemoryHandle memory)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_get_info(memory, out NativeCudaPitchedMemoryInfo info));
        return info;
    }

    public static void CopyPitchedFromHost2D(SafeCudaPitchedMemoryHandle memory, byte[] source, int sourcePitch, int widthInBytes, int height)
    {
        using PinnedByteBufferScope scope = Pin2D(source, sourcePitch, widthInBytes, height, new PinnedByteBufferDescriptor(nameof(CopyPitchedFromHost2D), nameof(source), "jyppx_cuda_pitched_memory_copy_from_host_2d", "host-to-device-2d"));
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_copy_from_host_2d(memory, scope.Pointer, (UIntPtr)sourcePitch, (UIntPtr)widthInBytes, (UIntPtr)height));
    }

    public static void CopyPitchedToHost2D(SafeCudaPitchedMemoryHandle memory, byte[] destination, int destinationPitch, int widthInBytes, int height)
    {
        using PinnedByteBufferScope scope = Pin2D(destination, destinationPitch, widthInBytes, height, new PinnedByteBufferDescriptor(nameof(CopyPitchedToHost2D), nameof(destination), "jyppx_cuda_pitched_memory_copy_to_host_2d", "device-to-host-2d"));
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_copy_to_host_2d(memory, scope.Pointer, (UIntPtr)destinationPitch, (UIntPtr)widthInBytes, (UIntPtr)height));
    }

    public static void CopyPitchedDeviceToDevice2D(SafeCudaPitchedMemoryHandle destination, SafeCudaPitchedMemoryHandle source, int widthInBytes, int height)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_copy_device_to_device_2d(destination, source, (UIntPtr)widthInBytes, (UIntPtr)height));
    }

    public static void CopyPitchedFromHost2DAsync(SafeCudaPitchedMemoryHandle memory, byte[] source, int sourcePitch, int widthInBytes, int height, SafeCudaStreamHandle stream)
    {
        using PinnedByteBufferScope scope = Pin2D(source, sourcePitch, widthInBytes, height, new PinnedByteBufferDescriptor(nameof(CopyPitchedFromHost2DAsync), nameof(source), "jyppx_cuda_pitched_memory_copy_from_host_2d_async", "host-to-device-2d-async"));
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_copy_from_host_2d_async(memory, scope.Pointer, (UIntPtr)sourcePitch, (UIntPtr)widthInBytes, (UIntPtr)height, stream));
    }

    public static void CopyPitchedFromPinnedHost2DAsync(SafeCudaPitchedMemoryHandle memory, SafeCudaPinnedMemoryHandle source, int sourcePitch, int widthInBytes, int height, SafeCudaStreamHandle stream)
    {
        Validate2DRegion(sourcePitch, widthInBytes, height, nameof(sourcePitch));
        IntPtr pointer = GetPinnedMemoryPointer(source);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_copy_from_host_2d_async(memory, pointer, (UIntPtr)sourcePitch, (UIntPtr)widthInBytes, (UIntPtr)height, stream));
    }

    public static void CopyPitchedToHost2DAsync(SafeCudaPitchedMemoryHandle memory, byte[] destination, int destinationPitch, int widthInBytes, int height, SafeCudaStreamHandle stream)
    {
        using PinnedByteBufferScope scope = Pin2D(destination, destinationPitch, widthInBytes, height, new PinnedByteBufferDescriptor(nameof(CopyPitchedToHost2DAsync), nameof(destination), "jyppx_cuda_pitched_memory_copy_to_host_2d_async", "device-to-host-2d-async"));
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_copy_to_host_2d_async(memory, scope.Pointer, (UIntPtr)destinationPitch, (UIntPtr)widthInBytes, (UIntPtr)height, stream));
    }

    public static void CopyPitchedToPinnedHost2DAsync(SafeCudaPitchedMemoryHandle memory, SafeCudaPinnedMemoryHandle destination, int destinationPitch, int widthInBytes, int height, SafeCudaStreamHandle stream)
    {
        Validate2DRegion(destinationPitch, widthInBytes, height, nameof(destinationPitch));
        IntPtr pointer = GetPinnedMemoryPointer(destination);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_copy_to_host_2d_async(memory, pointer, (UIntPtr)destinationPitch, (UIntPtr)widthInBytes, (UIntPtr)height, stream));
    }

    public static void CopyPitchedDeviceToDevice2DAsync(SafeCudaPitchedMemoryHandle destination, SafeCudaPitchedMemoryHandle source, int widthInBytes, int height, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_copy_device_to_device_2d_async(destination, source, (UIntPtr)widthInBytes, (UIntPtr)height, stream));
    }

    public static void CopyPitchedFromHost3D(SafeCudaPitchedMemoryHandle memory, byte[] source, int sourcePitch, int widthInBytes, int height, int depth)
    {
        using PinnedByteBufferScope scope = Pin3D(source, sourcePitch, widthInBytes, height, depth, new PinnedByteBufferDescriptor(nameof(CopyPitchedFromHost3D), nameof(source), "jyppx_cuda_pitched_memory_copy_from_host_3d", "host-to-device-3d"));
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_copy_from_host_3d(memory, scope.Pointer, (UIntPtr)sourcePitch, (UIntPtr)widthInBytes, (UIntPtr)height, (UIntPtr)depth));
    }

    public static void CopyPitchedToHost3D(SafeCudaPitchedMemoryHandle memory, byte[] destination, int destinationPitch, int widthInBytes, int height, int depth)
    {
        using PinnedByteBufferScope scope = Pin3D(destination, destinationPitch, widthInBytes, height, depth, new PinnedByteBufferDescriptor(nameof(CopyPitchedToHost3D), nameof(destination), "jyppx_cuda_pitched_memory_copy_to_host_3d", "device-to-host-3d"));
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_copy_to_host_3d(memory, scope.Pointer, (UIntPtr)destinationPitch, (UIntPtr)widthInBytes, (UIntPtr)height, (UIntPtr)depth));
    }

    public static void CopyPitchedDeviceToDevice3D(SafeCudaPitchedMemoryHandle destination, SafeCudaPitchedMemoryHandle source, int widthInBytes, int height, int depth)
    {
        Validate3DRegion(widthInBytes, height, depth);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_copy_device_to_device_3d(destination, source, (UIntPtr)widthInBytes, (UIntPtr)height, (UIntPtr)depth));
    }

    public static void CopyPitchedFromHost3DAsync(SafeCudaPitchedMemoryHandle memory, byte[] source, int sourcePitch, int widthInBytes, int height, int depth, SafeCudaStreamHandle stream)
    {
        using PinnedByteBufferScope scope = Pin3D(source, sourcePitch, widthInBytes, height, depth, new PinnedByteBufferDescriptor(nameof(CopyPitchedFromHost3DAsync), nameof(source), "jyppx_cuda_pitched_memory_copy_from_host_3d_async", "host-to-device-3d-async"));
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_copy_from_host_3d_async(memory, scope.Pointer, (UIntPtr)sourcePitch, (UIntPtr)widthInBytes, (UIntPtr)height, (UIntPtr)depth, stream));
    }

    public static void CopyPitchedFromPinnedHost3DAsync(SafeCudaPitchedMemoryHandle memory, SafeCudaPinnedMemoryHandle source, int sourcePitch, int widthInBytes, int height, int depth, SafeCudaStreamHandle stream)
    {
        Validate3DRegion(sourcePitch, widthInBytes, height, depth, nameof(sourcePitch));
        IntPtr pointer = GetPinnedMemoryPointer(source);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_copy_from_host_3d_async(memory, pointer, (UIntPtr)sourcePitch, (UIntPtr)widthInBytes, (UIntPtr)height, (UIntPtr)depth, stream));
    }

    public static void CopyPitchedToHost3DAsync(SafeCudaPitchedMemoryHandle memory, byte[] destination, int destinationPitch, int widthInBytes, int height, int depth, SafeCudaStreamHandle stream)
    {
        using PinnedByteBufferScope scope = Pin3D(destination, destinationPitch, widthInBytes, height, depth, new PinnedByteBufferDescriptor(nameof(CopyPitchedToHost3DAsync), nameof(destination), "jyppx_cuda_pitched_memory_copy_to_host_3d_async", "device-to-host-3d-async"));
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_copy_to_host_3d_async(memory, scope.Pointer, (UIntPtr)destinationPitch, (UIntPtr)widthInBytes, (UIntPtr)height, (UIntPtr)depth, stream));
    }

    public static void CopyPitchedToPinnedHost3DAsync(SafeCudaPitchedMemoryHandle memory, SafeCudaPinnedMemoryHandle destination, int destinationPitch, int widthInBytes, int height, int depth, SafeCudaStreamHandle stream)
    {
        Validate3DRegion(destinationPitch, widthInBytes, height, depth, nameof(destinationPitch));
        IntPtr pointer = GetPinnedMemoryPointer(destination);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_copy_to_host_3d_async(memory, pointer, (UIntPtr)destinationPitch, (UIntPtr)widthInBytes, (UIntPtr)height, (UIntPtr)depth, stream));
    }

    public static void CopyPitchedDeviceToDevice3DAsync(SafeCudaPitchedMemoryHandle destination, SafeCudaPitchedMemoryHandle source, int widthInBytes, int height, int depth, SafeCudaStreamHandle stream)
    {
        Validate3DRegion(widthInBytes, height, depth);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_copy_device_to_device_3d_async(destination, source, (UIntPtr)widthInBytes, (UIntPtr)height, (UIntPtr)depth, stream));
    }

    public static void FillPitched2D(SafeCudaPitchedMemoryHandle memory, byte value, int widthInBytes, int height)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_memset_2d(memory, value, (UIntPtr)widthInBytes, (UIntPtr)height));
    }

    public static void FillPitched2DAsync(SafeCudaPitchedMemoryHandle memory, byte value, int widthInBytes, int height, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_memset_2d_async(memory, value, (UIntPtr)widthInBytes, (UIntPtr)height, stream));
    }

    public static void FillPitched3D(SafeCudaPitchedMemoryHandle memory, byte value, int widthInBytes, int height, int depth)
    {
        Validate3DRegion(widthInBytes, height, depth);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_memset_3d(memory, value, (UIntPtr)widthInBytes, (UIntPtr)height, (UIntPtr)depth));
    }

    public static void FillPitched3DAsync(SafeCudaPitchedMemoryHandle memory, byte value, int widthInBytes, int height, int depth, SafeCudaStreamHandle stream)
    {
        Validate3DRegion(widthInBytes, height, depth);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_pitched_memory_memset_3d_async(memory, value, (UIntPtr)widthInBytes, (UIntPtr)height, (UIntPtr)depth, stream));
    }

    private static void Validate2DRegion(int pitch, int widthInBytes, int height, string pitchParameterName)
    {
        if (widthInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(widthInBytes));
        }

        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }

        if (pitch < widthInBytes)
        {
            throw new ArgumentOutOfRangeException(pitchParameterName);
        }

        _ = checked(pitch * height);
    }

    private static void Validate3DRegion(int widthInBytes, int height, int depth)
    {
        if (widthInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(widthInBytes));
        }

        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }

        if (depth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(depth));
        }

        _ = checked(widthInBytes * height * depth);
    }

    private static void Validate3DRegion(int pitch, int widthInBytes, int height, int depth, string pitchParameterName)
    {
        Validate3DRegion(widthInBytes, height, depth);
        if (pitch < widthInBytes)
        {
            throw new ArgumentOutOfRangeException(pitchParameterName);
        }

        _ = checked(pitch * height * depth);
    }

    private static PinnedByteBufferScope Pin2D(byte[] buffer, int pitch, int widthInBytes, int height, PinnedByteBufferDescriptor descriptor)
    {
        if (widthInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(widthInBytes));
        }

        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }

        if (pitch < widthInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(pitch));
        }

        int requiredBytes = checked(pitch * height);
        return PinnedByteBufferScope.Pin(buffer, requiredBytes, descriptor);
    }

    private static PinnedByteBufferScope Pin3D(byte[] buffer, int pitch, int widthInBytes, int height, int depth, PinnedByteBufferDescriptor descriptor)
    {
        Validate3DRegion(pitch, widthInBytes, height, depth, nameof(pitch));
        int requiredBytes = checked(pitch * height * depth);
        return PinnedByteBufferScope.Pin(buffer, requiredBytes, descriptor);
    }
}
