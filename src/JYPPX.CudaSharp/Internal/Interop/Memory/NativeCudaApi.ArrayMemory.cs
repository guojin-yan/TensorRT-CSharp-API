using System;
using JYPPX.CudaSharp.Internal.Handles;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static SafeCudaArrayHandle AllocateArray(CudaChannelFormatDescriptor descriptor, ulong width, ulong height, CudaArrayCreationFlags flags)
    {
        NativeCudaChannelFormatDesc nativeDescriptor = descriptor.ToNative();
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_malloc_array(
            ref nativeDescriptor,
            ToUIntPtr(width, nameof(width)),
            ToUIntPtr(height, nameof(height)),
            (uint)flags,
            out SafeCudaArrayHandle handle));
        return handle;
    }

    public static SafeCudaArrayHandle Allocate3DArray(CudaChannelFormatDescriptor descriptor, CudaArrayExtent extent, CudaArrayCreationFlags flags)
    {
        NativeCudaChannelFormatDesc nativeDescriptor = descriptor.ToNative();
        NativeCudaArrayExtent nativeExtent = extent.ToNative();
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_malloc_3d_array(
            ref nativeDescriptor,
            ref nativeExtent,
            (uint)flags,
            out SafeCudaArrayHandle handle));
        return handle;
    }

    public static NativeCudaArrayInfo GetArrayInfo(SafeCudaArrayHandle array)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_array_get_info(array, out NativeCudaArrayInfo info));
        return info;
    }

    public static NativeCudaChannelFormatDesc GetArrayChannelDescriptor(SafeCudaArrayHandle array)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_array_get_channel_desc(array, out NativeCudaChannelFormatDesc descriptor));
        return descriptor;
    }

    public static NativeCudaArrayMemoryRequirements GetArrayMemoryRequirements(SafeCudaArrayHandle array, int device)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_array_get_memory_requirements(array, device, out NativeCudaArrayMemoryRequirements requirements));
        return requirements;
    }

    public static NativeCudaArraySparseProperties GetArraySparseProperties(SafeCudaArrayHandle array)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_array_get_sparse_properties(array, out NativeCudaArraySparseProperties properties));
        return properties;
    }

    public static ulong GetTexture1DLinearMaxWidth(CudaChannelFormatDescriptor descriptor, int device)
    {
        NativeCudaChannelFormatDesc nativeDescriptor = descriptor.ToNative();
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_get_texture_1d_linear_max_width(ref nativeDescriptor, device, out UIntPtr maxWidth));
        return maxWidth.ToUInt64();
    }

    public static void CopyToArray(SafeCudaArrayHandle destination, int destinationXBytes, int destinationY, byte[] source, int byteCount)
    {
        using PinnedByteBufferScope scope = PinArray1D(source, byteCount, nameof(CopyToArray), nameof(source), "jyppx_cuda_memcpy_to_array", "host-to-array");
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memcpy_to_array(
            destination,
            ToUIntPtr(destinationXBytes, nameof(destinationXBytes)),
            ToUIntPtr(destinationY, nameof(destinationY)),
            scope.Pointer,
            scope.Size));
    }

    public static void CopyFromArray(byte[] destination, SafeCudaArrayHandle source, int sourceXBytes, int sourceY, int byteCount)
    {
        using PinnedByteBufferScope scope = PinArray1D(destination, byteCount, nameof(CopyFromArray), nameof(destination), "jyppx_cuda_memcpy_from_array", "array-to-host");
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memcpy_from_array(
            scope.Pointer,
            source,
            ToUIntPtr(sourceXBytes, nameof(sourceXBytes)),
            ToUIntPtr(sourceY, nameof(sourceY)),
            scope.Size));
    }

    public static void CopyArrayToArray(SafeCudaArrayHandle destination, int destinationXBytes, int destinationY, SafeCudaArrayHandle source, int sourceXBytes, int sourceY, int byteCount)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memcpy_array_to_array(
            destination,
            ToUIntPtr(destinationXBytes, nameof(destinationXBytes)),
            ToUIntPtr(destinationY, nameof(destinationY)),
            source,
            ToUIntPtr(sourceXBytes, nameof(sourceXBytes)),
            ToUIntPtr(sourceY, nameof(sourceY)),
            ToUIntPtr(byteCount, nameof(byteCount))));
    }

    public static void CopyToArrayAsync(SafeCudaArrayHandle destination, int destinationXBytes, int destinationY, SafeCudaPinnedMemoryHandle source, int byteCount, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memcpy_to_array_async(
            destination,
            ToUIntPtr(destinationXBytes, nameof(destinationXBytes)),
            ToUIntPtr(destinationY, nameof(destinationY)),
            GetPinnedMemoryPointer(source),
            ToUIntPtr(byteCount, nameof(byteCount)),
            stream));
    }

    public static void CopyFromArrayAsync(SafeCudaPinnedMemoryHandle destination, SafeCudaArrayHandle source, int sourceXBytes, int sourceY, int byteCount, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memcpy_from_array_async(
            GetPinnedMemoryPointer(destination),
            source,
            ToUIntPtr(sourceXBytes, nameof(sourceXBytes)),
            ToUIntPtr(sourceY, nameof(sourceY)),
            ToUIntPtr(byteCount, nameof(byteCount)),
            stream));
    }

    public static void Copy2DToArray(SafeCudaArrayHandle destination, int destinationXBytes, int destinationY, byte[] source, int sourcePitch, int widthBytes, int height)
    {
        using PinnedByteBufferScope scope = PinArray2D(source, sourcePitch, widthBytes, height, nameof(Copy2DToArray), nameof(source), "jyppx_cuda_memcpy_2d_to_array", "host-to-array-2d");
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memcpy_2d_to_array(
            destination,
            ToUIntPtr(destinationXBytes, nameof(destinationXBytes)),
            ToUIntPtr(destinationY, nameof(destinationY)),
            scope.Pointer,
            ToUIntPtr(sourcePitch, nameof(sourcePitch)),
            ToUIntPtr(widthBytes, nameof(widthBytes)),
            ToUIntPtr(height, nameof(height))));
    }

    public static void Copy2DFromArray(byte[] destination, int destinationPitch, SafeCudaArrayHandle source, int sourceXBytes, int sourceY, int widthBytes, int height)
    {
        using PinnedByteBufferScope scope = PinArray2D(destination, destinationPitch, widthBytes, height, nameof(Copy2DFromArray), nameof(destination), "jyppx_cuda_memcpy_2d_from_array", "array-to-host-2d");
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memcpy_2d_from_array(
            scope.Pointer,
            ToUIntPtr(destinationPitch, nameof(destinationPitch)),
            source,
            ToUIntPtr(sourceXBytes, nameof(sourceXBytes)),
            ToUIntPtr(sourceY, nameof(sourceY)),
            ToUIntPtr(widthBytes, nameof(widthBytes)),
            ToUIntPtr(height, nameof(height))));
    }

    public static void Copy2DArrayToArray(SafeCudaArrayHandle destination, int destinationXBytes, int destinationY, SafeCudaArrayHandle source, int sourceXBytes, int sourceY, int widthBytes, int height)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memcpy_2d_array_to_array(
            destination,
            ToUIntPtr(destinationXBytes, nameof(destinationXBytes)),
            ToUIntPtr(destinationY, nameof(destinationY)),
            source,
            ToUIntPtr(sourceXBytes, nameof(sourceXBytes)),
            ToUIntPtr(sourceY, nameof(sourceY)),
            ToUIntPtr(widthBytes, nameof(widthBytes)),
            ToUIntPtr(height, nameof(height))));
    }

    public static void Copy2DToArrayAsync(SafeCudaArrayHandle destination, int destinationXBytes, int destinationY, SafeCudaPinnedMemoryHandle source, int sourcePitch, int widthBytes, int height, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memcpy_2d_to_array_async(
            destination,
            ToUIntPtr(destinationXBytes, nameof(destinationXBytes)),
            ToUIntPtr(destinationY, nameof(destinationY)),
            GetPinnedMemoryPointer(source),
            ToUIntPtr(sourcePitch, nameof(sourcePitch)),
            ToUIntPtr(widthBytes, nameof(widthBytes)),
            ToUIntPtr(height, nameof(height)),
            stream));
    }

    public static void Copy2DFromArrayAsync(SafeCudaPinnedMemoryHandle destination, int destinationPitch, SafeCudaArrayHandle source, int sourceXBytes, int sourceY, int widthBytes, int height, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memcpy_2d_from_array_async(
            GetPinnedMemoryPointer(destination),
            ToUIntPtr(destinationPitch, nameof(destinationPitch)),
            source,
            ToUIntPtr(sourceXBytes, nameof(sourceXBytes)),
            ToUIntPtr(sourceY, nameof(sourceY)),
            ToUIntPtr(widthBytes, nameof(widthBytes)),
            ToUIntPtr(height, nameof(height)),
            stream));
    }

    public static void Copy3DToArray(SafeCudaArrayHandle destination, int destinationXBytes, int destinationY, int destinationZ, byte[] source, int sourcePitch, int sourceWidthBytes, int sourceHeight, int widthBytes, int height, int depth)
    {
        using PinnedByteBufferScope scope = PinArray3D(source, sourcePitch, sourceWidthBytes, sourceHeight, widthBytes, height, depth, nameof(Copy3DToArray), nameof(source), "jyppx_cuda_memcpy_3d_to_array", "host-to-array-3d");
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memcpy_3d_to_array(
            destination,
            ToUIntPtr(destinationXBytes, nameof(destinationXBytes)),
            ToUIntPtr(destinationY, nameof(destinationY)),
            ToUIntPtr(destinationZ, nameof(destinationZ)),
            scope.Pointer,
            ToUIntPtr(sourcePitch, nameof(sourcePitch)),
            ToUIntPtr(sourceWidthBytes, nameof(sourceWidthBytes)),
            ToUIntPtr(sourceHeight, nameof(sourceHeight)),
            ToUIntPtr(widthBytes, nameof(widthBytes)),
            ToUIntPtr(height, nameof(height)),
            ToUIntPtr(depth, nameof(depth))));
    }

    public static void Copy3DFromArray(byte[] destination, int destinationPitch, int destinationWidthBytes, int destinationHeight, SafeCudaArrayHandle source, int sourceXBytes, int sourceY, int sourceZ, int widthBytes, int height, int depth)
    {
        using PinnedByteBufferScope scope = PinArray3D(destination, destinationPitch, destinationWidthBytes, destinationHeight, widthBytes, height, depth, nameof(Copy3DFromArray), nameof(destination), "jyppx_cuda_memcpy_3d_from_array", "array-to-host-3d");
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memcpy_3d_from_array(
            scope.Pointer,
            ToUIntPtr(destinationPitch, nameof(destinationPitch)),
            ToUIntPtr(destinationWidthBytes, nameof(destinationWidthBytes)),
            ToUIntPtr(destinationHeight, nameof(destinationHeight)),
            source,
            ToUIntPtr(sourceXBytes, nameof(sourceXBytes)),
            ToUIntPtr(sourceY, nameof(sourceY)),
            ToUIntPtr(sourceZ, nameof(sourceZ)),
            ToUIntPtr(widthBytes, nameof(widthBytes)),
            ToUIntPtr(height, nameof(height)),
            ToUIntPtr(depth, nameof(depth))));
    }

    public static void Copy3DArrayToArray(SafeCudaArrayHandle destination, int destinationXBytes, int destinationY, int destinationZ, SafeCudaArrayHandle source, int sourceXBytes, int sourceY, int sourceZ, int widthBytes, int height, int depth)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memcpy_3d_array_to_array(
            destination,
            ToUIntPtr(destinationXBytes, nameof(destinationXBytes)),
            ToUIntPtr(destinationY, nameof(destinationY)),
            ToUIntPtr(destinationZ, nameof(destinationZ)),
            source,
            ToUIntPtr(sourceXBytes, nameof(sourceXBytes)),
            ToUIntPtr(sourceY, nameof(sourceY)),
            ToUIntPtr(sourceZ, nameof(sourceZ)),
            ToUIntPtr(widthBytes, nameof(widthBytes)),
            ToUIntPtr(height, nameof(height)),
            ToUIntPtr(depth, nameof(depth))));
    }

    public static void Copy3DToArrayAsync(SafeCudaArrayHandle destination, int destinationXBytes, int destinationY, int destinationZ, SafeCudaPinnedMemoryHandle source, int sourcePitch, int sourceWidthBytes, int sourceHeight, int widthBytes, int height, int depth, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memcpy_3d_to_array_async(
            destination,
            ToUIntPtr(destinationXBytes, nameof(destinationXBytes)),
            ToUIntPtr(destinationY, nameof(destinationY)),
            ToUIntPtr(destinationZ, nameof(destinationZ)),
            GetPinnedMemoryPointer(source),
            ToUIntPtr(sourcePitch, nameof(sourcePitch)),
            ToUIntPtr(sourceWidthBytes, nameof(sourceWidthBytes)),
            ToUIntPtr(sourceHeight, nameof(sourceHeight)),
            ToUIntPtr(widthBytes, nameof(widthBytes)),
            ToUIntPtr(height, nameof(height)),
            ToUIntPtr(depth, nameof(depth)),
            stream));
    }

    public static void Copy3DFromArrayAsync(SafeCudaPinnedMemoryHandle destination, int destinationPitch, int destinationWidthBytes, int destinationHeight, SafeCudaArrayHandle source, int sourceXBytes, int sourceY, int sourceZ, int widthBytes, int height, int depth, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memcpy_3d_from_array_async(
            GetPinnedMemoryPointer(destination),
            ToUIntPtr(destinationPitch, nameof(destinationPitch)),
            ToUIntPtr(destinationWidthBytes, nameof(destinationWidthBytes)),
            ToUIntPtr(destinationHeight, nameof(destinationHeight)),
            source,
            ToUIntPtr(sourceXBytes, nameof(sourceXBytes)),
            ToUIntPtr(sourceY, nameof(sourceY)),
            ToUIntPtr(sourceZ, nameof(sourceZ)),
            ToUIntPtr(widthBytes, nameof(widthBytes)),
            ToUIntPtr(height, nameof(height)),
            ToUIntPtr(depth, nameof(depth)),
            stream));
    }

    public static void Copy3DArrayToArrayAsync(SafeCudaArrayHandle destination, int destinationXBytes, int destinationY, int destinationZ, SafeCudaArrayHandle source, int sourceXBytes, int sourceY, int sourceZ, int widthBytes, int height, int depth, SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_memcpy_3d_array_to_array_async(
            destination,
            ToUIntPtr(destinationXBytes, nameof(destinationXBytes)),
            ToUIntPtr(destinationY, nameof(destinationY)),
            ToUIntPtr(destinationZ, nameof(destinationZ)),
            source,
            ToUIntPtr(sourceXBytes, nameof(sourceXBytes)),
            ToUIntPtr(sourceY, nameof(sourceY)),
            ToUIntPtr(sourceZ, nameof(sourceZ)),
            ToUIntPtr(widthBytes, nameof(widthBytes)),
            ToUIntPtr(height, nameof(height)),
            ToUIntPtr(depth, nameof(depth)),
            stream));
    }

    public static SafeCudaMipmappedArrayHandle AllocateMipmappedArray(CudaChannelFormatDescriptor descriptor, CudaArrayExtent extent, uint levelCount, CudaArrayCreationFlags flags)
    {
        NativeCudaChannelFormatDesc nativeDescriptor = descriptor.ToNative();
        NativeCudaArrayExtent nativeExtent = extent.ToNative();
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_malloc_mipmapped_array(
            ref nativeDescriptor,
            ref nativeExtent,
            levelCount,
            (uint)flags,
            out SafeCudaMipmappedArrayHandle handle));
        return handle;
    }

    public static NativeCudaArrayInfo GetMipmappedArrayLevelInfo(SafeCudaMipmappedArrayHandle array, uint level)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_get_mipmapped_array_level_info(array, level, out NativeCudaArrayInfo info));
        return info;
    }

    public static NativeCudaArrayMemoryRequirements GetMipmappedArrayMemoryRequirements(SafeCudaMipmappedArrayHandle array, int device)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_mipmapped_array_get_memory_requirements(array, device, out NativeCudaArrayMemoryRequirements requirements));
        return requirements;
    }

    public static NativeCudaArraySparseProperties GetMipmappedArraySparseProperties(SafeCudaMipmappedArrayHandle array)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_mipmapped_array_get_sparse_properties(array, out NativeCudaArraySparseProperties properties));
        return properties;
    }

    public static void ResetPersistingL2Cache()
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_ctx_reset_persisting_l2_cache());
    }

    public static void FlushGpuDirectRdmaWrites(CudaGpuDirectRdmaWritesTarget target, CudaGpuDirectRdmaWritesScope scope)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_flush_gpu_direct_rdma_writes((int)target, (int)scope));
    }

    private static UIntPtr ToUIntPtr(int value, string parameterName)
    {
        if (value < 0)
        {
            throw new ArgumentOutOfRangeException(parameterName);
        }

        return ToUIntPtr((ulong)value, parameterName);
    }

    private static UIntPtr ToUIntPtr(ulong value, string parameterName)
    {
        if (UIntPtr.Size == 4 && value > uint.MaxValue)
        {
            throw new ArgumentOutOfRangeException(parameterName);
        }

        return new UIntPtr(value);
    }

    private static PinnedByteBufferScope PinArray1D(byte[] buffer, int byteCount, string operationName, string bufferParameterName, string nativeEntryPoint, string transferDirection)
    {
        if (byteCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(byteCount));
        }

        return PinnedByteBufferScope.Pin(buffer, byteCount, new PinnedByteBufferDescriptor(operationName, bufferParameterName, nativeEntryPoint, transferDirection));
    }

    private static PinnedByteBufferScope PinArray2D(byte[] buffer, int pitch, int widthBytes, int height, string operationName, string bufferParameterName, string nativeEntryPoint, string transferDirection)
    {
        ValidateArray2DRegion(pitch, widthBytes, height, nameof(pitch));
        int requiredBytes = checked(pitch * height);
        return PinnedByteBufferScope.Pin(buffer, requiredBytes, new PinnedByteBufferDescriptor(operationName, bufferParameterName, nativeEntryPoint, transferDirection));
    }

    private static PinnedByteBufferScope PinArray3D(byte[] buffer, int pitch, int pitchedWidthBytes, int pitchedHeight, int widthBytes, int height, int depth, string operationName, string bufferParameterName, string nativeEntryPoint, string transferDirection)
    {
        ValidateArray3DRegion(pitch, pitchedWidthBytes, pitchedHeight, widthBytes, height, depth, nameof(pitch));
        int requiredBytes = checked(pitch * pitchedHeight * depth);
        return PinnedByteBufferScope.Pin(buffer, requiredBytes, new PinnedByteBufferDescriptor(operationName, bufferParameterName, nativeEntryPoint, transferDirection));
    }

    private static void ValidateArray2DRegion(int pitch, int widthBytes, int height, string pitchParameterName)
    {
        if (widthBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(widthBytes));
        }

        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }

        if (pitch < widthBytes)
        {
            throw new ArgumentOutOfRangeException(pitchParameterName);
        }

        _ = checked(pitch * height);
    }

    private static void ValidateArray3DRegion(int pitch, int pitchedWidthBytes, int pitchedHeight, int widthBytes, int height, int depth, string pitchParameterName)
    {
        ValidateArray2DRegion(pitch, widthBytes, height, pitchParameterName);

        if (depth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(depth));
        }

        if (pitchedWidthBytes < widthBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(pitchedWidthBytes));
        }

        if (pitchedHeight < height)
        {
            throw new ArgumentOutOfRangeException(nameof(pitchedHeight));
        }

        _ = checked(pitch * pitchedHeight * depth);
    }
}
