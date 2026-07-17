using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.Shared.Interop;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static SafeCudaSurfaceObjectHandle CreateSurfaceObject(SafeCudaArrayHandle array)
    {
        bool leaseAdded = false;
        array.DangerousAddRef(ref leaseAdded);
        try
        {
            CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_surface_object_create_array_owner_safe(
                array,
                out SafeCudaSurfaceObjectHandle handle));
            handle.AttachArrayOwnerLease(array);
            leaseAdded = false;
            return handle;
        }
        finally
        {
            if (leaseAdded)
            {
                array.DangerousRelease();
            }
        }
    }

    public static SafeCudaTextureObjectHandle CreateTextureObject(
        SafeCudaArrayHandle array,
        CudaTextureDescriptor descriptor,
        bool useCuda11Version2)
    {
        NativeCudaTextureDescriptor nativeDescriptor = descriptor.ToNative();
        bool leaseAdded = false;
        array.DangerousAddRef(ref leaseAdded);
        try
        {
            BridgeStatusCode status = useCuda11Version2
                ? NativeMethodsCuda.jyppx_cuda_texture_object_create_array_owner_v2_safe(
                    array,
                    ref nativeDescriptor,
                    out SafeCudaTextureObjectHandle handle)
                : NativeMethodsCuda.jyppx_cuda_texture_object_create_array_owner_safe(
                    array,
                    ref nativeDescriptor,
                    out handle);
            CudaNativeStatus.ThrowIfFailed(status);
            handle.AttachArrayOwnerLease(array);
            leaseAdded = false;
            return handle;
        }
        finally
        {
            if (leaseAdded)
            {
                array.DangerousRelease();
            }
        }
    }

    public static NativeCudaResourceDescriptorSnapshot GetSurfaceResourceSnapshot(SafeCudaSurfaceObjectHandle surface)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_surface_object_get_resource_snapshot_safe(
            surface,
            out NativeCudaResourceDescriptorSnapshot snapshot));
        return snapshot;
    }

    public static NativeCudaResourceDescriptorSnapshot GetTextureResourceSnapshot(SafeCudaTextureObjectHandle texture)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_texture_object_get_resource_snapshot_safe(
            texture,
            out NativeCudaResourceDescriptorSnapshot snapshot));
        return snapshot;
    }

    public static NativeCudaTextureDescriptor GetTextureDescriptor(SafeCudaTextureObjectHandle texture, bool useCuda11Version2)
    {
        BridgeStatusCode status = useCuda11Version2
            ? NativeMethodsCuda.jyppx_cuda_texture_object_get_descriptor_v2_safe(texture, out NativeCudaTextureDescriptor descriptor)
            : NativeMethodsCuda.jyppx_cuda_texture_object_get_descriptor_safe(texture, out descriptor);
        CudaNativeStatus.ThrowIfFailed(status);
        return descriptor;
    }

    public static NativeCudaTextureResourceViewSnapshot GetTextureResourceViewSnapshot(SafeCudaTextureObjectHandle texture)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_texture_object_get_resource_view_snapshot_safe(
            texture,
            out NativeCudaTextureResourceViewSnapshot snapshot));
        return snapshot;
    }
}
