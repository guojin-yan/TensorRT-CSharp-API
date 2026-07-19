using JYPPX.CudaSharp.Internal.Handles;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static CudaDevResourceSnapshot GetDeviceDevResourceSnapshot(int device, CudaDevResourceType resourceType)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_device_get_dev_resource_snapshot_safe(
            device,
            (int)resourceType,
            out NativeCudaDevResourceSnapshot snapshot));
        return new CudaDevResourceSnapshot(snapshot);
    }

    public static CudaDevResourceSnapshot GetExecutionContextDevResourceSnapshot(
        SafeCudaExecutionContextHandle context,
        CudaDevResourceType resourceType)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_execution_context_get_dev_resource_snapshot_safe(
            context,
            (int)resourceType,
            out NativeCudaDevResourceSnapshot snapshot));
        return new CudaDevResourceSnapshot(snapshot);
    }

    public static CudaDevResourceSnapshot GetStreamDevResourceSnapshot(
        SafeCudaStreamHandle stream,
        CudaDevResourceType resourceType)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_get_dev_resource_snapshot_safe(
            stream,
            (int)resourceType,
            out NativeCudaDevResourceSnapshot snapshot));
        return new CudaDevResourceSnapshot(snapshot);
    }
}
