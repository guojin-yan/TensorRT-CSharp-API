using JYPPX.CudaSharp.Internal.Handles;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static SafeCudaExecutionContextHandle GetPrimaryExecutionContext(int device)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_primary_execution_context_get_safe(
            device,
            out SafeCudaExecutionContextHandle context));
        return context;
    }

    public static int GetExecutionContextDevice(SafeCudaExecutionContextHandle context)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_execution_context_get_device_safe(
            context,
            out int device));
        return device;
    }

    public static ulong GetExecutionContextId(SafeCudaExecutionContextHandle context)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_execution_context_get_id_safe(
            context,
            out ulong contextId));
        return contextId;
    }

    public static void SynchronizeExecutionContext(SafeCudaExecutionContextHandle context)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_execution_context_synchronize_safe(context));
    }

    public static SafeCudaStreamHandle CreateExecutionContextStream(
        SafeCudaExecutionContextHandle context,
        CudaStreamCreationFlags flags,
        int priority)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_execution_context_create_stream_safe(
            context,
            (uint)flags,
            priority,
            out SafeCudaStreamHandle stream));
        return stream;
    }

    public static void RecordExecutionContextEvent(
        SafeCudaExecutionContextHandle context,
        SafeCudaEventHandle eventHandle)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_execution_context_record_event_safe(
            context,
            eventHandle));
    }

    public static void WaitExecutionContextEvent(
        SafeCudaExecutionContextHandle context,
        SafeCudaEventHandle eventHandle)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_execution_context_wait_event_safe(
            context,
            eventHandle));
    }
}
