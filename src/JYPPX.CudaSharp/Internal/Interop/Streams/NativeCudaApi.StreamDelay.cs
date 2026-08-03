using JYPPX.CudaSharp.Internal.Handles;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static void EnqueueStreamDelay(SafeCudaStreamHandle stream, uint milliseconds)
    {
        CudaNativeStatus.ThrowIfFailed(
            NativeMethodsCuda.jyppx_cuda_stream_enqueue_delay_safe(stream, milliseconds));
    }
}
