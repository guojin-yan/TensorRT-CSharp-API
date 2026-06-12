using JYPPX.Shared.Interop;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp.Internal.Handles;

internal sealed class SafeCudaStreamHandle : SafeBridgeHandle
{
    protected override bool ReleaseHandle()
    {
        return NativeMethodsCuda.jyppx_cuda_stream_destroy(handle) == BridgeStatusCode.Ok;
    }
}

