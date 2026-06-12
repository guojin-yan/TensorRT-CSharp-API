using JYPPX.Shared.Interop;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp.Internal.Handles;

internal sealed class SafeCudaArrayHandle : SafeBridgeHandle
{
    protected override bool ReleaseHandle()
    {
        return NativeMethodsCuda.jyppx_cuda_free_array(handle) == BridgeStatusCode.Ok;
    }
}
