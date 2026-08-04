using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp.Internal.Handles;

internal sealed class SafeCudaPitchedMemoryHandle : SafeBridgeHandle
{
    protected override bool ReleaseHandle()
    {
        return NativeMethodsCuda.jyppx_cuda_pitched_memory_free(handle) == BridgeStatusCode.Ok;
    }
}
