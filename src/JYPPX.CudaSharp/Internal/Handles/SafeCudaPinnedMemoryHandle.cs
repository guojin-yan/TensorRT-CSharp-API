using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp.Internal.Handles;

internal sealed class SafeCudaPinnedMemoryHandle : SafeBridgeHandle
{
    protected override bool ReleaseHandle()
    {
        return NativeMethodsCuda.jyppx_cuda_pinned_memory_free(handle) == BridgeStatusCode.Ok;
    }
}

