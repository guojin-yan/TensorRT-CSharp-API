using JYPPX.Shared.Interop;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp.Internal.Handles;

internal sealed class SafeCudaMemoryHandle : SafeBridgeHandle
{
    internal void MarkReleased()
    {
        SetHandleAsInvalid();
    }

    protected override bool ReleaseHandle()
    {
        return NativeMethodsCuda.jyppx_cuda_memory_free(handle) == BridgeStatusCode.Ok;
    }
}
