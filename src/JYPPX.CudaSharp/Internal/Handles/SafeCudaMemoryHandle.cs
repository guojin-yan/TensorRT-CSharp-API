using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp.Internal.Handles;

internal sealed class SafeCudaMemoryHandle : SafeBridgeHandle
{
    private bool _isIpcImported;

    internal void MarkIpcImported()
    {
        _isIpcImported = true;
    }

    internal void MarkReleased()
    {
        SetHandleAsInvalid();
    }

    protected override bool ReleaseHandle()
    {
        BridgeStatusCode status = _isIpcImported
            ? NativeMethodsCuda.jyppx_cuda_ipc_close_imported_memory_safe(handle)
            : NativeMethodsCuda.jyppx_cuda_memory_free(handle);
        return status == BridgeStatusCode.Ok;
    }
}
