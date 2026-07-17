using JYPPX.Shared.Interop;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp.Internal.Handles;

internal sealed class SafeCudaSurfaceObjectHandle : SafeBridgeHandle
{
    private SafeCudaArrayHandle? _arrayOwner;

    internal void AttachArrayOwnerLease(SafeCudaArrayHandle arrayOwner)
    {
        _arrayOwner = arrayOwner;
    }

    protected override bool ReleaseHandle()
    {
        try
        {
            return NativeMethodsCuda.jyppx_cuda_surface_object_destroy_safe(handle) == BridgeStatusCode.Ok;
        }
        finally
        {
            _arrayOwner?.DangerousRelease();
            _arrayOwner = null;
        }
    }
}
