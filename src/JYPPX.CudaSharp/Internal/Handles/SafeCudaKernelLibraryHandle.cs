using JYPPX.CudaSharp.Internal.Interop;
using JYPPX.Shared.Interop;

namespace JYPPX.CudaSharp.Internal.Handles;

internal sealed class SafeCudaKernelLibraryHandle : SafeBridgeHandle
{
    protected override bool ReleaseHandle()
    {
        return NativeMethodsCuda.jyppx_cuda_kernel_library_destroy_safe(handle) == BridgeStatusCode.Ok;
    }
}
