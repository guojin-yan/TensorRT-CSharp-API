using JYPPX.CudaSharp.Internal.Interop;
using JYPPX.Shared.Interop;

namespace JYPPX.CudaSharp.Internal.Handles;

internal sealed class SafeCudaKernelLaunchHandle : SafeBridgeHandle
{
    protected override bool ReleaseHandle()
    {
        _ = NativeMethodsCuda.jyppx_cuda_kernel_launch_destroy_safe(handle);
        return true;
    }
}
