using JYPPX.CudaSharp.Internal.Interop;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.CudaSharp.Internal.Handles;

internal sealed class SafeCudaDriverModuleHandle : SafeBridgeHandle
{
    protected override bool ReleaseHandle()
    {
        _ = NativeMethodsCuda.jyppx_cuda_driver_module_destroy_safe(handle);
        return true;
    }
}
