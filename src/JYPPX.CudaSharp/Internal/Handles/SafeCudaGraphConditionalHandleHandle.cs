using JYPPX.CudaSharp.Internal.Interop;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.CudaSharp.Internal.Handles;

internal sealed class SafeCudaGraphConditionalHandleHandle : SafeBridgeHandle
{
    protected override bool ReleaseHandle()
    {
        return NativeMethodsCuda.jyppx_cuda_graph_conditional_handle_destroy_safe(handle) == BridgeStatusCode.Ok;
    }
}
