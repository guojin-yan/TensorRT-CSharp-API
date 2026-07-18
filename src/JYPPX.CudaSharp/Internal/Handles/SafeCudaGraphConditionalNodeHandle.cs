using JYPPX.CudaSharp.Internal.Interop;
using JYPPX.Shared.Interop;

namespace JYPPX.CudaSharp.Internal.Handles;

internal sealed class SafeCudaGraphConditionalNodeHandle : SafeBridgeHandle
{
    protected override bool ReleaseHandle()
    {
        return NativeMethodsCuda.jyppx_cuda_graph_conditional_node_destroy_safe(handle) == BridgeStatusCode.Ok;
    }
}
