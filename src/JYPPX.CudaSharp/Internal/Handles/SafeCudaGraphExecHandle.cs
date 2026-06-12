using JYPPX.CudaSharp.Internal.Interop;
using JYPPX.Shared.Interop;

namespace JYPPX.CudaSharp.Internal.Handles;

internal sealed class SafeCudaGraphExecHandle : SafeBridgeHandle
{
    protected override bool ReleaseHandle()
    {
        return NativeMethodsCuda.jyppx_cuda_graph_exec_destroy(handle) == BridgeStatusCode.Ok;
    }
}
