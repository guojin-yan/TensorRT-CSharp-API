using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp.Internal.Handles;

internal sealed class SafeTensorRtObjectHandle : SafeBridgeHandle
{
    public SafeTensorRtObjectHandle()
        : base()
    {
    }

    protected override bool ReleaseHandle()
    {
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt_object_destroy(handle);
        return status == BridgeStatusCode.Ok;
    }
}
