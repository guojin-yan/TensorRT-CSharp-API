using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddElementWiseLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle left,
        SafeTensorRtObjectHandle right,
        TensorRtElementWiseOperation operation)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_elementwise(network, left, right, (int)operation, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_elementwise(network, left, right, (int)operation, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_elementwise(network, left, right, (int)operation, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

}
