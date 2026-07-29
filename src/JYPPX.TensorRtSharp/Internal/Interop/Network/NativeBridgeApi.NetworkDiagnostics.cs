using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static bool MarkNetworkDebugTensor(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle tensor)
    {
        int marked;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_mark_debug(network, tensor, out marked),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_mark_debug(network, tensor, out marked),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(MarkNetworkDebugTensor)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return marked != 0;
    }

    public static bool UnmarkNetworkDebugTensor(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle tensor)
    {
        int unmarked;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_unmark_debug(network, tensor, out unmarked),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_unmark_debug(network, tensor, out unmarked),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(UnmarkNetworkDebugTensor)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return unmarked != 0;
    }

    public static bool IsNetworkDebugTensor(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle tensor)
    {
        int isDebug;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_is_debug_tensor(network, tensor, out isDebug),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_is_debug_tensor(network, tensor, out isDebug),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(IsNetworkDebugTensor)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return isDebug != 0;
    }

    public static bool MarkNetworkUnfusedTensorsAsDebugTensors(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(MarkNetworkUnfusedTensorsAsDebugTensors));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_network_mark_unfused_tensors_as_debug_tensors(network, out int marked);
        NativeStatus.ThrowIfFailed(status);
        return marked != 0;
    }

    public static bool UnmarkNetworkUnfusedTensorsAsDebugTensors(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(UnmarkNetworkUnfusedTensorsAsDebugTensors));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_network_unmark_unfused_tensors_as_debug_tensors(network, out int unmarked);
        NativeStatus.ThrowIfFailed(status);
        return unmarked != 0;
    }

    public static bool MarkNetworkOutputForShapes(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle tensor)
    {
        int marked;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_mark_output_for_shapes(network, tensor, out marked),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_mark_output_for_shapes(network, tensor, out marked),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_mark_output_for_shapes(network, tensor, out marked),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return marked != 0;
    }

    public static bool UnmarkNetworkOutputForShapes(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle tensor)
    {
        int unmarked;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_unmark_output_for_shapes(network, tensor, out unmarked),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_unmark_output_for_shapes(network, tensor, out unmarked),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_unmark_output_for_shapes(network, tensor, out unmarked),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return unmarked != 0;
    }

}
