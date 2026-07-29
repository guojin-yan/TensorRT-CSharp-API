using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static bool HasNetworkErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        int hasRecorder;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_has_error_recorder(network, out hasRecorder),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_has_error_recorder(network, out hasRecorder),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_has_error_recorder(network, out hasRecorder),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return hasRecorder != 0;
    }

    public static void ClearNetworkErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_clear_error_recorder(network),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_clear_error_recorder(network),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_clear_error_recorder(network),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtErrorRecorderSnapshot GetNetworkErrorRecorderSnapshot(TensorRtApiLine line, SafeTensorRtObjectHandle network)
    {
        NativeTensorRtErrorRecorderSnapshotInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_error_recorder_snapshot_info(network, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_error_recorder_snapshot_info(network, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_error_recorder_snapshot_info(network, out info),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);

        return ReadOwnerErrorRecorderSnapshot(
            line,
            info,
            index =>
            {
                NativeTensorRtErrorRecordInfo error;
                BridgeStatusCode errorStatus = line switch
                {
                    TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_get_error_recorder_error(network, index, out error),
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_get_error_recorder_error(network, index, out error),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_get_error_recorder_error(network, index, out error),
                    _ => throw UnsupportedLine()
                };
                NativeStatus.ThrowIfFailed(errorStatus);
                return error;
            });
    }

    public static void RemoveNetworkTensor(TensorRtApiLine line, SafeTensorRtObjectHandle network, SafeTensorRtObjectHandle tensor)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(RemoveNetworkTensor));
        NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_network_remove_tensor(network, tensor));
    }

    public static SafeTensorRtObjectHandle AddTopKV2Layer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        TensorRtTopKOperation operation,
        int k,
        uint axes,
        TensorRtDataType indicesType)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(AddTopKV2Layer));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_network_add_topk_v2(network, input, (int)operation, k, axes, (int)indicesType, out SafeTensorRtObjectHandle layer);
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

}
