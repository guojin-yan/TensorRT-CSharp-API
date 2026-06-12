using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddGatherV2Layer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle data,
        SafeTensorRtObjectHandle indices,
        TensorRtGatherMode mode)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_gather_v2(network, data, indices, (int)mode, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_gather_v2(network, data, indices, (int)mode, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_gather_v2(network, data, indices, (int)mode, out layer),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddFillV2Layer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        TensorRtDims dimensions,
        TensorRtFillOperation operation,
        TensorRtDataType outputType)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(AddFillV2Layer));
        if (dimensions == null)
        {
            throw new ArgumentNullException(nameof(dimensions));
        }

        NativeTensorRtDims nativeDimensions = dimensions.ToNative();
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_fill_v2(network, ref nativeDimensions, (int)operation, (int)outputType, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_fill_v2(network, ref nativeDimensions, (int)operation, (int)outputType, out layer),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(AddFillV2Layer)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddParametricReluLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        SafeTensorRtObjectHandle slope)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_parametric_relu(network, input, slope, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_parametric_relu(network, input, slope, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_parametric_relu(network, input, slope, out layer),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddDequantizeV2Layer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        SafeTensorRtObjectHandle scale,
        TensorRtDataType outputType)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_dequantize_v2(network, input, scale, (int)outputType, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_dequantize_v2(network, input, scale, (int)outputType, out layer),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{nameof(AddDequantizeV2Layer)} is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle AddDistCollectiveLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle input,
        TensorRtCollectiveOperation collectiveOperation,
        TensorRtDistributedReduceOperation reduceOperation,
        long root,
        IReadOnlyList<long>? groups)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(AddDistCollectiveLayer));
        long[] nativeGroups = groups == null ? Array.Empty<long>() : ToArray(groups);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_network_add_dist_collective(
            network,
            input,
            (int)collectiveOperation,
            (int)reduceOperation,
            root,
            nativeGroups,
            nativeGroups.Length,
            out SafeTensorRtObjectHandle layer);
        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static SafeTensorRtObjectHandle GetAttentionFromBoundaryLayer(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetAttentionFromBoundaryLayer));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_attention_boundary_layer_get_attention(layer, out SafeTensorRtObjectHandle attention);
        NativeStatus.ThrowIfFailed(status);
        return attention;
    }

    private static long[] ToArray(IReadOnlyList<long> values)
    {
        long[] result = new long[values.Count];
        for (int index = 0; index < values.Count; ++index)
        {
            result[index] = values[index];
        }

        return result;
    }
}
