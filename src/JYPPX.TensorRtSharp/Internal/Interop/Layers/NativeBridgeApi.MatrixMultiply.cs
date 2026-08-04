using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle AddMatrixMultiplyLayer(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle left,
        TensorRtMatrixOperation leftOperation,
        SafeTensorRtObjectHandle right,
        TensorRtMatrixOperation rightOperation)
    {
        SafeTensorRtObjectHandle layer;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_network_add_matrix_multiply(network, left, (int)leftOperation, right, (int)rightOperation, out layer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_network_add_matrix_multiply(network, left, (int)leftOperation, right, (int)rightOperation, out layer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_network_add_matrix_multiply(network, left, (int)leftOperation, right, (int)rightOperation, out layer),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return layer;
    }

    public static void SetMatrixMultiplyOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int inputIndex, TensorRtMatrixOperation operation)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_matrix_multiply_layer_set_operation(layer, inputIndex, (int)operation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_matrix_multiply_layer_set_operation(layer, inputIndex, (int)operation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_matrix_multiply_layer_set_operation(layer, inputIndex, (int)operation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static TensorRtMatrixOperation GetMatrixMultiplyOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int inputIndex)
    {
        int operation;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_matrix_multiply_layer_get_operation(layer, inputIndex, out operation),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_matrix_multiply_layer_get_operation(layer, inputIndex, out operation),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_matrix_multiply_layer_get_operation(layer, inputIndex, out operation),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtMatrixOperation)operation;
    }

}
