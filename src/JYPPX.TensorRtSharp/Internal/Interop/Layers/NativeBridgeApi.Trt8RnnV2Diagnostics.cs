using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static int GetRnnV2LayerCount(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetRnnV2Int(line, layer, NativeMethodsTensorRt.jyppx_trt8_rnn_v2_layer_get_layer_count, nameof(GetRnnV2LayerCount));
    }

    public static int GetRnnV2HiddenSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetRnnV2Int(line, layer, NativeMethodsTensorRt.jyppx_trt8_rnn_v2_layer_get_hidden_size, nameof(GetRnnV2HiddenSize));
    }

    public static int GetRnnV2DataLength(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetRnnV2Int(line, layer, NativeMethodsTensorRt.jyppx_trt8_rnn_v2_layer_get_data_length, nameof(GetRnnV2DataLength));
    }

    public static int GetRnnV2MaxSequenceLength(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetRnnV2Int(line, layer, NativeMethodsTensorRt.jyppx_trt8_rnn_v2_layer_get_max_seq_length, nameof(GetRnnV2MaxSequenceLength));
    }

    public static TensorRtRnnOperation GetRnnV2Operation(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtRnnOperation)GetRnnV2Int(line, layer, NativeMethodsTensorRt.jyppx_trt8_rnn_v2_layer_get_operation, nameof(GetRnnV2Operation));
    }

    public static TensorRtRnnDirection GetRnnV2Direction(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtRnnDirection)GetRnnV2Int(line, layer, NativeMethodsTensorRt.jyppx_trt8_rnn_v2_layer_get_direction, nameof(GetRnnV2Direction));
    }

    public static TensorRtRnnInputMode GetRnnV2InputMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtRnnInputMode)GetRnnV2Int(line, layer, NativeMethodsTensorRt.jyppx_trt8_rnn_v2_layer_get_input_mode, nameof(GetRnnV2InputMode));
    }

    public static void SetRnnV2Operation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtRnnOperation operation)
    {
        SetRnnV2Int(line, layer, (int)operation, NativeMethodsTensorRt.jyppx_trt8_rnn_v2_layer_set_operation, nameof(SetRnnV2Operation));
    }

    public static void SetRnnV2Direction(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtRnnDirection direction)
    {
        SetRnnV2Int(line, layer, (int)direction, NativeMethodsTensorRt.jyppx_trt8_rnn_v2_layer_set_direction, nameof(SetRnnV2Direction));
    }

    public static void SetRnnV2InputMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtRnnInputMode inputMode)
    {
        SetRnnV2Int(line, layer, (int)inputMode, NativeMethodsTensorRt.jyppx_trt8_rnn_v2_layer_set_input_mode, nameof(SetRnnV2InputMode));
    }

    public static void SetRnnV2CellState(TensorRtApiLine line, SafeTensorRtObjectHandle layer, SafeTensorRtObjectHandle tensor)
    {
        SetRnnV2Tensor(line, layer, tensor, NativeMethodsTensorRt.jyppx_trt8_rnn_v2_layer_set_cell_state, nameof(SetRnnV2CellState));
    }

    public static void SetRnnV2HiddenState(TensorRtApiLine line, SafeTensorRtObjectHandle layer, SafeTensorRtObjectHandle tensor)
    {
        SetRnnV2Tensor(line, layer, tensor, NativeMethodsTensorRt.jyppx_trt8_rnn_v2_layer_set_hidden_state, nameof(SetRnnV2HiddenState));
    }

    public static void SetRnnV2SequenceLengths(TensorRtApiLine line, SafeTensorRtObjectHandle layer, SafeTensorRtObjectHandle tensor)
    {
        SetRnnV2Tensor(line, layer, tensor, NativeMethodsTensorRt.jyppx_trt8_rnn_v2_layer_set_sequence_lengths, nameof(SetRnnV2SequenceLengths));
    }

    public static SafeTensorRtObjectHandle? GetRnnV2CellState(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetRnnV2OptionalTensor(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt8_rnn_v2_layer_get_cell_state,
            nameof(GetRnnV2CellState));
    }

    public static SafeTensorRtObjectHandle? GetRnnV2HiddenState(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetRnnV2OptionalTensor(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt8_rnn_v2_layer_get_hidden_state,
            nameof(GetRnnV2HiddenState));
    }

    public static SafeTensorRtObjectHandle? GetRnnV2SequenceLengths(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetRnnV2OptionalTensor(
            line,
            layer,
            NativeMethodsTensorRt.jyppx_trt8_rnn_v2_layer_get_sequence_lengths,
            nameof(GetRnnV2SequenceLengths));
    }

    public static TensorRtRnnV2GateWeightsSnapshot GetRnnV2WeightsForGate(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        int layerIndex,
        TensorRtRnnGateType gate,
        bool isInputWeights)
    {
        return GetRnnV2GateWeights(
            line,
            layer,
            layerIndex,
            gate,
            isInputWeights,
            isBias: false,
            NativeMethodsTensorRt.jyppx_trt8_rnn_v2_layer_get_weights_for_gate_copy,
            nameof(GetRnnV2WeightsForGate));
    }

    public static TensorRtRnnV2GateWeightsSnapshot GetRnnV2BiasForGate(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        int layerIndex,
        TensorRtRnnGateType gate,
        bool isInputWeights)
    {
        return GetRnnV2GateWeights(
            line,
            layer,
            layerIndex,
            gate,
            isInputWeights,
            isBias: true,
            NativeMethodsTensorRt.jyppx_trt8_rnn_v2_layer_get_bias_for_gate_copy,
            nameof(GetRnnV2BiasForGate));
    }

    private static int GetRnnV2Int(TensorRtApiLine line, SafeTensorRtObjectHandle layer, LayerIntGetter trt8, string apiName)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{apiName} is available only for TensorRT 8 RNNv2 layers.");
        }

        BridgeStatusCode status = trt8(layer, out int value);
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    private static void SetRnnV2Int(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int value, LayerIntSetter trt8, string apiName)
    {
        EnsureTensorRt8RnnV2(line, apiName);
        BridgeStatusCode status = trt8(layer, value);
        NativeStatus.ThrowIfFailed(status);
    }

    private static void SetRnnV2Tensor(TensorRtApiLine line, SafeTensorRtObjectHandle layer, SafeTensorRtObjectHandle tensor, LayerTensorSetter trt8, string apiName)
    {
        EnsureTensorRt8RnnV2(line, apiName);
        BridgeStatusCode status = trt8(layer, tensor);
        NativeStatus.ThrowIfFailed(status);
    }

    private static SafeTensorRtObjectHandle? GetRnnV2OptionalTensor(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        LayerTensorGetter trt8,
        string apiName)
    {
        EnsureTensorRt8RnnV2(line, apiName);

        BridgeStatusCode status = trt8(layer, out SafeTensorRtObjectHandle tensor);
        if (status != BridgeStatusCode.Ok)
        {
            tensor?.Dispose();
            NativeStatus.ThrowIfFailed(status);
        }

        if (tensor == null || tensor.IsInvalid)
        {
            tensor?.Dispose();
            return null;
        }

        return tensor;
    }

    private static TensorRtRnnV2GateWeightsSnapshot GetRnnV2GateWeights(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle layer,
        int layerIndex,
        TensorRtRnnGateType gate,
        bool isInputWeights,
        bool isBias,
        RnnV2GateWeightsGetter trt8,
        string apiName)
    {
        EnsureTensorRt8RnnV2(line, apiName);

        BridgeStatusCode status = trt8(
            layer,
            layerIndex,
            (int)gate,
            isInputWeights ? 1 : 0,
            out NativeTensorRtWeightsInfo info,
            Array.Empty<byte>(),
            UIntPtr.Zero,
            out UIntPtr requiredSize);
        NativeStatus.ThrowIfFailed(status);

        ulong requiredByteCount = requiredSize.ToUInt64();
        if (requiredByteCount > int.MaxValue)
        {
            throw new BridgeProbeException(
                BridgeStatusCode.NotSupported,
                BridgeErrorCategory.TensorRt,
                $"{apiName} returned {requiredByteCount} bytes, which exceeds the managed snapshot limit.");
        }

        byte[] values = new byte[(int)requiredByteCount];
        if (values.Length != 0)
        {
            status = trt8(
                layer,
                layerIndex,
                (int)gate,
                isInputWeights ? 1 : 0,
                out info,
                values,
                (UIntPtr)(uint)values.Length,
                out UIntPtr copiedSize);
            NativeStatus.ThrowIfFailed(status);

            if (copiedSize.ToUInt64() != requiredByteCount)
            {
                throw new BridgeProbeException(
                    BridgeStatusCode.InvalidState,
                    BridgeErrorCategory.TensorRt,
                    $"{apiName} changed its required byte count during the copy.");
            }
        }

        return new TensorRtRnnV2GateWeightsSnapshot(
            layerIndex,
            gate,
            isInputWeights,
            isBias,
            (TensorRtDataType)info.DataType,
            info.Count,
            values);
    }

    private static void EnsureTensorRt8RnnV2(TensorRtApiLine line, string apiName)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(
                BridgeStatusCode.NotSupported,
                BridgeErrorCategory.TensorRt,
                $"{apiName} is available only for TensorRT 8 RNNv2 layers.");
        }
    }

    private delegate BridgeStatusCode LayerTensorGetter(
        SafeTensorRtObjectHandle layer,
        out SafeTensorRtObjectHandle outTensor);

    private delegate BridgeStatusCode LayerTensorSetter(SafeTensorRtObjectHandle layer, SafeTensorRtObjectHandle tensor);

    private delegate BridgeStatusCode RnnV2GateWeightsGetter(
        SafeTensorRtObjectHandle layer,
        int layerIndex,
        int gate,
        int isW,
        out NativeTensorRtWeightsInfo outInfo,
        byte[] outputBuffer,
        UIntPtr outputBufferSize,
        out UIntPtr outRequiredSize);
}
