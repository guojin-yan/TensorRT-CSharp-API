using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtLayer
{
    /// <summary>
    /// Gets the TensorRT 8 RNNv2 stacked layer count.
    /// 获取 TensorRT 8 RNNv2 堆叠层数。
    /// </summary>
    public int GetRnnV2LayerCount()
    {
        return NativeBridgeApi.GetRnnV2LayerCount(Line, _handle);
    }

    /// <summary>
    /// Gets the TensorRT 8 RNNv2 hidden-state size.
    /// 获取 TensorRT 8 RNNv2 hidden-state 大小。
    /// </summary>
    public int GetRnnV2HiddenSize()
    {
        return NativeBridgeApi.GetRnnV2HiddenSize(Line, _handle);
    }

    /// <summary>
    /// Gets the TensorRT 8 RNNv2 per-step input data length.
    /// 获取 TensorRT 8 RNNv2 每个时间步的输入数据长度。
    /// </summary>
    /// <remarks>
    /// This is a copied scalar query. It does not expose TensorRT-owned tensor or weight pointers.
    /// 这是 copied scalar 查询，不会暴露 TensorRT-owned tensor 或 weights 指针。
    /// </remarks>
    public int GetRnnV2DataLength()
    {
        return NativeBridgeApi.GetRnnV2DataLength(Line, _handle);
    }

    /// <summary>
    /// Gets the TensorRT 8 RNNv2 maximum sequence length.
    /// 获取 TensorRT 8 RNNv2 最大序列长度。
    /// </summary>
    public int GetRnnV2MaxSequenceLength()
    {
        return NativeBridgeApi.GetRnnV2MaxSequenceLength(Line, _handle);
    }

    /// <summary>
    /// Gets the TensorRT 8 RNNv2 operation kind.
    /// 获取 TensorRT 8 RNNv2 运算类型。
    /// </summary>
    public TensorRtRnnOperation GetRnnV2Operation()
    {
        return NativeBridgeApi.GetRnnV2Operation(Line, _handle);
    }

    /// <summary>
    /// Gets the TensorRT 8 RNNv2 direction.
    /// 获取 TensorRT 8 RNNv2 方向。
    /// </summary>
    public TensorRtRnnDirection GetRnnV2Direction()
    {
        return NativeBridgeApi.GetRnnV2Direction(Line, _handle);
    }

    /// <summary>
    /// Gets the TensorRT 8 RNNv2 input mode.
    /// 获取 TensorRT 8 RNNv2 输入模式。
    /// </summary>
    public TensorRtRnnInputMode GetRnnV2InputMode()
    {
        return NativeBridgeApi.GetRnnV2InputMode(Line, _handle);
    }

    /// <summary>
    /// Gets the optional TensorRT 8 RNNv2 initial cell state.
    /// 获取可选的 TensorRT 8 RNNv2 初始 cell state。
    /// </summary>
    /// <returns>
    /// An owner-bound borrowed tensor wrapper, or <see langword="null"/> when no cell state is configured.
    /// 绑定 owner 生命周期的 borrowed tensor 包装；未配置 cell state 时返回 <see langword="null"/>。
    /// </returns>
    public TensorRtTensor? GetRnnV2CellState()
    {
        SafeTensorRtObjectHandleLease ownerLease = CloneRequiredOwnerLease();
        try
        {
            SafeTensorRtObjectHandle? tensor = NativeBridgeApi.GetRnnV2CellState(Line, _handle);
            if (tensor == null)
            {
                ownerLease.Dispose();
                return null;
            }

            return new TensorRtTensor(Line, tensor, ownerLease);
        }
        catch
        {
            ownerLease.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Gets the optional TensorRT 8 RNNv2 initial hidden state.
    /// 获取可选的 TensorRT 8 RNNv2 初始 hidden state。
    /// </summary>
    public TensorRtTensor? GetRnnV2HiddenState()
    {
        SafeTensorRtObjectHandleLease ownerLease = CloneRequiredOwnerLease();
        try
        {
            SafeTensorRtObjectHandle? tensor = NativeBridgeApi.GetRnnV2HiddenState(Line, _handle);
            if (tensor == null)
            {
                ownerLease.Dispose();
                return null;
            }

            return new TensorRtTensor(Line, tensor, ownerLease);
        }
        catch
        {
            ownerLease.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Gets the optional TensorRT 8 RNNv2 sequence-length tensor.
    /// 获取可选的 TensorRT 8 RNNv2 sequence-length tensor。
    /// </summary>
    public TensorRtTensor? GetRnnV2SequenceLengths()
    {
        SafeTensorRtObjectHandleLease ownerLease = CloneRequiredOwnerLease();
        try
        {
            SafeTensorRtObjectHandle? tensor = NativeBridgeApi.GetRnnV2SequenceLengths(Line, _handle);
            if (tensor == null)
            {
                ownerLease.Dispose();
                return null;
            }

            return new TensorRtTensor(Line, tensor, ownerLease);
        }
        catch
        {
            ownerLease.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Copies one TensorRT 8 RNNv2 gate weight matrix into managed storage.
    /// 将一个 TensorRT 8 RNNv2 gate weight matrix 复制到托管存储。
    /// </summary>
    public TensorRtRnnV2GateWeightsSnapshot GetRnnV2WeightsForGate(
        int layerIndex,
        TensorRtRnnGateType gate,
        bool isInputWeights)
    {
        return NativeBridgeApi.GetRnnV2WeightsForGate(
            Line,
            _handle,
            layerIndex,
            gate,
            isInputWeights);
    }

    /// <summary>
    /// Copies one TensorRT 8 RNNv2 gate bias vector into managed storage.
    /// 将一个 TensorRT 8 RNNv2 gate bias vector 复制到托管存储。
    /// </summary>
    public TensorRtRnnV2GateWeightsSnapshot GetRnnV2BiasForGate(
        int layerIndex,
        TensorRtRnnGateType gate,
        bool isInputWeights)
    {
        return NativeBridgeApi.GetRnnV2BiasForGate(
            Line,
            _handle,
            layerIndex,
            gate,
            isInputWeights);
    }
}
