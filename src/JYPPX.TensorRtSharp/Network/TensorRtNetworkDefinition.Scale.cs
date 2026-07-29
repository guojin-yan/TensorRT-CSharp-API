using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Network Definition wrapper.
/// 表示托管 TensorRT Tensor Rt Network Definition 包装器。
/// </summary>
public sealed partial class TensorRtNetworkDefinition
{
    /// <summary>
    /// Adds a Scale layer or object.
    /// 添加 Scale 层或对象。
    /// </summary>
    public TensorRtLayer AddScale(
        TensorRtTensor input,
        TensorRtScaleMode mode,
        TensorRtWeights? shift = null,
        TensorRtWeights? scale = null,
        TensorRtWeights? power = null,
        int channelAxis = 0)
    {
        ValidateInputTensor(input, nameof(input));
        return new TensorRtLayer(Line, NativeBridgeApi.AddScaleLayer(Line, _handle, input.Handle, mode, shift, scale, power, channelAxis));
    }

}
