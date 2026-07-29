using System;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes TensorRT attention input/output tensor packing.
/// 描述 TensorRT attention 输入/输出张量排布方式。
/// </summary>
public enum TensorRtAttentionIoForm
{
    /// <summary>
    /// Represents the PaddedBhnd value of TensorRtAttentionIoForm.
    /// 表示 TensorRtAttentionIoForm 的 PaddedBhnd 取值。
    /// </summary>
    PaddedBhnd = 0,
    /// <summary>
    /// Represents the PackedNhd value of TensorRtAttentionIoForm.
    /// 表示 TensorRtAttentionIoForm 的 PackedNhd 取值。
    /// </summary>
    PackedNhd = 1
}
/// <summary>
/// Describes the normalization operation used inside TensorRT attention.
/// 描述 TensorRT attention 内部使用的归一化操作。
/// </summary>
public enum TensorRtAttentionNormalizationOperation
{
    /// <summary>
    /// Represents the None value of TensorRtAttentionNormalizationOperation.
    /// 表示 TensorRtAttentionNormalizationOperation 的 None 取值。
    /// </summary>
    None = 0,
    /// <summary>
    /// Represents the Softmax value of TensorRtAttentionNormalizationOperation.
    /// 表示 TensorRtAttentionNormalizationOperation 的 Softmax 取值。
    /// </summary>
    Softmax = 1
}

/// <summary>
/// Describes TensorRT attention causal-mask alignment.
/// 描述 TensorRT attention 因果 mask 的对齐方向。
/// </summary>
public enum TensorRtCausalMaskKind
{
    /// <summary>
    /// Represents the None value of TensorRtCausalMaskKind.
    /// 表示 TensorRtCausalMaskKind 的 None 取值。
    /// </summary>
    None = 0,
    /// <summary>
    /// Represents the UpperLeft value of TensorRtCausalMaskKind.
    /// 表示 TensorRtCausalMaskKind 的 UpperLeft 取值。
    /// </summary>
    UpperLeft = 1,
    /// <summary>
    /// Represents the LowerRight value of TensorRtCausalMaskKind.
    /// 表示 TensorRtCausalMaskKind 的 LowerRight 取值。
    /// </summary>
    LowerRight = 2
}

/// <summary>
/// Describes the activation used by a TensorRT MoE layer.
/// 描述 TensorRT MoE 层使用的激活函数。
/// </summary>
public enum TensorRtMoEActivationType
{
    /// <summary>
    /// Represents the None value of TensorRtMoEActivationType.
    /// 表示 TensorRtMoEActivationType 的 None 取值。
    /// </summary>
    None = 0,
    /// <summary>
    /// Represents the SiLU value of TensorRtMoEActivationType.
    /// 表示 TensorRtMoEActivationType 的 SiLU 取值。
    /// </summary>
    SiLU = 1
}
