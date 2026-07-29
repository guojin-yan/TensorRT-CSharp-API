using System;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents TensorRT TensorRtLayerType values.
/// 表示 TensorRT TensorRtLayerType 枚举值。
/// </summary>
public enum TensorRtLayerType
{
    /// <summary>
    /// Represents the Unknown value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Unknown 取值。
    /// </summary>
    Unknown = -1,
    /// <summary>
    /// Represents the Convolution value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Convolution 取值。
    /// </summary>
    Convolution = 0,
    /// <summary>
    /// Represents the Activation value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Activation 取值。
    /// </summary>
    Activation = 2,
    /// <summary>
    /// Represents the Pooling value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Pooling 取值。
    /// </summary>
    Pooling = 3,
    /// <summary>
    /// Represents the Lrn value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Lrn 取值。
    /// </summary>
    Lrn = 4,
    /// <summary>
    /// Represents the Scale value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Scale 取值。
    /// </summary>
    Scale = 5,
    /// <summary>
    /// Represents the SoftMax value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 SoftMax 取值。
    /// </summary>
    SoftMax = 6,
    /// <summary>
    /// Represents the Deconvolution value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Deconvolution 取值。
    /// </summary>
    Deconvolution = 7,
    /// <summary>
    /// Represents the Concatenation value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Concatenation 取值。
    /// </summary>
    Concatenation = 8,
    /// <summary>
    /// Represents the ElementWise value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 ElementWise 取值。
    /// </summary>
    ElementWise = 9,
    /// <summary>
    /// Represents the Cast value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Cast 取值。
    /// </summary>
    Cast = 1,
    /// <summary>
    /// Represents the Plugin value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Plugin 取值。
    /// </summary>
    Plugin = 10,
    /// <summary>
    /// Represents the Unary value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Unary 取值。
    /// </summary>
    Unary = 11,
    /// <summary>
    /// Represents the Padding value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Padding 取值。
    /// </summary>
    Padding = 12,
    /// <summary>
    /// Represents the TopK value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 TopK 取值。
    /// </summary>
    TopK = 15,
    /// <summary>
    /// Represents the Gather value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Gather 取值。
    /// </summary>
    Gather = 16,
    /// <summary>
    /// Represents the MatrixMultiply value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 MatrixMultiply 取值。
    /// </summary>
    MatrixMultiply = 17,
    /// <summary>
    /// Represents the RaggedSoftMax value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 RaggedSoftMax 取值。
    /// </summary>
    RaggedSoftMax = 18,
    /// <summary>
    /// Represents the Shuffle value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Shuffle 取值。
    /// </summary>
    Shuffle = 13,
    /// <summary>
    /// Represents the Reduce value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Reduce 取值。
    /// </summary>
    Reduce = 14,
    /// <summary>
    /// Represents the Constant value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Constant 取值。
    /// </summary>
    Constant = 19,
    /// <summary>
    /// Represents the IdentityTrt10 value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 IdentityTrt10 取值。
    /// </summary>
    IdentityTrt10 = 20,
    /// <summary>
    /// Represents the IdentityTrt8 value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 IdentityTrt8 取值。
    /// </summary>
    IdentityTrt8 = 100021,
    /// <summary>
    /// Represents the SliceTrt10 value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 SliceTrt10 取值。
    /// </summary>
    SliceTrt10 = 22,
    /// <summary>
    /// Represents the SliceTrt8 value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 SliceTrt8 取值。
    /// </summary>
    SliceTrt8 = 100023,
    /// <summary>
    /// Represents the ShapeTrt10 value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 ShapeTrt10 取值。
    /// </summary>
    ShapeTrt10 = 23,
    /// <summary>
    /// Represents the ShapeTrt8 value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 ShapeTrt8 取值。
    /// </summary>
    ShapeTrt8 = 100024,
    /// <summary>
    /// Represents the ParametricReLU value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 ParametricReLU 取值。
    /// </summary>
    ParametricReLU = 24,
    /// <summary>
    /// Represents the ResizeTrt10 value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 ResizeTrt10 取值。
    /// </summary>
    ResizeTrt10 = 25,
    /// <summary>
    /// Represents the ResizeTrt8 value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 ResizeTrt8 取值。
    /// </summary>
    ResizeTrt8 = 100026,
    /// <summary>
    /// Represents the TripLimit value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 TripLimit 取值。
    /// </summary>
    TripLimit = 26,
    /// <summary>
    /// Represents the Recurrence value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Recurrence 取值。
    /// </summary>
    Recurrence = 27,
    /// <summary>
    /// Represents the Iterator value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Iterator 取值。
    /// </summary>
    Iterator = 28,
    /// <summary>
    /// Represents the LoopOutput value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 LoopOutput 取值。
    /// </summary>
    LoopOutput = 29,
    /// <summary>
    /// Represents the SelectTrt10 value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 SelectTrt10 取值。
    /// </summary>
    SelectTrt10 = 30,
    /// <summary>
    /// Represents the SelectTrt8 value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 SelectTrt8 取值。
    /// </summary>
    SelectTrt8 = 100031,
    /// <summary>
    /// Represents the FillTrt10 value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 FillTrt10 取值。
    /// </summary>
    FillTrt10 = 31,
    /// <summary>
    /// Represents the FillTrt8 value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 FillTrt8 取值。
    /// </summary>
    FillTrt8 = 100032,
    /// <summary>
    /// Represents the QuantizeTrt10 value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 QuantizeTrt10 取值。
    /// </summary>
    QuantizeTrt10 = 32,
    /// <summary>
    /// Represents the QuantizeTrt8 value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 QuantizeTrt8 取值。
    /// </summary>
    QuantizeTrt8 = 100033,
    /// <summary>
    /// Represents the DequantizeTrt10 value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 DequantizeTrt10 取值。
    /// </summary>
    DequantizeTrt10 = 33,
    /// <summary>
    /// Represents the DequantizeTrt8 value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 DequantizeTrt8 取值。
    /// </summary>
    DequantizeTrt8 = 100034,
    /// <summary>
    /// Represents the Condition value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Condition 取值。
    /// </summary>
    Condition = 34,
    /// <summary>
    /// Represents the ConditionalInput value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 ConditionalInput 取值。
    /// </summary>
    ConditionalInput = 35,
    /// <summary>
    /// Represents the ConditionalOutput value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 ConditionalOutput 取值。
    /// </summary>
    ConditionalOutput = 36,
    /// <summary>
    /// Represents the Scatter value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Scatter 取值。
    /// </summary>
    Scatter = 37,
    /// <summary>
    /// Represents the Einsum value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Einsum 取值。
    /// </summary>
    Einsum = 38,
    /// <summary>
    /// Represents the Assertion value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Assertion 取值。
    /// </summary>
    Assertion = 39,
    /// <summary>
    /// Represents the OneHot value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 OneHot 取值。
    /// </summary>
    OneHot = 40,
    /// <summary>
    /// Represents the NonZero value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 NonZero 取值。
    /// </summary>
    NonZero = 41,
    /// <summary>
    /// Represents the GridSample value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 GridSample 取值。
    /// </summary>
    GridSample = 42,
    /// <summary>
    /// Represents the Nms value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Nms 取值。
    /// </summary>
    Nms = 43,
    /// <summary>
    /// Represents the ReverseSequence value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 ReverseSequence 取值。
    /// </summary>
    ReverseSequence = 44,
    /// <summary>
    /// Represents the NormalizationTrt10 value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 NormalizationTrt10 取值。
    /// </summary>
    NormalizationTrt10 = 45,
    /// <summary>
    /// Represents the NormalizationTrt8 value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 NormalizationTrt8 取值。
    /// </summary>
    NormalizationTrt8 = 100046,
    /// <summary>
    /// Represents the Squeeze value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Squeeze 取值。
    /// </summary>
    Squeeze = 47,
    /// <summary>
    /// Represents the Unsqueeze value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Unsqueeze 取值。
    /// </summary>
    Unsqueeze = 48,
    /// <summary>
    /// Represents the Cumulative value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Cumulative 取值。
    /// </summary>
    Cumulative = 49,
    /// <summary>
    /// Represents the DynamicQuantize value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 DynamicQuantize 取值。
    /// </summary>
    DynamicQuantize = 50,
    /// <summary>
    /// Represents the AttentionInput value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 AttentionInput 取值。
    /// </summary>
    AttentionInput = 51,
    /// <summary>
    /// Represents the AttentionOutput value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 AttentionOutput 取值。
    /// </summary>
    AttentionOutput = 52,
    /// <summary>
    /// Represents the RotaryEmbedding value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 RotaryEmbedding 取值。
    /// </summary>
    RotaryEmbedding = 53,
    /// <summary>
    /// Represents the KvCacheUpdate value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 KvCacheUpdate 取值。
    /// </summary>
    KvCacheUpdate = 54,
    /// <summary>
    /// Represents the Moe value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 Moe 取值。
    /// </summary>
    Moe = 55,
    /// <summary>
    /// Represents the DistCollective value of TensorRtLayerType.
    /// 表示 TensorRtLayerType 的 DistCollective 取值。
    /// </summary>
    DistCollective = 56
}
/// <summary>
/// Describes the coordinate encoding used by a TensorRT NMS layer.
/// 描述 TensorRT NMS 层使用的边界框坐标编码方式。
/// </summary>
public enum TensorRtBoundingBoxFormat
{
    /// <summary>
    /// Boxes are encoded as diagonal corner pairs: (x1, y1, x2, y2).
    /// 边界框按对角角点编码：(x1, y1, x2, y2)。
    /// </summary>
    CornerPairs = 0,

    /// <summary>
    /// Boxes are encoded as center point and size: (x_center, y_center, width, height).
    /// 边界框按中心点和尺寸编码：(x_center, y_center, width, height)。
    /// </summary>
    CenterSizes = 1
}
