using System;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents TensorRT TensorRtElementWiseOperation values.
/// 表示 TensorRT TensorRtElementWiseOperation 枚举值。
/// </summary>
public enum TensorRtElementWiseOperation
{
    /// <summary>
    /// Represents the Sum value of TensorRtElementWiseOperation.
    /// 表示 TensorRtElementWiseOperation 的 Sum 取值。
    /// </summary>
    Sum = 0,
    /// <summary>
    /// Represents the Product value of TensorRtElementWiseOperation.
    /// 表示 TensorRtElementWiseOperation 的 Product 取值。
    /// </summary>
    Product = 1,
    /// <summary>
    /// Represents the Max value of TensorRtElementWiseOperation.
    /// 表示 TensorRtElementWiseOperation 的 Max 取值。
    /// </summary>
    Max = 2,
    /// <summary>
    /// Represents the Min value of TensorRtElementWiseOperation.
    /// 表示 TensorRtElementWiseOperation 的 Min 取值。
    /// </summary>
    Min = 3,
    /// <summary>
    /// Represents the Subtract value of TensorRtElementWiseOperation.
    /// 表示 TensorRtElementWiseOperation 的 Subtract 取值。
    /// </summary>
    Subtract = 4,
    /// <summary>
    /// Represents the Divide value of TensorRtElementWiseOperation.
    /// 表示 TensorRtElementWiseOperation 的 Divide 取值。
    /// </summary>
    Divide = 5,
    /// <summary>
    /// Represents the Power value of TensorRtElementWiseOperation.
    /// 表示 TensorRtElementWiseOperation 的 Power 取值。
    /// </summary>
    Power = 6,
    /// <summary>
    /// Represents the FloorDivide value of TensorRtElementWiseOperation.
    /// 表示 TensorRtElementWiseOperation 的 FloorDivide 取值。
    /// </summary>
    FloorDivide = 7,
    /// <summary>
    /// Represents the And value of TensorRtElementWiseOperation.
    /// 表示 TensorRtElementWiseOperation 的 And 取值。
    /// </summary>
    And = 8,
    /// <summary>
    /// Represents the Or value of TensorRtElementWiseOperation.
    /// 表示 TensorRtElementWiseOperation 的 Or 取值。
    /// </summary>
    Or = 9,
    /// <summary>
    /// Represents the Xor value of TensorRtElementWiseOperation.
    /// 表示 TensorRtElementWiseOperation 的 Xor 取值。
    /// </summary>
    Xor = 10,
    /// <summary>
    /// Represents the Equal value of TensorRtElementWiseOperation.
    /// 表示 TensorRtElementWiseOperation 的 Equal 取值。
    /// </summary>
    Equal = 11,
    /// <summary>
    /// Represents the Greater value of TensorRtElementWiseOperation.
    /// 表示 TensorRtElementWiseOperation 的 Greater 取值。
    /// </summary>
    Greater = 12,
    /// <summary>
    /// Represents the Less value of TensorRtElementWiseOperation.
    /// 表示 TensorRtElementWiseOperation 的 Less 取值。
    /// </summary>
    Less = 13
}
/// <summary>
/// Represents TensorRT TensorRtMatrixOperation values.
/// 表示 TensorRT TensorRtMatrixOperation 枚举值。
/// </summary>
public enum TensorRtMatrixOperation
{
    /// <summary>
    /// Represents the None value of TensorRtMatrixOperation.
    /// 表示 TensorRtMatrixOperation 的 None 取值。
    /// </summary>
    None = 0,
    /// <summary>
    /// Represents the Transpose value of TensorRtMatrixOperation.
    /// 表示 TensorRtMatrixOperation 的 Transpose 取值。
    /// </summary>
    Transpose = 1,
    /// <summary>
    /// Represents the Vector value of TensorRtMatrixOperation.
    /// 表示 TensorRtMatrixOperation 的 Vector 取值。
    /// </summary>
    Vector = 2
}

/// <summary>
/// Represents TensorRT TensorRtReduceOperation values.
/// 表示 TensorRT TensorRtReduceOperation 枚举值。
/// </summary>
public enum TensorRtReduceOperation
{
    /// <summary>
    /// Represents the Sum value of TensorRtReduceOperation.
    /// 表示 TensorRtReduceOperation 的 Sum 取值。
    /// </summary>
    Sum = 0,
    /// <summary>
    /// Represents the Product value of TensorRtReduceOperation.
    /// 表示 TensorRtReduceOperation 的 Product 取值。
    /// </summary>
    Product = 1,
    /// <summary>
    /// Represents the Max value of TensorRtReduceOperation.
    /// 表示 TensorRtReduceOperation 的 Max 取值。
    /// </summary>
    Max = 2,
    /// <summary>
    /// Represents the Min value of TensorRtReduceOperation.
    /// 表示 TensorRtReduceOperation 的 Min 取值。
    /// </summary>
    Min = 3,
    /// <summary>
    /// Represents the Average value of TensorRtReduceOperation.
    /// 表示 TensorRtReduceOperation 的 Average 取值。
    /// </summary>
    Average = 4
}

/// <summary>
/// Selects the distributed reduction operation for a TensorRT 11 DistCollective layer.
/// 选择 TensorRT 11 DistCollective 层使用的分布式归约操作。
/// </summary>
public enum TensorRtDistributedReduceOperation
{
    /// <summary>
    /// Represents the Sum value of TensorRtDistributedReduceOperation.
    /// 表示 TensorRtDistributedReduceOperation 的 Sum 取值。
    /// </summary>
    Sum = 0,
    /// <summary>
    /// Represents the Product value of TensorRtDistributedReduceOperation.
    /// 表示 TensorRtDistributedReduceOperation 的 Product 取值。
    /// </summary>
    Product = 1,
    /// <summary>
    /// Represents the Max value of TensorRtDistributedReduceOperation.
    /// 表示 TensorRtDistributedReduceOperation 的 Max 取值。
    /// </summary>
    Max = 2,
    /// <summary>
    /// Represents the Min value of TensorRtDistributedReduceOperation.
    /// 表示 TensorRtDistributedReduceOperation 的 Min 取值。
    /// </summary>
    Min = 3,
    /// <summary>
    /// Represents the Average value of TensorRtDistributedReduceOperation.
    /// 表示 TensorRtDistributedReduceOperation 的 Average 取值。
    /// </summary>
    Average = 4,
    /// <summary>
    /// Represents the None value of TensorRtDistributedReduceOperation.
    /// 表示 TensorRtDistributedReduceOperation 的 None 取值。
    /// </summary>
    None = 5
}

/// <summary>
/// Represents TensorRT TensorRtUnaryOperation values.
/// 表示 TensorRT TensorRtUnaryOperation 枚举值。
/// </summary>
public enum TensorRtUnaryOperation
{
    /// <summary>
    /// Represents the Exp value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Exp 取值。
    /// </summary>
    Exp = 0,
    /// <summary>
    /// Represents the Log value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Log 取值。
    /// </summary>
    Log = 1,
    /// <summary>
    /// Represents the Sqrt value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Sqrt 取值。
    /// </summary>
    Sqrt = 2,
    /// <summary>
    /// Represents the Recip value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Recip 取值。
    /// </summary>
    Recip = 3,
    /// <summary>
    /// Represents the Abs value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Abs 取值。
    /// </summary>
    Abs = 4,
    /// <summary>
    /// Represents the Neg value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Neg 取值。
    /// </summary>
    Neg = 5,
    /// <summary>
    /// Represents the Sin value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Sin 取值。
    /// </summary>
    Sin = 6,
    /// <summary>
    /// Represents the Cos value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Cos 取值。
    /// </summary>
    Cos = 7,
    /// <summary>
    /// Represents the Tan value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Tan 取值。
    /// </summary>
    Tan = 8,
    /// <summary>
    /// Represents the Sinh value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Sinh 取值。
    /// </summary>
    Sinh = 9,
    /// <summary>
    /// Represents the Cosh value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Cosh 取值。
    /// </summary>
    Cosh = 10,
    /// <summary>
    /// Represents the Asin value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Asin 取值。
    /// </summary>
    Asin = 11,
    /// <summary>
    /// Represents the Acos value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Acos 取值。
    /// </summary>
    Acos = 12,
    /// <summary>
    /// Represents the Atan value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Atan 取值。
    /// </summary>
    Atan = 13,
    /// <summary>
    /// Represents the Asinh value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Asinh 取值。
    /// </summary>
    Asinh = 14,
    /// <summary>
    /// Represents the Acosh value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Acosh 取值。
    /// </summary>
    Acosh = 15,
    /// <summary>
    /// Represents the Atanh value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Atanh 取值。
    /// </summary>
    Atanh = 16,
    /// <summary>
    /// Represents the Ceil value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Ceil 取值。
    /// </summary>
    Ceil = 17,
    /// <summary>
    /// Represents the Floor value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Floor 取值。
    /// </summary>
    Floor = 18,
    /// <summary>
    /// Represents the Erf value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Erf 取值。
    /// </summary>
    Erf = 19,
    /// <summary>
    /// Represents the Not value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Not 取值。
    /// </summary>
    Not = 20,
    /// <summary>
    /// Represents the Sign value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Sign 取值。
    /// </summary>
    Sign = 21,
    /// <summary>
    /// Represents the Round value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 Round 取值。
    /// </summary>
    Round = 22,
    /// <summary>
    /// Represents the IsInf value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 IsInf 取值。
    /// </summary>
    IsInf = 23,
    /// <summary>
    /// Represents the IsNaN value of TensorRtUnaryOperation.
    /// 表示 TensorRtUnaryOperation 的 IsNaN 取值。
    /// </summary>
    IsNaN = 24
}

/// <summary>
/// Represents TensorRT TensorRtActivationType values.
/// 表示 TensorRT TensorRtActivationType 枚举值。
/// </summary>
public enum TensorRtActivationType
{
    /// <summary>
    /// Represents the Relu value of TensorRtActivationType.
    /// 表示 TensorRtActivationType 的 Relu 取值。
    /// </summary>
    Relu = 0,
    /// <summary>
    /// Represents the Sigmoid value of TensorRtActivationType.
    /// 表示 TensorRtActivationType 的 Sigmoid 取值。
    /// </summary>
    Sigmoid = 1,
    /// <summary>
    /// Represents the Tanh value of TensorRtActivationType.
    /// 表示 TensorRtActivationType 的 Tanh 取值。
    /// </summary>
    Tanh = 2,
    /// <summary>
    /// Represents the LeakyRelu value of TensorRtActivationType.
    /// 表示 TensorRtActivationType 的 LeakyRelu 取值。
    /// </summary>
    LeakyRelu = 3,
    /// <summary>
    /// Represents the Elu value of TensorRtActivationType.
    /// 表示 TensorRtActivationType 的 Elu 取值。
    /// </summary>
    Elu = 4,
    /// <summary>
    /// Represents the Selu value of TensorRtActivationType.
    /// 表示 TensorRtActivationType 的 Selu 取值。
    /// </summary>
    Selu = 5,
    /// <summary>
    /// Represents the SoftSign value of TensorRtActivationType.
    /// 表示 TensorRtActivationType 的 SoftSign 取值。
    /// </summary>
    SoftSign = 6,
    /// <summary>
    /// Represents the SoftPlus value of TensorRtActivationType.
    /// 表示 TensorRtActivationType 的 SoftPlus 取值。
    /// </summary>
    SoftPlus = 7,
    /// <summary>
    /// Represents the Clip value of TensorRtActivationType.
    /// 表示 TensorRtActivationType 的 Clip 取值。
    /// </summary>
    Clip = 8,
    /// <summary>
    /// Represents the HardSigmoid value of TensorRtActivationType.
    /// 表示 TensorRtActivationType 的 HardSigmoid 取值。
    /// </summary>
    HardSigmoid = 9,
    /// <summary>
    /// Represents the ScaledTanh value of TensorRtActivationType.
    /// 表示 TensorRtActivationType 的 ScaledTanh 取值。
    /// </summary>
    ScaledTanh = 10,
    /// <summary>
    /// Represents the ThresholdedRelu value of TensorRtActivationType.
    /// 表示 TensorRtActivationType 的 ThresholdedRelu 取值。
    /// </summary>
    ThresholdedRelu = 11,
    /// <summary>
    /// Represents the GeluErf value of TensorRtActivationType.
    /// 表示 TensorRtActivationType 的 GeluErf 取值。
    /// </summary>
    GeluErf = 12,
    /// <summary>
    /// Represents the GeluTanh value of TensorRtActivationType.
    /// 表示 TensorRtActivationType 的 GeluTanh 取值。
    /// </summary>
    GeluTanh = 13
}

/// <summary>
/// Represents TensorRT TensorRtPoolingType values.
/// 表示 TensorRT TensorRtPoolingType 枚举值。
/// </summary>
public enum TensorRtPoolingType
{
    /// <summary>
    /// Represents the Max value of TensorRtPoolingType.
    /// 表示 TensorRtPoolingType 的 Max 取值。
    /// </summary>
    Max = 0,
    /// <summary>
    /// Represents the Average value of TensorRtPoolingType.
    /// 表示 TensorRtPoolingType 的 Average 取值。
    /// </summary>
    Average = 1,
    /// <summary>
    /// Represents the MaxAverageBlend value of TensorRtPoolingType.
    /// 表示 TensorRtPoolingType 的 MaxAverageBlend 取值。
    /// </summary>
    MaxAverageBlend = 2
}

/// <summary>
/// Represents TensorRT TensorRtScaleMode values.
/// 表示 TensorRT TensorRtScaleMode 枚举值。
/// </summary>
public enum TensorRtScaleMode
{
    /// <summary>
    /// Represents the Uniform value of TensorRtScaleMode.
    /// 表示 TensorRtScaleMode 的 Uniform 取值。
    /// </summary>
    Uniform = 0,
    /// <summary>
    /// Represents the Channel value of TensorRtScaleMode.
    /// 表示 TensorRtScaleMode 的 Channel 取值。
    /// </summary>
    Channel = 1,
    /// <summary>
    /// Represents the ElementWise value of TensorRtScaleMode.
    /// 表示 TensorRtScaleMode 的 ElementWise 取值。
    /// </summary>
    ElementWise = 2
}

/// <summary>
/// Represents TensorRT TensorRtPaddingMode values.
/// 表示 TensorRT TensorRtPaddingMode 枚举值。
/// </summary>
public enum TensorRtPaddingMode
{
    /// <summary>
    /// Represents the ExplicitRoundDown value of TensorRtPaddingMode.
    /// 表示 TensorRtPaddingMode 的 ExplicitRoundDown 取值。
    /// </summary>
    ExplicitRoundDown = 0,
    /// <summary>
    /// Represents the ExplicitRoundUp value of TensorRtPaddingMode.
    /// 表示 TensorRtPaddingMode 的 ExplicitRoundUp 取值。
    /// </summary>
    ExplicitRoundUp = 1,
    /// <summary>
    /// Represents the SameUpper value of TensorRtPaddingMode.
    /// 表示 TensorRtPaddingMode 的 SameUpper 取值。
    /// </summary>
    SameUpper = 2,
    /// <summary>
    /// Represents the SameLower value of TensorRtPaddingMode.
    /// 表示 TensorRtPaddingMode 的 SameLower 取值。
    /// </summary>
    SameLower = 3
}

/// <summary>
/// Represents TensorRT TensorRtSliceMode values.
/// 表示 TensorRT TensorRtSliceMode 枚举值。
/// </summary>
public enum TensorRtSliceMode
{
    /// <summary>
    /// Represents the StrictBounds value of TensorRtSliceMode.
    /// 表示 TensorRtSliceMode 的 StrictBounds 取值。
    /// </summary>
    StrictBounds = 0,
    /// <summary>
    /// Represents the Wrap value of TensorRtSliceMode.
    /// 表示 TensorRtSliceMode 的 Wrap 取值。
    /// </summary>
    Wrap = 1,
    /// <summary>
    /// Represents the Clamp value of TensorRtSliceMode.
    /// 表示 TensorRtSliceMode 的 Clamp 取值。
    /// </summary>
    Clamp = 2,
    /// <summary>
    /// Represents the Fill value of TensorRtSliceMode.
    /// 表示 TensorRtSliceMode 的 Fill 取值。
    /// </summary>
    Fill = 3,
    /// <summary>
    /// Represents the Reflect value of TensorRtSliceMode.
    /// 表示 TensorRtSliceMode 的 Reflect 取值。
    /// </summary>
    Reflect = 4
}

/// <summary>
/// Controls how TensorRT samples out-of-bounds coordinates.
/// 控制 TensorRT 对越界坐标的采样策略。
/// </summary>
public enum TensorRtSampleMode
{
    /// <summary>
    /// Fail when coordinates are out of bounds.
    /// 坐标越界时失败。
    /// </summary>
    StrictBounds = 0,

    /// <summary>
    /// Wrap coordinates periodically.
    /// 周期性回绕坐标。
    /// </summary>
    Wrap = 1,

    /// <summary>
    /// Clamp out-of-bounds coordinates to the valid range.
    /// 将越界坐标钳制到有效范围。
    /// </summary>
    Clamp = 2,

    /// <summary>
    /// Use the configured fill value for out-of-bounds coordinates.
    /// 越界坐标使用填充值。
    /// </summary>
    Fill = 3,

    /// <summary>
    /// Reflect coordinates at the tensor boundary.
    /// 在张量边界处反射坐标。
    /// </summary>
    Reflect = 4
}

/// <summary>
/// Represents TensorRT TensorRtFillOperation values.
/// 表示 TensorRT TensorRtFillOperation 枚举值。
/// </summary>
public enum TensorRtFillOperation
{
    /// <summary>
    /// Represents the Linspace value of TensorRtFillOperation.
    /// 表示 TensorRtFillOperation 的 Linspace 取值。
    /// </summary>
    Linspace = 0,
    /// <summary>
    /// Represents the RandomUniform value of TensorRtFillOperation.
    /// 表示 TensorRtFillOperation 的 RandomUniform 取值。
    /// </summary>
    RandomUniform = 1,
    /// <summary>
    /// Represents the RandomNormal value of TensorRtFillOperation.
    /// 表示 TensorRtFillOperation 的 RandomNormal 取值。
    /// </summary>
    RandomNormal = 2
}

/// <summary>
/// Represents TensorRT TensorRtTopKOperation values.
/// 表示 TensorRT TensorRtTopKOperation 枚举值。
/// </summary>
public enum TensorRtTopKOperation
{
    /// <summary>
    /// Represents the Max value of TensorRtTopKOperation.
    /// 表示 TensorRtTopKOperation 的 Max 取值。
    /// </summary>
    Max = 0,
    /// <summary>
    /// Represents the Min value of TensorRtTopKOperation.
    /// 表示 TensorRtTopKOperation 的 Min 取值。
    /// </summary>
    Min = 1
}

/// <summary>
/// Selects the TensorRT gather semantics.
/// 选择 TensorRT gather 层的索引语义。
/// </summary>
public enum TensorRtGatherMode
{
    /// <summary>
    /// ONNX Gather-like mode.
    /// 类似 ONNX Gather 的模式。
    /// </summary>
    Default = 0,

    /// <summary>
    /// ONNX GatherElements-like mode.
    /// 类似 ONNX GatherElements 的模式。
    /// </summary>
    Element = 1,

    /// <summary>
    /// ONNX GatherND-like mode.
    /// 类似 ONNX GatherND 的模式。
    /// </summary>
    Nd = 2
}

/// <summary>
/// Selects the TensorRT scatter semantics.
/// 选择 TensorRT scatter 层的写入语义。
/// </summary>
public enum TensorRtScatterMode
{
    /// <summary>
    /// ONNX ScatterElements-like mode.
    /// 类似 ONNX ScatterElements 的模式。
    /// </summary>
    Element = 0,

    /// <summary>
    /// ONNX ScatterND-like mode.
    /// 类似 ONNX ScatterND 的模式。
    /// </summary>
    Nd = 1
}

/// <summary>
/// Selects the TensorRT cumulative operation.
/// 选择 TensorRT cumulative 层的累计运算。
/// </summary>
public enum TensorRtCumulativeOperation
{
    /// <summary>
    /// Cumulative sum.
    /// 累计求和。
    /// </summary>
    Sum = 0
}

/// <summary>
/// Selects the collective communication primitive for a TensorRT 11 DistCollective layer.
/// 选择 TensorRT 11 DistCollective 层的集合通信原语。
/// </summary>
public enum TensorRtCollectiveOperation
{
    /// <summary>
    /// Represents the AllReduce value of TensorRtCollectiveOperation.
    /// 表示 TensorRtCollectiveOperation 的 AllReduce 取值。
    /// </summary>
    AllReduce = 0,
    /// <summary>
    /// Represents the AllGather value of TensorRtCollectiveOperation.
    /// 表示 TensorRtCollectiveOperation 的 AllGather 取值。
    /// </summary>
    AllGather = 1,
    /// <summary>
    /// Represents the Broadcast value of TensorRtCollectiveOperation.
    /// 表示 TensorRtCollectiveOperation 的 Broadcast 取值。
    /// </summary>
    Broadcast = 2,
    /// <summary>
    /// Represents the Reduce value of TensorRtCollectiveOperation.
    /// 表示 TensorRtCollectiveOperation 的 Reduce 取值。
    /// </summary>
    Reduce = 3,
    /// <summary>
    /// Represents the ReduceScatter value of TensorRtCollectiveOperation.
    /// 表示 TensorRtCollectiveOperation 的 ReduceScatter 取值。
    /// </summary>
    ReduceScatter = 4,
    /// <summary>
    /// Represents the AllToAll value of TensorRtCollectiveOperation.
    /// 表示 TensorRtCollectiveOperation 的 AllToAll 取值。
    /// </summary>
    AllToAll = 5,
    /// <summary>
    /// Represents the Gather value of TensorRtCollectiveOperation.
    /// 表示 TensorRtCollectiveOperation 的 Gather 取值。
    /// </summary>
    Gather = 6,
    /// <summary>
    /// Represents the Scatter value of TensorRtCollectiveOperation.
    /// 表示 TensorRtCollectiveOperation 的 Scatter 取值。
    /// </summary>
    Scatter = 7
}
