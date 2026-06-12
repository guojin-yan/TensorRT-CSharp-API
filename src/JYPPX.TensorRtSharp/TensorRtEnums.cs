using System;

namespace JYPPX.TensorRtSharp;

public enum TensorRtNetworkDefinitionCreationFlags : uint
{
    None = 0,
    ExplicitBatch = 1
}

public enum TensorRtDataType
{
    Float = 0,
    Half = 1,
    Int8 = 2,
    Int32 = 3,
    Bool = 4,
    UInt8 = 5,
    Float8 = 6,
    BFloat16 = 7,
    Int64 = 8,
    Int4 = 9,
    Float4 = 10,
    E8M0 = 11,
    Unknown = 999
}

public enum TensorRtIOMode
{
    Unknown = 0,
    Input = 1,
    Output = 2
}

/// <summary>
/// Represents a bitmask of ONNX parser flags.
/// 表示 ONNX parser 标志位掩码。
/// </summary>
[Flags]
public enum TensorRtOnnxParserFlags : uint
{
    /// <summary>
    /// No parser flags are enabled.
    /// 不启用任何 parser 标志。
    /// </summary>
    None = 0,

    /// <summary>
    /// Prefer TensorRT native instance-normalization handling when supported.
    /// 在支持时优先使用 TensorRT 原生 instance normalization 处理。
    /// </summary>
    NativeInstanceNormalization = 1u << (int)TensorRtOnnxParserFlag.NativeInstanceNormalization,

    /// <summary>
    /// Enables TensorRT 10 parser handling for UInt8 and asymmetric quantization on DLA.
    /// 启用 TensorRT 10 parser 对 DLA UInt8 和非对称量化的处理。
    /// </summary>
    EnableUInt8AndAsymmetricQuantizationDla = 1u << (int)TensorRtOnnxParserFlag.EnableUInt8AndAsymmetricQuantizationDla
}

/// <summary>
/// Identifies an individual ONNX parser flag.
/// 标识单个 ONNX parser 标志。
/// </summary>
public enum TensorRtOnnxParserFlag
{
    /// <summary>
    /// Native instance-normalization parser flag.
    /// 原生 instance normalization parser 标志。
    /// </summary>
    NativeInstanceNormalization = 0,

    /// <summary>
    /// TensorRT 10 UInt8 and asymmetric quantization DLA parser flag.
    /// TensorRT 10 的 UInt8 与 DLA 非对称量化 parser 标志。
    /// </summary>
    EnableUInt8AndAsymmetricQuantizationDla = 1
}

public enum TensorRtTensorLocation
{
    Device = 0,
    Host = 1
}

/// <summary>
/// Selects how TensorRT 11 allocates execution-context device memory.
/// 选择 TensorRT 11 execution context 的设备内存分配策略。
/// </summary>
public enum TensorRtExecutionContextAllocationStrategy
{
    /// <summary>
    /// Allocate statically for the maximum requirement across profiles.
    /// 按所有 profile 中的最大需求静态分配。
    /// </summary>
    Static = 0,

    /// <summary>
    /// Reallocate when the active optimization profile changes.
    /// 当 active optimization profile 改变时重新分配。
    /// </summary>
    OnProfileChange = 1,

    /// <summary>
    /// The application provides device memory explicitly.
    /// 由应用程序显式提供 device memory。
    /// </summary>
    UserManaged = 2
}

/// <summary>
/// Identifies TensorRT 11 serialization flags.
/// 标识 TensorRT 11 serialization config 的单个标志。
/// </summary>
public enum TensorRtSerializationFlag
{
    /// <summary>
    /// Exclude refittable weights from the serialized plan.
    /// 从序列化 plan 中排除可 refit 权重。
    /// </summary>
    ExcludeWeights = 0,

    /// <summary>
    /// Exclude the lean runtime from the serialized plan.
    /// 从序列化 plan 中排除 lean runtime。
    /// </summary>
    ExcludeLeanRuntime = 1,

    /// <summary>
    /// Keep refit metadata in the serialized plan when supported.
    /// 在支持时保留 refit 元数据。
    /// </summary>
    IncludeRefit = 2
}

/// <summary>
/// Represents TensorRT 11 serialization flags as a bitmask.
/// 以位掩码形式表示 TensorRT 11 serialization flags。
/// </summary>
[Flags]
public enum TensorRtSerializationFlags : uint
{
    /// <summary>
    /// No serialization flags are set.
    /// 不设置任何 serialization flag。
    /// </summary>
    None = 0,

    /// <summary>
    /// Exclude refittable weights from the serialized plan.
    /// 从序列化 plan 中排除可 refit 权重。
    /// </summary>
    ExcludeWeights = 1u << (int)TensorRtSerializationFlag.ExcludeWeights,

    /// <summary>
    /// Exclude the lean runtime from the serialized plan.
    /// 从序列化 plan 中排除 lean runtime。
    /// </summary>
    ExcludeLeanRuntime = 1u << (int)TensorRtSerializationFlag.ExcludeLeanRuntime,

    /// <summary>
    /// Keep refit metadata in the serialized plan when supported.
    /// 在支持时保留 refit 元数据。
    /// </summary>
    IncludeRefit = 1u << (int)TensorRtSerializationFlag.IncludeRefit
}

/// <summary>
/// Identifies TensorRT runtime temporary-file control flags.
/// 标识 TensorRT runtime 临时文件控制标志。
/// </summary>
public enum TensorRtTempfileControlFlag
{
    /// <summary>
    /// Allow TensorRT to use in-memory or unnamed temporary files.
    /// 允许 TensorRT 使用内存内或未命名临时文件。
    /// </summary>
    AllowInMemoryFiles = 0,

    /// <summary>
    /// Allow TensorRT to create regular temporary files.
    /// 允许 TensorRT 创建普通临时文件。
    /// </summary>
    AllowTemporaryFiles = 1
}

/// <summary>
/// Represents TensorRT runtime temporary-file control flags as a bitmask.
/// 以位掩码形式表示 TensorRT runtime 临时文件控制标志。
/// </summary>
[Flags]
public enum TensorRtTempfileControlFlags : uint
{
    /// <summary>
    /// No temporary-file mechanism is enabled.
    /// 不启用任何临时文件机制。
    /// </summary>
    None = 0,

    /// <summary>
    /// Allow TensorRT to use in-memory or unnamed temporary files.
    /// 允许 TensorRT 使用内存内或未命名临时文件。
    /// </summary>
    AllowInMemoryFiles = 1u << (int)TensorRtTempfileControlFlag.AllowInMemoryFiles,

    /// <summary>
    /// Allow TensorRT to create regular temporary files.
    /// 允许 TensorRT 创建普通临时文件。
    /// </summary>
    AllowTemporaryFiles = 1u << (int)TensorRtTempfileControlFlag.AllowTemporaryFiles,

    /// <summary>
    /// TensorRT default behavior: allow all tempfile mechanisms.
    /// TensorRT 默认行为：允许全部临时文件机制。
    /// </summary>
    All = AllowInMemoryFiles | AllowTemporaryFiles
}

public enum TensorRtTensorFormat
{
    Linear = 0,
    Chw2 = 1,
    Hwc8 = 2,
    Chw4 = 3,
    Chw16 = 4,
    Chw32 = 5,
    Dhwc8 = 6,
    Cdhw32 = 7,
    Hwc = 8,
    DlaLinear = 9,
    DlaHwc4 = 10,
    Hwc16 = 11,
    Dhwc = 12,
    Unknown = 999
}

[Flags]
public enum TensorRtTensorFormats : uint
{
    None = 0,
    Linear = 1u << (int)TensorRtTensorFormat.Linear,
    Chw2 = 1u << (int)TensorRtTensorFormat.Chw2,
    Hwc8 = 1u << (int)TensorRtTensorFormat.Hwc8,
    Chw4 = 1u << (int)TensorRtTensorFormat.Chw4,
    Chw16 = 1u << (int)TensorRtTensorFormat.Chw16,
    Chw32 = 1u << (int)TensorRtTensorFormat.Chw32,
    Dhwc8 = 1u << (int)TensorRtTensorFormat.Dhwc8,
    Cdhw32 = 1u << (int)TensorRtTensorFormat.Cdhw32,
    Hwc = 1u << (int)TensorRtTensorFormat.Hwc,
    DlaLinear = 1u << (int)TensorRtTensorFormat.DlaLinear,
    DlaHwc4 = 1u << (int)TensorRtTensorFormat.DlaHwc4,
    Hwc16 = 1u << (int)TensorRtTensorFormat.Hwc16,
    Dhwc = 1u << (int)TensorRtTensorFormat.Dhwc
}

public enum TensorRtEngineCapability
{
    Standard = 0,
    Safety = 1,
    DlaStandalone = 2
}

/// <summary>
/// Selects TensorRT builder preview features. Some values are valid only on specific TensorRT major versions.
/// 选择 TensorRT builder 预览特性；部分取值仅在特定 TensorRT 大版本中有效。
/// </summary>
public enum TensorRtPreviewFeature
{
    /// <summary>
    /// TensorRT 8.x dynamic-shape compiler preview feature. Deprecated by NVIDIA.
    /// TensorRT 8.x 动态 shape 编译器预览特性，NVIDIA 已标记为弃用。
    /// </summary>
    FasterDynamicShapes0805 = 0,

    /// <summary>
    /// TensorRT 10.x profile-sharing feature. Deprecated and always enabled by NVIDIA.
    /// TensorRT 10.x profile 共享特性；NVIDIA 已弃用并默认始终启用。
    /// </summary>
    ProfileSharing0806Trt10 = 0,

    /// <summary>
    /// TensorRT 8.x option that disables external library tactics for TensorRT core.
    /// TensorRT 8.x 中禁用 TensorRT core 外部库 tactics 的选项。
    /// </summary>
    DisableExternalTacticSourcesForCore0805 = 1,

    /// <summary>
    /// TensorRT 8.x profile-sharing feature. In TensorRT 10.x use <see cref="ProfileSharing0806Trt10"/> instead.
    /// TensorRT 8.x profile 共享特性；在 TensorRT 10.x 中请使用 <see cref="ProfileSharing0806Trt10"/>。
    /// </summary>
    ProfileSharing0806 = 2,

    /// <summary>
    /// TensorRT 10.x plugin I/O aliasing preview feature.
    /// TensorRT 10.x plugin I/O 别名预览特性。
    /// </summary>
    AliasedPluginIo1003 = 1,

    /// <summary>
    /// TensorRT 10.x runtime activation-memory resize preview feature.
    /// TensorRT 10.x 运行时 activation 内存 resize 预览特性。
    /// </summary>
    RuntimeActivationResize1010 = 2
}

/// <summary>
/// Describes TensorRT engine hardware-compatibility requirements.
/// 描述 TensorRT engine 的硬件兼容性要求。
/// </summary>
public enum TensorRtHardwareCompatibilityLevel
{
    /// <summary>
    /// No cross-GPU architecture compatibility requirement.
    /// 不要求跨 GPU 架构兼容。
    /// </summary>
    None = 0,

    /// <summary>
    /// Require compatibility with Ampere and newer GPUs.
    /// 要求兼容 Ampere 及更新 GPU。
    /// </summary>
    AmperePlus = 1,

    /// <summary>
    /// TensorRT 10.x only: require compatibility with GPUs of the same compute capability.
    /// 仅 TensorRT 10.x：要求兼容相同 compute capability 的 GPU。
    /// </summary>
    SameComputeCapability = 2
}

/// <summary>
/// Selects TensorRT 10.x cross-platform runtime target.
/// 选择 TensorRT 10.x 跨平台 runtime 目标。
/// </summary>
public enum TensorRtRuntimePlatform
{
    /// <summary>
    /// Engine can run only on the same platform family it was built for.
    /// Engine 仅面向构建时相同的平台家族运行。
    /// </summary>
    SameAsBuild = 0,

    /// <summary>
    /// TensorRT 10.x Windows AMD64 target. NVIDIA currently documents this as a Linux-build-to-Windows-target mode.
    /// TensorRT 10.x Windows AMD64 目标；NVIDIA 当前文档将其定义为 Linux 构建面向 Windows 运行的模式。
    /// </summary>
    WindowsAmd64 = 1
}

public enum TensorRtDeviceType
{
    Gpu = 0,
    Dla = 1
}

public enum TensorRtElementWiseOperation
{
    Sum = 0,
    Product = 1,
    Max = 2,
    Min = 3,
    Subtract = 4,
    Divide = 5,
    Power = 6,
    FloorDivide = 7,
    And = 8,
    Or = 9,
    Xor = 10,
    Equal = 11,
    Greater = 12,
    Less = 13
}

public enum TensorRtMatrixOperation
{
    None = 0,
    Transpose = 1,
    Vector = 2
}

public enum TensorRtReduceOperation
{
    Sum = 0,
    Product = 1,
    Max = 2,
    Min = 3,
    Average = 4
}

/// <summary>
/// Selects the distributed reduction operation for a TensorRT 11 DistCollective layer.
/// 选择 TensorRT 11 DistCollective 层使用的分布式归约操作。
/// </summary>
public enum TensorRtDistributedReduceOperation
{
    Sum = 0,
    Product = 1,
    Max = 2,
    Min = 3,
    Average = 4,
    None = 5
}

public enum TensorRtUnaryOperation
{
    Exp = 0,
    Log = 1,
    Sqrt = 2,
    Recip = 3,
    Abs = 4,
    Neg = 5,
    Sin = 6,
    Cos = 7,
    Tan = 8,
    Sinh = 9,
    Cosh = 10,
    Asin = 11,
    Acos = 12,
    Atan = 13,
    Asinh = 14,
    Acosh = 15,
    Atanh = 16,
    Ceil = 17,
    Floor = 18,
    Erf = 19,
    Not = 20,
    Sign = 21,
    Round = 22,
    IsInf = 23,
    IsNaN = 24
}

public enum TensorRtActivationType
{
    Relu = 0,
    Sigmoid = 1,
    Tanh = 2,
    LeakyRelu = 3,
    Elu = 4,
    Selu = 5,
    SoftSign = 6,
    SoftPlus = 7,
    Clip = 8,
    HardSigmoid = 9,
    ScaledTanh = 10,
    ThresholdedRelu = 11,
    GeluErf = 12,
    GeluTanh = 13
}

public enum TensorRtPoolingType
{
    Max = 0,
    Average = 1,
    MaxAverageBlend = 2
}

public enum TensorRtScaleMode
{
    Uniform = 0,
    Channel = 1,
    ElementWise = 2
}

public enum TensorRtPaddingMode
{
    ExplicitRoundDown = 0,
    ExplicitRoundUp = 1,
    SameUpper = 2,
    SameLower = 3
}

public enum TensorRtResizeMode
{
    Nearest = 0,
    Linear = 1,
    Cubic = 2
}

/// <summary>
/// Selects the TensorRT interpolation algorithm for resize and grid-sample style layers.
/// 选择 TensorRT resize / grid-sample 等层使用的插值算法。
/// </summary>
public enum TensorRtInterpolationMode
{
    /// <summary>
    /// Nearest-neighbor interpolation.
    /// 最近邻插值。
    /// </summary>
    Nearest = 0,

    /// <summary>
    /// Linear interpolation; TensorRT maps this to linear, bilinear, or trilinear by rank.
    /// 线性插值；TensorRT 会按张量维度映射为 linear、bilinear 或 trilinear。
    /// </summary>
    Linear = 1,

    /// <summary>
    /// Cubic interpolation.
    /// 三次插值。
    /// </summary>
    Cubic = 2
}

public enum TensorRtResizeCoordinateTransformation
{
    AlignCorners = 0,
    Asymmetric = 1,
    HalfPixel = 2
}

public enum TensorRtResizeSelector
{
    Formula = 0,
    Upper = 1
}

public enum TensorRtResizeRoundMode
{
    HalfUp = 0,
    HalfDown = 1,
    Floor = 2,
    Ceil = 3
}

public enum TensorRtSliceMode
{
    StrictBounds = 0,
    Wrap = 1,
    Clamp = 2,
    Fill = 3,
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

public enum TensorRtFillOperation
{
    Linspace = 0,
    RandomUniform = 1,
    RandomNormal = 2
}

public enum TensorRtTopKOperation
{
    Max = 0,
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
    AllReduce = 0,
    AllGather = 1,
    Broadcast = 2,
    Reduce = 3,
    ReduceScatter = 4,
    AllToAll = 5,
    Gather = 6,
    Scatter = 7
}

/// <summary>
/// Selects the output semantics of a TensorRT loop output layer.
/// 选择 TensorRT loop output 层的输出语义。
/// </summary>
public enum TensorRtLoopOutputKind
{
    /// <summary>
    /// Output the tensor value from the last loop iteration.
    /// 输出最后一次循环迭代的张量值。
    /// </summary>
    LastValue = 0,

    /// <summary>
    /// Concatenate values from all iterations in forward order.
    /// 按正向顺序拼接每次迭代的值。
    /// </summary>
    Concatenate = 1,

    /// <summary>
    /// Concatenate values from all iterations in reverse order.
    /// 按反向顺序拼接每次迭代的值。
    /// </summary>
    Reverse = 2
}

/// <summary>
/// Selects how TensorRT limits loop iteration count.
/// 选择 TensorRT 如何限制循环迭代次数。
/// </summary>
public enum TensorRtTripLimitKind
{
    /// <summary>
    /// A scalar Int32/Int64 tensor provides the maximum trip count.
    /// 使用 Int32/Int64 标量张量提供最大迭代次数。
    /// </summary>
    Count = 0,

    /// <summary>
    /// A scalar Bool tensor controls whether the loop should continue.
    /// 使用 Bool 标量张量控制循环是否继续。
    /// </summary>
    While = 1
}

/// <summary>
/// Selects an engine statistic reported by TensorRT.
/// 选择 TensorRT 返回的 engine 统计项。
/// </summary>
public enum TensorRtEngineStat
{
    /// <summary>
    /// Total engine weight size in bytes.
    /// Engine 权重总大小，单位为字节。
    /// </summary>
    TotalWeightsSize = 0,

    /// <summary>
    /// Stripped weight size in bytes for stripped-plan engines.
    /// Strip-plan engine 的剥离权重大小，单位为字节。
    /// </summary>
    StrippedWeightsSize = 1
}

/// <summary>
/// Selects TensorRT 11 tiling optimization effort.
/// 选择 TensorRT 11 tiling 优化搜索强度。
/// </summary>
public enum TensorRtTilingOptimizationLevel
{
    /// <summary>
    /// Disable tiling optimization.
    /// 禁用 tiling 优化。
    /// </summary>
    None = 0,

    /// <summary>
    /// Use a fast heuristic strategy.
    /// 使用快速启发式策略。
    /// </summary>
    Fast = 1,

    /// <summary>
    /// Use a moderate mixed heuristic/profiling strategy.
    /// 使用中等强度的启发式与 profiling 混合策略。
    /// </summary>
    Moderate = 2,

    /// <summary>
    /// Use the most exhaustive tiling search.
    /// 使用更完整的 tiling 搜索。
    /// </summary>
    Full = 3
}

[Flags]
public enum TensorRtTacticSources : uint
{
    None = 0,
    CuBlas = 1u << 0,
    CuBlasLt = 1u << 1,
    CuDnn = 1u << 2,
    EdgeMaskConvolutions = 1u << 3,
    JitConvolutions = 1u << 4
}

public enum TensorRtLayerType
{
    Unknown = -1,
    Convolution = 0,
    Activation = 2,
    Pooling = 3,
    Lrn = 4,
    Scale = 5,
    SoftMax = 6,
    Deconvolution = 7,
    Concatenation = 8,
    ElementWise = 9,
    Cast = 1,
    Plugin = 10,
    Unary = 11,
    Padding = 12,
    TopK = 15,
    Gather = 16,
    MatrixMultiply = 17,
    RaggedSoftMax = 18,
    Shuffle = 13,
    Reduce = 14,
    Constant = 19,
    IdentityTrt10 = 20,
    IdentityTrt8 = 100021,
    SliceTrt10 = 22,
    SliceTrt8 = 100023,
    ShapeTrt10 = 23,
    ShapeTrt8 = 100024,
    ParametricReLU = 24,
    ResizeTrt10 = 25,
    ResizeTrt8 = 100026,
    TripLimit = 26,
    Recurrence = 27,
    Iterator = 28,
    LoopOutput = 29,
    SelectTrt10 = 30,
    SelectTrt8 = 100031,
    FillTrt10 = 31,
    FillTrt8 = 100032,
    QuantizeTrt10 = 32,
    QuantizeTrt8 = 100033,
    DequantizeTrt10 = 33,
    DequantizeTrt8 = 100034,
    Condition = 34,
    ConditionalInput = 35,
    ConditionalOutput = 36,
    Scatter = 37,
    Einsum = 38,
    Assertion = 39,
    OneHot = 40,
    NonZero = 41,
    GridSample = 42,
    Nms = 43,
    ReverseSequence = 44,
    NormalizationTrt10 = 45,
    NormalizationTrt8 = 100046,
    Squeeze = 47,
    Unsqueeze = 48,
    Cumulative = 49,
    DynamicQuantize = 50,
    AttentionInput = 51,
    AttentionOutput = 52,
    RotaryEmbedding = 53,
    KvCacheUpdate = 54,
    Moe = 55,
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

public enum TensorRtOptimizationProfileSelector
{
    Min = 0,
    Opt = 1,
    Max = 2
}

public enum TensorRtLayerInformationFormat
{
    Oneline = 0,
    Json = 1
}

/// <summary>
/// Describes the TensorRT KV cache update mode.
/// 描述 TensorRT KV cache update 模式。
/// </summary>
public enum TensorRtKvCacheMode
{
    Linear = 0
}

/// <summary>
/// Describes TensorRT attention input/output tensor packing.
/// 描述 TensorRT attention 输入/输出张量排布方式。
/// </summary>
public enum TensorRtAttentionIoForm
{
    PaddedBhnd = 0,
    PackedNhd = 1
}

/// <summary>
/// Describes the normalization operation used inside TensorRT attention.
/// 描述 TensorRT attention 内部使用的归一化操作。
/// </summary>
public enum TensorRtAttentionNormalizationOperation
{
    None = 0,
    Softmax = 1
}

/// <summary>
/// Describes TensorRT attention causal-mask alignment.
/// 描述 TensorRT attention 因果 mask 的对齐方向。
/// </summary>
public enum TensorRtCausalMaskKind
{
    None = 0,
    UpperLeft = 1,
    LowerRight = 2
}

/// <summary>
/// Describes the activation used by a TensorRT MoE layer.
/// 描述 TensorRT MoE 层使用的激活函数。
/// </summary>
public enum TensorRtMoEActivationType
{
    None = 0,
    SiLU = 1
}

public enum TensorRtProfilingVerbosity
{
    LayerNamesOnly = 0,
    None = 1,
    Detailed = 2
}

public enum TensorRtBuilderFlag
{
    Fp16 = 0,
    Int8 = 1,
    Debug = 2,
    GpuFallback = 3,
    Refit = 4,
    DisableTimingCache = 5,
    Tf32 = 6,
    SparseWeights = 7,
    SafetyScope = 8,
    ObeyPrecisionConstraints = 9,
    PreferPrecisionConstraints = 10,
    DirectIO = 11,
    RejectEmptyAlgorithms = 12,
    VersionCompatible = 13,
    ExcludeLeanRuntime = 14,
    Fp8 = 15,
    ErrorOnTimingCacheMiss = 16,
    Bf16 = 17,
    DisableCompilationCache = 18,
    StripPlan = 19,
    RefitIdentical = 20,
    WeightStreaming = 21
}

/// <summary>
/// Represents a TensorRT builder flag bitmask.
/// 表示 TensorRT builder flag 位掩码。
/// </summary>
[Flags]
public enum TensorRtBuilderFlags : uint
{
    /// <summary>
    /// No builder flags are enabled.
    /// 不启用任何 builder flag。
    /// </summary>
    None = 0,
    Fp16 = 1u << (int)TensorRtBuilderFlag.Fp16,
    Int8 = 1u << (int)TensorRtBuilderFlag.Int8,
    Debug = 1u << (int)TensorRtBuilderFlag.Debug,
    GpuFallback = 1u << (int)TensorRtBuilderFlag.GpuFallback,
    Refit = 1u << (int)TensorRtBuilderFlag.Refit,
    DisableTimingCache = 1u << (int)TensorRtBuilderFlag.DisableTimingCache,
    Tf32 = 1u << (int)TensorRtBuilderFlag.Tf32,
    SparseWeights = 1u << (int)TensorRtBuilderFlag.SparseWeights,
    SafetyScope = 1u << (int)TensorRtBuilderFlag.SafetyScope,
    ObeyPrecisionConstraints = 1u << (int)TensorRtBuilderFlag.ObeyPrecisionConstraints,
    PreferPrecisionConstraints = 1u << (int)TensorRtBuilderFlag.PreferPrecisionConstraints,
    DirectIO = 1u << (int)TensorRtBuilderFlag.DirectIO,
    RejectEmptyAlgorithms = 1u << (int)TensorRtBuilderFlag.RejectEmptyAlgorithms,
    VersionCompatible = 1u << (int)TensorRtBuilderFlag.VersionCompatible,
    ExcludeLeanRuntime = 1u << (int)TensorRtBuilderFlag.ExcludeLeanRuntime,
    Fp8 = 1u << (int)TensorRtBuilderFlag.Fp8,
    ErrorOnTimingCacheMiss = 1u << (int)TensorRtBuilderFlag.ErrorOnTimingCacheMiss,
    Bf16 = 1u << (int)TensorRtBuilderFlag.Bf16,
    DisableCompilationCache = 1u << (int)TensorRtBuilderFlag.DisableCompilationCache,
    StripPlan = 1u << (int)TensorRtBuilderFlag.StripPlan,
    RefitIdentical = 1u << (int)TensorRtBuilderFlag.RefitIdentical,
    WeightStreaming = 1u << (int)TensorRtBuilderFlag.WeightStreaming
}

public enum TensorRtMemoryPoolType
{
    Workspace = 0,
    DlaManagedSram = 1,
    DlaLocalDram = 2,
    DlaGlobalDram = 3,
    TacticDram = 4,
    TacticSharedMemory = 5
}
