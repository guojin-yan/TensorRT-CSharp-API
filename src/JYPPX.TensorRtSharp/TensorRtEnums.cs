using System;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents TensorRT TensorRtNetworkDefinitionCreationFlags values.
/// 表示 TensorRT TensorRtNetworkDefinitionCreationFlags 枚举值。
/// </summary>
public enum TensorRtNetworkDefinitionCreationFlags : uint
{
    /// <summary>
    /// Represents the None value of TensorRtNetworkDefinitionCreationFlags.
    /// 表示 TensorRtNetworkDefinitionCreationFlags 的 None 取值。
    /// </summary>
    None = 0,
    /// <summary>
    /// Represents the ExplicitBatch value of TensorRtNetworkDefinitionCreationFlags.
    /// 表示 TensorRtNetworkDefinitionCreationFlags 的 ExplicitBatch 取值。
    /// </summary>
    ExplicitBatch = 1
}

/// <summary>
/// Represents TensorRT TensorRtDataType values.
/// 表示 TensorRT TensorRtDataType 枚举值。
/// </summary>
public enum TensorRtDataType
{
    /// <summary>
    /// Represents the Float value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Float 取值。
    /// </summary>
    Float = 0,
    /// <summary>
    /// Represents the Half value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Half 取值。
    /// </summary>
    Half = 1,
    /// <summary>
    /// Represents the Int8 value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Int8 取值。
    /// </summary>
    Int8 = 2,
    /// <summary>
    /// Represents the Int32 value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Int32 取值。
    /// </summary>
    Int32 = 3,
    /// <summary>
    /// Represents the Bool value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Bool 取值。
    /// </summary>
    Bool = 4,
    /// <summary>
    /// Represents the UInt8 value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 UInt8 取值。
    /// </summary>
    UInt8 = 5,
    /// <summary>
    /// Represents the Float8 value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Float8 取值。
    /// </summary>
    Float8 = 6,
    /// <summary>
    /// Represents the BFloat16 value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 BFloat16 取值。
    /// </summary>
    BFloat16 = 7,
    /// <summary>
    /// Represents the Int64 value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Int64 取值。
    /// </summary>
    Int64 = 8,
    /// <summary>
    /// Represents the Int4 value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Int4 取值。
    /// </summary>
    Int4 = 9,
    /// <summary>
    /// Represents the Float4 value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Float4 取值。
    /// </summary>
    Float4 = 10,
    /// <summary>
    /// Represents the E8M0 value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 E8M0 取值。
    /// </summary>
    E8M0 = 11,
    /// <summary>
    /// Represents the Unknown value of TensorRtDataType.
    /// 表示 TensorRtDataType 的 Unknown 取值。
    /// </summary>
    Unknown = 999
}

/// <summary>
/// Represents TensorRT TensorRtIOMode values.
/// 表示 TensorRT TensorRtIOMode 枚举值。
/// </summary>
public enum TensorRtIOMode
{
    /// <summary>
    /// Represents the Unknown value of TensorRtIOMode.
    /// 表示 TensorRtIOMode 的 Unknown 取值。
    /// </summary>
    Unknown = 0,
    /// <summary>
    /// Represents the Input value of TensorRtIOMode.
    /// 表示 TensorRtIOMode 的 Input 取值。
    /// </summary>
    Input = 1,
    /// <summary>
    /// Represents the Output value of TensorRtIOMode.
    /// 表示 TensorRtIOMode 的 Output 取值。
    /// </summary>
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

/// <summary>
/// Represents TensorRT TensorRtTensorLocation values.
/// 表示 TensorRT TensorRtTensorLocation 枚举值。
/// </summary>
public enum TensorRtTensorLocation
{
    /// <summary>
    /// Represents the Device value of TensorRtTensorLocation.
    /// 表示 TensorRtTensorLocation 的 Device 取值。
    /// </summary>
    Device = 0,
    /// <summary>
    /// Represents the Host value of TensorRtTensorLocation.
    /// 表示 TensorRtTensorLocation 的 Host 取值。
    /// </summary>
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

/// <summary>
/// Represents TensorRT TensorRtTensorFormat values.
/// 表示 TensorRT TensorRtTensorFormat 枚举值。
/// </summary>
public enum TensorRtTensorFormat
{
    /// <summary>
    /// Represents the Linear value of TensorRtTensorFormat.
    /// 表示 TensorRtTensorFormat 的 Linear 取值。
    /// </summary>
    Linear = 0,
    /// <summary>
    /// Represents the Chw2 value of TensorRtTensorFormat.
    /// 表示 TensorRtTensorFormat 的 Chw2 取值。
    /// </summary>
    Chw2 = 1,
    /// <summary>
    /// Represents the Hwc8 value of TensorRtTensorFormat.
    /// 表示 TensorRtTensorFormat 的 Hwc8 取值。
    /// </summary>
    Hwc8 = 2,
    /// <summary>
    /// Represents the Chw4 value of TensorRtTensorFormat.
    /// 表示 TensorRtTensorFormat 的 Chw4 取值。
    /// </summary>
    Chw4 = 3,
    /// <summary>
    /// Represents the Chw16 value of TensorRtTensorFormat.
    /// 表示 TensorRtTensorFormat 的 Chw16 取值。
    /// </summary>
    Chw16 = 4,
    /// <summary>
    /// Represents the Chw32 value of TensorRtTensorFormat.
    /// 表示 TensorRtTensorFormat 的 Chw32 取值。
    /// </summary>
    Chw32 = 5,
    /// <summary>
    /// Represents the Dhwc8 value of TensorRtTensorFormat.
    /// 表示 TensorRtTensorFormat 的 Dhwc8 取值。
    /// </summary>
    Dhwc8 = 6,
    /// <summary>
    /// Represents the Cdhw32 value of TensorRtTensorFormat.
    /// 表示 TensorRtTensorFormat 的 Cdhw32 取值。
    /// </summary>
    Cdhw32 = 7,
    /// <summary>
    /// Represents the Hwc value of TensorRtTensorFormat.
    /// 表示 TensorRtTensorFormat 的 Hwc 取值。
    /// </summary>
    Hwc = 8,
    /// <summary>
    /// Represents the DlaLinear value of TensorRtTensorFormat.
    /// 表示 TensorRtTensorFormat 的 DlaLinear 取值。
    /// </summary>
    DlaLinear = 9,
    /// <summary>
    /// Represents the DlaHwc4 value of TensorRtTensorFormat.
    /// 表示 TensorRtTensorFormat 的 DlaHwc4 取值。
    /// </summary>
    DlaHwc4 = 10,
    /// <summary>
    /// Represents the Hwc16 value of TensorRtTensorFormat.
    /// 表示 TensorRtTensorFormat 的 Hwc16 取值。
    /// </summary>
    Hwc16 = 11,
    /// <summary>
    /// Represents the Dhwc value of TensorRtTensorFormat.
    /// 表示 TensorRtTensorFormat 的 Dhwc 取值。
    /// </summary>
    Dhwc = 12,
    /// <summary>
    /// Represents the Unknown value of TensorRtTensorFormat.
    /// 表示 TensorRtTensorFormat 的 Unknown 取值。
    /// </summary>
    Unknown = 999
}

/// <summary>
/// Represents TensorRT TensorRtTensorFormats values.
/// 表示 TensorRT TensorRtTensorFormats 枚举值。
/// </summary>
[Flags]
public enum TensorRtTensorFormats : uint
{
    /// <summary>
    /// Represents the None value of TensorRtTensorFormats.
    /// 表示 TensorRtTensorFormats 的 None 取值。
    /// </summary>
    None = 0,
    /// <summary>
    /// Represents the Linear value of TensorRtTensorFormats.
    /// 表示 TensorRtTensorFormats 的 Linear 取值。
    /// </summary>
    Linear = 1u << (int)TensorRtTensorFormat.Linear,
    /// <summary>
    /// Represents the Chw2 value of TensorRtTensorFormats.
    /// 表示 TensorRtTensorFormats 的 Chw2 取值。
    /// </summary>
    Chw2 = 1u << (int)TensorRtTensorFormat.Chw2,
    /// <summary>
    /// Represents the Hwc8 value of TensorRtTensorFormats.
    /// 表示 TensorRtTensorFormats 的 Hwc8 取值。
    /// </summary>
    Hwc8 = 1u << (int)TensorRtTensorFormat.Hwc8,
    /// <summary>
    /// Represents the Chw4 value of TensorRtTensorFormats.
    /// 表示 TensorRtTensorFormats 的 Chw4 取值。
    /// </summary>
    Chw4 = 1u << (int)TensorRtTensorFormat.Chw4,
    /// <summary>
    /// Represents the Chw16 value of TensorRtTensorFormats.
    /// 表示 TensorRtTensorFormats 的 Chw16 取值。
    /// </summary>
    Chw16 = 1u << (int)TensorRtTensorFormat.Chw16,
    /// <summary>
    /// Represents the Chw32 value of TensorRtTensorFormats.
    /// 表示 TensorRtTensorFormats 的 Chw32 取值。
    /// </summary>
    Chw32 = 1u << (int)TensorRtTensorFormat.Chw32,
    /// <summary>
    /// Represents the Dhwc8 value of TensorRtTensorFormats.
    /// 表示 TensorRtTensorFormats 的 Dhwc8 取值。
    /// </summary>
    Dhwc8 = 1u << (int)TensorRtTensorFormat.Dhwc8,
    /// <summary>
    /// Represents the Cdhw32 value of TensorRtTensorFormats.
    /// 表示 TensorRtTensorFormats 的 Cdhw32 取值。
    /// </summary>
    Cdhw32 = 1u << (int)TensorRtTensorFormat.Cdhw32,
    /// <summary>
    /// Represents the Hwc value of TensorRtTensorFormats.
    /// 表示 TensorRtTensorFormats 的 Hwc 取值。
    /// </summary>
    Hwc = 1u << (int)TensorRtTensorFormat.Hwc,
    /// <summary>
    /// Represents the DlaLinear value of TensorRtTensorFormats.
    /// 表示 TensorRtTensorFormats 的 DlaLinear 取值。
    /// </summary>
    DlaLinear = 1u << (int)TensorRtTensorFormat.DlaLinear,
    /// <summary>
    /// Represents the DlaHwc4 value of TensorRtTensorFormats.
    /// 表示 TensorRtTensorFormats 的 DlaHwc4 取值。
    /// </summary>
    DlaHwc4 = 1u << (int)TensorRtTensorFormat.DlaHwc4,
    /// <summary>
    /// Represents the Hwc16 value of TensorRtTensorFormats.
    /// 表示 TensorRtTensorFormats 的 Hwc16 取值。
    /// </summary>
    Hwc16 = 1u << (int)TensorRtTensorFormat.Hwc16,
    /// <summary>
    /// Represents the Dhwc value of TensorRtTensorFormats.
    /// 表示 TensorRtTensorFormats 的 Dhwc 取值。
    /// </summary>
    Dhwc = 1u << (int)TensorRtTensorFormat.Dhwc
}

/// <summary>
/// Represents TensorRT TensorRtEngineCapability values.
/// 表示 TensorRT TensorRtEngineCapability 枚举值。
/// </summary>
public enum TensorRtEngineCapability
{
    /// <summary>
    /// Represents the Standard value of TensorRtEngineCapability.
    /// 表示 TensorRtEngineCapability 的 Standard 取值。
    /// </summary>
    Standard = 0,
    /// <summary>
    /// Represents the Safety value of TensorRtEngineCapability.
    /// 表示 TensorRtEngineCapability 的 Safety 取值。
    /// </summary>
    Safety = 1,
    /// <summary>
    /// Represents the DlaStandalone value of TensorRtEngineCapability.
    /// 表示 TensorRtEngineCapability 的 DlaStandalone 取值。
    /// </summary>
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

/// <summary>
/// Represents TensorRT 8 RNNv2 operation kinds.
/// 表示 TensorRT 8 RNNv2 运算类型。
/// </summary>
public enum TensorRtRnnOperation
{
    /// <summary>
    /// Single-gate RNN with ReLU activation.
    /// 使用 ReLU 激活的单门 RNN。
    /// </summary>
    Relu = 0,

    /// <summary>
    /// Single-gate RNN with tanh activation.
    /// 使用 tanh 激活的单门 RNN。
    /// </summary>
    Tanh = 1,

    /// <summary>
    /// Four-gate LSTM network without peephole connections.
    /// 不含 peephole 连接的四门 LSTM 网络。
    /// </summary>
    Lstm = 2,

    /// <summary>
    /// Three-gate gated recurrent unit network.
    /// 三门 GRU 网络。
    /// </summary>
    Gru = 3
}

/// <summary>
/// Represents TensorRT 8 RNNv2 direction modes.
/// 表示 TensorRT 8 RNNv2 方向模式。
/// </summary>
public enum TensorRtRnnDirection
{
    /// <summary>
    /// Iterate from the first input to the last input.
    /// 从第一个输入迭代到最后一个输入。
    /// </summary>
    Unidirection = 0,

    /// <summary>
    /// Iterate in both directions and concatenate outputs.
    /// 双向迭代并拼接输出。
    /// </summary>
    Bidirection = 1
}

/// <summary>
/// Represents TensorRT 8 RNNv2 input modes.
/// 表示 TensorRT 8 RNNv2 输入模式。
/// </summary>
public enum TensorRtRnnInputMode
{
    /// <summary>
    /// Perform the normal matrix multiplication in the first recurrent layer.
    /// 在第一个 recurrent layer 执行常规矩阵乘法。
    /// </summary>
    Linear = 0,

    /// <summary>
    /// Skip the first recurrent layer input matrix multiplication.
    /// 跳过第一个 recurrent layer 的输入矩阵乘法。
    /// </summary>
    Skip = 1
}

/// <summary>
/// Represents an individual TensorRT 8 RNNv2 gate.
/// 表示 TensorRT 8 RNNv2 单个门类型。
/// </summary>
public enum TensorRtRnnGateType
{
    /// <summary>Input gate (i). 输入门（i）。</summary>
    Input = 0,

    /// <summary>Output gate (o). 输出门（o）。</summary>
    Output = 1,

    /// <summary>Forget gate (f). 遗忘门（f）。</summary>
    Forget = 2,

    /// <summary>Update gate (z). 更新门（z）。</summary>
    Update = 3,

    /// <summary>Reset gate (r). 重置门（r）。</summary>
    Reset = 4,

    /// <summary>Cell gate (c). 单元门（c）。</summary>
    Cell = 5,

    /// <summary>Hidden gate (h). 隐状态门（h）。</summary>
    Hidden = 6
}

/// <summary>
/// Represents TensorRT TensorRtDeviceType values.
/// 表示 TensorRT TensorRtDeviceType 枚举值。
/// </summary>
public enum TensorRtDeviceType
{
    /// <summary>
    /// Represents the Gpu value of TensorRtDeviceType.
    /// 表示 TensorRtDeviceType 的 Gpu 取值。
    /// </summary>
    Gpu = 0,
    /// <summary>
    /// Represents the Dla value of TensorRtDeviceType.
    /// 表示 TensorRtDeviceType 的 Dla 取值。
    /// </summary>
    Dla = 1
}

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
/// Represents TensorRT TensorRtResizeMode values.
/// 表示 TensorRT TensorRtResizeMode 枚举值。
/// </summary>
public enum TensorRtResizeMode
{
    /// <summary>
    /// Represents the Nearest value of TensorRtResizeMode.
    /// 表示 TensorRtResizeMode 的 Nearest 取值。
    /// </summary>
    Nearest = 0,
    /// <summary>
    /// Represents the Linear value of TensorRtResizeMode.
    /// 表示 TensorRtResizeMode 的 Linear 取值。
    /// </summary>
    Linear = 1,
    /// <summary>
    /// Represents the Cubic value of TensorRtResizeMode.
    /// 表示 TensorRtResizeMode 的 Cubic 取值。
    /// </summary>
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

/// <summary>
/// Represents TensorRT TensorRtResizeCoordinateTransformation values.
/// 表示 TensorRT TensorRtResizeCoordinateTransformation 枚举值。
/// </summary>
public enum TensorRtResizeCoordinateTransformation
{
    /// <summary>
    /// Represents the AlignCorners value of TensorRtResizeCoordinateTransformation.
    /// 表示 TensorRtResizeCoordinateTransformation 的 AlignCorners 取值。
    /// </summary>
    AlignCorners = 0,
    /// <summary>
    /// Represents the Asymmetric value of TensorRtResizeCoordinateTransformation.
    /// 表示 TensorRtResizeCoordinateTransformation 的 Asymmetric 取值。
    /// </summary>
    Asymmetric = 1,
    /// <summary>
    /// Represents the HalfPixel value of TensorRtResizeCoordinateTransformation.
    /// 表示 TensorRtResizeCoordinateTransformation 的 HalfPixel 取值。
    /// </summary>
    HalfPixel = 2
}

/// <summary>
/// Represents TensorRT TensorRtResizeSelector values.
/// 表示 TensorRT TensorRtResizeSelector 枚举值。
/// </summary>
public enum TensorRtResizeSelector
{
    /// <summary>
    /// Represents the Formula value of TensorRtResizeSelector.
    /// 表示 TensorRtResizeSelector 的 Formula 取值。
    /// </summary>
    Formula = 0,
    /// <summary>
    /// Represents the Upper value of TensorRtResizeSelector.
    /// 表示 TensorRtResizeSelector 的 Upper 取值。
    /// </summary>
    Upper = 1
}

/// <summary>
/// Represents TensorRT TensorRtResizeRoundMode values.
/// 表示 TensorRT TensorRtResizeRoundMode 枚举值。
/// </summary>
public enum TensorRtResizeRoundMode
{
    /// <summary>
    /// Represents the HalfUp value of TensorRtResizeRoundMode.
    /// 表示 TensorRtResizeRoundMode 的 HalfUp 取值。
    /// </summary>
    HalfUp = 0,
    /// <summary>
    /// Represents the HalfDown value of TensorRtResizeRoundMode.
    /// 表示 TensorRtResizeRoundMode 的 HalfDown 取值。
    /// </summary>
    HalfDown = 1,
    /// <summary>
    /// Represents the Floor value of TensorRtResizeRoundMode.
    /// 表示 TensorRtResizeRoundMode 的 Floor 取值。
    /// </summary>
    Floor = 2,
    /// <summary>
    /// Represents the Ceil value of TensorRtResizeRoundMode.
    /// 表示 TensorRtResizeRoundMode 的 Ceil 取值。
    /// </summary>
    Ceil = 3
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

/// <summary>
/// Selects TensorRT 8/10 quantization calibration behavior.
/// 选择 TensorRT 8/10 quantization calibration 行为。
/// </summary>
public enum TensorRtQuantizationFlag
{
    /// <summary>
    /// Run the INT8 calibration pass before layer fusion.
    /// 在 layer fusion 前运行 INT8 calibration pass。
    /// </summary>
    CalibrateBeforeFusion = 0
}

/// <summary>
/// Represents a TensorRT 8/10 quantization flag bitmask.
/// 表示 TensorRT 8/10 quantization flag 位掩码。
/// </summary>
[Flags]
public enum TensorRtQuantizationFlags : uint
{
    /// <summary>
    /// No quantization flags are enabled.
    /// 不启用任何 quantization flag。
    /// </summary>
    None = 0,

    /// <summary>
    /// Run the INT8 calibration pass before layer fusion.
    /// 在 layer fusion 前运行 INT8 calibration pass。
    /// </summary>
    CalibrateBeforeFusion = 1u << (int)TensorRtQuantizationFlag.CalibrateBeforeFusion
}

/// <summary>
/// Represents TensorRT TensorRtTacticSources values.
/// 表示 TensorRT TensorRtTacticSources 枚举值。
/// </summary>
[Flags]
public enum TensorRtTacticSources : uint
{
    /// <summary>
    /// Represents the None value of TensorRtTacticSources.
    /// 表示 TensorRtTacticSources 的 None 取值。
    /// </summary>
    None = 0,
    /// <summary>
    /// Represents the CuBlas value of TensorRtTacticSources.
    /// 表示 TensorRtTacticSources 的 CuBlas 取值。
    /// </summary>
    CuBlas = 1u << 0,
    /// <summary>
    /// Represents the CuBlasLt value of TensorRtTacticSources.
    /// 表示 TensorRtTacticSources 的 CuBlasLt 取值。
    /// </summary>
    CuBlasLt = 1u << 1,
    /// <summary>
    /// Represents the CuDnn value of TensorRtTacticSources.
    /// 表示 TensorRtTacticSources 的 CuDnn 取值。
    /// </summary>
    CuDnn = 1u << 2,
    /// <summary>
    /// Represents the EdgeMaskConvolutions value of TensorRtTacticSources.
    /// 表示 TensorRtTacticSources 的 EdgeMaskConvolutions 取值。
    /// </summary>
    EdgeMaskConvolutions = 1u << 3,
    /// <summary>
    /// Represents the JitConvolutions value of TensorRtTacticSources.
    /// 表示 TensorRtTacticSources 的 JitConvolutions 取值。
    /// </summary>
    JitConvolutions = 1u << 4
}

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

/// <summary>
/// Represents TensorRT TensorRtOptimizationProfileSelector values.
/// 表示 TensorRT TensorRtOptimizationProfileSelector 枚举值。
/// </summary>
public enum TensorRtOptimizationProfileSelector
{
    /// <summary>
    /// Represents the Min value of TensorRtOptimizationProfileSelector.
    /// 表示 TensorRtOptimizationProfileSelector 的 Min 取值。
    /// </summary>
    Min = 0,
    /// <summary>
    /// Represents the Opt value of TensorRtOptimizationProfileSelector.
    /// 表示 TensorRtOptimizationProfileSelector 的 Opt 取值。
    /// </summary>
    Opt = 1,
    /// <summary>
    /// Represents the Max value of TensorRtOptimizationProfileSelector.
    /// 表示 TensorRtOptimizationProfileSelector 的 Max 取值。
    /// </summary>
    Max = 2
}

/// <summary>
/// Represents TensorRT TensorRtLayerInformationFormat values.
/// 表示 TensorRT TensorRtLayerInformationFormat 枚举值。
/// </summary>
public enum TensorRtLayerInformationFormat
{
    /// <summary>
    /// Represents the Oneline value of TensorRtLayerInformationFormat.
    /// 表示 TensorRtLayerInformationFormat 的 Oneline 取值。
    /// </summary>
    Oneline = 0,
    /// <summary>
    /// Represents the Json value of TensorRtLayerInformationFormat.
    /// 表示 TensorRtLayerInformationFormat 的 Json 取值。
    /// </summary>
    Json = 1
}

/// <summary>
/// Describes the TensorRT KV cache update mode.
/// 描述 TensorRT KV cache update 模式。
/// </summary>
public enum TensorRtKvCacheMode
{
    /// <summary>
    /// Represents the Linear value of TensorRtKvCacheMode.
    /// 表示 TensorRtKvCacheMode 的 Linear 取值。
    /// </summary>
    Linear = 0
}

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

/// <summary>
/// Represents TensorRT TensorRtProfilingVerbosity values.
/// 表示 TensorRT TensorRtProfilingVerbosity 枚举值。
/// </summary>
public enum TensorRtProfilingVerbosity
{
    /// <summary>
    /// Represents the LayerNamesOnly value of TensorRtProfilingVerbosity.
    /// 表示 TensorRtProfilingVerbosity 的 LayerNamesOnly 取值。
    /// </summary>
    LayerNamesOnly = 0,
    /// <summary>
    /// Represents the None value of TensorRtProfilingVerbosity.
    /// 表示 TensorRtProfilingVerbosity 的 None 取值。
    /// </summary>
    None = 1,
    /// <summary>
    /// Represents the Detailed value of TensorRtProfilingVerbosity.
    /// 表示 TensorRtProfilingVerbosity 的 Detailed 取值。
    /// </summary>
    Detailed = 2
}

/// <summary>
/// Represents TensorRT TensorRtBuilderFlag values.
/// 表示 TensorRT TensorRtBuilderFlag 枚举值。
/// </summary>
public enum TensorRtBuilderFlag
{
    /// <summary>
    /// Represents the Fp16 value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 Fp16 取值。
    /// </summary>
    Fp16 = 0,
    /// <summary>
    /// Represents the Int8 value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 Int8 取值。
    /// </summary>
    Int8 = 1,
    /// <summary>
    /// Represents the Debug value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 Debug 取值。
    /// </summary>
    Debug = 2,
    /// <summary>
    /// Represents the GpuFallback value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 GpuFallback 取值。
    /// </summary>
    GpuFallback = 3,
    /// <summary>
    /// Represents the Refit value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 Refit 取值。
    /// </summary>
    Refit = 4,
    /// <summary>
    /// Represents the DisableTimingCache value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 DisableTimingCache 取值。
    /// </summary>
    DisableTimingCache = 5,
    /// <summary>
    /// Represents the Tf32 value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 Tf32 取值。
    /// </summary>
    Tf32 = 6,
    /// <summary>
    /// Represents the SparseWeights value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 SparseWeights 取值。
    /// </summary>
    SparseWeights = 7,
    /// <summary>
    /// Represents the SafetyScope value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 SafetyScope 取值。
    /// </summary>
    SafetyScope = 8,
    /// <summary>
    /// Represents the ObeyPrecisionConstraints value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 ObeyPrecisionConstraints 取值。
    /// </summary>
    ObeyPrecisionConstraints = 9,
    /// <summary>
    /// Represents the PreferPrecisionConstraints value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 PreferPrecisionConstraints 取值。
    /// </summary>
    PreferPrecisionConstraints = 10,
    /// <summary>
    /// Represents the DirectIO value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 DirectIO 取值。
    /// </summary>
    DirectIO = 11,
    /// <summary>
    /// Represents the RejectEmptyAlgorithms value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 RejectEmptyAlgorithms 取值。
    /// </summary>
    RejectEmptyAlgorithms = 12,
    /// <summary>
    /// Represents the VersionCompatible value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 VersionCompatible 取值。
    /// </summary>
    VersionCompatible = 13,
    /// <summary>
    /// Represents the ExcludeLeanRuntime value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 ExcludeLeanRuntime 取值。
    /// </summary>
    ExcludeLeanRuntime = 14,
    /// <summary>
    /// Represents the Fp8 value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 Fp8 取值。
    /// </summary>
    Fp8 = 15,
    /// <summary>
    /// Represents the ErrorOnTimingCacheMiss value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 ErrorOnTimingCacheMiss 取值。
    /// </summary>
    ErrorOnTimingCacheMiss = 16,
    /// <summary>
    /// Represents the Bf16 value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 Bf16 取值。
    /// </summary>
    Bf16 = 17,
    /// <summary>
    /// Represents the DisableCompilationCache value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 DisableCompilationCache 取值。
    /// </summary>
    DisableCompilationCache = 18,
    /// <summary>
    /// Represents the StripPlan value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 StripPlan 取值。
    /// </summary>
    StripPlan = 19,
    /// <summary>
    /// Represents the RefitIdentical value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 RefitIdentical 取值。
    /// </summary>
    RefitIdentical = 20,
    /// <summary>
    /// Represents the WeightStreaming value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 WeightStreaming 取值。
    /// </summary>
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
    /// <summary>
    /// Represents the Fp16 value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 Fp16 取值。
    /// </summary>
    Fp16 = 1u << (int)TensorRtBuilderFlag.Fp16,
    /// <summary>
    /// Represents the Int8 value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 Int8 取值。
    /// </summary>
    Int8 = 1u << (int)TensorRtBuilderFlag.Int8,
    /// <summary>
    /// Represents the Debug value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 Debug 取值。
    /// </summary>
    Debug = 1u << (int)TensorRtBuilderFlag.Debug,
    /// <summary>
    /// Represents the GpuFallback value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 GpuFallback 取值。
    /// </summary>
    GpuFallback = 1u << (int)TensorRtBuilderFlag.GpuFallback,
    /// <summary>
    /// Represents the Refit value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 Refit 取值。
    /// </summary>
    Refit = 1u << (int)TensorRtBuilderFlag.Refit,
    /// <summary>
    /// Represents the DisableTimingCache value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 DisableTimingCache 取值。
    /// </summary>
    DisableTimingCache = 1u << (int)TensorRtBuilderFlag.DisableTimingCache,
    /// <summary>
    /// Represents the Tf32 value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 Tf32 取值。
    /// </summary>
    Tf32 = 1u << (int)TensorRtBuilderFlag.Tf32,
    /// <summary>
    /// Represents the SparseWeights value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 SparseWeights 取值。
    /// </summary>
    SparseWeights = 1u << (int)TensorRtBuilderFlag.SparseWeights,
    /// <summary>
    /// Represents the SafetyScope value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 SafetyScope 取值。
    /// </summary>
    SafetyScope = 1u << (int)TensorRtBuilderFlag.SafetyScope,
    /// <summary>
    /// Represents the ObeyPrecisionConstraints value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 ObeyPrecisionConstraints 取值。
    /// </summary>
    ObeyPrecisionConstraints = 1u << (int)TensorRtBuilderFlag.ObeyPrecisionConstraints,
    /// <summary>
    /// Represents the PreferPrecisionConstraints value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 PreferPrecisionConstraints 取值。
    /// </summary>
    PreferPrecisionConstraints = 1u << (int)TensorRtBuilderFlag.PreferPrecisionConstraints,
    /// <summary>
    /// Represents the DirectIO value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 DirectIO 取值。
    /// </summary>
    DirectIO = 1u << (int)TensorRtBuilderFlag.DirectIO,
    /// <summary>
    /// Represents the RejectEmptyAlgorithms value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 RejectEmptyAlgorithms 取值。
    /// </summary>
    RejectEmptyAlgorithms = 1u << (int)TensorRtBuilderFlag.RejectEmptyAlgorithms,
    /// <summary>
    /// Represents the VersionCompatible value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 VersionCompatible 取值。
    /// </summary>
    VersionCompatible = 1u << (int)TensorRtBuilderFlag.VersionCompatible,
    /// <summary>
    /// Represents the ExcludeLeanRuntime value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 ExcludeLeanRuntime 取值。
    /// </summary>
    ExcludeLeanRuntime = 1u << (int)TensorRtBuilderFlag.ExcludeLeanRuntime,
    /// <summary>
    /// Represents the Fp8 value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 Fp8 取值。
    /// </summary>
    Fp8 = 1u << (int)TensorRtBuilderFlag.Fp8,
    /// <summary>
    /// Represents the ErrorOnTimingCacheMiss value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 ErrorOnTimingCacheMiss 取值。
    /// </summary>
    ErrorOnTimingCacheMiss = 1u << (int)TensorRtBuilderFlag.ErrorOnTimingCacheMiss,
    /// <summary>
    /// Represents the Bf16 value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 Bf16 取值。
    /// </summary>
    Bf16 = 1u << (int)TensorRtBuilderFlag.Bf16,
    /// <summary>
    /// Represents the DisableCompilationCache value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 DisableCompilationCache 取值。
    /// </summary>
    DisableCompilationCache = 1u << (int)TensorRtBuilderFlag.DisableCompilationCache,
    /// <summary>
    /// Represents the StripPlan value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 StripPlan 取值。
    /// </summary>
    StripPlan = 1u << (int)TensorRtBuilderFlag.StripPlan,
    /// <summary>
    /// Represents the RefitIdentical value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 RefitIdentical 取值。
    /// </summary>
    RefitIdentical = 1u << (int)TensorRtBuilderFlag.RefitIdentical,
    /// <summary>
    /// Represents the WeightStreaming value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 WeightStreaming 取值。
    /// </summary>
    WeightStreaming = 1u << (int)TensorRtBuilderFlag.WeightStreaming
}

/// <summary>
/// Represents TensorRT TensorRtMemoryPoolType values.
/// 表示 TensorRT TensorRtMemoryPoolType 枚举值。
/// </summary>
public enum TensorRtMemoryPoolType
{
    /// <summary>
    /// Represents the Workspace value of TensorRtMemoryPoolType.
    /// 表示 TensorRtMemoryPoolType 的 Workspace 取值。
    /// </summary>
    Workspace = 0,
    /// <summary>
    /// Represents the DlaManagedSram value of TensorRtMemoryPoolType.
    /// 表示 TensorRtMemoryPoolType 的 DlaManagedSram 取值。
    /// </summary>
    DlaManagedSram = 1,
    /// <summary>
    /// Represents the DlaLocalDram value of TensorRtMemoryPoolType.
    /// 表示 TensorRtMemoryPoolType 的 DlaLocalDram 取值。
    /// </summary>
    DlaLocalDram = 2,
    /// <summary>
    /// Represents the DlaGlobalDram value of TensorRtMemoryPoolType.
    /// 表示 TensorRtMemoryPoolType 的 DlaGlobalDram 取值。
    /// </summary>
    DlaGlobalDram = 3,
    /// <summary>
    /// Represents the TacticDram value of TensorRtMemoryPoolType.
    /// 表示 TensorRtMemoryPoolType 的 TacticDram 取值。
    /// </summary>
    TacticDram = 4,
    /// <summary>
    /// Represents the TacticSharedMemory value of TensorRtMemoryPoolType.
    /// 表示 TensorRtMemoryPoolType 的 TacticSharedMemory 取值。
    /// </summary>
    TacticSharedMemory = 5
}
