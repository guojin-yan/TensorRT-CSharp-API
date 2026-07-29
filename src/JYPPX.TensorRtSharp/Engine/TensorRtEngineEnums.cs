using System;

namespace JYPPX.TensorRtSharp;

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
