using System;

namespace JYPPX.TensorRtSharp;

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
