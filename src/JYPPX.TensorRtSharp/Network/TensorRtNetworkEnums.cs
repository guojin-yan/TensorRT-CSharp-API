using System;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents TensorRT TensorRtNetworkDefinitionCreationFlags values.
/// 表示 TensorRT TensorRtNetworkDefinitionCreationFlags 枚举值。
/// </summary>
[Flags]
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
    ExplicitBatch = 1u << 0,
    /// <summary>
    /// Represents the TensorRT 10 raw strongly-typed creation bit.
    /// 表示仅适用于 TensorRT 10 的 strongly typed 原始创建位。
    /// </summary>
    /// <remarks>
    /// TensorRT 11 uses a different enum position and always creates strongly typed networks. Prefer <c>TensorRtBuilder.CreateNetwork(bool)</c> for cross-version code.
    /// TensorRT 11 使用不同的枚举位置且始终创建 strongly typed network；跨版本代码应优先使用 <c>TensorRtBuilder.CreateNetwork(bool)</c>。
    /// </remarks>
    StronglyTypedTensorRt10 = 1u << 1
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
