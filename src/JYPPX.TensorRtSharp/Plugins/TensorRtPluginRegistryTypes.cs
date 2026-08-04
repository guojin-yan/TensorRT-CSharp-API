using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// TensorRT plugin field data type reported by plugin creators.
/// TensorRT plugin creator 报告的字段数据类型。
/// </summary>
public enum TensorRtPluginFieldType
{
    /// <summary>
    /// Represents the Float16 value of TensorRtPluginFieldType.
    /// 表示 TensorRtPluginFieldType 的 Float16 取值。
    /// </summary>
    Float16 = 0,
    /// <summary>
    /// Represents the Float32 value of TensorRtPluginFieldType.
    /// 表示 TensorRtPluginFieldType 的 Float32 取值。
    /// </summary>
    Float32 = 1,
    /// <summary>
    /// Represents the Float64 value of TensorRtPluginFieldType.
    /// 表示 TensorRtPluginFieldType 的 Float64 取值。
    /// </summary>
    Float64 = 2,
    /// <summary>
    /// Represents the Int8 value of TensorRtPluginFieldType.
    /// 表示 TensorRtPluginFieldType 的 Int8 取值。
    /// </summary>
    Int8 = 3,
    /// <summary>
    /// Represents the Int16 value of TensorRtPluginFieldType.
    /// 表示 TensorRtPluginFieldType 的 Int16 取值。
    /// </summary>
    Int16 = 4,
    /// <summary>
    /// Represents the Int32 value of TensorRtPluginFieldType.
    /// 表示 TensorRtPluginFieldType 的 Int32 取值。
    /// </summary>
    Int32 = 5,
    /// <summary>
    /// Represents the Char value of TensorRtPluginFieldType.
    /// 表示 TensorRtPluginFieldType 的 Char 取值。
    /// </summary>
    Char = 6,
    /// <summary>
    /// Represents the Dims value of TensorRtPluginFieldType.
    /// 表示 TensorRtPluginFieldType 的 Dims 取值。
    /// </summary>
    Dims = 7,
    /// <summary>
    /// Represents the Unknown value of TensorRtPluginFieldType.
    /// 表示 TensorRtPluginFieldType 的 Unknown 取值。
    /// </summary>
    Unknown = 8,
    /// <summary>
    /// Represents the BFloat16 value of TensorRtPluginFieldType.
    /// 表示 TensorRtPluginFieldType 的 BFloat16 取值。
    /// </summary>
    BFloat16 = 9,
    /// <summary>
    /// Represents the Int64 value of TensorRtPluginFieldType.
    /// 表示 TensorRtPluginFieldType 的 Int64 取值。
    /// </summary>
    Int64 = 10,
    /// <summary>
    /// Represents the Float8 value of TensorRtPluginFieldType.
    /// 表示 TensorRtPluginFieldType 的 Float8 取值。
    /// </summary>
    Float8 = 11,
    /// <summary>
    /// Represents the Int4 value of TensorRtPluginFieldType.
    /// 表示 TensorRtPluginFieldType 的 Int4 取值。
    /// </summary>
    Int4 = 12,
    /// <summary>
    /// Represents the Float4 value of TensorRtPluginFieldType.
    /// 表示 TensorRtPluginFieldType 的 Float4 取值。
    /// </summary>
    Float4 = 13
}

/// <summary>
/// Identifies the TensorRT registry source used for a plugin creator inventory.
/// 标识 plugin creator inventory 使用的 TensorRT registry 来源。
/// </summary>
public enum TensorRtPluginRegistrySource
{
    /// <summary>
    /// The inventory was collected from a builder-visible registry.
    /// inventory 来自 builder 可见 registry。
    /// </summary>
    Builder = 0,

    /// <summary>
    /// The inventory was collected from TensorRT's global runtime registry.
    /// inventory 来自 TensorRT 全局 runtime registry。
    /// </summary>
    Global = 1,

    /// <summary>
    /// The inventory was collected from TensorRT's builder capability registry.
    /// inventory 来自 TensorRT builder capability registry。
    /// </summary>
    BuilderCapability = 2,

    /// <summary>
    /// The inventory was collected from a runtime-local plugin registry.
    /// inventory 来自 runtime-local plugin registry。
    /// </summary>
    Runtime = 3
}
