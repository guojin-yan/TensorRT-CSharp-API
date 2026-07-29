using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Read-only metadata for a TensorRT plugin creator field.
/// TensorRT plugin creator 字段的只读元数据。
/// </summary>
public sealed class TensorRtPluginFieldInfo
{
    internal TensorRtPluginFieldInfo(string name, TensorRtPluginFieldType fieldType, int length, bool hasData)
    {
        Name = name ?? string.Empty;
        FieldType = fieldType;
        Length = length;
        HasData = hasData;
    }

    /// <summary>
    /// Gets the field name reported by TensorRT.
    /// 获取 TensorRT 报告的字段名。
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the TensorRT plugin field type.
    /// 获取 TensorRT plugin 字段类型。
    /// </summary>
    public TensorRtPluginFieldType FieldType { get; }

    /// <summary>
    /// Gets the number of entries described by this field.
    /// 获取该字段描述的元素数量。
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Gets whether TensorRT reported a non-null field data pointer.
    /// 获取 TensorRT 是否报告非空字段数据指针。
    /// </summary>
    public bool HasData { get; }

    /// <summary>
    /// Returns a compact display string for the plugin field.
    /// 返回该 plugin 字段的简短显示字符串。
    /// </summary>
    /// <returns>A display string containing the field name, type, and length. 包含字段名、类型和长度的显示字符串。</returns>
    public override string ToString() => $"{Name}:{FieldType}[{Length}]";
}
