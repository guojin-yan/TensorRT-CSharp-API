using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Pointer-free summary metadata for one TensorRT plugin creator field.
/// TensorRT plugin creator 字段的无指针摘要元数据。
/// </summary>
public sealed class TensorRtPluginFieldSummary
{
    internal TensorRtPluginFieldSummary(
        int creatorIndex,
        string creatorName,
        string creatorVersion,
        string creatorNamespace,
        int fieldIndex,
        string fieldName,
        TensorRtPluginFieldType fieldType,
        int length,
        bool hasData)
    {
        CreatorIndex = creatorIndex;
        CreatorName = creatorName ?? string.Empty;
        CreatorVersion = creatorVersion ?? string.Empty;
        CreatorNamespace = creatorNamespace ?? string.Empty;
        FieldIndex = fieldIndex;
        FieldName = fieldName ?? string.Empty;
        FieldType = fieldType;
        Length = length;
        HasData = hasData;
    }

    /// <summary>
    /// Gets the creator index in the registry snapshot.
    /// 获取 creator 在 registry 快照中的索引。
    /// </summary>
    public int CreatorIndex { get; }

    /// <summary>
    /// Gets the plugin creator name copied from TensorRT.
    /// 获取从 TensorRT 复制出的 plugin creator 名称。
    /// </summary>
    public string CreatorName { get; }

    /// <summary>
    /// Gets the plugin creator version copied from TensorRT.
    /// 获取从 TensorRT 复制出的 plugin creator 版本。
    /// </summary>
    public string CreatorVersion { get; }

    /// <summary>
    /// Gets the plugin creator namespace copied from TensorRT.
    /// 获取从 TensorRT 复制出的 plugin creator namespace。
    /// </summary>
    public string CreatorNamespace { get; }

    /// <summary>
    /// Gets the field index inside the copied creator field collection.
    /// 获取字段在已复制 creator 字段集合中的索引。
    /// </summary>
    public int FieldIndex { get; }

    /// <summary>
    /// Gets the field name copied from TensorRT.
    /// 获取从 TensorRT 复制出的字段名。
    /// </summary>
    public string FieldName { get; }

    /// <summary>
    /// Gets the TensorRT plugin field type copied from TensorRT metadata.
    /// 获取从 TensorRT 元数据复制出的 plugin 字段类型。
    /// </summary>
    public TensorRtPluginFieldType FieldType { get; }

    /// <summary>
    /// Gets the number of entries described by this copied field.
    /// 获取该已复制字段描述的元素数量。
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Gets whether TensorRT reported a non-null field data pointer. The pointer value itself is never exposed.
    /// 获取 TensorRT 是否报告非空字段数据指针；指针值本身永不暴露。
    /// </summary>
    public bool HasData { get; }

    /// <summary>
    /// Returns a compact display string for the plugin field summary.
    /// 返回该 plugin 字段摘要的简短显示字符串。
    /// </summary>
    /// <returns>A display string containing creator identity and field metadata. 包含 creator 标识和字段元数据的显示字符串。</returns>
    public override string ToString() => $"{CreatorIndex}:{CreatorName}:{CreatorVersion}:{CreatorNamespace}:field={FieldIndex}:{FieldName}:{FieldType}[{Length}]:hasData={HasData}";
}
