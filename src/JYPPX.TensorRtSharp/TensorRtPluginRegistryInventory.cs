using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// TensorRT plugin field data type reported by plugin creators.
/// TensorRT plugin creator 报告的字段数据类型。
/// </summary>
public enum TensorRtPluginFieldType
{
    Float16 = 0,
    Float32 = 1,
    Float64 = 2,
    Int8 = 3,
    Int16 = 4,
    Int32 = 5,
    Char = 6,
    Dims = 7,
    Unknown = 8,
    BFloat16 = 9,
    Int64 = 10,
    Float8 = 11,
    Int4 = 12,
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
    /// </summary>
    BuilderCapability = 2
}

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

    public override string ToString() => $"{Name}:{FieldType}[{Length}]";
}

/// <summary>
/// Read-only metadata for a TensorRT plugin creator.
/// TensorRT plugin creator 的只读元数据。
/// </summary>
public sealed class TensorRtPluginCreatorInfo
{
    internal TensorRtPluginCreatorInfo(
        int index,
        string name,
        string version,
        string pluginNamespace,
        string interfaceKind,
        int interfaceMajor,
        int interfaceMinor,
        IReadOnlyList<TensorRtPluginFieldInfo> fields)
    {
        Index = index;
        Name = name ?? string.Empty;
        Version = version ?? string.Empty;
        Namespace = pluginNamespace ?? string.Empty;
        InterfaceKind = interfaceKind ?? string.Empty;
        InterfaceMajor = interfaceMajor;
        InterfaceMinor = interfaceMinor;
        Fields = fields ?? Array.Empty<TensorRtPluginFieldInfo>();
    }

    /// <summary>
    /// Gets the creator index in the registry snapshot.
    /// 获取 creator 在 registry 快照中的索引。
    /// </summary>
    public int Index { get; }

    /// <summary>
    /// Gets the plugin creator name.
    /// 获取 plugin creator 名称。
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the plugin creator version.
    /// 获取 plugin creator 版本。
    /// </summary>
    public string Version { get; }

    /// <summary>
    /// Gets the plugin creator namespace.
    /// 获取 plugin creator namespace。
    /// </summary>
    public string Namespace { get; }

    /// <summary>
    /// Gets the TensorRT interface kind string.
    /// 获取 TensorRT interface kind 字符串。
    /// </summary>
    public string InterfaceKind { get; }

    /// <summary>
    /// Gets the TensorRT interface major version.
    /// 获取 TensorRT interface major 版本。
    /// </summary>
    public int InterfaceMajor { get; }

    /// <summary>
    /// Gets the TensorRT interface minor version.
    /// 获取 TensorRT interface minor 版本。
    /// </summary>
    public int InterfaceMinor { get; }

    /// <summary>
    /// Gets the plugin fields reported by this creator.
    /// 获取该 creator 报告的 plugin 字段。
    /// </summary>
    public IReadOnlyList<TensorRtPluginFieldInfo> Fields { get; }

    public override string ToString() => $"{Index}:{Name}:{Version}:{Namespace}:{InterfaceKind}:{Fields.Count}";
}

/// <summary>
/// Read-only snapshot of a TensorRT plugin registry.
/// TensorRT plugin registry 的只读快照。
/// </summary>
public sealed class TensorRtPluginRegistryInventory
{
    internal TensorRtPluginRegistryInventory(
        TensorRtApiLine line,
        TensorRtPluginRegistrySource source,
        bool hasErrorRecorder,
        bool parentSearchEnabled,
        int? recursiveCreatorCount,
        IReadOnlyList<TensorRtPluginCreatorInfo> creators)
    {
        Line = line;
        Source = source;
        HasErrorRecorder = hasErrorRecorder;
        ParentSearchEnabled = parentSearchEnabled;
        RecursiveCreatorCount = recursiveCreatorCount;
        Creators = creators ?? Array.Empty<TensorRtPluginCreatorInfo>();
    }

    /// <summary>
    /// Gets the TensorRT API line used to collect this inventory.
    /// 获取采集该 inventory 的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the registry source used to collect this inventory.
    /// 获取采集该 inventory 的 registry 来源。
    /// </summary>
    public TensorRtPluginRegistrySource Source { get; }

    /// <summary>
    /// Gets whether the registry has an error recorder.
    /// 获取 registry 是否绑定 error recorder。
    /// </summary>
    public bool HasErrorRecorder { get; }

    /// <summary>
    /// Gets whether parent registry search is enabled.
    /// 获取是否启用 parent registry search。
    /// </summary>
    public bool ParentSearchEnabled { get; }

    /// <summary>
    /// Gets the plugin creators copied from the registry snapshot.
    /// 获取从 registry 快照复制出的 plugin creator 列表。
    /// </summary>
    public IReadOnlyList<TensorRtPluginCreatorInfo> Creators { get; }

    /// <summary>
    /// Gets the number of plugin creators in the snapshot.
    /// 获取快照中的 plugin creator 数量。
    /// </summary>
    public int CreatorCount => Creators.Count;

    /// <summary>
    /// Gets the recursive creator count when the source supports a separate recursive query.
    /// 当 registry 来源支持单独递归查询时，获取递归 creator 数量。
    /// </summary>
    public int? RecursiveCreatorCount { get; }

    public override string ToString() => $"{Line}:{Source}:creators={CreatorCount}:recursive={RecursiveCreatorCount?.ToString() ?? "n/a"}:parentSearch={ParentSearchEnabled}:errorRecorder={HasErrorRecorder}";
}
