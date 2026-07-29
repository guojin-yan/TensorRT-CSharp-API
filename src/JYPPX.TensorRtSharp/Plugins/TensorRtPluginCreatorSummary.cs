using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Pointer-free summary metadata for a TensorRT plugin creator.
/// TensorRT plugin creator 的无指针摘要元数据。
/// </summary>
public sealed class TensorRtPluginCreatorSummary
{
    internal TensorRtPluginCreatorSummary(
        int index,
        string name,
        string version,
        string pluginNamespace,
        string interfaceKind,
        int interfaceMajor,
        int interfaceMinor,
        TensorRtApiLanguage apiLanguage,
        int fieldCount,
        int? tensorRtVersion = null)
    {
        Index = index;
        Name = name ?? string.Empty;
        Version = version ?? string.Empty;
        Namespace = pluginNamespace ?? string.Empty;
        InterfaceKind = interfaceKind ?? string.Empty;
        InterfaceMajor = interfaceMajor;
        InterfaceMinor = interfaceMinor;
        ApiLanguage = apiLanguage;
        FieldCount = fieldCount;
        TensorRtVersion = tensorRtVersion;
    }

    /// <summary>
    /// Gets the creator index in the registry snapshot.
    /// 获取 creator 在 registry 快照中的索引。
    /// </summary>
    public int Index { get; }

    /// <summary>
    /// Gets the plugin creator name copied from TensorRT.
    /// 获取从 TensorRT 复制出的 plugin creator 名称。
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the plugin creator version copied from TensorRT.
    /// 获取从 TensorRT 复制出的 plugin creator 版本。
    /// </summary>
    public string Version { get; }

    /// <summary>
    /// Gets the plugin creator namespace copied from TensorRT.
    /// 获取从 TensorRT 复制出的 plugin creator namespace。
    /// </summary>
    public string Namespace { get; }

    /// <summary>
    /// Gets the TensorRT interface kind string copied from TensorRT.
    /// 获取从 TensorRT 复制出的 interface kind 字符串。
    /// </summary>
    public string InterfaceKind { get; }

    /// <summary>
    /// Gets the TensorRT interface major version copied from TensorRT.
    /// 获取从 TensorRT 复制出的 interface major 版本。
    /// </summary>
    public int InterfaceMajor { get; }

    /// <summary>
    /// Gets the TensorRT interface minor version copied from TensorRT.
    /// 获取从 TensorRT 复制出的 interface minor 版本。
    /// </summary>
    public int InterfaceMinor { get; }

    /// <summary>
    /// Gets the TensorRT API language copied from TensorRT.
    /// 获取从 TensorRT 复制出的 API language。
    /// </summary>
    public TensorRtApiLanguage ApiLanguage { get; }

    /// <summary>Gets the copied creator TensorRT API version when available. 获取可用时复制出的 creator TensorRT API 版本。</summary>
    public int? TensorRtVersion { get; }

    /// <summary>
    /// Gets the number of plugin fields reported by this creator.
    /// 获取该 creator 报告的 plugin 字段数量。
    /// </summary>
    public int FieldCount { get; }

    /// <summary>
    /// Returns a compact display string for the plugin creator summary.
    /// 返回该 plugin creator 摘要的简短显示字符串。
    /// </summary>
    /// <returns>A display string containing creator identity and field count. 包含 creator 标识和字段数量的显示字符串。</returns>
    public override string ToString() => $"{Index}:{Name}:{Version}:{Namespace}:{InterfaceKind}:{ApiLanguage}:trt={TensorRtVersion?.ToString() ?? "n/a"}:fields={FieldCount}";
}
