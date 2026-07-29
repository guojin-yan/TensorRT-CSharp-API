using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

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
        TensorRtApiLanguage apiLanguage,
        IReadOnlyList<TensorRtPluginFieldInfo> fields,
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
        Fields = fields ?? Array.Empty<TensorRtPluginFieldInfo>();
        TensorRtVersion = tensorRtVersion;
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
    /// Gets the API language reported by this plugin creator.
    /// 获取该 plugin creator 报告的 API language。
    /// </summary>
    public TensorRtApiLanguage ApiLanguage { get; }

    /// <summary>
    /// Gets the TensorRT API version used to compile this creator when TensorRT exposes it.
    /// 当 TensorRT 提供该信息时，获取 creator 编译时使用的 TensorRT API 数字版本。
    /// </summary>
    /// <remarks>TensorRT 8 exposes this scalar on <c>IPluginCreator</c>; later API lines return <see langword="null"/>. TensorRT 8 在 <c>IPluginCreator</c> 上公开该标量；后续 API 版本线返回 <see langword="null"/>。</remarks>
    public int? TensorRtVersion { get; }

    /// <summary>
    /// Gets the plugin fields reported by this creator.
    /// 获取该 creator 报告的 plugin 字段。
    /// </summary>
    public IReadOnlyList<TensorRtPluginFieldInfo> Fields { get; }

    /// <summary>
    /// Returns a compact display string for the plugin creator.
    /// 返回该 plugin creator 的简短显示字符串。
    /// </summary>
    /// <returns>A display string containing creator identity and field count. 包含 creator 标识和字段数量的显示字符串。</returns>
    public override string ToString() => $"{Index}:{Name}:{Version}:{Namespace}:{InterfaceKind}:{ApiLanguage}:trt={TensorRtVersion?.ToString() ?? "n/a"}:{Fields.Count}";
}
