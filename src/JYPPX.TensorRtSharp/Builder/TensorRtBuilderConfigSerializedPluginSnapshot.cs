using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a copied read-only snapshot of TensorRT builder config serialized plugin paths.
/// 表示 TensorRT builder config serialized plugin path 的复制型只读快照。
/// </summary>
public sealed class TensorRtBuilderConfigSerializedPluginSnapshot
{
    internal TensorRtBuilderConfigSerializedPluginSnapshot(
        TensorRtApiLine line,
        int count,
        IReadOnlyList<string> pluginLibraryPaths,
        bool hasPathInventory,
        string diagnostic)
    {
        Line = line;
        Count = count;
        PluginLibraryPaths = pluginLibraryPaths ?? Array.Empty<string>();
        HasPathInventory = hasPathInventory;
        Diagnostic = diagnostic ?? string.Empty;
    }

    /// <summary>
    /// Gets the TensorRT API line used for the snapshot.
    /// 获取该快照对应的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the serialized plugin path count reported by TensorRT.
    /// 获取 TensorRT 报告的 serialized plugin path 数量。
    /// </summary>
    public int Count { get; }

    /// <summary>
    /// Gets copied serialized plugin library paths when the TensorRT line supports path copying.
    /// 获取当前 TensorRT 版本线支持 path copying 时复制出的 serialized plugin library path。
    /// </summary>
    public IReadOnlyList<string> PluginLibraryPaths { get; }

    /// <summary>
    /// Gets whether copied plugin path inventory is available.
    /// 获取是否存在已复制的 plugin path inventory。
    /// </summary>
    public bool HasPathInventory { get; }

    /// <summary>
    /// Gets a diagnostic string describing path inventory availability.
    /// 获取描述 path inventory 可用性的诊断字符串。
    /// </summary>
    public string Diagnostic { get; }

    /// <summary>
    /// Formats this snapshot for diagnostics.
    /// 将该快照格式化为诊断字符串。
    /// </summary>
    public override string ToString()
    {
        return $"Line={(int)Line}, Count={Count}, Paths={PluginLibraryPaths.Count}, HasPathInventory={HasPathInventory}, Diagnostic={Diagnostic}";
    }
}
