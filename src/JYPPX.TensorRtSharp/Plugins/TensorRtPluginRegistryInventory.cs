using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

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

    /// <summary>
    /// Finds the first creator in this managed snapshot matching the supplied identity.
    /// 在当前托管快照中查找第一个匹配给定标识的 creator。
    /// </summary>
    /// <remarks>
    /// This method only scans already-copied metadata. It does not call TensorRT, return native creator pointers, or change ownership.
    /// 该方法只扫描已经复制出的元数据；不会调用 TensorRT、返回 native creator 指针或改变所有权。
    /// </remarks>
    /// <param name="pluginName">The plugin creator name. plugin creator 名称。</param>
    /// <param name="pluginVersion">The plugin creator version. plugin creator 版本。</param>
    /// <param name="pluginNamespace">The plugin creator namespace. plugin creator 命名空间。</param>
    /// <returns>The matching creator metadata when found; otherwise <see langword="null"/>. 找到时返回匹配 creator 元数据，否则返回 <see langword="null"/>。</returns>
    public TensorRtPluginCreatorInfo? FindCreator(string pluginName, string pluginVersion, string pluginNamespace)
    {
        string name = pluginName ?? string.Empty;
        string version = pluginVersion ?? string.Empty;
        string creatorNamespace = pluginNamespace ?? string.Empty;

        foreach (TensorRtPluginCreatorInfo creator in Creators)
        {
            if (string.Equals(creator.Name, name, StringComparison.Ordinal) &&
                string.Equals(creator.Version, version, StringComparison.Ordinal) &&
                string.Equals(creator.Namespace, creatorNamespace, StringComparison.Ordinal))
            {
                return creator;
            }
        }

        return null;
    }

    /// <summary>
    /// Tries to find a creator in this managed snapshot matching the supplied identity.
    /// 尝试在当前托管快照中查找匹配给定标识的 creator。
    /// </summary>
    /// <remarks>
    /// This is a managed snapshot lookup. It does not expose borrowed TensorRT pointers or invoke plugin creation APIs.
    /// 这是托管快照查询；不会暴露 borrowed TensorRT 指针，也不会调用 plugin 创建 API。
    /// </remarks>
    /// <param name="pluginName">The plugin creator name. plugin creator 名称。</param>
    /// <param name="pluginVersion">The plugin creator version. plugin creator 版本。</param>
    /// <param name="pluginNamespace">The plugin creator namespace. plugin creator 命名空间。</param>
    /// <param name="creator">The matching creator metadata when found. 找到时返回匹配 creator 元数据。</param>
    /// <returns><see langword="true"/> when a matching creator exists in the snapshot. 快照中存在匹配 creator 时返回 <see langword="true"/>。</returns>
    public bool TryFindCreator(string pluginName, string pluginVersion, string pluginNamespace, out TensorRtPluginCreatorInfo? creator)
    {
        creator = FindCreator(pluginName, pluginVersion, pluginNamespace);
        return creator != null;
    }

    /// <summary>
    /// Returns pointer-free summaries for creators already copied into this managed snapshot.
    /// 返回当前托管快照中已复制 creator 的无指针摘要。
    /// </summary>
    /// <remarks>
    /// This method does not call TensorRT, expose native creator pointers, or copy plugin field data payloads.
    /// 该方法不会调用 TensorRT、暴露 native creator 指针，也不会复制 plugin 字段数据载荷。
    /// </remarks>
    /// <param name="maxCreators">Maximum number of creator summaries to return. 要返回的最大 creator 摘要数量。</param>
    /// <returns>Pointer-free creator summaries copied from the managed snapshot. 从托管快照复制出的无指针 creator 摘要。</returns>
    public IReadOnlyList<TensorRtPluginCreatorSummary> GetCreatorSummaries(int maxCreators = int.MaxValue)
    {
        if (maxCreators < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxCreators), "Maximum creator count cannot be negative.");
        }

        int count = Math.Min(maxCreators, Creators.Count);
        if (count == 0)
        {
            return Array.Empty<TensorRtPluginCreatorSummary>();
        }

        List<TensorRtPluginCreatorSummary> summaries = new List<TensorRtPluginCreatorSummary>(count);
        for (int index = 0; index < count; index++)
        {
            TensorRtPluginCreatorInfo creator = Creators[index];
            summaries.Add(new TensorRtPluginCreatorSummary(
                creator.Index,
                creator.Name,
                creator.Version,
                creator.Namespace,
                creator.InterfaceKind,
                creator.InterfaceMajor,
                creator.InterfaceMinor,
                creator.ApiLanguage,
                creator.Fields.Count,
                creator.TensorRtVersion));
        }

        return summaries;
    }

    /// <summary>
    /// Returns pointer-free field summaries for metadata already copied into this managed snapshot.
    /// 返回当前托管快照中已复制字段元数据的无指针摘要。
    /// </summary>
    /// <remarks>
    /// This method scans the managed snapshot only. It does not call TensorRT, expose native creator pointers, or copy plugin field data payloads.
    /// 该方法只扫描托管快照；不会调用 TensorRT、暴露 native creator 指针，也不会复制 plugin 字段数据载荷。
    /// </remarks>
    /// <param name="maxCreators">Maximum number of creators to scan. 要扫描的最大 creator 数量。</param>
    /// <param name="maxFieldsPerCreator">Maximum number of fields to include for each creator. 每个 creator 最多包含的字段数量。</param>
    /// <returns>Pointer-free field summaries copied from the managed snapshot. 从托管快照复制出的无指针字段摘要。</returns>
    public IReadOnlyList<TensorRtPluginFieldSummary> GetFieldSummaries(int maxCreators = int.MaxValue, int maxFieldsPerCreator = int.MaxValue)
    {
        if (maxCreators < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxCreators), "Maximum creator count cannot be negative.");
        }

        if (maxFieldsPerCreator < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxFieldsPerCreator), "Maximum field count cannot be negative.");
        }

        int creatorCount = Math.Min(maxCreators, Creators.Count);
        if (creatorCount == 0 || maxFieldsPerCreator == 0)
        {
            return Array.Empty<TensorRtPluginFieldSummary>();
        }

        List<TensorRtPluginFieldSummary> summaries = new List<TensorRtPluginFieldSummary>();
        for (int creatorIndex = 0; creatorIndex < creatorCount; creatorIndex++)
        {
            TensorRtPluginCreatorInfo creator = Creators[creatorIndex];
            int fieldCount = Math.Min(maxFieldsPerCreator, creator.Fields.Count);
            for (int fieldIndex = 0; fieldIndex < fieldCount; fieldIndex++)
            {
                TensorRtPluginFieldInfo field = creator.Fields[fieldIndex];
                summaries.Add(new TensorRtPluginFieldSummary(
                    creator.Index,
                    creator.Name,
                    creator.Version,
                    creator.Namespace,
                    fieldIndex,
                    field.Name,
                    field.FieldType,
                    field.Length,
                    field.HasData));
            }
        }

        return summaries;
    }

    /// <summary>
    /// Returns consistency diagnostics for metadata already copied into this managed snapshot.
    /// 返回当前托管快照中已复制元数据的一致性诊断。
    /// </summary>
    /// <remarks>
    /// This method only inspects already-copied metadata. It does not call TensorRT, expose native creator pointers, or copy plugin field data payloads.
    /// 该方法只检查已经复制出的元数据；不会调用 TensorRT、暴露 native creator 指针，也不会复制 plugin 字段数据载荷。
    /// </remarks>
    /// <returns>Pointer-free consistency diagnostics for the copied inventory metadata. 已复制 inventory 元数据的无指针一致性诊断。</returns>
    public TensorRtPluginRegistryInventoryDiagnostics GetDiagnostics()
    {
        int emptyNameCount = 0;
        int emptyVersionCount = 0;
        int emptyNamespaceCount = 0;
        int creatorWithFieldCount = 0;
        int totalFieldCount = 0;
        int emptyFieldNameCount = 0;
        int negativeFieldLengthCount = 0;

        foreach (TensorRtPluginCreatorInfo creator in Creators)
        {
            if (string.IsNullOrWhiteSpace(creator.Name))
            {
                emptyNameCount++;
            }

            if (string.IsNullOrWhiteSpace(creator.Version))
            {
                emptyVersionCount++;
            }

            if (string.IsNullOrWhiteSpace(creator.Namespace))
            {
                emptyNamespaceCount++;
            }

            if (creator.Fields.Count > 0)
            {
                creatorWithFieldCount++;
            }

            foreach (TensorRtPluginFieldInfo field in creator.Fields)
            {
                totalFieldCount++;
                if (string.IsNullOrWhiteSpace(field.Name))
                {
                    emptyFieldNameCount++;
                }

                if (field.Length < 0)
                {
                    negativeFieldLengthCount++;
                }
            }
        }

        IReadOnlyList<TensorRtPluginCreatorSummary> summaries = GetCreatorSummaries();
        return new TensorRtPluginRegistryInventoryDiagnostics(
            CreatorCount,
            RecursiveCreatorCount.GetValueOrDefault(),
            RecursiveCreatorCount.HasValue,
            summaries.Count,
            emptyNameCount,
            emptyVersionCount,
            emptyNamespaceCount,
            creatorWithFieldCount,
            totalFieldCount,
            emptyFieldNameCount,
            negativeFieldLengthCount);
    }

    /// <summary>
    /// Returns a compact display string for the registry inventory.
    /// 返回该 registry inventory 的简短显示字符串。
    /// </summary>
    /// <returns>A display string containing source, creator counts, and registry options. 包含来源、creator 数量和 registry 选项的显示字符串。</returns>
    public override string ToString() => $"{Line}:{Source}:creators={CreatorCount}:recursive={RecursiveCreatorCount?.ToString() ?? "n/a"}:parentSearch={ParentSearchEnabled}:errorRecorder={HasErrorRecorder}";
}
