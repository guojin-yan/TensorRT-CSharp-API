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
        IReadOnlyList<TensorRtPluginFieldInfo> fields)
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
    /// Gets the plugin fields reported by this creator.
    /// 获取该 creator 报告的 plugin 字段。
    /// </summary>
    public IReadOnlyList<TensorRtPluginFieldInfo> Fields { get; }

    /// <summary>
    /// Returns a compact display string for the plugin creator.
    /// 返回该 plugin creator 的简短显示字符串。
    /// </summary>
    /// <returns>A display string containing creator identity and field count. 包含 creator 标识和字段数量的显示字符串。</returns>
    public override string ToString() => $"{Index}:{Name}:{Version}:{Namespace}:{InterfaceKind}:{ApiLanguage}:{Fields.Count}";
}

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
        int fieldCount)
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
    public override string ToString() => $"{Index}:{Name}:{Version}:{Namespace}:{InterfaceKind}:{ApiLanguage}:fields={FieldCount}";
}

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

/// <summary>
/// Diagnostics for already-copied TensorRT plugin registry metadata.
/// 已复制 TensorRT plugin registry 元数据的诊断信息。
/// </summary>
public sealed class TensorRtPluginRegistryInventoryDiagnostics
{
    internal TensorRtPluginRegistryInventoryDiagnostics(
        int creatorCount,
        int recursiveCreatorCount,
        bool hasRecursiveCreatorCount,
        int summaryCount,
        int emptyNameCount,
        int emptyVersionCount,
        int emptyNamespaceCount,
        int creatorWithFieldCount,
        int totalFieldCount,
        int emptyFieldNameCount,
        int negativeFieldLengthCount)
    {
        CreatorCount = creatorCount;
        RecursiveCreatorCount = recursiveCreatorCount;
        HasRecursiveCreatorCount = hasRecursiveCreatorCount;
        SummaryCount = summaryCount;
        EmptyNameCount = emptyNameCount;
        EmptyVersionCount = emptyVersionCount;
        EmptyNamespaceCount = emptyNamespaceCount;
        CreatorWithFieldCount = creatorWithFieldCount;
        TotalFieldCount = totalFieldCount;
        EmptyFieldNameCount = emptyFieldNameCount;
        NegativeFieldLengthCount = negativeFieldLengthCount;
    }

    /// <summary>
    /// Gets the number of creators copied into the managed inventory.
    /// 获取托管 inventory 中已复制的 creator 数量。
    /// </summary>
    public int CreatorCount { get; }

    /// <summary>
    /// Gets the recursive creator count, or zero when the source did not report one.
    /// 获取递归 creator 数量；当来源未报告该值时为零。
    /// </summary>
    public int RecursiveCreatorCount { get; }

    /// <summary>
    /// Gets whether the source reported a recursive creator count.
    /// 获取来源是否报告了递归 creator 数量。
    /// </summary>
    public bool HasRecursiveCreatorCount { get; }

    /// <summary>
    /// Gets the number of pointer-free summaries generated from the copied creators.
    /// 获取从已复制 creator 生成的无指针摘要数量。
    /// </summary>
    public int SummaryCount { get; }

    /// <summary>
    /// Gets whether summary generation preserved the copied creator count.
    /// 获取摘要生成是否保留了已复制 creator 数量。
    /// </summary>
    public bool HasCreatorCountMismatch => SummaryCount != CreatorCount;

    /// <summary>
    /// Gets whether the recursive count is smaller than the copied creator count.
    /// 获取递归计数是否小于已复制 creator 数量。
    /// </summary>
    public bool HasRecursiveCountMismatch => HasRecursiveCreatorCount && RecursiveCreatorCount < CreatorCount;

    /// <summary>
    /// Gets the number of creators with an empty name.
    /// 获取 name 为空的 creator 数量。
    /// </summary>
    public int EmptyNameCount { get; }

    /// <summary>
    /// Gets the number of creators with an empty version.
    /// 获取 version 为空的 creator 数量。
    /// </summary>
    public int EmptyVersionCount { get; }

    /// <summary>
    /// Gets the number of creators with an empty namespace.
    /// 获取 namespace 为空的 creator 数量。
    /// </summary>
    public int EmptyNamespaceCount { get; }

    /// <summary>
    /// Gets the number of creators that reported at least one field.
    /// 获取至少报告一个字段的 creator 数量。
    /// </summary>
    public int CreatorWithFieldCount { get; }

    /// <summary>
    /// Gets the total number of field descriptors copied from all creators.
    /// 获取从所有 creator 复制出的字段描述符总数。
    /// </summary>
    public int TotalFieldCount { get; }

    /// <summary>
    /// Gets the number of fields with an empty name.
    /// 获取 name 为空的字段数量。
    /// </summary>
    public int EmptyFieldNameCount { get; }

    /// <summary>
    /// Gets the number of fields with a negative length.
    /// 获取 length 为负数的字段数量。
    /// </summary>
    public int NegativeFieldLengthCount { get; }

    /// <summary>
    /// Gets whether any copied field metadata appears invalid.
    /// 获取是否存在无效的已复制字段元数据。
    /// </summary>
    public bool HasInvalidFieldMetadata => EmptyFieldNameCount > 0 || NegativeFieldLengthCount > 0;

    /// <summary>
    /// Gets whether the copied metadata is internally consistent.
    /// 获取已复制元数据是否内部自洽。
    /// </summary>
    public bool IsConsistent =>
        !HasCreatorCountMismatch &&
        !HasRecursiveCountMismatch &&
        EmptyNameCount == 0 &&
        EmptyVersionCount == 0 &&
        !HasInvalidFieldMetadata;

    /// <summary>
    /// Gets whether this readonly inventory diagnostic can be promoted to runtime proof.
    /// 获取该只读 inventory 诊断是否可晋级为 runtime proof。
    /// </summary>
    public bool CanPromoteRuntimeProof => false;

    /// <summary>
    /// Gets whether this readonly inventory diagnostic permits deleting deferred history records.
    /// 获取该只读 inventory 诊断是否允许删除 deferred 历史记录。
    /// </summary>
    public bool CanDeleteDeferredRecord => false;

    /// <summary>
    /// Gets a compact diagnostic summary for logging and smoke tests.
    /// 获取用于日志和 smoke 测试的简短诊断摘要。
    /// </summary>
    public string DiagnosticSummary =>
        $"consistent={IsConsistent}:creators={CreatorCount}:recursive={(HasRecursiveCreatorCount ? RecursiveCreatorCount.ToString() : "n/a")}:summaries={SummaryCount}:fields={TotalFieldCount}:emptyNames={EmptyNameCount}:emptyVersions={EmptyVersionCount}:emptyFieldNames={EmptyFieldNameCount}:negativeFieldLengths={NegativeFieldLengthCount}";

    /// <summary>
    /// Returns a compact display string for the plugin registry inventory diagnostics.
    /// 返回 plugin registry inventory 诊断的简短显示字符串。
    /// </summary>
    /// <returns>A compact diagnostic summary. 简短诊断摘要。</returns>
    public override string ToString() => DiagnosticSummary;
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
                creator.Fields.Count));
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
