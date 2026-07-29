using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

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
