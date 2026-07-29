using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports the aggregated callback owner closure matrix.
/// 报告聚合后的 callback owner 闭环矩阵。
/// </summary>
public sealed class TensorRtCallbackOwnerClosureMatrixResult
{
    private readonly TensorRtCallbackOwnerClosureMatrixRow[] _rows;
    private readonly string[] _blockedPrerequisites;

    internal TensorRtCallbackOwnerClosureMatrixResult(
        TensorRtCallbackOwnerClosureMatrixRow[] rows,
        string[] blockedPrerequisites)
    {
        _rows = rows == null ? Array.Empty<TensorRtCallbackOwnerClosureMatrixRow>() : (TensorRtCallbackOwnerClosureMatrixRow[])rows.Clone();
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this matrix. 获取该矩阵的 evidence marker。</summary>
    public string EvidenceKind => "callback-owner-closure-matrix";

    /// <summary>Gets the runtime evidence kind. 获取 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind => "closure-matrix";

    /// <summary>Gets whether this matrix proves real TensorRT callback runtime. 获取该矩阵是否证明真实 TensorRT callback runtime。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether this matrix is promotable as real callback runtime proof. 获取该矩阵是否可提升为真实 callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets matrix rows. 获取矩阵行。</summary>
    public ReadOnlyCollection<TensorRtCallbackOwnerClosureMatrixRow> Rows =>
        Array.AsReadOnly(_rows ?? Array.Empty<TensorRtCallbackOwnerClosureMatrixRow>());

    /// <summary>Gets row count. 获取矩阵行数。</summary>
    public int FamilyCount => (_rows ?? Array.Empty<TensorRtCallbackOwnerClosureMatrixRow>()).Length;

    /// <summary>Gets how many families have design-gate evidence ready. 获取 design gate 证据就绪的 family 数量。</summary>
    public int DesignGateReadyFamilyCount => CountRows(static row => row.DesignGateReady);

    /// <summary>Gets how many families have full owner closure. 获取 owner 闭环完成的 family 数量。</summary>
    public int ClosureReadyFamilyCount => CountRows(static row => row.ClosureReady);

    /// <summary>Gets how many families can attempt runtime proof. 获取可尝试 runtime proof 的 family 数量。</summary>
    public int RuntimeProofAttemptReadyFamilyCount => CountRows(static row => row.CanAttemptRuntimeProof);

    /// <summary>Gets how many families have package-consumer runtime proof. 获取已有 package-consumer runtime proof 的 family 数量。</summary>
    public int PackageConsumerRuntimeProofReadyFamilyCount => CountRows(static row => row.PackageConsumerRuntimeProofReady);

    /// <summary>Gets whether public surfaces remain pointer-free across all rows. 获取所有行 public surface 是否保持无裸指针。</summary>
    public bool PointerFreeSurfaceReady => CountRows(static row => row.BorrowedPointerEscapeBlocked) == FamilyCount;

    /// <summary>Gets whether all rows have closure complete. 获取全部 family 是否已闭环。</summary>
    public bool AllFamiliesClosureReady => FamilyCount > 0 && ClosureReadyFamilyCount == FamilyCount;

    /// <summary>Gets whether all rows can attempt real runtime proof. 获取全部 family 是否可尝试真实 runtime proof。</summary>
    public bool CanAttemptRuntimeProof => FamilyCount > 0 && RuntimeProofAttemptReadyFamilyCount == FamilyCount;

    /// <summary>Gets whether real runtime proof remains blocked. 获取真实 runtime proof 是否仍被阻塞。</summary>
    public bool RuntimeProofBlocked => !CanAttemptRuntimeProof || PackageConsumerRuntimeProofReadyFamilyCount != FamilyCount;

    /// <summary>Gets whether direct callback deferred rows still must remain deferred. 获取 direct callback deferred 行是否仍必须保留。</summary>
    public bool DeferredRowsStillRequired => CountRows(static row => row.DeferredRowsStillRequired) > 0;

    /// <summary>Gets copied blocker details. 获取复制出的阻塞项。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets blocker count. 获取阻塞项数量。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets matrix status. 获取矩阵状态。</summary>
    public string Status => AllFamiliesClosureReady ? "closure-ready" : "closure-blocked";

    /// <summary>Gets a compact matrix summary. 获取紧凑矩阵摘要。</summary>
    public string Summary =>
        "callback-owner-closure-matrix; RuntimeEvidenceKind=closure-matrix; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; FamilyCount=" + FamilyCount + "; DesignGateReadyFamilyCount=" + DesignGateReadyFamilyCount + "; " +
        "ClosureReadyFamilyCount=" + ClosureReadyFamilyCount + "; RuntimeProofAttemptReadyFamilyCount=" + RuntimeProofAttemptReadyFamilyCount + "; " +
        "PackageConsumerRuntimeProofReadyFamilyCount=" + PackageConsumerRuntimeProofReadyFamilyCount + "; " +
        "PointerFreeSurfaceReady=" + PointerFreeSurfaceReady + "; CanAttemptRuntimeProof=" + CanAttemptRuntimeProof + "; " +
        "RuntimeProofBlocked=" + RuntimeProofBlocked + "; DeferredRowsStillRequired=" + DeferredRowsStillRequired + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 返回紧凑诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:families={FamilyCount}:closure={ClosureReadyFamilyCount}:proof={IsRealCallbackRuntimeProof}:blocked={BlockedPrerequisiteCount}";
    }

    private int CountRows(Func<TensorRtCallbackOwnerClosureMatrixRow, bool> predicate)
    {
        int count = 0;
        foreach (TensorRtCallbackOwnerClosureMatrixRow row in _rows ?? Array.Empty<TensorRtCallbackOwnerClosureMatrixRow>())
        {
            if (predicate(row))
            {
                count++;
            }
        }

        return count;
    }
}
