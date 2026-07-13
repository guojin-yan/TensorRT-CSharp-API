using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates the pointer-free TensorRT ErrorRecorder diagnostics design gate.
/// 评估 TensorRT ErrorRecorder 只读诊断的无裸指针设计门。
/// </summary>
/// <remarks>
/// This gate only describes copied diagnostics readiness. It does not expose an <c>IErrorRecorder*</c>, does not
/// control native recorder reference counts, and is not runtime execution proof.
/// 该门禁只描述已复制诊断的边界状态；不会暴露 <c>IErrorRecorder*</c>，不会控制原生 recorder 引用计数，也不是 runtime execution proof。
/// </remarks>
public static class TensorRtErrorRecorderDiagnosticsDesignGate
{
    /// <summary>
    /// Evaluates the known public ErrorRecorder diagnostics surface for a TensorRT API line.
    /// 基于已知 public ErrorRecorder 诊断边界评估指定 TensorRT API line。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API line。</param>
    /// <returns>A pointer-free diagnostics design gate result. 无裸指针诊断设计门结果。</returns>
    public static TensorRtErrorRecorderDiagnosticsDesignGateResult EvaluateKnownSurface(TensorRtApiLine line)
    {
        return EvaluateCore(
            line,
            copiedSnapshotObserved: false,
            hasRecorder: false,
            errorCount: 0,
            hasOverflowed: false,
            copiedRecordCount: 0,
            runtimeSnapshotAvailable: true,
            refitterSnapshotAvailable: true,
            presenceControlsAvailable: true,
            clearControlsAvailable: true);
    }

    /// <summary>
    /// Evaluates the ErrorRecorder diagnostics design gate from a copied snapshot.
    /// 根据已复制的 ErrorRecorder snapshot 评估诊断设计门。
    /// </summary>
    /// <param name="snapshot">The copied ErrorRecorder snapshot. 已复制的 ErrorRecorder snapshot。</param>
    /// <returns>A pointer-free diagnostics design gate result. 无裸指针诊断设计门结果。</returns>
    public static TensorRtErrorRecorderDiagnosticsDesignGateResult Evaluate(TensorRtErrorRecorderSnapshot snapshot)
    {
        return Evaluate(
            snapshot,
            runtimeSnapshotAvailable: true,
            refitterSnapshotAvailable: true,
            presenceControlsAvailable: true,
            clearControlsAvailable: true);
    }

    /// <summary>
    /// Evaluates the ErrorRecorder diagnostics design gate from a copied snapshot and explicit capability flags.
    /// 根据已复制 snapshot 和显式能力标记评估 ErrorRecorder 诊断设计门。
    /// </summary>
    /// <param name="snapshot">The copied ErrorRecorder snapshot. 已复制的 ErrorRecorder snapshot。</param>
    /// <param name="runtimeSnapshotAvailable">Whether runtime copied snapshot APIs are available. Runtime copied snapshot API 是否可用。</param>
    /// <param name="refitterSnapshotAvailable">Whether refitter copied snapshot APIs are available. Refitter copied snapshot API 是否可用。</param>
    /// <param name="presenceControlsAvailable">Whether owner presence controls are available. Owner presence 控制是否可用。</param>
    /// <param name="clearControlsAvailable">Whether owner clear controls are available. Owner clear 控制是否可用。</param>
    /// <returns>A pointer-free diagnostics design gate result. 无裸指针诊断设计门结果。</returns>
    public static TensorRtErrorRecorderDiagnosticsDesignGateResult Evaluate(
        TensorRtErrorRecorderSnapshot snapshot,
        bool runtimeSnapshotAvailable,
        bool refitterSnapshotAvailable,
        bool presenceControlsAvailable,
        bool clearControlsAvailable)
    {
        if (snapshot == null)
        {
            throw new ArgumentNullException(nameof(snapshot));
        }

        return EvaluateCore(
            snapshot.Line,
            copiedSnapshotObserved: true,
            snapshot.HasRecorder,
            snapshot.ErrorCount,
            snapshot.HasOverflowed,
            snapshot.Records.Count,
            runtimeSnapshotAvailable,
            refitterSnapshotAvailable,
            presenceControlsAvailable,
            clearControlsAvailable);
    }

    private static TensorRtErrorRecorderDiagnosticsDesignGateResult EvaluateCore(
        TensorRtApiLine line,
        bool copiedSnapshotObserved,
        bool hasRecorder,
        int errorCount,
        bool hasOverflowed,
        int copiedRecordCount,
        bool runtimeSnapshotAvailable,
        bool refitterSnapshotAvailable,
        bool presenceControlsAvailable,
        bool clearControlsAvailable)
    {
        bool lineSupportsErrorRecorder =
            line == TensorRtApiLine.TensorRt8 ||
            line == TensorRtApiLine.TensorRt10 ||
            line == TensorRtApiLine.TensorRt11;
        int normalizedErrorCount = errorCount < 0 ? 0 : errorCount;
        int normalizedCopiedRecordCount = copiedRecordCount < 0 ? 0 : copiedRecordCount;
        bool snapshotRecordCopyReady =
            !copiedSnapshotObserved ||
            !hasRecorder ||
            normalizedErrorCount == normalizedCopiedRecordCount;

        List<string> blockers = new List<string>();
        if (!lineSupportsErrorRecorder)
        {
            blockers.Add("TensorRT 8, 10, or 11 ErrorRecorder line support has not been selected.");
        }

        if (!runtimeSnapshotAvailable)
        {
            blockers.Add("runtime copied ErrorRecorder snapshot APIs are not available.");
        }

        if (!refitterSnapshotAvailable)
        {
            blockers.Add("refitter copied ErrorRecorder snapshot APIs are not available.");
        }

        if (!presenceControlsAvailable)
        {
            blockers.Add("owner ErrorRecorder presence controls are not available.");
        }

        if (!clearControlsAvailable)
        {
            blockers.Add("owner ErrorRecorder clear controls are not available.");
        }

        if (!snapshotRecordCopyReady)
        {
            blockers.Add("copied ErrorRecorder record count does not match the snapshot error count.");
        }

        blockers.Add("direct IErrorRecorder ref-count ownership remains deferred by design.");
        blockers.Add("direct IErrorRecorder interface-info ownership remains deferred by design.");
        blockers.Add("full package consumer runtime execution proof has not been promoted from this design gate.");

        return new TensorRtErrorRecorderDiagnosticsDesignGateResult(
            line,
            lineSupportsErrorRecorder,
            copiedSnapshotObserved,
            hasRecorder,
            normalizedErrorCount,
            hasOverflowed,
            normalizedCopiedRecordCount,
            runtimeSnapshotAvailable,
            refitterSnapshotAvailable,
            presenceControlsAvailable,
            clearControlsAvailable,
            snapshotRecordCopyReady,
            blockers.ToArray());
    }
}

/// <summary>
/// Reports copied ErrorRecorder diagnostics design status without exposing native recorder pointers.
/// 报告 ErrorRecorder 已复制诊断的设计状态，不暴露原生 recorder 指针。
/// </summary>
/// <remarks>
/// This result is a design gate. It keeps direct recorder ownership and reference-count operations deferred until a
/// separate ownership model and runtime proof exist.
/// 该结果是设计门禁；在独立 ownership 模型和 runtime proof 出现前，direct recorder ownership 与引用计数操作继续 deferred。
/// </remarks>
public readonly struct TensorRtErrorRecorderDiagnosticsDesignGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtErrorRecorderDiagnosticsDesignGateResult(
        TensorRtApiLine line,
        bool lineSupportsErrorRecorder,
        bool copiedSnapshotObserved,
        bool hasRecorder,
        int errorCount,
        bool hasOverflowed,
        int copiedRecordCount,
        bool runtimeSnapshotAvailable,
        bool refitterSnapshotAvailable,
        bool presenceControlsAvailable,
        bool clearControlsAvailable,
        bool snapshotRecordCopyReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        LineSupportsErrorRecorder = lineSupportsErrorRecorder;
        CopiedSnapshotObserved = copiedSnapshotObserved;
        HasRecorder = hasRecorder;
        ErrorCount = errorCount;
        HasOverflowed = hasOverflowed;
        CopiedRecordCount = copiedRecordCount;
        RuntimeSnapshotAvailable = runtimeSnapshotAvailable;
        RefitterSnapshotAvailable = refitterSnapshotAvailable;
        PresenceControlsAvailable = presenceControlsAvailable;
        ClearControlsAvailable = clearControlsAvailable;
        SnapshotRecordCopyReady = snapshotRecordCopyReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 获取该设计门的证据标记。</summary>
    public string EvidenceKind => "error-recorder-diagnostics-design-gate";

    /// <summary>Gets the diagnostics kind represented by this gate. 获取该设计门代表的诊断类型。</summary>
    public string DiagnosticsKind => "error-recorder-diagnostics";

    /// <summary>Gets the runtime evidence kind. 获取 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind => "design-gate";

    /// <summary>Gets whether this gate is runtime execution evidence. 获取该设计门是否为 runtime execution evidence。</summary>
    public bool IsRuntimeExecutionEvidence => false;

    /// <summary>Gets whether this gate may be promoted as runtime execution proof. 获取该设计门是否可晋级为 runtime execution proof。</summary>
    public bool IsRuntimeExecutionProof => false;

    /// <summary>Gets the TensorRT API line. 获取 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether the selected TensorRT line supports the ErrorRecorder diagnostics boundary. 获取当前 TensorRT line 是否支持 ErrorRecorder 诊断边界。</summary>
    public bool LineSupportsErrorRecorder { get; }

    /// <summary>Gets whether <see cref="TensorRtErrorRecorderSnapshot"/> is available as the copied snapshot type. 获取 copied snapshot 类型是否可用。</summary>
    public bool SnapshotTypeAvailable => true;

    /// <summary>Gets whether runtime copied snapshot APIs are available. 获取 runtime copied snapshot API 是否可用。</summary>
    public bool RuntimeSnapshotAvailable { get; }

    /// <summary>Gets whether refitter copied snapshot APIs are available. 获取 refitter copied snapshot API 是否可用。</summary>
    public bool RefitterSnapshotAvailable { get; }

    /// <summary>Gets whether owner presence controls are available. 获取 owner presence 控制是否可用。</summary>
    public bool PresenceControlsAvailable { get; }

    /// <summary>Gets whether owner clear controls are available. 获取 owner clear 控制是否可用。</summary>
    public bool ClearControlsAvailable { get; }

    /// <summary>Gets whether a copied snapshot was observed while evaluating this gate. 获取评估时是否传入了已复制 snapshot。</summary>
    public bool CopiedSnapshotObserved { get; }

    /// <summary>Gets whether the copied snapshot reported an attached recorder. 获取已复制 snapshot 是否报告 recorder 已附加。</summary>
    public bool HasRecorder { get; }

    /// <summary>Gets the copied error count. 获取已复制 error count。</summary>
    public int ErrorCount { get; }

    /// <summary>Gets whether the copied snapshot reported overflow. 获取已复制 snapshot 是否报告 overflow。</summary>
    public bool HasOverflowed { get; }

    /// <summary>Gets the copied record count. 获取已复制 error record 数量。</summary>
    public int CopiedRecordCount { get; }

    /// <summary>Gets whether copied diagnostics APIs are ready. 获取 copied diagnostics API 是否就绪。</summary>
    public bool CopiedDiagnosticsReady => SnapshotTypeAvailable && RuntimeSnapshotAvailable && RefitterSnapshotAvailable;

    /// <summary>Gets whether the observed snapshot copied all reported records. 获取已观察 snapshot 是否复制了全部记录。</summary>
    public bool SnapshotRecordCopyReady { get; }

    /// <summary>Gets whether a native recorder pointer is exposed through this public surface. 获取 public surface 是否暴露原生 recorder 指针。</summary>
    public bool RecorderPointerExposed => false;

    /// <summary>Gets whether this gate produces a native recorder pointer. 获取该设计门是否产生原生 recorder 指针。</summary>
    public bool RecorderPointerProduced => false;

    /// <summary>Gets whether a borrowed recorder pointer can escape the public surface. 获取 borrowed recorder 指针是否可能逃逸 public surface。</summary>
    public bool BorrowedRecorderPointerEscaped => false;

    /// <summary>Gets whether public APIs expose recorder reference-count ownership control. 获取 public API 是否暴露 recorder 引用计数 ownership 控制。</summary>
    public bool RefCountPublicOwnershipControl => false;

    /// <summary>Gets whether public APIs expose direct recorder interface-info ownership control. 获取 public API 是否暴露 direct recorder interface-info ownership 控制。</summary>
    public bool InterfaceInfoPublicOwnershipControl => false;

    /// <summary>Gets whether direct recorder ownership rows intentionally remain deferred. 获取 direct recorder ownership 行是否继续 deferred。</summary>
    public bool DirectRecorderOwnershipDeferred => true;

    /// <summary>Gets whether the public surface remains pointer-free. 获取 public surface 是否保持无裸指针。</summary>
    public bool PointerFreeSurfaceReady =>
        !RecorderPointerExposed &&
        !RecorderPointerProduced &&
        !BorrowedRecorderPointerEscaped &&
        !RefCountPublicOwnershipControl &&
        !InterfaceInfoPublicOwnershipControl;

    /// <summary>Gets whether the design gate has enough copied diagnostics to be considered ready. 获取 copied diagnostics 设计门是否就绪。</summary>
    public bool DesignGateReady =>
        LineSupportsErrorRecorder &&
        CopiedDiagnosticsReady &&
        PresenceControlsAvailable &&
        ClearControlsAvailable &&
        SnapshotRecordCopyReady &&
        PointerFreeSurfaceReady;

    /// <summary>Gets whether this gate can be promoted without a design gate. 获取该结果是否可在无 design gate 情况下晋级。</summary>
    public bool CanPromoteWithoutDesignGate => false;

    /// <summary>Gets whether this gate can be promoted without runtime proof. 获取该结果是否可在无 runtime proof 情况下晋级。</summary>
    public bool CanPromoteWithoutRuntimeProof => false;

    /// <summary>Gets whether a full package consumer runtime proof is ready. 获取完整 package consumer runtime proof 是否就绪。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady => false;

    /// <summary>Gets whether this result can be promoted as runtime proof. 获取该结果是否可晋级为 runtime proof。</summary>
    public bool CanPromoteRuntimeProof => false;

    /// <summary>Gets whether runtime proof remains blocked. 获取 runtime proof 是否仍被阻塞。</summary>
    public bool RuntimeProofBlocked => !CanPromoteRuntimeProof;

    /// <summary>Gets whether direct deferred rows are still required. 获取 direct deferred 行是否仍必须保留。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets the candidate TensorRT interfaces covered by this design gate. 获取该设计门覆盖的候选 TensorRT 接口。</summary>
    public ReadOnlyCollection<string> CandidateInterfaces =>
        Array.AsReadOnly(new[]
        {
            "IErrorRecorder"
        });

    /// <summary>Gets the direct deferred methods covered by this design gate. 获取该设计门覆盖的 direct deferred 方法。</summary>
    public ReadOnlyCollection<string> CandidateMethods =>
        Array.AsReadOnly(new[]
        {
            "IErrorRecorder::getInterfaceInfo",
            "IErrorRecorder::getNbErrors",
            "IErrorRecorder::getErrorCode",
            "IErrorRecorder::getErrorDesc",
            "IErrorRecorder::hasOverflowed",
            "IErrorRecorder::incRefCount",
            "IErrorRecorder::decRefCount"
        });

    /// <summary>Gets the copied output mode required before implementation. 获取实现前要求的 copied 输出模式。</summary>
    public string RequiredOutputMode => "owner-scoped copied diagnostics and interface metadata snapshot";

    /// <summary>Gets the next safe implementation step. 获取下一步安全实现动作。</summary>
    public string NextSafeImplementationStep =>
        "Continue proving owner-scoped snapshots and keep direct IErrorRecorder pointer, ref-count, and ownership APIs deferred.";

    /// <summary>Gets the direct candidate method count. 获取 direct 候选方法数量。</summary>
    public int CandidateMethodCount => CandidateMethods.Count;

    /// <summary>Gets copied blocked prerequisites. 获取已复制的阻塞前置项。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 获取已复制阻塞前置项数量。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the design gate status. 获取设计门状态。</summary>
    public string Status => DesignGateReady ? "design-gate-ready" : "design-gate-blocked";

    /// <summary>Gets a compact diagnostic summary. 获取简短诊断摘要。</summary>
    public string Diagnostic =>
        "error-recorder-diagnostics-design-gate; RuntimeEvidenceKind=design-gate; " +
        "IsRuntimeExecutionEvidence=False; IsRuntimeExecutionProof=False; " +
        "CopiedDiagnosticsReady=" + CopiedDiagnosticsReady + "; " +
        "PointerFreeSurfaceReady=" + PointerFreeSurfaceReady + "; " +
        "RecorderPointerExposed=False; RecorderPointerProduced=False; " +
        "RefCountPublicOwnershipControl=False; InterfaceInfoPublicOwnershipControl=False; " +
        "RequiredOutputMode=" + RequiredOutputMode + "; CandidateMethodCount=" + CandidateMethodCount + "; " +
        "DirectRecorderOwnershipDeferred=True; CanPromoteWithoutRuntimeProof=False; " +
        "RuntimeProofBlocked=True; BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 返回简短诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:copied={CopiedRecordCount}:proof={IsRuntimeExecutionProof}";
    }
}
