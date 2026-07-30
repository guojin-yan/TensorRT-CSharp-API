using System;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports RNNv2 borrowed-state design status without exposing native pointers.
/// 报告 RNNv2 borrowed-state 设计状态，不暴露原生指针。
/// </summary>
public readonly struct TensorRtRnnV2BorrowedStateDesignGateResult
{
    private static readonly string[] SelectedMethods =
    {
        "IRNNv2Layer::getDataLength",
        "IRNNv2Layer::getBiasForGate",
        "IRNNv2Layer::getCellState",
        "IRNNv2Layer::getHiddenState",
        "IRNNv2Layer::getSequenceLengths",
        "IRNNv2Layer::getWeightsForGate"
    };

    private static readonly string[] DeferredMethods = Array.Empty<string>();

    private readonly string[] _blockedPrerequisites;

    internal TensorRtRnnV2BorrowedStateDesignGateResult(
        TensorRtApiLine line,
        bool dataLengthScalarPromoted,
        bool networkOwnedTensorReferencePolicyReady,
        bool gateWeightSnapshotCopyReady,
        bool ownerLifetimeKnown,
        string[] blockedPrerequisites)
    {
        Line = line;
        DataLengthScalarPromoted = dataLengthScalarPromoted;
        NetworkOwnedTensorReferencePolicyReady = networkOwnedTensorReferencePolicyReady;
        GateWeightSnapshotCopyReady = gateWeightSnapshotCopyReady;
        OwnerLifetimeKnown = ownerLifetimeKnown;
        _blockedPrerequisites = blockedPrerequisites == null
            ? Array.Empty<string>()
            : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the design-gate evidence marker. 获取设计门证据标记。</summary>
    public string EvidenceKind => "rnnv2-borrowed-state-design-gate";

    /// <summary>Gets the diagnostics kind. 获取诊断类型。</summary>
    public string DiagnosticsKind => "rnnv2-borrowed-state";

    /// <summary>Gets the runtime evidence kind. 获取 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind => "design-gate";

    /// <summary>Gets the TensorRT API line. 获取 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether the selected line supports RNNv2. 获取所选 line 是否支持 RNNv2。</summary>
    public bool LineSupportsRnnV2 => Line == TensorRtApiLine.TensorRt8;

    /// <summary>Gets whether getDataLength is promoted as a copied scalar. 获取 getDataLength 是否已按 copied scalar 提升。</summary>
    public bool DataLengthScalarPromoted { get; }

    /// <summary>Gets whether a safe network-owned tensor reference policy exists. 获取安全 network-owned tensor reference policy 是否存在。</summary>
    public bool NetworkOwnedTensorReferencePolicyReady { get; }

    /// <summary>Gets whether copied gate-weight snapshots are ready. 获取 copied gate-weight snapshot 是否就绪。</summary>
    public bool GateWeightSnapshotCopyReady { get; }

    /// <summary>Gets whether the borrowed state owner lifetime is proven. 获取 borrowed state owner lifetime 是否已证明。</summary>
    public bool OwnerLifetimeKnown { get; }

    /// <summary>Gets whether a borrowed tensor pointer is exposed. 获取是否暴露 borrowed tensor pointer。</summary>
    public bool BorrowedTensorPointerExposed => false;

    /// <summary>Gets whether a borrowed weights pointer is exposed. 获取是否暴露 borrowed weights pointer。</summary>
    public bool BorrowedWeightsPointerExposed => false;

    /// <summary>Gets whether borrowed state can escape the call. 获取 borrowed state 是否可逃逸调用边界。</summary>
    public bool BorrowedStateEscapesCall => false;

    /// <summary>Gets whether this public design surface is pointer-free. 获取 public design surface 是否无裸指针。</summary>
    public bool PointerFreeSurfaceReady =>
        !BorrowedTensorPointerExposed &&
        !BorrowedWeightsPointerExposed &&
        !BorrowedStateEscapesCall;

    /// <summary>Gets whether borrowed snapshots may be promoted. 获取 borrowed snapshot 是否可提升。</summary>
    public bool BorrowedSnapshotPromotionReady =>
        NetworkOwnedTensorReferencePolicyReady &&
        GateWeightSnapshotCopyReady &&
        OwnerLifetimeKnown;

    /// <summary>Gets whether the design boundary is complete. 获取设计边界是否完整。</summary>
    public bool DesignGateReady =>
        LineSupportsRnnV2 &&
        DataLengthScalarPromoted &&
        BorrowedSnapshotPromotionReady &&
        PointerFreeSurfaceReady;

    /// <summary>Gets whether this is runtime execution evidence. 获取是否为 runtime execution evidence。</summary>
    public bool IsRuntimeExecutionEvidence => false;

    /// <summary>Gets whether this is runtime execution proof. 获取是否为 runtime execution proof。</summary>
    public bool IsRuntimeExecutionProof => false;

    /// <summary>Gets whether runtime proof can be promoted. 获取是否可提升 runtime proof。</summary>
    public bool CanPromoteRuntimeProof => false;

    /// <summary>Gets whether direct borrowed rows remain deferred. 获取 direct borrowed 行是否继续 deferred。</summary>
    public bool DeferredBorrowedRowsStillRequired => false;

    /// <summary>Gets the selected triage row count before promotion. 获取提升前选中的 triage 行数。</summary>
    public int SelectedTriageRowCount => 12;

    /// <summary>Gets the promoted scalar triage row count. 获取已提升 scalar triage 行数。</summary>
    public int PromotedScalarTriageRowCount => 12;

    /// <summary>Gets the remaining deferred triage row count. 获取剩余 deferred triage 行数。</summary>
    public int RemainingDeferredTriageRowCount => 0;

    /// <summary>Gets selected methods. 获取选中方法。</summary>
    public ReadOnlyCollection<string> SelectedCandidateMethods =>
        Array.AsReadOnly(SelectedMethods);

    /// <summary>Gets methods that remain deferred. 获取继续 deferred 的方法。</summary>
    public ReadOnlyCollection<string> DeferredBorrowedMethods =>
        Array.AsReadOnly(DeferredMethods);

    /// <summary>Gets copied blockers. 获取已复制阻塞项。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the current status. 获取当前状态。</summary>
    public string Status => DesignGateReady
        ? "safe-wrapper-surface-ready-runtime-proof-pending"
        : "design-gate-blocked";

    /// <summary>Gets the next safe implementation step. 获取下一步安全实现动作。</summary>
    public string NextSafeImplementationStep =>
        "Construct a real TensorRT 8 RNNv2 network on a compatible host and capture package-consumer runtime proof.";

    /// <summary>Gets a compact diagnostic summary. 获取简短诊断摘要。</summary>
    public string Diagnostic =>
        "rnnv2-borrowed-state-design-gate; RuntimeEvidenceKind=design-gate; " +
        "DataLengthScalarPromoted=" + DataLengthScalarPromoted + "; " +
        "SelectedTriageRowCount=12; PromotedScalarTriageRowCount=" + PromotedScalarTriageRowCount +
        "; RemainingDeferredTriageRowCount=" + RemainingDeferredTriageRowCount + "; " +
        "BorrowedTensorPointerExposed=False; BorrowedWeightsPointerExposed=False; BorrowedStateEscapesCall=False; " +
        "BorrowedSnapshotPromotionReady=" + BorrowedSnapshotPromotionReady + "; " +
        "DeferredBorrowedRowsStillRequired=" + DeferredBorrowedRowsStillRequired + "; CanPromoteRuntimeProof=False.";
}
