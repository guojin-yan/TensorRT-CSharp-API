using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports algorithm snapshot design status without exposing TensorRT algorithm pointers.
/// 报告 algorithm snapshot 设计状态，不暴露 TensorRT algorithm 指针。
/// </summary>
public readonly struct TensorRtAlgorithmSnapshotDesignGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtAlgorithmSnapshotDesignGateResult(
        TensorRtApiLine line,
        bool lineSupportsAlgorithmSelector,
        bool selectorCallbackOwnerModeled,
        bool algorithmResultLifetimeModeled,
        bool copiedTimingWorkspaceShapeReady,
        bool copiedContextShapeReady,
        bool copiedIoInfoShapeReady,
        bool copiedVariantShapeReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        LineSupportsAlgorithmSelector = lineSupportsAlgorithmSelector;
        SelectorCallbackOwnerModeled = selectorCallbackOwnerModeled;
        AlgorithmResultLifetimeModeled = algorithmResultLifetimeModeled;
        CopiedTimingWorkspaceShapeReady = copiedTimingWorkspaceShapeReady;
        CopiedContextShapeReady = copiedContextShapeReady;
        CopiedIoInfoShapeReady = copiedIoInfoShapeReady;
        CopiedVariantShapeReady = copiedVariantShapeReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 获取该设计门的证据标记。</summary>
    public string EvidenceKind => "algorithm-snapshot-design-gate";

    /// <summary>Gets the diagnostics kind represented by this gate. 获取该设计门代表的诊断类型。</summary>
    public string DiagnosticsKind => "algorithm-snapshot";

    /// <summary>Gets the runtime evidence kind. 获取 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind => "design-gate";

    /// <summary>Gets whether this gate is runtime execution evidence. 获取该设计门是否为 runtime execution evidence。</summary>
    public bool IsRuntimeExecutionEvidence => false;

    /// <summary>Gets whether this gate may be promoted as runtime execution proof. 获取该设计门是否可晋级为 runtime execution proof。</summary>
    public bool IsRuntimeExecutionProof => false;

    /// <summary>Gets the TensorRT API line. 获取 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether the selected TensorRT line supports algorithm selector objects. 获取当前 TensorRT line 是否支持 algorithm selector 对象。</summary>
    public bool LineSupportsAlgorithmSelector { get; }

    /// <summary>Gets whether selector callback owner lifetime has been modeled. 获取 selector callback owner lifetime 是否已建模。</summary>
    public bool SelectorCallbackOwnerModeled { get; }

    /// <summary>Gets whether algorithm callback result lifetime has been modeled. 获取 algorithm callback result lifetime 是否已建模。</summary>
    public bool AlgorithmResultLifetimeModeled { get; }

    /// <summary>Gets whether timing/workspace copied shape is ready. 获取 timing/workspace copied 形态是否就绪。</summary>
    public bool CopiedTimingWorkspaceShapeReady { get; }

    /// <summary>Gets whether context copied shape is ready. 获取 context copied 形态是否就绪。</summary>
    public bool CopiedContextShapeReady { get; }

    /// <summary>Gets whether IO info copied shape is ready. 获取 IO info copied 形态是否就绪。</summary>
    public bool CopiedIoInfoShapeReady { get; }

    /// <summary>Gets whether variant copied shape is ready. 获取 variant copied 形态是否就绪。</summary>
    public bool CopiedVariantShapeReady { get; }

    /// <summary>Gets whether public APIs expose native algorithm pointers. 获取 public API 是否暴露原生 algorithm 指针。</summary>
    public bool AlgorithmPointerExposed => false;

    /// <summary>Gets whether public APIs expose native algorithm context pointers. 获取 public API 是否暴露原生 algorithm context 指针。</summary>
    public bool AlgorithmContextPointerExposed => false;

    /// <summary>Gets whether public APIs expose native algorithm IO info pointers. 获取 public API 是否暴露原生 algorithm IO info 指针。</summary>
    public bool AlgorithmIoInfoPointerExposed => false;

    /// <summary>Gets whether public APIs expose native algorithm variant pointers. 获取 public API 是否暴露原生 algorithm variant 指针。</summary>
    public bool AlgorithmVariantPointerExposed => false;

    /// <summary>Gets whether algorithm selector callback trampolines are public. 获取 algorithm selector callback trampoline 是否公开。</summary>
    public bool AlgorithmSelectorCallbackTrampolineEnabled => false;

    /// <summary>Gets whether direct algorithm rows intentionally remain deferred. 获取 direct algorithm 行是否继续 deferred。</summary>
    public bool DirectAlgorithmRowsDeferred => true;

    /// <summary>Gets whether direct algorithm selector callback rows intentionally remain deferred. 获取 direct algorithm selector callback 行是否继续 deferred。</summary>
    public bool DirectSelectorCallbackRowsDeferred => true;

    /// <summary>Gets whether copied algorithm metadata shape is ready. 获取 copied algorithm metadata 形态是否就绪。</summary>
    public bool CopiedMetadataShapeReady =>
        CopiedTimingWorkspaceShapeReady &&
        CopiedContextShapeReady &&
        CopiedIoInfoShapeReady &&
        CopiedVariantShapeReady;

    /// <summary>Gets whether the public surface remains pointer-free. 获取 public surface 是否保持无裸指针。</summary>
    public bool PointerFreeSurfaceReady =>
        !AlgorithmPointerExposed &&
        !AlgorithmContextPointerExposed &&
        !AlgorithmIoInfoPointerExposed &&
        !AlgorithmVariantPointerExposed &&
        !AlgorithmSelectorCallbackTrampolineEnabled;

    /// <summary>Gets whether this design gate is ready as non-proof evidence. 获取该 design gate 作为非 proof 证据是否就绪。</summary>
    public bool DesignGateReady =>
        LineSupportsAlgorithmSelector &&
        CopiedMetadataShapeReady &&
        PointerFreeSurfaceReady;

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
            "IAlgorithm",
            "IAlgorithmContext",
            "IAlgorithmIOInfo",
            "IAlgorithmVariant",
            "IAlgorithmSelector"
        });

    /// <summary>Gets the direct deferred methods covered by this design gate. 获取该设计门覆盖的 direct deferred 方法。</summary>
    public ReadOnlyCollection<string> CandidateMethods =>
        Array.AsReadOnly(new[]
        {
            "IAlgorithm::getTimingMSec",
            "IAlgorithm::getWorkspaceSize",
            "IAlgorithm::getAlgorithmVariant",
            "IAlgorithm::getAlgorithmIOInfoByIndex",
            "IAlgorithmContext::getName",
            "IAlgorithmContext::getNbInputs",
            "IAlgorithmContext::getNbOutputs",
            "IAlgorithmContext::getDimensions",
            "IAlgorithmIOInfo::getDataType",
            "IAlgorithmIOInfo::getStrides",
            "IAlgorithmIOInfo::getVectorizedDim",
            "IAlgorithmVariant::getImplementation",
            "IAlgorithmVariant::getTactic",
            "IAlgorithmSelector::getInterfaceInfo"
        });

    /// <summary>Gets the copied output mode required before implementation. 获取实现前要求的 copied 输出模式。</summary>
    public string RequiredOutputMode => "callback-scoped copied algorithm timing, workspace, context, IO, and variant snapshot";

    /// <summary>Gets the next safe implementation step. 获取下一步安全实现动作。</summary>
    public string NextSafeImplementationStep =>
        "Design an algorithm selector owner that copies callback-scoped algorithm result metadata before callback return without exposing borrowed handles.";

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
        "algorithm-snapshot-design-gate; RuntimeEvidenceKind=design-gate; " +
        "IsRuntimeExecutionEvidence=False; IsRuntimeExecutionProof=False; " +
        "SelectorCallbackOwnerModeled=" + SelectorCallbackOwnerModeled + "; " +
        "AlgorithmResultLifetimeModeled=" + AlgorithmResultLifetimeModeled + "; " +
        "CopiedMetadataShapeReady=" + CopiedMetadataShapeReady + "; " +
        "PointerFreeSurfaceReady=" + PointerFreeSurfaceReady + "; " +
        "AlgorithmPointerExposed=False; AlgorithmContextPointerExposed=False; " +
        "AlgorithmIoInfoPointerExposed=False; AlgorithmVariantPointerExposed=False; " +
        "AlgorithmSelectorCallbackTrampolineEnabled=False; " +
        "DirectAlgorithmRowsDeferred=True; DirectSelectorCallbackRowsDeferred=True; " +
        "RequiredOutputMode=" + RequiredOutputMode + "; CandidateMethodCount=" + CandidateMethodCount + "; " +
        "CanPromoteWithoutRuntimeProof=False; RuntimeProofBlocked=True; " +
        "DeferredRowsStillRequired=True; BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 返回简短诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:pointerFree={PointerFreeSurfaceReady}:proof={IsRuntimeExecutionProof}";
    }
}
