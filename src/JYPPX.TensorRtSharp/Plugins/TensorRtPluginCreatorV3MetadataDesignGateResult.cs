using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports PluginCreatorV3 metadata design status without exposing plugin creator pointers.
/// 报告 PluginCreatorV3 metadata 设计状态，不暴露 plugin creator 指针。
/// </summary>
public readonly struct TensorRtPluginCreatorV3MetadataDesignGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtPluginCreatorV3MetadataDesignGateResult(
        TensorRtApiLine line,
        bool lineSupportsPluginCreatorV3,
        bool registryInventorySnapshotReady,
        bool copiedIdentityReady,
        bool copiedFieldMetadataReady,
        bool copiedInterfaceInfoReady,
        bool pluginCreationModeled,
        bool borrowedCreatorLifetimeModeled,
        string[] blockedPrerequisites)
    {
        Line = line;
        LineSupportsPluginCreatorV3 = lineSupportsPluginCreatorV3;
        RegistryInventorySnapshotReady = registryInventorySnapshotReady;
        CopiedIdentityReady = copiedIdentityReady;
        CopiedFieldMetadataReady = copiedFieldMetadataReady;
        CopiedInterfaceInfoReady = copiedInterfaceInfoReady;
        PluginCreationModeled = pluginCreationModeled;
        BorrowedCreatorLifetimeModeled = borrowedCreatorLifetimeModeled;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 获取该设计门的证据标记。</summary>
    public string EvidenceKind => "plugin-creator-v3-metadata-design-gate";

    /// <summary>Gets the diagnostics kind represented by this gate. 获取该设计门代表的诊断类型。</summary>
    public string DiagnosticsKind => "plugin-creator-v3-metadata";

    /// <summary>Gets the runtime evidence kind. 获取 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind => "design-gate";

    /// <summary>Gets whether this gate is runtime execution evidence. 获取该设计门是否为 runtime execution evidence。</summary>
    public bool IsRuntimeExecutionEvidence => false;

    /// <summary>Gets whether this gate may be promoted as runtime execution proof. 获取该设计门是否可晋级为 runtime execution proof。</summary>
    public bool IsRuntimeExecutionProof => false;

    /// <summary>Gets the TensorRT API line. 获取 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether the selected TensorRT line supports PluginCreatorV3 metadata. 获取当前 TensorRT line 是否支持 PluginCreatorV3 metadata。</summary>
    public bool LineSupportsPluginCreatorV3 { get; }

    /// <summary>Gets whether copied plugin registry inventory snapshots are ready. 获取 copied plugin registry inventory snapshot 是否就绪。</summary>
    public bool RegistryInventorySnapshotReady { get; }

    /// <summary>Gets whether creator name/version/namespace copies are ready. 获取 creator name/version/namespace copied 输出是否就绪。</summary>
    public bool CopiedIdentityReady { get; }

    /// <summary>Gets whether plugin field metadata copies are ready. 获取 plugin field metadata copied 输出是否就绪。</summary>
    public bool CopiedFieldMetadataReady { get; }

    /// <summary>Gets whether interface metadata copies are ready. 获取 interface metadata copied 输出是否就绪。</summary>
    public bool CopiedInterfaceInfoReady { get; }

    /// <summary>Gets whether plugin creation ownership has been modeled. 获取 plugin creation ownership 是否已建模。</summary>
    public bool PluginCreationModeled { get; }

    /// <summary>Gets whether borrowed creator lifetime has been modeled as a public ownership surface. 获取 borrowed creator lifetime 是否已建模为 public ownership surface。</summary>
    public bool BorrowedCreatorLifetimeModeled { get; }

    /// <summary>Gets whether public APIs expose native plugin creator pointers. 获取 public API 是否暴露原生 plugin creator 指针。</summary>
    public bool PluginCreatorPointerExposed => false;

    /// <summary>Gets whether public APIs expose borrowed plugin creator handles. 获取 public API 是否暴露 borrowed plugin creator handle。</summary>
    public bool BorrowedPluginCreatorHandleExposed => false;

    /// <summary>Gets whether public APIs can create plugin instances. 获取 public API 是否可创建 plugin instance。</summary>
    public bool PluginInstanceCreationEnabled => false;

    /// <summary>Gets whether plugin resource acquire/release is public. 获取 plugin resource acquire/release 是否公开。</summary>
    public bool PluginResourceOwnershipControlEnabled => false;

    /// <summary>Gets whether direct PluginCreatorV3 createPlugin rows intentionally remain deferred. 获取 direct PluginCreatorV3 createPlugin 行是否继续 deferred。</summary>
    public bool DirectCreatePluginRowsDeferred => true;

    /// <summary>Gets whether direct borrowed creator list rows intentionally remain deferred. 获取 direct borrowed creator list 行是否继续 deferred。</summary>
    public bool DirectBorrowedCreatorListRowsDeferred => true;

    /// <summary>Gets whether copied metadata output shape is ready. 获取 copied metadata 输出形态是否就绪。</summary>
    public bool CopiedMetadataShapeReady =>
        RegistryInventorySnapshotReady &&
        CopiedIdentityReady &&
        CopiedFieldMetadataReady &&
        CopiedInterfaceInfoReady;

    /// <summary>Gets whether the public surface remains pointer-free. 获取 public surface 是否保持无裸指针。</summary>
    public bool PointerFreeSurfaceReady =>
        !PluginCreatorPointerExposed &&
        !BorrowedPluginCreatorHandleExposed &&
        !PluginInstanceCreationEnabled &&
        !PluginResourceOwnershipControlEnabled;

    /// <summary>Gets whether this design gate is ready as non-proof evidence. 获取该 design gate 作为非 proof 证据是否就绪。</summary>
    public bool DesignGateReady =>
        LineSupportsPluginCreatorV3 &&
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

    /// <summary>Gets whether this design gate allows deleting deferred records. 获取该设计门是否允许删除 deferred 记录。</summary>
    public bool CanDeleteDeferredRecord => false;

    /// <summary>Gets the candidate TensorRT interfaces covered by this design gate. 获取该设计门覆盖的候选 TensorRT 接口。</summary>
    public ReadOnlyCollection<string> CandidateInterfaces =>
        Array.AsReadOnly(new[]
        {
            "IPluginCreatorV3One",
            "IVersionedInterface"
        });

    /// <summary>Gets the direct deferred or safe-alternative methods covered by this design gate. 获取该设计门覆盖的 direct deferred 或 safe-alternative 方法。</summary>
    public ReadOnlyCollection<string> CandidateMethods =>
        Array.AsReadOnly(new[]
        {
            "IPluginCreatorV3One::getPluginName",
            "IPluginCreatorV3One::getPluginVersion",
            "IPluginCreatorV3One::getPluginNamespace",
            "IPluginCreatorV3One::getFieldNames",
            "IPluginCreatorV3One::getInterfaceInfo",
            "IVersionedInterface::getInterfaceInfo"
        });

    /// <summary>Gets the copied output mode required before implementation. 获取实现前要求的 copied 输出模式。</summary>
    public string RequiredOutputMode => "copied string, copied field metadata, and managed interface metadata snapshot";

    /// <summary>Gets the next safe implementation step. 获取下一步安全实现动作。</summary>
    public string NextSafeImplementationStep =>
        "Reuse plugin registry inventory snapshots for PluginCreatorV3 metadata, then keep createPlugin/resource/callback ownership deferred.";

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
        "plugin-creator-v3-metadata-design-gate; RuntimeEvidenceKind=design-gate; " +
        "IsRuntimeExecutionEvidence=False; IsRuntimeExecutionProof=False; " +
        "RegistryInventorySnapshotReady=" + RegistryInventorySnapshotReady + "; " +
        "CopiedMetadataShapeReady=" + CopiedMetadataShapeReady + "; " +
        "PointerFreeSurfaceReady=" + PointerFreeSurfaceReady + "; " +
        "PluginCreatorPointerExposed=False; BorrowedPluginCreatorHandleExposed=False; " +
        "PluginInstanceCreationEnabled=False; PluginResourceOwnershipControlEnabled=False; " +
        "DirectCreatePluginRowsDeferred=True; DirectBorrowedCreatorListRowsDeferred=True; " +
        "RequiredOutputMode=" + RequiredOutputMode + "; CandidateMethodCount=" + CandidateMethodCount + "; " +
        "CanPromoteWithoutRuntimeProof=False; RuntimeProofBlocked=True; " +
        "DeferredRowsStillRequired=True; BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 返回简短诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:pointerFree={PointerFreeSurfaceReady}:proof={IsRuntimeExecutionProof}:runtimeProof={CanPromoteRuntimeProof}:deleteDeferred={CanDeleteDeferredRecord}";
    }
}
