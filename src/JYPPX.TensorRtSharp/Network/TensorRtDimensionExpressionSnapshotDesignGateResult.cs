using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports dimension expression snapshot design status without exposing native expression pointers.
/// 报告 dimension expression snapshot 设计状态，不暴露原生 expression 指针。
/// </summary>
/// <remarks>
/// This result is a design gate. It keeps direct <c>IDimensionExpr</c> and <c>IExprBuilder</c> rows deferred
/// until owner lifetime, callback lifetime, and runtime proof exist.
/// 该结果是设计门禁；在 owner lifetime、callback lifetime 和 runtime proof 出现前，direct <c>IDimensionExpr</c>
/// 与 <c>IExprBuilder</c> 行继续 deferred。
/// </remarks>
public readonly struct TensorRtDimensionExpressionSnapshotDesignGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtDimensionExpressionSnapshotDesignGateResult(
        TensorRtApiLine line,
        bool lineSupportsDimensionExpression,
        bool lineSupportsSizeTensor,
        bool ownerLifetimeKnown,
        bool constantSnapshotCopyReady,
        bool sizeTensorMetadataCopyReady,
        bool exprBuilderOwnershipModeled,
        bool pluginShapeCallbackLifetimeModeled,
        string[] blockedPrerequisites)
    {
        Line = line;
        LineSupportsDimensionExpression = lineSupportsDimensionExpression;
        LineSupportsSizeTensor = lineSupportsSizeTensor;
        OwnerLifetimeKnown = ownerLifetimeKnown;
        ConstantSnapshotCopyReady = constantSnapshotCopyReady;
        SizeTensorMetadataCopyReady = sizeTensorMetadataCopyReady;
        ExprBuilderOwnershipModeled = exprBuilderOwnershipModeled;
        PluginShapeCallbackLifetimeModeled = pluginShapeCallbackLifetimeModeled;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 获取该设计门的证据标记。</summary>
    public string EvidenceKind => "dimension-expression-snapshot-design-gate";

    /// <summary>Gets the diagnostics kind represented by this gate. 获取该设计门代表的诊断类型。</summary>
    public string DiagnosticsKind => "dimension-expression-snapshot";

    /// <summary>Gets the runtime evidence kind. 获取 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind => "design-gate";

    /// <summary>Gets whether this gate is runtime execution evidence. 获取该设计门是否为 runtime execution evidence。</summary>
    public bool IsRuntimeExecutionEvidence => false;

    /// <summary>Gets whether this gate may be promoted as runtime execution proof. 获取该设计门是否可晋级为 runtime execution proof。</summary>
    public bool IsRuntimeExecutionProof => false;

    /// <summary>Gets the TensorRT API line. 获取 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether the selected TensorRT line supports dimension expressions. 获取当前 TensorRT line 是否支持 dimension expression。</summary>
    public bool LineSupportsDimensionExpression { get; }

    /// <summary>Gets whether the selected TensorRT line has IDimensionExpr::isSizeTensor. 获取当前 TensorRT line 是否有 IDimensionExpr::isSizeTensor。</summary>
    public bool LineSupportsSizeTensor { get; }

    /// <summary>Gets whether a pointer-free snapshot result type is ready. 获取无裸指针 snapshot result 类型是否就绪。</summary>
    public bool SnapshotTypeReady => true;

    /// <summary>Gets whether copied constant metadata shape is ready. 获取 copied constant metadata 形态是否就绪。</summary>
    public bool ConstantSnapshotCopyReady { get; }

    /// <summary>Gets whether copied size-tensor metadata shape is ready for TensorRT 10/11. 获取 TRT10/11 copied size-tensor metadata 形态是否就绪。</summary>
    public bool SizeTensorMetadataCopyReady { get; }

    /// <summary>Gets whether a safe owner object lifetime has been proven. 获取安全 owner object lifetime 是否已证明。</summary>
    public bool OwnerLifetimeKnown { get; }

    /// <summary>Gets whether expression builder ownership and lifetime are modeled. 获取 expression builder ownership/lifetime 是否已建模。</summary>
    public bool ExprBuilderOwnershipModeled { get; }

    /// <summary>Gets whether plugin shape callback dimension expression lifetime is modeled. 获取 plugin shape callback dimension expression lifetime 是否已建模。</summary>
    public bool PluginShapeCallbackLifetimeModeled { get; }

    /// <summary>Gets whether a native dimension expression pointer is exposed through this public surface. 获取 public surface 是否暴露原生 dimension expression 指针。</summary>
    public bool ExpressionPointerExposed => false;

    /// <summary>Gets whether this gate produces a native dimension expression pointer. 获取该设计门是否产生原生 dimension expression 指针。</summary>
    public bool ExpressionPointerProduced => false;

    /// <summary>Gets whether a borrowed dimension expression pointer can escape the public surface. 获取 borrowed dimension expression 指针是否可能逃逸 public surface。</summary>
    public bool BorrowedExpressionPointerEscaped => false;

    /// <summary>Gets whether public APIs expose an expression builder pointer. 获取 public API 是否暴露 expression builder 指针。</summary>
    public bool ExprBuilderPointerExposed => false;

    /// <summary>Gets whether public APIs can create expression nodes. 获取 public API 是否可创建 expression node。</summary>
    public bool ExprBuilderCreationEnabled => false;

    /// <summary>Gets whether public APIs expose expression node ownership control. 获取 public API 是否暴露 expression node ownership 控制。</summary>
    public bool ExpressionNodePublicOwnershipControl => false;

    /// <summary>Gets whether direct dimension expression rows intentionally remain deferred. 获取 direct dimension expression 行是否继续 deferred。</summary>
    public bool DirectDimensionExpressionRowsDeferred => true;

    /// <summary>Gets whether direct expression builder rows intentionally remain deferred. 获取 direct expression builder 行是否继续 deferred。</summary>
    public bool DirectExpressionBuilderRowsDeferred => true;

    /// <summary>Gets whether the public surface remains pointer-free. 获取 public surface 是否保持无裸指针。</summary>
    public bool PointerFreeSurfaceReady =>
        !ExpressionPointerExposed &&
        !ExpressionPointerProduced &&
        !BorrowedExpressionPointerEscaped &&
        !ExprBuilderPointerExposed &&
        !ExprBuilderCreationEnabled &&
        !ExpressionNodePublicOwnershipControl;

    /// <summary>Gets whether copied dimension metadata shape is ready. 获取 copied dimension metadata 形态是否就绪。</summary>
    public bool CopiedMetadataShapeReady =>
        SnapshotTypeReady &&
        ConstantSnapshotCopyReady &&
        SizeTensorMetadataCopyReady;

    /// <summary>Gets whether this design gate is ready as non-proof evidence. 获取该 design gate 作为非 proof 证据是否就绪。</summary>
    public bool DesignGateReady =>
        LineSupportsDimensionExpression &&
        CopiedMetadataShapeReady &&
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
            "IDimensionExpr",
            "IExprBuilder"
        });

    /// <summary>Gets the direct deferred methods covered by this design gate. 获取该设计门覆盖的 direct deferred 方法。</summary>
    public ReadOnlyCollection<string> CandidateMethods =>
        Array.AsReadOnly(new[]
        {
            "IDimensionExpr::isConstant",
            "IDimensionExpr::isSizeTensor",
            "IDimensionExpr::getConstantValue",
            "IExprBuilder::constant",
            "IExprBuilder::operation",
            "IExprBuilder::declareSizeTensor"
        });

    /// <summary>Gets the copied output mode required before implementation. 获取实现前要求的 copied 输出模式。</summary>
    public string RequiredOutputMode => "copied bool/int64 scalar snapshot tied to a proven owner object";

    /// <summary>Gets the next safe implementation step. 获取下一步安全实现动作。</summary>
    public string NextSafeImplementationStep =>
        "Identify a high-level owner that can copy IDimensionExpr constant and size-tensor metadata without exposing borrowed pointers.";

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
        "dimension-expression-snapshot-design-gate; RuntimeEvidenceKind=design-gate; " +
        "IsRuntimeExecutionEvidence=False; IsRuntimeExecutionProof=False; " +
        "SnapshotTypeReady=" + SnapshotTypeReady + "; " +
        "CopiedMetadataShapeReady=" + CopiedMetadataShapeReady + "; " +
        "OwnerLifetimeKnown=" + OwnerLifetimeKnown + "; " +
        "PointerFreeSurfaceReady=" + PointerFreeSurfaceReady + "; " +
        "ExpressionPointerExposed=False; ExpressionPointerProduced=False; " +
        "BorrowedExpressionPointerEscaped=False; ExprBuilderCreationEnabled=False; " +
        "DirectDimensionExpressionRowsDeferred=True; DirectExpressionBuilderRowsDeferred=True; " +
        "RequiredOutputMode=" + RequiredOutputMode + "; CandidateMethodCount=" + CandidateMethodCount + "; " +
        "CanPromoteWithoutRuntimeProof=False; RuntimeProofBlocked=True; " +
        "DeferredRowsStillRequired=True; BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 返回简短诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:ownerLifetime={OwnerLifetimeKnown}:proof={IsRuntimeExecutionProof}";
    }
}
