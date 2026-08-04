using System;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports ILoggerFinder metadata design status without exposing native finder pointers.
/// 报告 ILoggerFinder metadata 设计状态，不暴露原生 finder 指针。
/// </summary>
public readonly struct TensorRtLoggerFinderMetadataDesignGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtLoggerFinderMetadataDesignGateResult(
        TensorRtApiLine line,
        bool lineSupportsLoggerFinder,
        bool copiedInterfaceInfoMetadataReady,
        bool finderOwnerLifetimeModeled,
        bool loggerCallbackOwnershipModeled,
        string[] blockedPrerequisites)
    {
        Line = line;
        LineSupportsLoggerFinder = lineSupportsLoggerFinder;
        CopiedInterfaceInfoMetadataReady = copiedInterfaceInfoMetadataReady;
        FinderOwnerLifetimeModeled = finderOwnerLifetimeModeled;
        LoggerCallbackOwnershipModeled = loggerCallbackOwnershipModeled;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 获取该设计门的证据标记。</summary>
    public string EvidenceKind => "logger-finder-metadata-design-gate";

    /// <summary>Gets the diagnostics kind represented by this gate. 获取该设计门代表的诊断类型。</summary>
    public string DiagnosticsKind => "logger-finder-metadata";

    /// <summary>Gets the runtime evidence kind. 获取 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind => "design-gate";

    /// <summary>Gets whether this gate is runtime execution evidence. 获取该设计门是否为 runtime execution evidence。</summary>
    public bool IsRuntimeExecutionEvidence => false;

    /// <summary>Gets whether this gate may be promoted as runtime execution proof. 获取该设计门是否可晋级为 runtime execution proof。</summary>
    public bool IsRuntimeExecutionProof => false;

    /// <summary>Gets the TensorRT API line. 获取 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether the selected TensorRT line supports ILoggerFinder. 获取当前 TensorRT line 是否支持 ILoggerFinder。</summary>
    public bool LineSupportsLoggerFinder { get; }

    /// <summary>Gets whether copied interface-info metadata shape is ready. 获取 copied interface-info metadata 形态是否就绪。</summary>
    public bool CopiedInterfaceInfoMetadataReady { get; }

    /// <summary>Gets whether finder owner lifetime has been modeled. 获取 finder owner lifetime 是否已建模。</summary>
    public bool FinderOwnerLifetimeModeled { get; }

    /// <summary>Gets whether logger callback ownership has been modeled. 获取 logger callback ownership 是否已建模。</summary>
    public bool LoggerCallbackOwnershipModeled { get; }

    /// <summary>Gets whether public APIs expose native finder pointers. 获取 public API 是否暴露原生 finder 指针。</summary>
    public bool LoggerFinderPointerExposed => false;

    /// <summary>Gets whether public APIs expose logger callback pointers. 获取 public API 是否暴露 logger callback 指针。</summary>
    public bool LoggerCallbackPointerExposed => false;

    /// <summary>Gets whether public APIs can invoke logger finder callbacks. 获取 public API 是否可调用 logger finder callback。</summary>
    public bool LoggerFinderCallbackInvocationEnabled => false;

    /// <summary>Gets whether direct logger finder rows intentionally remain deferred. 获取 direct logger finder 行是否继续 deferred。</summary>
    public bool DirectLoggerFinderRowsDeferred => true;

    /// <summary>Gets whether copied logger finder metadata shape is ready. 获取 copied logger finder metadata 形态是否就绪。</summary>
    public bool CopiedMetadataShapeReady => CopiedInterfaceInfoMetadataReady;

    /// <summary>Gets whether the public surface remains pointer-free. 获取 public surface 是否保持无裸指针。</summary>
    public bool PointerFreeSurfaceReady =>
        !LoggerFinderPointerExposed &&
        !LoggerCallbackPointerExposed &&
        !LoggerFinderCallbackInvocationEnabled;

    /// <summary>Gets whether this design gate is ready as non-proof evidence. 获取该 design gate 作为非 proof 证据是否就绪。</summary>
    public bool DesignGateReady =>
        LineSupportsLoggerFinder &&
        CopiedMetadataShapeReady &&
        PointerFreeSurfaceReady;

    /// <summary>Gets whether this gate can be promoted without runtime proof. 获取该结果是否可在无 runtime proof 情况下晋级。</summary>
    public bool CanPromoteWithoutRuntimeProof => false;

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
            "ILoggerFinder"
        });

    /// <summary>Gets the direct deferred methods covered by this design gate. 获取该设计门覆盖的 direct deferred 方法。</summary>
    public ReadOnlyCollection<string> CandidateMethods =>
        Array.AsReadOnly(new[]
        {
            "ILoggerFinder::getInterfaceInfo"
        });

    /// <summary>Gets the copied output mode required before implementation. 获取实现前要求的 copied 输出模式。</summary>
    public string RequiredOutputMode => "copied logger finder interface metadata snapshot";

    /// <summary>Gets the next safe implementation step. 获取下一步安全实现动作。</summary>
    public string NextSafeImplementationStep =>
        "Model ILoggerFinder owner lifetime and copied interface metadata before enabling logger callback lookup or invocation.";

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
        "logger-finder-metadata-design-gate; RuntimeEvidenceKind=design-gate; " +
        "IsRuntimeExecutionEvidence=False; IsRuntimeExecutionProof=False; " +
        "CopiedMetadataShapeReady=" + CopiedMetadataShapeReady + "; " +
        "PointerFreeSurfaceReady=" + PointerFreeSurfaceReady + "; " +
        "LoggerFinderPointerExposed=False; LoggerCallbackPointerExposed=False; " +
        "LoggerFinderCallbackInvocationEnabled=False; DirectLoggerFinderRowsDeferred=True; " +
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
