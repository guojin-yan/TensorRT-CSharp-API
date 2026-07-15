using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates the pointer-free TensorRT INT8 calibrator metadata design gate.
/// 评估 TensorRT INT8 calibrator metadata 的无裸指针设计门。
/// </summary>
/// <remarks>
/// This gate only allows presence and copied metadata planning. It does not invoke calibration callbacks,
/// does not read or write calibration caches, and is not runtime execution proof.
/// 该门禁只允许 presence 与 copied metadata 规划；不会调用 calibration callback，不读写 calibration cache，也不是 runtime execution proof。
/// </remarks>
public static class TensorRtCalibratorMetadataDesignGate
{
    /// <summary>
    /// Evaluates the known public calibrator metadata surface for a TensorRT API line.
    /// 基于已知 public calibrator metadata 边界评估指定 TensorRT API line。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <returns>A pointer-free calibrator metadata design gate result. 无裸指针 calibrator metadata 设计门结果。</returns>
    public static TensorRtCalibratorMetadataDesignGateResult EvaluateKnownSurface(TensorRtApiLine line)
    {
        return Evaluate(
            line,
            presenceProbeAvailable: line == TensorRtApiLine.TensorRt8 || line == TensorRtApiLine.TensorRt10,
            copiedAlgorithmMetadataReady: true,
            copiedInterfaceInfoMetadataReady: line == TensorRtApiLine.TensorRt10 || line == TensorRtApiLine.TensorRt11,
            batchCallbackOwnershipModeled: false,
            cacheBufferOwnershipModeled: false);
    }

    /// <summary>
    /// Evaluates the calibrator metadata design gate from explicit capability flags.
    /// 根据显式能力标记评估 calibrator metadata 设计门。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <param name="presenceProbeAvailable">Whether safe calibrator presence probe is available. 是否已有安全 calibrator presence 查询。</param>
    /// <param name="copiedAlgorithmMetadataReady">Whether copied algorithm metadata shape is ready. copied algorithm metadata 形态是否就绪。</param>
    /// <param name="copiedInterfaceInfoMetadataReady">Whether copied interface-info metadata shape is ready where available. 可用时 copied interface-info metadata 形态是否就绪。</param>
    /// <param name="batchCallbackOwnershipModeled">Whether getBatch callback ownership is modeled. getBatch callback ownership 是否已建模。</param>
    /// <param name="cacheBufferOwnershipModeled">Whether calibration cache buffer ownership is modeled. calibration cache buffer ownership 是否已建模。</param>
    /// <returns>A pointer-free calibrator metadata design gate result. 无裸指针 calibrator metadata 设计门结果。</returns>
    public static TensorRtCalibratorMetadataDesignGateResult Evaluate(
        TensorRtApiLine line,
        bool presenceProbeAvailable,
        bool copiedAlgorithmMetadataReady,
        bool copiedInterfaceInfoMetadataReady,
        bool batchCallbackOwnershipModeled,
        bool cacheBufferOwnershipModeled)
    {
        bool lineSupportsCalibrator =
            line == TensorRtApiLine.TensorRt8 ||
            line == TensorRtApiLine.TensorRt10 ||
            line == TensorRtApiLine.TensorRt11;

        List<string> blockers = new List<string>();
        if (!lineSupportsCalibrator)
        {
            blockers.Add("TensorRT 8, 10, or 11 calibrator line support has not been selected.");
        }

        if (!presenceProbeAvailable)
        {
            blockers.Add("safe builder-config INT8 calibrator presence probe is not available for this TensorRT line.");
        }

        if (!copiedAlgorithmMetadataReady)
        {
            blockers.Add("copied calibrator algorithm metadata shape is not ready.");
        }

        if (!copiedInterfaceInfoMetadataReady)
        {
            blockers.Add("copied calibrator interface-info metadata shape is not ready for this TensorRT line.");
        }

        if (!batchCallbackOwnershipModeled)
        {
            blockers.Add("IInt8Calibrator::getBatch callback buffer ownership remains deferred.");
        }

        if (!cacheBufferOwnershipModeled)
        {
            blockers.Add("calibration cache and histogram cache buffer ownership remains deferred.");
        }

        blockers.Add("direct INT8 calibrator callback invocation remains deferred by design.");
        blockers.Add("direct INT8 calibrator cache read/write remains deferred by design.");
        blockers.Add("full package consumer runtime execution proof has not been promoted from this design gate.");

        return new TensorRtCalibratorMetadataDesignGateResult(
            line,
            lineSupportsCalibrator,
            presenceProbeAvailable,
            copiedAlgorithmMetadataReady,
            copiedInterfaceInfoMetadataReady,
            batchCallbackOwnershipModeled,
            cacheBufferOwnershipModeled,
            blockers.ToArray());
    }
}

/// <summary>
/// Reports INT8 calibrator metadata design status without exposing native calibrator pointers.
/// 报告 INT8 calibrator metadata 设计状态，不暴露原生 calibrator 指针。
/// </summary>
/// <remarks>
/// This result is a design gate. It keeps direct calibrator callback, batch-buffer, and cache-buffer rows deferred
/// until ownership and runtime proof exist.
/// 该结果是设计门禁；在 ownership 与 runtime proof 出现前，direct calibrator callback、batch buffer 与 cache buffer 行继续 deferred。
/// </remarks>
public readonly struct TensorRtCalibratorMetadataDesignGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtCalibratorMetadataDesignGateResult(
        TensorRtApiLine line,
        bool lineSupportsCalibrator,
        bool presenceProbeAvailable,
        bool copiedAlgorithmMetadataReady,
        bool copiedInterfaceInfoMetadataReady,
        bool batchCallbackOwnershipModeled,
        bool cacheBufferOwnershipModeled,
        string[] blockedPrerequisites)
    {
        Line = line;
        LineSupportsCalibrator = lineSupportsCalibrator;
        PresenceProbeAvailable = presenceProbeAvailable;
        CopiedAlgorithmMetadataReady = copiedAlgorithmMetadataReady;
        CopiedInterfaceInfoMetadataReady = copiedInterfaceInfoMetadataReady;
        BatchCallbackOwnershipModeled = batchCallbackOwnershipModeled;
        CacheBufferOwnershipModeled = cacheBufferOwnershipModeled;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 获取该设计门的证据标记。</summary>
    public string EvidenceKind => "calibrator-metadata-design-gate";

    /// <summary>Gets the diagnostics kind represented by this gate. 获取该设计门代表的诊断类型。</summary>
    public string DiagnosticsKind => "calibrator-metadata";

    /// <summary>Gets the runtime evidence kind. 获取 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind => "design-gate";

    /// <summary>Gets whether this gate is runtime execution evidence. 获取该设计门是否为 runtime execution evidence。</summary>
    public bool IsRuntimeExecutionEvidence => false;

    /// <summary>Gets whether this gate may be promoted as runtime execution proof. 获取该设计门是否可晋级为 runtime execution proof。</summary>
    public bool IsRuntimeExecutionProof => false;

    /// <summary>Gets the TensorRT API line. 获取 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether the selected TensorRT line supports INT8 calibrator metadata planning. 获取当前 TensorRT line 是否支持 INT8 calibrator metadata 规划。</summary>
    public bool LineSupportsCalibrator { get; }

    /// <summary>Gets whether safe builder-config INT8 calibrator presence probe is available. 获取安全 builder-config INT8 calibrator presence 查询是否可用。</summary>
    public bool PresenceProbeAvailable { get; }

    /// <summary>Gets whether copied calibrator algorithm metadata shape is ready. 获取 copied calibrator algorithm metadata 形态是否就绪。</summary>
    public bool CopiedAlgorithmMetadataReady { get; }

    /// <summary>Gets whether copied calibrator interface-info metadata shape is ready. 获取 copied calibrator interface-info metadata 形态是否就绪。</summary>
    public bool CopiedInterfaceInfoMetadataReady { get; }

    /// <summary>Gets whether getBatch callback ownership is modeled. 获取 getBatch callback ownership 是否已建模。</summary>
    public bool BatchCallbackOwnershipModeled { get; }

    /// <summary>Gets whether calibration cache buffer ownership is modeled. 获取 calibration cache buffer ownership 是否已建模。</summary>
    public bool CacheBufferOwnershipModeled { get; }

    /// <summary>Gets whether a native calibrator pointer is exposed through this public surface. 获取 public surface 是否暴露原生 calibrator 指针。</summary>
    public bool CalibratorPointerExposed => false;

    /// <summary>Gets whether this gate produces a native calibrator pointer. 获取该设计门是否产生原生 calibrator 指针。</summary>
    public bool CalibratorPointerProduced => false;

    /// <summary>Gets whether a borrowed calibrator pointer can escape the public surface. 获取 borrowed calibrator 指针是否可能逃逸 public surface。</summary>
    public bool BorrowedCalibratorPointerEscaped => false;

    /// <summary>Gets whether public APIs can invoke calibration callbacks. 获取 public API 是否可调用 calibration callback。</summary>
    public bool CallbackInvocationEnabled => false;

    /// <summary>Gets whether public APIs can access calibrator batch buffers. 获取 public API 是否可访问 calibrator batch buffer。</summary>
    public bool BatchBufferAccessEnabled => false;

    /// <summary>Gets whether public APIs can access calibration cache buffers. 获取 public API 是否可访问 calibration cache buffer。</summary>
    public bool CacheBufferAccessEnabled => false;

    /// <summary>Gets whether direct calibrator callback rows intentionally remain deferred. 获取 direct calibrator callback 行是否继续 deferred。</summary>
    public bool DirectCalibratorCallbackRowsDeferred => true;

    /// <summary>Gets whether direct calibrator cache rows intentionally remain deferred. 获取 direct calibrator cache 行是否继续 deferred。</summary>
    public bool DirectCalibratorCacheRowsDeferred => true;

    /// <summary>Gets whether the public surface remains pointer-free. 获取 public surface 是否保持无裸指针。</summary>
    public bool PointerFreeSurfaceReady =>
        !CalibratorPointerExposed &&
        !CalibratorPointerProduced &&
        !BorrowedCalibratorPointerEscaped &&
        !CallbackInvocationEnabled &&
        !BatchBufferAccessEnabled &&
        !CacheBufferAccessEnabled;

    /// <summary>Gets whether copied calibrator metadata shape is ready. 获取 copied calibrator metadata 形态是否就绪。</summary>
    public bool CopiedMetadataShapeReady =>
        PresenceProbeAvailable &&
        CopiedAlgorithmMetadataReady &&
        CopiedInterfaceInfoMetadataReady;

    /// <summary>Gets whether this design gate is ready as non-proof evidence. 获取该 design gate 作为非 proof 证据是否就绪。</summary>
    public bool DesignGateReady =>
        LineSupportsCalibrator &&
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

    /// <summary>Gets copied blocked prerequisites. 获取已复制的阻塞前置项。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 获取已复制阻塞前置项数量。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the design gate status. 获取设计门状态。</summary>
    public string Status => DesignGateReady ? "design-gate-ready" : "design-gate-blocked";

    /// <summary>Gets a compact diagnostic summary. 获取简短诊断摘要。</summary>
    public string Diagnostic =>
        "calibrator-metadata-design-gate; RuntimeEvidenceKind=design-gate; " +
        "IsRuntimeExecutionEvidence=False; IsRuntimeExecutionProof=False; " +
        "PresenceProbeAvailable=" + PresenceProbeAvailable + "; " +
        "CopiedMetadataShapeReady=" + CopiedMetadataShapeReady + "; " +
        "PointerFreeSurfaceReady=" + PointerFreeSurfaceReady + "; " +
        "CalibratorPointerExposed=False; BorrowedCalibratorPointerEscaped=False; " +
        "CallbackInvocationEnabled=False; BatchBufferAccessEnabled=False; CacheBufferAccessEnabled=False; " +
        "DirectCalibratorCallbackRowsDeferred=True; DirectCalibratorCacheRowsDeferred=True; " +
        "CanPromoteWithoutRuntimeProof=False; RuntimeProofBlocked=True; " +
        "DeferredRowsStillRequired=True; BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 返回简短诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:presence={PresenceProbeAvailable}:proof={IsRuntimeExecutionProof}";
    }
}
