using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates dependency diagnostics for TensorRT runtime deserialization without promoting runtime proof.
/// 评估 TensorRT runtime deserialization 的依赖诊断边界，但不提升 runtime proof。
/// </summary>
/// <remarks>
/// This diagnostic layer consumes the pointer-free managed deserialization precheck and package-consumer
/// classification facts. It does not load plugin host code, does not call <c>IRuntime::loadRuntime</c>,
/// and does not expose native runtime, engine, plugin, or creator pointers.
/// 该诊断层消费无裸指针的托管反序列化预检和 package-consumer 分类事实；不会加载 plugin host code，
/// 不会调用 <c>IRuntime::loadRuntime</c>，也不会暴露 native runtime、engine、plugin 或 creator 指针。
/// </remarks>
public static class TensorRtRuntimeDeserializationDependencyDiagnostics
{
    /// <summary>
    /// Evaluates the known dependency diagnostic shape for a TensorRT API line.
    /// 基于已知 public surface 评估指定 TensorRT API line 的依赖诊断形态。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <returns>A pointer-free dependency diagnostics result. 无裸指针依赖诊断结果。</returns>
    public static TensorRtRuntimeDeserializationDependencyDiagnosticsResult EvaluateKnownSurface(TensorRtApiLine line)
    {
        return Evaluate(
            TensorRtRuntimeDeserializationBoundaryPrecheck.EvaluateKnownSurface(line),
            fullPackageConsumerReportPresent: false,
            fullPackageConsumerSmokeRequested: false,
            fullPackageConsumerSmokeResult: "not-present",
            dependencyProbeOnly: true,
            blockedByCudaDriver: false);
    }

    /// <summary>
    /// Evaluates dependency diagnostics from explicit package-consumer classification facts.
    /// 根据显式 package-consumer 分类事实评估依赖诊断。
    /// </summary>
    /// <param name="precheck">The runtime deserialization boundary precheck result. runtime deserialization 边界预检结果。</param>
    /// <param name="fullPackageConsumerReportPresent">Whether a full package consumer report is present. 是否存在 full package consumer report。</param>
    /// <param name="fullPackageConsumerSmokeRequested">Whether full package consumer smoke was requested. 是否请求过 full package consumer smoke。</param>
    /// <param name="fullPackageConsumerSmokeResult">The full package consumer smoke result. full package consumer smoke 结果。</param>
    /// <param name="dependencyProbeOnly">Whether the available evidence is dependency-probe only. 当前证据是否仅为 dependency-probe。</param>
    /// <param name="blockedByCudaDriver">Whether the smoke reached CUDA and was blocked by driver/runtime compatibility. smoke 是否到达 CUDA 后被 driver/runtime 兼容性阻塞。</param>
    /// <returns>A pointer-free dependency diagnostics result. 无裸指针依赖诊断结果。</returns>
    public static TensorRtRuntimeDeserializationDependencyDiagnosticsResult Evaluate(
        TensorRtRuntimeDeserializationBoundaryPrecheckResult precheck,
        bool fullPackageConsumerReportPresent,
        bool fullPackageConsumerSmokeRequested,
        string? fullPackageConsumerSmokeResult,
        bool dependencyProbeOnly,
        bool blockedByCudaDriver)
    {
        string rawSmokeResult = fullPackageConsumerSmokeResult ?? string.Empty;
        string normalizedSmokeResult = string.IsNullOrWhiteSpace(rawSmokeResult)
            ? "not-present"
            : rawSmokeResult.Trim();
        bool normalizedBlockedByCudaDriver =
            blockedByCudaDriver ||
            string.Equals(normalizedSmokeResult, "blocked-by-cuda-driver", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(normalizedSmokeResult, "runtime-smoke-driver-blocked", StringComparison.OrdinalIgnoreCase);

        List<string> blockers = new List<string>();
        AddBlockerIfFalse(blockers, precheck.PrecheckReady, "runtime deserialization boundary precheck is not ready.");
        AddBlockerIfFalse(blockers, precheck.ManagedDeserializeSurfaceReady, "managed runtime deserialization surface is not ready.");
        AddBlockerIfFalse(blockers, fullPackageConsumerReportPresent, "full package consumer report is not present.");
        AddBlockerIfFalse(blockers, fullPackageConsumerSmokeRequested, "full package consumer smoke has not been requested.");
        if (dependencyProbeOnly)
        {
            AddBlocker(blockers, "available evidence is dependency-probe-only and not runtime execution proof.");
        }

        if (normalizedBlockedByCudaDriver)
        {
            AddBlocker(blockers, "full package consumer smoke is blocked by CUDA driver/runtime compatibility.");
        }

        AddBlocker(blockers, "plugin library dependency diagnostics are not complete.");
        AddBlocker(blockers, "IRuntime::loadRuntime returned runtime ownership is not modeled.");
        if (precheck.LineSupportsDeserializeCudaEngineV2)
        {
            AddBlocker(blockers, "direct IRuntime::deserializeCudaEngineV2 rows remain deferred by design.");
        }

        AddBlocker(blockers, "IRuntime::loadRuntime remains deferred by design.");

        return new TensorRtRuntimeDeserializationDependencyDiagnosticsResult(
            precheck.Line,
            precheck.PrecheckReady,
            precheck.ManagedDeserializeSurfaceReady,
            precheck.SafeDeserializeBridgeReady,
            precheck.LineSupportsDeserializeCudaEngineV2,
            precheck.DirectDeserializeCudaEngineRowsDeferred,
            precheck.DirectDeserializeCudaEngineV2RowsDeferred,
            precheck.LoadRuntimeDeferred,
            fullPackageConsumerReportPresent,
            fullPackageConsumerSmokeRequested,
            normalizedSmokeResult,
            dependencyProbeOnly,
            normalizedBlockedByCudaDriver,
            blockers.ToArray());
    }

    private static void AddBlockerIfFalse(List<string> blockers, bool condition, string blocker)
    {
        if (!condition)
        {
            AddBlocker(blockers, blocker);
        }
    }

    private static void AddBlocker(List<string> blockers, string blocker)
    {
        if (!string.IsNullOrWhiteSpace(blocker) && !blockers.Contains(blocker))
        {
            blockers.Add(blocker);
        }
    }
}

/// <summary>
/// Reports runtime deserialization dependency diagnostics without exposing native pointers.
/// 报告 runtime deserialization 依赖诊断，不暴露 native 指针。
/// </summary>
/// <remarks>
/// This result separates dependency-probe-only, driver-blocked, and package-consumer runtime proof states.
/// It is dependency diagnostics, not runtime execution evidence.
/// 该结果区分 dependency-probe-only、driver-blocked 和 package-consumer runtime proof 状态；
/// 它是 dependency diagnostics，不是 runtime execution evidence。
/// </remarks>
public readonly struct TensorRtRuntimeDeserializationDependencyDiagnosticsResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtRuntimeDeserializationDependencyDiagnosticsResult(
        TensorRtApiLine line,
        bool precheckReady,
        bool managedDeserializeSurfaceReady,
        bool safeDeserializeBridgeReady,
        bool lineSupportsDeserializeCudaEngineV2,
        bool directDeserializeCudaEngineRowsDeferred,
        bool directDeserializeCudaEngineV2RowsDeferred,
        bool loadRuntimeDeferred,
        bool fullPackageConsumerReportPresent,
        bool fullPackageConsumerSmokeRequested,
        string fullPackageConsumerSmokeResult,
        bool dependencyProbeOnly,
        bool blockedByCudaDriver,
        string[] blockedPrerequisites)
    {
        Line = line;
        PrecheckReady = precheckReady;
        ManagedDeserializeSurfaceReady = managedDeserializeSurfaceReady;
        SafeDeserializeBridgeReady = safeDeserializeBridgeReady;
        LineSupportsDeserializeCudaEngineV2 = lineSupportsDeserializeCudaEngineV2;
        DirectDeserializeCudaEngineRowsDeferred = directDeserializeCudaEngineRowsDeferred;
        DirectDeserializeCudaEngineV2RowsDeferred = directDeserializeCudaEngineV2RowsDeferred;
        LoadRuntimeDeferred = loadRuntimeDeferred;
        FullPackageConsumerReportPresent = fullPackageConsumerReportPresent;
        FullPackageConsumerSmokeRequested = fullPackageConsumerSmokeRequested;
        FullPackageConsumerSmokeResult = fullPackageConsumerSmokeResult;
        DependencyProbeOnly = dependencyProbeOnly;
        BlockedByCudaDriver = blockedByCudaDriver;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this diagnostics layer. 获取 readiness 使用的证据标记。</summary>
    public string EvidenceKind => "runtime-deserialization-dependency-diagnostics";

    /// <summary>Gets the diagnostics kind. 获取诊断类型。</summary>
    public string DiagnosticsKind => "runtime-deserialization-dependency-diagnostics";

    /// <summary>Gets the runtime evidence kind. 获取 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind => "dependency-diagnostics";

    /// <summary>Gets whether this result is runtime execution evidence. 获取该结果是否为 runtime execution evidence。</summary>
    public bool IsRuntimeExecutionEvidence => false;

    /// <summary>Gets whether this result can be used as runtime execution proof. 获取该结果是否可作为 runtime execution proof。</summary>
    public bool IsRuntimeExecutionProof => false;

    /// <summary>Gets the TensorRT API line. 获取 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether the boundary precheck is ready. 获取边界预检是否就绪。</summary>
    public bool PrecheckReady { get; }

    /// <summary>Gets whether managed deserialization overloads are ready. 获取托管反序列化 overload 是否就绪。</summary>
    public bool ManagedDeserializeSurfaceReady { get; }

    /// <summary>Gets whether the pointer-free deserialize bridge is ready. 获取无裸指针 deserialize bridge 是否就绪。</summary>
    public bool SafeDeserializeBridgeReady { get; }

    /// <summary>Gets whether the selected line has deserializeCudaEngineV2 rows. 获取当前 line 是否有 deserializeCudaEngineV2 行。</summary>
    public bool LineSupportsDeserializeCudaEngineV2 { get; }

    /// <summary>Gets whether direct deserializeCudaEngine rows still remain deferred. 获取 direct deserializeCudaEngine 行是否仍 deferred。</summary>
    public bool DirectDeserializeCudaEngineRowsDeferred { get; }

    /// <summary>Gets whether direct deserializeCudaEngineV2 rows still remain deferred where present. 获取存在时 direct deserializeCudaEngineV2 行是否仍 deferred。</summary>
    public bool DirectDeserializeCudaEngineV2RowsDeferred { get; }

    /// <summary>Gets whether IRuntime::loadRuntime remains deferred. 获取 IRuntime::loadRuntime 是否仍 deferred。</summary>
    public bool LoadRuntimeDeferred { get; }

    /// <summary>Gets whether a full package consumer report is present. 获取 full package consumer report 是否存在。</summary>
    public bool FullPackageConsumerReportPresent { get; }

    /// <summary>Gets whether full package consumer smoke was requested. 获取 full package consumer smoke 是否已请求。</summary>
    public bool FullPackageConsumerSmokeRequested { get; }

    /// <summary>Gets the full package consumer smoke result. 获取 full package consumer smoke 结果。</summary>
    public string FullPackageConsumerSmokeResult { get; }

    /// <summary>Gets whether available evidence is dependency-probe only. 获取当前证据是否仅为 dependency-probe。</summary>
    public bool DependencyProbeOnly { get; }

    /// <summary>Gets whether CUDA driver/runtime compatibility blocks runtime smoke. 获取 CUDA driver/runtime 兼容性是否阻塞 runtime smoke。</summary>
    public bool BlockedByCudaDriver { get; }

    /// <summary>Gets whether driver/runtime mismatch has been classified. 获取是否已分类 driver/runtime 不匹配。</summary>
    public bool DriverRuntimeMismatchClassified => BlockedByCudaDriver;

    /// <summary>Gets whether a promotable package-consumer runtime proof is present. 获取是否存在可晋级的 package-consumer runtime proof。</summary>
    public bool PackageConsumerRuntimeProofPresent => false;

    /// <summary>Gets whether an external runtime proof record is required. 获取是否需要外部 runtime proof 记录。</summary>
    public bool ExternalRuntimeProofRequired => true;

    /// <summary>Gets whether release-owner action is required. 获取是否需要 release owner 操作。</summary>
    public bool RuntimeProofOwnerActionRequired => true;

    /// <summary>Gets the blocker category for release-owner action. 获取 release owner 操作的阻塞分类。</summary>
    public string RuntimeProofBlockerCategory
    {
        get
        {
            if (!PrecheckReady || !ManagedDeserializeSurfaceReady)
            {
                return "runtime-deserialization-precheck-incomplete";
            }

            if (!FullPackageConsumerReportPresent)
            {
                return "full-package-consumer-report-missing";
            }

            if (!FullPackageConsumerSmokeRequested)
            {
                return "runtime-smoke-not-requested";
            }

            if (BlockedByCudaDriver)
            {
                return "cuda-driver-runtime-compatibility";
            }

            if (DependencyProbeOnly)
            {
                return "dependency-probe-only";
            }

            if (!PluginLibraryDependencyDiagnosticsComplete)
            {
                return "plugin-library-dependency-diagnostics-incomplete";
            }

            if (!LoadRuntimeOwnershipModeled)
            {
                return "load-runtime-ownership-deferred";
            }

            return "runtime-proof-incomplete";
        }
    }

    /// <summary>Gets the current package-consumer evidence classification. 获取当前 package-consumer 证据分类。</summary>
    public string PackageConsumerEvidenceClassification
    {
        get
        {
            if (BlockedByCudaDriver)
            {
                return "runtime-smoke-driver-blocked";
            }

            if (DependencyProbeOnly)
            {
                return "dependency-probe-only";
            }

            if (!FullPackageConsumerSmokeRequested)
            {
                return "runtime-smoke-not-requested";
            }

            return "runtime-proof-incomplete";
        }
    }

    /// <summary>Gets whether plugin library dependency diagnostics are complete. 获取 plugin library dependency 诊断是否完整。</summary>
    public bool PluginLibraryDependencyDiagnosticsComplete => false;

    /// <summary>Gets whether IRuntime::loadRuntime returned runtime ownership is modeled. 获取 loadRuntime 返回 runtime 的 ownership 是否已建模。</summary>
    public bool LoadRuntimeOwnershipModeled => false;

    /// <summary>Gets whether this diagnostic can attempt runtime proof. 获取该诊断是否可尝试 runtime proof。</summary>
    public bool CanAttemptRuntimeProof => false;

    /// <summary>Gets whether this diagnostic can be promoted as runtime proof. 获取该诊断是否可提升为 runtime proof。</summary>
    public bool CanPromoteRuntimeProof => false;

    /// <summary>Gets whether runtime proof remains blocked. 获取 runtime proof 是否仍被阻塞。</summary>
    public bool RuntimeProofBlocked => !CanPromoteRuntimeProof;

    /// <summary>Gets whether deferred rows are still required. 获取 deferred 行是否仍必须保留。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets whether public diagnostics remain pointer-free. 获取 public diagnostics 是否保持无裸指针。</summary>
    public bool PointerFreeSurfaceReady => true;

    /// <summary>Gets the owner-facing reason why this diagnostics result is not runtime proof. 获取面向 owner 的非 runtime proof 原因。</summary>
    public string WhyNotRuntimeProof =>
        "runtime deserialization dependency diagnostics, dependency-probe-only output, blocked-by-cuda-driver, " +
        "precheck, design-gate, build-only, and deferred loadRuntime ownership are not runtime execution proof.";

    /// <summary>Gets the next owner action required to unblock runtime proof. 获取解除 runtime proof 阻塞所需的下一步 owner 操作。</summary>
    public string NextOwnerAction
    {
        get
        {
            switch (RuntimeProofBlockerCategory)
            {
                case "cuda-driver-runtime-compatibility":
                    return "Run full package consumer smoke on a host with a compatible NVIDIA driver, then attach a promotable package-consumer-runtime external proof record.";
                case "runtime-smoke-not-requested":
                    return "Run Test-PackageConsumer.ps1 with -RunSmoke for the selected runtime package key and refresh runtime readiness evidence.";
                case "full-package-consumer-report-missing":
                    return "Generate the full package consumer report before evaluating runtime proof.";
                case "dependency-probe-only":
                    return "Replace dependency-probe-only evidence with successful full package consumer runtime smoke evidence.";
                case "plugin-library-dependency-diagnostics-incomplete":
                    return "Complete plugin library dependency diagnostics before attempting runtime proof promotion.";
                case "load-runtime-ownership-deferred":
                    return "Model IRuntime::loadRuntime returned runtime ownership before enabling loadRuntime proof promotion.";
                default:
                    return "Inspect blocked prerequisites, refresh package consumer evidence, and provide a promotable package-consumer-runtime proof record.";
            }
        }
    }

    /// <summary>Gets copied blocked prerequisites. 获取已复制的阻塞前置项。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets copied blocked prerequisite count. 获取阻塞前置项数量。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the diagnostics status. 获取诊断状态。</summary>
    public string Status => PrecheckReady ? "dependency-diagnostics-ready" : "dependency-diagnostics-blocked";

    /// <summary>Gets a compact diagnostic summary. 获取简短诊断摘要。</summary>
    public string Diagnostic =>
        "runtime-deserialization-dependency-diagnostics; RuntimeEvidenceKind=dependency-diagnostics; " +
        "IsRuntimeExecutionEvidence=False; IsRuntimeExecutionProof=False; " +
        "PrecheckReady=" + PrecheckReady + "; " +
        "ManagedDeserializeSurfaceReady=" + ManagedDeserializeSurfaceReady + "; " +
        "FullPackageConsumerReportPresent=" + FullPackageConsumerReportPresent + "; " +
        "FullPackageConsumerSmokeRequested=" + FullPackageConsumerSmokeRequested + "; " +
        "FullPackageConsumerSmokeResult=" + FullPackageConsumerSmokeResult + "; " +
        "DependencyProbeOnly=" + DependencyProbeOnly + "; " +
        "BlockedByCudaDriver=" + BlockedByCudaDriver + "; " +
        "DriverRuntimeMismatchClassified=" + DriverRuntimeMismatchClassified + "; " +
        "PackageConsumerEvidenceClassification=" + PackageConsumerEvidenceClassification + "; " +
        "PluginLibraryDependencyDiagnosticsComplete=False; " +
        "LoadRuntimeOwnershipModeled=False; " +
        "RuntimeProofBlockerCategory=" + RuntimeProofBlockerCategory + "; " +
        "RuntimeProofOwnerActionRequired=True; ExternalRuntimeProofRequired=True; " +
        "CanAttemptRuntimeProof=False; CanPromoteRuntimeProof=False; " +
        "RuntimeProofBlocked=True; DeferredRowsStillRequired=True; " +
        "WhyNotRuntimeProof=" + WhyNotRuntimeProof + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 返回简短诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:smoke={FullPackageConsumerSmokeResult}:proof={IsRuntimeExecutionProof}";
    }
}
