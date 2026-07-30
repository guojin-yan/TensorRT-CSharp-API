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
