using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates the pointer-free TensorRT algorithm selector result snapshot design gate.
/// 评估 TensorRT algorithm selector result snapshot 的无裸指针设计门。
/// </summary>
/// <remarks>
/// TensorRT algorithm objects are borrowed during algorithm selector callbacks. This gate records the
/// copied snapshot shape required before any public API can expose algorithm timing, workspace, context,
/// IO info, or variant metadata.
/// TensorRT algorithm 对象来自 algorithm selector callback 期间的 borrowed 值；该门禁记录在公开 timing、
/// workspace、context、IO info 或 variant metadata 前必须满足的 copied snapshot 形态。
/// </remarks>
public static class TensorRtAlgorithmSnapshotDesignGate
{
    /// <summary>
    /// Evaluates the known public algorithm snapshot design surface for a TensorRT API line.
    /// 基于已知 public algorithm snapshot 设计边界评估指定 TensorRT API line。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <returns>A pointer-free algorithm snapshot design gate result. 无裸指针 algorithm snapshot 设计门结果。</returns>
    public static TensorRtAlgorithmSnapshotDesignGateResult EvaluateKnownSurface(TensorRtApiLine line)
    {
        return Evaluate(
            line,
            selectorCallbackOwnerModeled: false,
            algorithmResultLifetimeModeled: false,
            copiedTimingWorkspaceShapeReady: true,
            copiedContextShapeReady: true,
            copiedIoInfoShapeReady: true,
            copiedVariantShapeReady: true);
    }

    /// <summary>
    /// Evaluates algorithm snapshot readiness from explicit capability flags.
    /// 根据显式能力标记评估 algorithm snapshot 就绪状态。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <param name="selectorCallbackOwnerModeled">Whether selector callback owner lifetime has been modeled. selector callback owner lifetime 是否已建模。</param>
    /// <param name="algorithmResultLifetimeModeled">Whether algorithm result lifetime has been modeled. algorithm result lifetime 是否已建模。</param>
    /// <param name="copiedTimingWorkspaceShapeReady">Whether timing/workspace copied shape is ready. timing/workspace copied 形态是否就绪。</param>
    /// <param name="copiedContextShapeReady">Whether context copied shape is ready. context copied 形态是否就绪。</param>
    /// <param name="copiedIoInfoShapeReady">Whether IO info copied shape is ready. IO info copied 形态是否就绪。</param>
    /// <param name="copiedVariantShapeReady">Whether variant copied shape is ready. variant copied 形态是否就绪。</param>
    /// <returns>A pointer-free algorithm snapshot design gate result. 无裸指针 algorithm snapshot 设计门结果。</returns>
    public static TensorRtAlgorithmSnapshotDesignGateResult Evaluate(
        TensorRtApiLine line,
        bool selectorCallbackOwnerModeled,
        bool algorithmResultLifetimeModeled,
        bool copiedTimingWorkspaceShapeReady,
        bool copiedContextShapeReady,
        bool copiedIoInfoShapeReady,
        bool copiedVariantShapeReady)
    {
        bool lineSupportsAlgorithmSelector = line == TensorRtApiLine.TensorRt8 || line == TensorRtApiLine.TensorRt10;

        List<string> blockers = new List<string>();
        if (!lineSupportsAlgorithmSelector)
        {
            blockers.Add("TensorRT 8 or 10 algorithm selector line support has not been selected.");
        }

        if (!selectorCallbackOwnerModeled)
        {
            blockers.Add("algorithm selector callback owner lifetime is not modeled.");
        }

        if (!algorithmResultLifetimeModeled)
        {
            blockers.Add("borrowed IAlgorithm/IAlgorithmContext/IAlgorithmIOInfo/IAlgorithmVariant result lifetime is not modeled.");
        }

        if (!copiedTimingWorkspaceShapeReady)
        {
            blockers.Add("copied algorithm timing/workspace snapshot shape is not ready.");
        }

        if (!copiedContextShapeReady)
        {
            blockers.Add("copied algorithm context name/input/output/dimensions snapshot shape is not ready.");
        }

        if (!copiedIoInfoShapeReady)
        {
            blockers.Add("copied algorithm IO info data type/strides/vectorized-dim snapshot shape is not ready.");
        }

        if (!copiedVariantShapeReady)
        {
            blockers.Add("copied algorithm variant implementation/tactic snapshot shape is not ready.");
        }

        blockers.Add("direct IAlgorithm pointer access remains deferred by design.");
        blockers.Add("algorithm selector select/report callback trampolines remain deferred by design.");
        blockers.Add("full package consumer runtime execution proof has not been promoted from this design gate.");

        return new TensorRtAlgorithmSnapshotDesignGateResult(
            line,
            lineSupportsAlgorithmSelector,
            selectorCallbackOwnerModeled,
            algorithmResultLifetimeModeled,
            copiedTimingWorkspaceShapeReady,
            copiedContextShapeReady,
            copiedIoInfoShapeReady,
            copiedVariantShapeReady,
            blockers.ToArray());
    }
}
