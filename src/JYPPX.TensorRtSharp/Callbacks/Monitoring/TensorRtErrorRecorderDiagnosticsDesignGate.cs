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
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
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
