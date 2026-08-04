using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates the pointer-free TensorRT ILoggerFinder metadata design gate.
/// 评估 TensorRT ILoggerFinder metadata 的无裸指针设计门。
/// </summary>
/// <remarks>
/// ILoggerFinder is an application callback provider in TensorRT 11. This gate records copied metadata
/// requirements only and does not expose finder handles or take ownership of logger callbacks.
/// ILoggerFinder 是 TensorRT 11 的应用侧 callback provider；该门禁只记录 copied metadata 要求，
/// 不暴露 finder handle，也不接管 logger callback ownership。
/// </remarks>
public static class TensorRtLoggerFinderMetadataDesignGate
{
    /// <summary>
    /// Evaluates the known public ILoggerFinder metadata design surface for a TensorRT API line.
    /// 基于已知 public ILoggerFinder metadata 设计边界评估指定 TensorRT API line。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <returns>A pointer-free ILoggerFinder metadata design gate result. 无裸指针 ILoggerFinder metadata 设计门结果。</returns>
    public static TensorRtLoggerFinderMetadataDesignGateResult EvaluateKnownSurface(TensorRtApiLine line)
    {
        return Evaluate(
            line,
            copiedInterfaceInfoMetadataReady: true,
            finderOwnerLifetimeModeled: false,
            loggerCallbackOwnershipModeled: false);
    }

    /// <summary>
    /// Evaluates ILoggerFinder metadata readiness from explicit capability flags.
    /// 根据显式能力标记评估 ILoggerFinder metadata 就绪状态。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <param name="copiedInterfaceInfoMetadataReady">Whether copied interface-info metadata shape is ready. copied interface-info metadata 形态是否就绪。</param>
    /// <param name="finderOwnerLifetimeModeled">Whether finder owner lifetime has been modeled. finder owner lifetime 是否已建模。</param>
    /// <param name="loggerCallbackOwnershipModeled">Whether logger callback ownership has been modeled. logger callback ownership 是否已建模。</param>
    /// <returns>A pointer-free ILoggerFinder metadata design gate result. 无裸指针 ILoggerFinder metadata 设计门结果。</returns>
    public static TensorRtLoggerFinderMetadataDesignGateResult Evaluate(
        TensorRtApiLine line,
        bool copiedInterfaceInfoMetadataReady,
        bool finderOwnerLifetimeModeled,
        bool loggerCallbackOwnershipModeled)
    {
        bool lineSupportsLoggerFinder = line == TensorRtApiLine.TensorRt11;

        List<string> blockers = new List<string>();
        if (!lineSupportsLoggerFinder)
        {
            blockers.Add("TensorRT 11 ILoggerFinder line support has not been selected.");
        }

        if (!copiedInterfaceInfoMetadataReady)
        {
            blockers.Add("copied ILoggerFinder interface-info metadata shape is not ready.");
        }

        if (!finderOwnerLifetimeModeled)
        {
            blockers.Add("ILoggerFinder owner lifetime is not modeled.");
        }

        if (!loggerCallbackOwnershipModeled)
        {
            blockers.Add("logger callback provider ownership remains deferred.");
        }

        blockers.Add("direct ILoggerFinder pointer access remains deferred by design.");
        blockers.Add("logger callback lookup/invocation remains deferred by design.");
        blockers.Add("full package consumer runtime execution proof has not been promoted from this design gate.");

        return new TensorRtLoggerFinderMetadataDesignGateResult(
            line,
            lineSupportsLoggerFinder,
            copiedInterfaceInfoMetadataReady,
            finderOwnerLifetimeModeled,
            loggerCallbackOwnershipModeled,
            blockers.ToArray());
    }
}
