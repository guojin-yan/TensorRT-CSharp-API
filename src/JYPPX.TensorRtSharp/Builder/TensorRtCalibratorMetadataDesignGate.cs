using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

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
