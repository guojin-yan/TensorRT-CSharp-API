using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates the pointer-free TensorRT dimension expression snapshot design gate.
/// 评估 TensorRT dimension expression snapshot 的无裸指针设计门。
/// </summary>
/// <remarks>
/// This gate does not expose an <c>IDimensionExpr*</c>, does not create expression nodes through
/// <c>IExprBuilder</c>, and is not runtime execution proof.
/// 该门禁不会暴露 <c>IDimensionExpr*</c>，不会通过 <c>IExprBuilder</c> 创建表达式节点，也不是 runtime execution proof。
/// </remarks>
public static class TensorRtDimensionExpressionSnapshotDesignGate
{
    /// <summary>
    /// Evaluates the known public dimension expression design surface for a TensorRT API line.
    /// 基于已知 public dimension expression 设计边界评估指定 TensorRT API line。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <returns>A pointer-free dimension expression design gate result. 无裸指针 dimension expression 设计门结果。</returns>
    public static TensorRtDimensionExpressionSnapshotDesignGateResult EvaluateKnownSurface(TensorRtApiLine line)
    {
        return Evaluate(
            line,
            ownerLifetimeKnown: false,
            constantSnapshotCopyReady: true,
            sizeTensorMetadataCopyReady: line != TensorRtApiLine.TensorRt8,
            exprBuilderOwnershipModeled: false,
            pluginShapeCallbackLifetimeModeled: false);
    }

    /// <summary>
    /// Evaluates the dimension expression snapshot design gate from explicit capability flags.
    /// 根据显式能力标记评估 dimension expression snapshot 设计门。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <param name="ownerLifetimeKnown">Whether a safe owner lifetime has been proven. 是否已证明安全 owner lifetime。</param>
    /// <param name="constantSnapshotCopyReady">Whether copied constant metadata shape is ready. copied constant metadata 形态是否就绪。</param>
    /// <param name="sizeTensorMetadataCopyReady">Whether copied size-tensor metadata shape is ready where available. 可用时 copied size-tensor metadata 形态是否就绪。</param>
    /// <param name="exprBuilderOwnershipModeled">Whether expression builder ownership has been modeled. expression builder ownership 是否已建模。</param>
    /// <param name="pluginShapeCallbackLifetimeModeled">Whether plugin shape callback lifetime has been modeled. plugin shape callback lifetime 是否已建模。</param>
    /// <returns>A pointer-free dimension expression design gate result. 无裸指针 dimension expression 设计门结果。</returns>
    public static TensorRtDimensionExpressionSnapshotDesignGateResult Evaluate(
        TensorRtApiLine line,
        bool ownerLifetimeKnown,
        bool constantSnapshotCopyReady,
        bool sizeTensorMetadataCopyReady,
        bool exprBuilderOwnershipModeled,
        bool pluginShapeCallbackLifetimeModeled)
    {
        bool lineSupportsDimensionExpression =
            line == TensorRtApiLine.TensorRt8 ||
            line == TensorRtApiLine.TensorRt10 ||
            line == TensorRtApiLine.TensorRt11;
        bool lineSupportsSizeTensor = line == TensorRtApiLine.TensorRt10 || line == TensorRtApiLine.TensorRt11;
        bool normalizedSizeTensorMetadataCopyReady = !lineSupportsSizeTensor || sizeTensorMetadataCopyReady;

        List<string> blockers = new List<string>();
        if (!lineSupportsDimensionExpression)
        {
            blockers.Add("TensorRT 8, 10, or 11 dimension expression line support has not been selected.");
        }

        if (!constantSnapshotCopyReady)
        {
            blockers.Add("copied IDimensionExpr constant metadata snapshot shape is not ready.");
        }

        if (lineSupportsSizeTensor && !sizeTensorMetadataCopyReady)
        {
            blockers.Add("copied IDimensionExpr size-tensor metadata snapshot shape is not ready for TensorRT 10/11.");
        }

        if (!ownerLifetimeKnown)
        {
            blockers.Add("a known TensorRT owner object lifetime for borrowed IDimensionExpr values has not been proven.");
        }

        if (!exprBuilderOwnershipModeled)
        {
            blockers.Add("IExprBuilder expression node ownership and lifetime remain deferred.");
        }

        if (!pluginShapeCallbackLifetimeModeled)
        {
            blockers.Add("plugin shape callback dimension expression lifetime remains deferred.");
        }

        blockers.Add("direct IDimensionExpr pointer access remains deferred by design.");
        blockers.Add("direct IExprBuilder expression creation remains deferred by design.");
        blockers.Add("full package consumer runtime execution proof has not been promoted from this design gate.");

        return new TensorRtDimensionExpressionSnapshotDesignGateResult(
            line,
            lineSupportsDimensionExpression,
            lineSupportsSizeTensor,
            ownerLifetimeKnown,
            constantSnapshotCopyReady,
            normalizedSizeTensorMetadataCopyReady,
            exprBuilderOwnershipModeled,
            pluginShapeCallbackLifetimeModeled,
            blockers.ToArray());
    }
}
