using System;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates the TensorRT 8 RNNv2 borrowed tensor and gate-weight design boundary.
/// 评估 TensorRT 8 RNNv2 borrowed tensor 与 gate weights 的设计边界。
/// </summary>
public static class TensorRtRnnV2BorrowedStateDesignGate
{
    /// <summary>
    /// Evaluates the current known-safe RNNv2 surface.
    /// 评估当前已知安全的 RNNv2 surface。
    /// </summary>
    /// <returns>A pointer-free design gate result. 无裸指针设计门结果。</returns>
    public static TensorRtRnnV2BorrowedStateDesignGateResult EvaluateKnownSurface()
    {
        return Evaluate(
            TensorRtApiLine.TensorRt8,
            dataLengthScalarPromoted: true,
            networkOwnedTensorReferencePolicyReady: true,
            gateWeightSnapshotCopyReady: true,
            ownerLifetimeKnown: true);
    }

    /// <summary>
    /// Evaluates the RNNv2 borrowed-state boundary from explicit capability flags.
    /// 根据显式能力标记评估 RNNv2 borrowed-state 边界。
    /// </summary>
    public static TensorRtRnnV2BorrowedStateDesignGateResult Evaluate(
        TensorRtApiLine line,
        bool dataLengthScalarPromoted,
        bool networkOwnedTensorReferencePolicyReady,
        bool gateWeightSnapshotCopyReady,
        bool ownerLifetimeKnown)
    {
        string[] blockers =
        {
            "TensorRT 8 RNNv2 runtime construction is not exercised by this design gate.",
            "full package-consumer runtime proof is not provided by this design gate."
        };

        return new TensorRtRnnV2BorrowedStateDesignGateResult(
            line,
            dataLengthScalarPromoted,
            networkOwnedTensorReferencePolicyReady,
            gateWeightSnapshotCopyReady,
            ownerLifetimeKnown,
            blockers);
    }
}
