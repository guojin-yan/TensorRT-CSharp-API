using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates the pointer-free TensorRT PluginCreatorV3 and IVersionedInterface metadata design gate.
/// 评估 TensorRT PluginCreatorV3 与 IVersionedInterface metadata 的无裸指针设计门。
/// </summary>
/// <remarks>
/// This gate documents the safe alternative already provided by plugin registry inventory snapshots.
/// It does not expose plugin creator pointers, create plugin instances, or load plugin libraries.
/// 该门禁记录 plugin registry inventory snapshot 已提供的安全替代能力；不会暴露 plugin creator 指针、
/// 创建 plugin instance，也不会加载 plugin library。
/// </remarks>
public static class TensorRtPluginCreatorV3MetadataDesignGate
{
    /// <summary>
    /// Evaluates the known public PluginCreatorV3 metadata surface for a TensorRT API line.
    /// 基于已知 public PluginCreatorV3 metadata 边界评估指定 TensorRT API line。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <returns>A pointer-free PluginCreatorV3 metadata design gate result. 无裸指针 PluginCreatorV3 metadata 设计门结果。</returns>
    public static TensorRtPluginCreatorV3MetadataDesignGateResult EvaluateKnownSurface(TensorRtApiLine line)
    {
        return Evaluate(
            line,
            registryInventorySnapshotReady: true,
            copiedIdentityReady: true,
            copiedFieldMetadataReady: true,
            copiedInterfaceInfoReady: true,
            pluginCreationModeled: false,
            borrowedCreatorLifetimeModeled: false);
    }

    /// <summary>
    /// Evaluates PluginCreatorV3 metadata readiness from explicit capability flags.
    /// 根据显式能力标记评估 PluginCreatorV3 metadata 就绪状态。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <param name="registryInventorySnapshotReady">Whether copied registry inventory snapshots are ready. copied registry inventory snapshot 是否就绪。</param>
    /// <param name="copiedIdentityReady">Whether creator name/version/namespace copies are ready. creator name/version/namespace copied 输出是否就绪。</param>
    /// <param name="copiedFieldMetadataReady">Whether plugin field metadata copies are ready. plugin field metadata copied 输出是否就绪。</param>
    /// <param name="copiedInterfaceInfoReady">Whether interface metadata copies are ready. interface metadata copied 输出是否就绪。</param>
    /// <param name="pluginCreationModeled">Whether plugin creation ownership has been modeled. plugin creation ownership 是否已建模。</param>
    /// <param name="borrowedCreatorLifetimeModeled">Whether borrowed creator lifetime has been modeled. borrowed creator lifetime 是否已建模。</param>
    /// <returns>A pointer-free PluginCreatorV3 metadata design gate result. 无裸指针 PluginCreatorV3 metadata 设计门结果。</returns>
    public static TensorRtPluginCreatorV3MetadataDesignGateResult Evaluate(
        TensorRtApiLine line,
        bool registryInventorySnapshotReady,
        bool copiedIdentityReady,
        bool copiedFieldMetadataReady,
        bool copiedInterfaceInfoReady,
        bool pluginCreationModeled,
        bool borrowedCreatorLifetimeModeled)
    {
        bool lineSupportsPluginCreatorV3 = line == TensorRtApiLine.TensorRt10 || line == TensorRtApiLine.TensorRt11;

        List<string> blockers = new List<string>();
        if (!lineSupportsPluginCreatorV3)
        {
            blockers.Add("TensorRT 10 or 11 PluginCreatorV3 metadata line support has not been selected.");
        }

        if (!registryInventorySnapshotReady)
        {
            blockers.Add("copied plugin registry inventory snapshots are not ready.");
        }

        if (!copiedIdentityReady)
        {
            blockers.Add("copied plugin creator name/version/namespace metadata is not ready.");
        }

        if (!copiedFieldMetadataReady)
        {
            blockers.Add("copied plugin field metadata snapshot is not ready.");
        }

        if (!copiedInterfaceInfoReady)
        {
            blockers.Add("copied IVersionedInterface metadata snapshot is not ready.");
        }

        if (!borrowedCreatorLifetimeModeled)
        {
            blockers.Add("borrowed plugin creator pointer lifetime is not modeled as a public ownership surface.");
        }

        if (!pluginCreationModeled)
        {
            blockers.Add("plugin create/clone/enqueue ownership remains deferred.");
        }

        blockers.Add("direct plugin creator pointer export remains deferred by design.");
        blockers.Add("IPluginCreatorV3One::createPlugin remains deferred by design.");
        blockers.Add("plugin resource acquire/release remains deferred by design.");
        blockers.Add("full package consumer runtime execution proof has not been promoted from this design gate.");

        return new TensorRtPluginCreatorV3MetadataDesignGateResult(
            line,
            lineSupportsPluginCreatorV3,
            registryInventorySnapshotReady,
            copiedIdentityReady,
            copiedFieldMetadataReady,
            copiedInterfaceInfoReady,
            pluginCreationModeled,
            borrowedCreatorLifetimeModeled,
            blockers.ToArray());
    }
}
