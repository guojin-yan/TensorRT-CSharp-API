using System;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

/// <summary>
/// Represents copied evidence for a refitted-engine persistence and independent reload lifecycle.
/// 表示 refitted engine 持久化与独立重新加载生命周期的复制证据。
/// </summary>
public sealed class OnnxEngineRefitPersistenceSnapshot
{
    public OnnxEngineRefitPersistenceSnapshot(
        bool attempted,
        bool succeeded,
        string state,
        string strippedPlanPath,
        long strippedPlanLengthBytes,
        string strippedPlanSha256,
        string persistedPlanPath,
        long persistedPlanLengthBytes,
        string persistedPlanSha256,
        TensorRtSerializationFlags serializationFlagsBefore,
        TensorRtSerializationFlags serializationFlagsAfter,
        bool refittableWeightsIncludedInSerialization,
        bool artifactDiffersFromStrippedPlan,
        bool originalRefittedEngineDisposedBeforeReload,
        bool reloadAttempted,
        bool reloadSucceeded,
        bool reloadEngineRefittable,
        int reloadIoTensorCount,
        int reloadLayerCount,
        int reloadOptimizationProfileCount,
        bool reloadContextCreationAllowed,
        bool reloadEngineSelectedForRuntime,
        bool inferenceRanFromReloadedEngine,
        string evidenceBoundary)
    {
        Attempted = attempted;
        Succeeded = succeeded;
        State = state ?? string.Empty;
        StrippedPlanPath = strippedPlanPath ?? string.Empty;
        StrippedPlanLengthBytes = strippedPlanLengthBytes;
        StrippedPlanSha256 = strippedPlanSha256 ?? string.Empty;
        PersistedPlanPath = persistedPlanPath ?? string.Empty;
        PersistedPlanLengthBytes = persistedPlanLengthBytes;
        PersistedPlanSha256 = persistedPlanSha256 ?? string.Empty;
        SerializationFlagsBefore = serializationFlagsBefore;
        SerializationFlagsAfter = serializationFlagsAfter;
        RefittableWeightsIncludedInSerialization = refittableWeightsIncludedInSerialization;
        ArtifactDiffersFromStrippedPlan = artifactDiffersFromStrippedPlan;
        OriginalRefittedEngineDisposedBeforeReload = originalRefittedEngineDisposedBeforeReload;
        ReloadAttempted = reloadAttempted;
        ReloadSucceeded = reloadSucceeded;
        ReloadEngineRefittable = reloadEngineRefittable;
        ReloadIOTensorCount = reloadIoTensorCount;
        ReloadLayerCount = reloadLayerCount;
        ReloadOptimizationProfileCount = reloadOptimizationProfileCount;
        ReloadContextCreationAllowed = reloadContextCreationAllowed;
        ReloadEngineSelectedForRuntime = reloadEngineSelectedForRuntime;
        InferenceRanFromReloadedEngine = inferenceRanFromReloadedEngine;
        EvidenceBoundary = evidenceBoundary ?? string.Empty;
    }

    public static OnnxEngineRefitPersistenceSnapshot Empty { get; } = new OnnxEngineRefitPersistenceSnapshot(
        false, false, "not-requested", string.Empty, 0, string.Empty, string.Empty, 0, string.Empty,
        TensorRtSerializationFlags.None, TensorRtSerializationFlags.None, false,
        false, false, false, false, false, 0, 0, 0, false, false, false, string.Empty);

    public bool Attempted { get; }
    public bool Succeeded { get; }
    public string State { get; }
    public string StrippedPlanPath { get; }
    public long StrippedPlanLengthBytes { get; }
    public string StrippedPlanSha256 { get; }
    public string PersistedPlanPath { get; }
    public long PersistedPlanLengthBytes { get; }
    public string PersistedPlanSha256 { get; }
    public TensorRtSerializationFlags SerializationFlagsBefore { get; }
    public TensorRtSerializationFlags SerializationFlagsAfter { get; }
    public bool RefittableWeightsIncludedInSerialization { get; }
    public bool ArtifactDiffersFromStrippedPlan { get; }
    public bool OriginalRefittedEngineDisposedBeforeReload { get; }
    public bool ReloadAttempted { get; }
    public bool ReloadSucceeded { get; }
    public bool ReloadEngineRefittable { get; }
    public int ReloadIOTensorCount { get; }
    public int ReloadLayerCount { get; }
    public int ReloadOptimizationProfileCount { get; }
    public bool ReloadContextCreationAllowed { get; }
    public bool ReloadEngineSelectedForRuntime { get; }
    public bool InferenceRanFromReloadedEngine { get; }
    public string EvidenceBoundary { get; }

    public OnnxEngineRefitPersistenceSnapshot WithRuntimeOutcome(bool selectedForRuntime, bool inferenceRan)
    {
        return new OnnxEngineRefitPersistenceSnapshot(
            Attempted, Succeeded, State, StrippedPlanPath, StrippedPlanLengthBytes, StrippedPlanSha256,
            PersistedPlanPath, PersistedPlanLengthBytes, PersistedPlanSha256, SerializationFlagsBefore,
            SerializationFlagsAfter, RefittableWeightsIncludedInSerialization, ArtifactDiffersFromStrippedPlan,
            OriginalRefittedEngineDisposedBeforeReload, ReloadAttempted, ReloadSucceeded, ReloadEngineRefittable,
            ReloadIOTensorCount, ReloadLayerCount, ReloadOptimizationProfileCount, ReloadContextCreationAllowed,
            selectedForRuntime, inferenceRan, EvidenceBoundary);
    }
}
