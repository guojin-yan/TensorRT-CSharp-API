using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class OnnxEngineBuildService
{
    private static (TensorRtEngine Engine, OnnxEngineRefitPersistenceSnapshot Snapshot) PersistAndReloadRefittedEngine(
        TensorRtRuntime runtime,
        TensorRtEngine refittedEngine,
        string strippedPlanPath,
        string persistedPlanPath,
        List<string> log)
    {
        const string boundary = "The snapshot proves that this committed engine was serialized, the original engine was disposed, and a distinct managed engine reloaded the persisted bytes. Runtime output still requires an explicit enqueue/baseline comparison and this is not package-consumer or public-release proof.";
        byte[] strippedPlan = File.ReadAllBytes(strippedPlanPath);
        byte[] persistedPlan;
        TensorRtSerializationFlags serializationFlagsBefore = TensorRtSerializationFlags.None;
        TensorRtSerializationFlags serializationFlagsAfter = TensorRtSerializationFlags.None;
        bool refittableWeightsIncluded = false;
        bool originalDisposed = false;
        try
        {
            using TensorRtSerializationConfig serializationConfig = refittedEngine.CreateSerializationConfig();
            serializationFlagsBefore = serializationConfig.Flags;
            bool excludeWeightsCleared = serializationConfig.ClearFlag(TensorRtSerializationFlag.ExcludeWeights);
            serializationFlagsAfter = serializationConfig.Flags;
            refittableWeightsIncluded = excludeWeightsCleared &&
                (serializationFlagsAfter & TensorRtSerializationFlags.ExcludeWeights) == 0;
            if (!refittableWeightsIncluded)
            {
                throw new InvalidOperationException("TensorRT did not clear ExcludeWeights before refitted-engine serialization.");
            }

            using TensorRtHostMemory serialized = refittedEngine.Serialize(serializationConfig);
            persistedPlan = serialized.ToArray();
            string? directory = Path.GetDirectoryName(persistedPlanPath);
            if (!string.IsNullOrWhiteSpace(directory))
            {
                Directory.CreateDirectory(directory);
            }

            File.WriteAllBytes(persistedPlanPath, persistedPlan);
        }
        finally
        {
            refittedEngine.Dispose();
            originalDisposed = true;
        }

        string strippedSha256 = ComputeSha256(strippedPlan);
        string persistedSha256 = ComputeSha256(persistedPlan);
        bool differsFromStripped = strippedPlan.LongLength != persistedPlan.LongLength ||
            !string.Equals(strippedSha256, persistedSha256, StringComparison.Ordinal);
        TensorRtEngine? reloadedEngine = null;
        try
        {
            reloadedEngine = runtime.DeserializeFromFile(persistedPlanPath);
            bool reloadRefittable = reloadedEngine.IsRefittable;
            int ioTensorCount = reloadedEngine.IOTensorCount;
            int layerCount = reloadedEngine.LayerCount;
            int profileCount = reloadedEngine.OptimizationProfileCount;
            bool reloadGate = ioTensorCount > 0 && layerCount > 0 && profileCount > 0;
            bool succeeded = persistedPlan.LongLength > 0 &&
                refittableWeightsIncluded &&
                originalDisposed &&
                differsFromStripped &&
                reloadGate;
            OnnxEngineRefitPersistenceSnapshot snapshot = new OnnxEngineRefitPersistenceSnapshot(
                attempted: true,
                succeeded,
                state: succeeded ? "refitted-plan-persisted-and-reloaded" : "refitted-plan-persistence-incomplete",
                strippedPlanPath,
                strippedPlanLengthBytes: strippedPlan.LongLength,
                strippedPlanSha256: strippedSha256,
                persistedPlanPath,
                persistedPlanLengthBytes: persistedPlan.LongLength,
                persistedPlanSha256: persistedSha256,
                serializationFlagsBefore,
                serializationFlagsAfter,
                refittableWeightsIncludedInSerialization: refittableWeightsIncluded,
                artifactDiffersFromStrippedPlan: differsFromStripped,
                originalRefittedEngineDisposedBeforeReload: originalDisposed,
                reloadAttempted: true,
                reloadSucceeded: true,
                reloadEngineRefittable: reloadRefittable,
                reloadIoTensorCount: ioTensorCount,
                reloadLayerCount: layerCount,
                reloadOptimizationProfileCount: profileCount,
                reloadContextCreationAllowed: succeeded,
                reloadEngineSelectedForRuntime: false,
                inferenceRanFromReloadedEngine: false,
                evidenceBoundary: boundary);
            log.Add(
                $"OnnxRefitPersistence Attempted=True Succeeded={succeeded} StrippedPlan={strippedPlanPath} StrippedLengthBytes={strippedPlan.LongLength} StrippedSha256={strippedSha256} " +
                $"PersistedPlan={persistedPlanPath} PersistedLengthBytes={persistedPlan.LongLength} PersistedSha256={persistedSha256} ArtifactDiffers={differsFromStripped} " +
                $"SerializationFlagsBefore={serializationFlagsBefore} SerializationFlagsAfter={serializationFlagsAfter} RefittableWeightsIncluded={refittableWeightsIncluded} " +
                $"OriginalDisposedBeforeReload={originalDisposed} ReloadSucceeded=True ReloadRefittable={reloadRefittable} IOTensors={ioTensorCount} Layers={layerCount} Profiles={profileCount} ContextCreationAllowed={snapshot.ReloadContextCreationAllowed}");
            if (!succeeded)
            {
                throw new InvalidOperationException("The refitted plan was written but did not pass the independent reload gate.");
            }

            TensorRtEngine result = reloadedEngine;
            reloadedEngine = null;
            return (result, snapshot);
        }
        finally
        {
            reloadedEngine?.Dispose();
        }
    }

    private static OnnxEngineRefitSnapshot RefitStrippedEngineFromOnnx(
        TensorRtEngine engine,
        TensorRtLogger logger,
        string sourcePath,
        List<string> log)
    {
        const string boundary = "Copied refit inventory and parser diagnostics prove only this in-memory engine lifecycle; they do not prove model accuracy, persistence of refitted weights in the stripped plan, package-consumer runtime, or public release readiness.";
        if (engine.Line == TensorRtApiLine.TensorRt8)
        {
            throw new NotSupportedException("ONNX parser-refitter is available only for TensorRT 10 and TensorRT 11.");
        }

        byte[] sourceBytes = File.ReadAllBytes(sourcePath);
        bool refittableBefore = engine.IsRefittable;
        if (!refittableBefore)
        {
            throw new InvalidOperationException("The deserialized stripped plan is not refittable.");
        }

        using TensorRtRefitter refitter = engine.CreateRefitter(logger);
        IReadOnlyList<string> missingBefore = CopyRefitEntries(refitter.GetMissingEntries());
        IReadOnlyList<string> allBefore = CopyRefitEntries(refitter.GetAllEntries());
        using TensorRtOnnxParserRefitter parserRefitter = refitter.CreateOnnxParserRefitter(logger);
        parserRefitter.ClearErrors();
        bool parserRefitReturned = parserRefitter.RefitFromFile(sourcePath);
        TensorRtOnnxParserRefitterDiagnosticSnapshot parserSnapshot = parserRefitter.GetDiagnosticSnapshot();
        bool engineRefitReturned = parserRefitReturned && parserSnapshot.ErrorCount == 0 && refitter.RefitCudaEngine();
        IReadOnlyList<string> missingAfter = CopyRefitEntries(refitter.GetMissingEntries());
        IReadOnlyList<string> allAfter = CopyRefitEntries(refitter.GetAllEntries());
        bool refittableAfter = engine.IsRefittable;
        bool succeeded = parserRefitReturned &&
            engineRefitReturned &&
            parserSnapshot.ErrorCount == 0 &&
            missingAfter.Count == 0 &&
            refittableAfter;

        OnnxEngineRefitSnapshot snapshot = new OnnxEngineRefitSnapshot(
            attempted: true,
            succeeded,
            state: succeeded ? "onnx-refit-complete" : "onnx-refit-incomplete",
            sourcePath,
            sourceLengthBytes: sourceBytes.LongLength,
            sourceSha256: ComputeSha256(sourceBytes),
            engineRefittableBefore: refittableBefore,
            engineRefittableAfter: refittableAfter,
            parserRefitReturned,
            engineRefitReturned,
            missingWeightsBefore: missingBefore,
            allWeightsBefore: allBefore,
            missingWeightsAfter: missingAfter,
            allWeightsAfter: allAfter,
            parserErrorCount: parserSnapshot.ErrorCount,
            copiedDiagnosticCount: parserSnapshot.Diagnostics.Count,
            diagnosticSummary: parserSnapshot.DiagnosticSummary,
            contextCreationAllowed: succeeded,
            evidenceBoundary: boundary);

        log.Add(
            $"OnnxRefitLifecycle Attempted=True Succeeded={succeeded} Source={sourcePath} SourceLengthBytes={sourceBytes.LongLength} " +
            $"SourceSha256={snapshot.SourceSha256} EngineRefittableBefore={refittableBefore} EngineRefittableAfter={refittableAfter} " +
            $"ParserRefitReturned={parserRefitReturned} EngineRefitReturned={engineRefitReturned} ParserErrors={parserSnapshot.ErrorCount} CopiedDiagnostics={parserSnapshot.Diagnostics.Count} " +
            $"MissingBefore={missingBefore.Count} AllBefore={allBefore.Count} MissingAfter={missingAfter.Count} AllAfter={allAfter.Count} " +
            $"ContextCreationAllowed={snapshot.ContextCreationAllowed}");
        log.Add(
            $"OnnxRefitInventory MissingBeforeSha256={ComputeSha256(string.Join("\n", missingBefore))} " +
            $"AllBeforeSha256={ComputeSha256(string.Join("\n", allBefore))} " +
            $"MissingAfterSha256={ComputeSha256(string.Join("\n", missingAfter))} " +
            $"AllAfterSha256={ComputeSha256(string.Join("\n", allAfter))}");
        return snapshot;
    }

    private static IReadOnlyList<string> CopyRefitEntries(IReadOnlyList<TensorRtRefitEntry> entries)
    {
        return entries.Select(static entry => entry.ToString()).ToArray();
    }

}
