using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class TrtexecLikeOptionCapability
{
    public TrtexecLikeOptionCapability(
        string option,
        string group,
        string status,
        string implementationClass,
        string proofBoundary,
        params string[] aliases)
    {
        Option = option ?? throw new ArgumentNullException(nameof(option));
        Group = group ?? throw new ArgumentNullException(nameof(group));
        Status = status ?? throw new ArgumentNullException(nameof(status));
        ImplementationClass = implementationClass ?? throw new ArgumentNullException(nameof(implementationClass));
        ProofBoundary = proofBoundary ?? throw new ArgumentNullException(nameof(proofBoundary));
        Aliases = aliases ?? Array.Empty<string>();
    }

    public string Option { get; }

    public string Group { get; }

    public string Status { get; }

    public string ImplementationClass { get; }

    public string ProofBoundary { get; }

    public IReadOnlyList<string> Aliases { get; }
}

public static class TrtexecLikeOptionCapabilities
{
    public static IReadOnlyList<TrtexecLikeOptionCapability> Entries { get; } =
        new[]
        {
            new TrtexecLikeOptionCapability("--onnx", "model", "implemented", "applied-build-input", "ONNX path selection is build input only; it is not runtime proof.", "--model", "--onnxFile"),
            new TrtexecLikeOptionCapability("--saveEngine", "model", "implemented", "applied-build-artifact", "Serialized engine output is build artifact evidence only.", "--save-engine", "--plan", "--engineFile"),
            new TrtexecLikeOptionCapability("--loadEngine", "model", "bounded-runtime-output", "readonly-preflight-plus-bounded-runtime", "Engine diagnostics and bounded enqueue/readback are not real-model or package-consumer runtime proof.", "--load-engine"),
            new TrtexecLikeOptionCapability("--minShapes/--optShapes/--maxShapes", "profiles", "implemented", "applied-build-option", "Profile shape application is builder evidence until real sample output is validated."),
            new TrtexecLikeOptionCapability("--shapes/--inputShapes", "profiles", "implemented-alias", "applied-build-option", "Shape alias normalization does not prove runtime output correctness."),
            new TrtexecLikeOptionCapability("--fp16", "precision", "implemented", "applied-build-option", "FP16 builder flag is not proof of model accuracy or runtime package consumption."),
            new TrtexecLikeOptionCapability("--bf16/--noTF32", "precision", "diagnostic-or-conditional", "capability-conditional", "Conditional precision availability is not promoted without compatible host runtime logs."),
            new TrtexecLikeOptionCapability("--fp8/--best", "precision", "parse-report-only", "parse-only", "Parse/report-only precision intent cannot be claimed as implemented TensorRT behavior."),
            new TrtexecLikeOptionCapability("--int8/--calib", "precision", "blocked-calibrator-lifecycle", "blocked-owner-lifecycle", "INT8/calibrator ownership remains blocked until safe calibrator lifecycle and real calibration proof exist."),
            new TrtexecLikeOptionCapability("--workspace", "memory", "implemented", "applied-build-option", "Workspace limit application is not runtime output proof."),
            new TrtexecLikeOptionCapability("--memPoolSize", "memory", "implemented-build-readback", "applied-build-option-with-readback", "Memory pool set/get readback proves builder-config acceptance only."),
            new TrtexecLikeOptionCapability("--inputIOFormats/--outputIOFormats", "builder", "implemented-build-readback-with-version-guards", "applied-network-io-policy", "Tensor type/allowed-format readback proves parsed-network policy application only."),
            new TrtexecLikeOptionCapability("--precisionConstraints/--layerPrecisions/--layerOutputTypes", "precision", "implemented-build-readback-with-version-guards", "applied-layer-policy", "Constraint and layer setter readback is builder evidence only."),
            new TrtexecLikeOptionCapability("--tacticSources", "builder", "implemented-build-readback", "applied-build-option-with-readback", "Tactic source readback proves config acceptance only, not selected tactic or performance."),
            new TrtexecLikeOptionCapability("--timingCacheFile/--timingCache", "builder", "implemented-build-cache-lifecycle", "applied-build-cache-lifecycle", "Timing cache import SHA256 is build-cache evidence only."),
            new TrtexecLikeOptionCapability("--exportTimingCache", "builder", "implemented-build-cache-lifecycle", "applied-build-cache-lifecycle", "Timing cache export SHA256 is build-cache evidence only."),
            new TrtexecLikeOptionCapability("--plugins/--plugin/--dynamicPlugins/--setPluginsToSerialize", "plugin", "diagnostic-alias-compatible", "diagnostic-only", "Plugin path normalization does not load, register, deregister, or serialize plugin libraries."),
            new TrtexecLikeOptionCapability("--profilingVerbosity/--verbose", "diagnostics", "implemented-report", "report-only", "Profile verbosity reports are not enqueue/profile runtime proof."),
            new TrtexecLikeOptionCapability("--buildOnly/--skipInference/--dryRun", "execution", "implemented-boundary", "execution-mode", "Build-only, skip-inference, and dry-run modes cannot promote runtime proof."),
            new TrtexecLikeOptionCapability("--iterations/--warmUp/--duration/--streams/--infStreams", "runtime", "bounded-runtime-control", "bounded-scheduler-control", "Scheduler controls are bounded runtime infrastructure, not model correctness proof."),
            new TrtexecLikeOptionCapability("--noDataTransfers/--useSpinWait/--threads/--avgRuns/--percentile/--idleTime", "runtime", "bounded-runtime-control", "bounded-scheduler-control", "Runtime controls remain bounded enqueue/readback infrastructure evidence."),
            new TrtexecLikeOptionCapability("--sleepTime", "runtime", "parse-report-only", "parse-only", "CPU sleep parsing is retained for report parity and is not official device-side launch-gap behavior."),
            new TrtexecLikeOptionCapability("--loadInputs", "runtime-artifacts", "implemented-bounded-float-input", "applied-bounded-input-capture", "Named float input files feed bounded runtime but do not validate expected output."),
            new TrtexecLikeOptionCapability("--dumpOutput/--dumpRawBindingsToFile/--exportOutput", "runtime-artifacts", "implemented-bounded-multi-output-capture", "applied-bounded-output-capture", "All float outputs are captured with bounded metadata and deterministic raw manifests; capture is not validation."),
            new TrtexecLikeOptionCapability("--exportTimes/--exportProfile/--saveProfile", "runtime-artifacts", "bounded-artifact", "bounded-runtime-artifact", "Exported timing and profile artifacts are not package-consumer runtime proof."),
            new TrtexecLikeOptionCapability("--versionCompatible/--excludeLeanRuntime/--stripWeights/--refit", "engine-packaging", "implemented-build-readback", "applied-build-option", "Engine packaging/refit flags are local lifecycle evidence only."),
            new TrtexecLikeOptionCapability("--refitFromOnnx/--saveRefittedEngine", "engine-packaging", "implemented-local-lifecycle", "managed-refit-persistence", "Stripped-plan refit persistence and reload do not prove package-consumer runtime."),
            new TrtexecLikeOptionCapability("--allowWeightStreaming/--weightStreamingBudget", "engine-packaging", "implemented-build-readback", "applied-build-option-with-readback", "Weight-streaming budget readback is engine/config evidence only."),
            new TrtexecLikeOptionCapability("--builderOptimizationLevel/--maxAuxStreams/--maxNbTactics", "builder", "implemented-or-version-guarded", "applied-build-option", "Builder scalar controls are source-quality builder evidence only."),
            new TrtexecLikeOptionCapability("--tilingOptimizationLevel/--l2LimitForTiling/--quantizationFlags", "builder", "implemented-or-version-guarded", "applied-build-option-with-readback", "Version-guarded scalar readbacks are not runtime proof."),
            new TrtexecLikeOptionCapability("--exportReport/--report", "report", "implemented", "structured-report", "Reports are evidence carriers and cannot substitute real runtime logs or release proof."),
            new TrtexecLikeOptionCapability("--evidenceSidecar", "evidence", "implemented", "owner-evidence-input", "Sidecars remain owner-filled evidence inputs and cannot promote templates.")
        };

    public static string FormatJson(string toolName)
    {
        string normalizedToolName = string.IsNullOrWhiteSpace(toolName) ? "trtexec-like" : toolName;
        var payload = new
        {
            schema = "trtexec-like-option-capabilities.v1",
            tool = normalizedToolName,
            generatedFrom = "JYPPX.TensorRtSharp.Tools.TrtexecLikeOptionCapabilities",
            matrixState = "source-quality-capability-surface",
            entryCount = Entries.Count,
            implementedCount = Entries.Count(static entry => entry.Status.StartsWith("implemented", StringComparison.Ordinal) || entry.Status.StartsWith("bounded", StringComparison.Ordinal)),
            parseOrDiagnosticOnlyCount = Entries.Count(static entry => entry.Status.Contains("parse", StringComparison.Ordinal) || entry.Status.Contains("diagnostic", StringComparison.Ordinal)),
            blockedCount = Entries.Count(static entry => entry.Status.Contains("blocked", StringComparison.Ordinal)),
            proofBoundary = "machine-readable help and capability metadata only; not runtime proof; not real-model-runtime proof; not package-consumer-runtime proof; not release proof",
            releaseFrozen = true,
            canPromoteRuntimeProof = false,
            entries = Entries.Select(static entry => new
            {
                option = entry.Option,
                aliases = entry.Aliases,
                group = entry.Group,
                status = entry.Status,
                implementationClass = entry.ImplementationClass,
                proofBoundary = entry.ProofBoundary,
                requiresOwnerEvidence = true,
                canPromoteRuntimeProof = false,
                canPromotePackageConsumerRuntime = false
            }).ToArray()
        };

        return JsonSerializer.Serialize(payload, new JsonSerializerOptions
        {
            WriteIndented = true
        });
    }
}
