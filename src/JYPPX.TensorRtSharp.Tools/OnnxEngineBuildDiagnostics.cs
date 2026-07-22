using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Tools;

public static class OnnxEngineBuildDiagnostics
{
    public static void WriteReport(OnnxEngineBuildResult result, string reportPath)
    {
        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        if (string.IsNullOrWhiteSpace(reportPath))
        {
            return;
        }

        string? directory = Path.GetDirectoryName(Path.GetFullPath(reportPath));
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        string extension = Path.GetExtension(reportPath);
        string content = string.Equals(extension, ".md", StringComparison.OrdinalIgnoreCase)
            ? ToMarkdown(result)
            : ToJson(result);
        File.WriteAllText(reportPath, content);
    }

    public static string ToJson(OnnxEngineBuildResult result)
    {
        return JsonSerializer.Serialize(new
        {
            result.Success,
            result.Skipped,
            result.State,
            TensorRtLine = (int)result.TensorRtLine,
            result.ModelSource,
            result.EnginePath,
            result.Parsed,
            result.EngineSaved,
            result.EngineFileRoundTrip,
            DryRun = string.Equals(result.ProofClassification, "precheck", StringComparison.Ordinal),
            result.InferenceRan,
            result.OutputMatch,
            result.ProfileIndex,
            result.ElapsedMilliseconds,
            result.SkipReason,
            result.NormalizedCommandLine,
            result.NormalizedCommandSha256,
            result.DeploymentOptions,
            result.RuntimeOptions,
            result.PreflightMetadata,
            result.LoadedEngineDiagnostics,
            result.TimingCacheArtifact,
            result.CapabilityProbe,
            result.WorkspaceBytes,
            result.BuilderConfigDeploymentSnapshot,
            result.ParserPreflightSnapshot,
            result.RefitSnapshot,
            result.RefitPersistenceSnapshot,
            OptionImplementationStatus = CreateOptionImplementationStatus(result),
            result.BenchmarkSummary,
            result.ProofClassification,
            result.EvidenceClassifications,
            result.BuildEvidenceOnly,
            result.IsRealModelRuntimeProof,
            result.IsPackageConsumerRuntimeProof,
            result.StdoutSummary,
            result.StderrSummary,
            result.ModelEvidence,
            EvidenceSidecarPath = result.EvidenceSidecar.Path,
            EvidenceSidecarProofClassification = result.EvidenceSidecar.ProofClassification,
            EvidenceSidecarDiagnostics = result.EvidenceSidecar.Diagnostics,
            result.Diagnostics,
            result.LogLines,
            result.IsRuntimeExecutionProof,
            ReportBoundary = CreateReportBoundary(result)
        }, new JsonSerializerOptions { WriteIndented = true });
    }

    public static string ToMarkdown(OnnxEngineBuildResult result)
    {
        OnnxEngineBuildOptionImplementationStatus optionImplementationStatus = CreateOptionImplementationStatus(result);
        OnnxEngineBuildReportBoundary reportBoundary = CreateReportBoundary(result);

        return string.Join(Environment.NewLine, new[]
        {
            "# ONNX Engine Build Report",
            string.Empty,
            $"State: `{result.State}`",
            $"Success: `{result.Success}`",
            $"Skipped: `{result.Skipped}`",
            $"TensorRT line: `{(int)result.TensorRtLine}`",
            $"Model source: `{result.ModelSource}`",
            $"Engine path: `{result.EnginePath}`",
            $"Parsed: `{result.Parsed}`",
            $"Engine saved: `{result.EngineSaved}`",
            $"Engine file round trip: `{result.EngineFileRoundTrip}`",
            $"Preflight kind: `{result.PreflightMetadata.Kind}`",
            $"Preflight path: `{result.PreflightMetadata.Path}`",
            $"Preflight exists: `{result.PreflightMetadata.Exists}`",
            $"Preflight length bytes: `{result.PreflightMetadata.LengthBytes}`",
            $"Preflight SHA256: `{result.PreflightMetadata.Sha256}`",
            $"Preflight state: `{result.PreflightMetadata.PreflightState}`",
            $"Preflight proof classification: `{result.PreflightMetadata.ProofClassification}`",
            $"Preflight evidence boundary: `{result.PreflightMetadata.EvidenceBoundary}`",
            $"Workspace bytes: `{result.WorkspaceBytes}`",
            $"Builder config deployment snapshot: `{(result.BuilderConfigDeploymentSnapshot == null ? "unavailable" : "copied-readback")}`",
            $"Builder config deployment values: `{result.BuilderConfigDeploymentSnapshot?.ToString() ?? string.Empty}`",
            $"Builder scalar controls: `{result.DeploymentOptions.ScalarControlSummary}`",
            $"Parser preflight snapshot: `{result.ParserPreflightSnapshot}`",
            $"Parser preflight diagnostics state: `{result.ParserPreflightSnapshot.DiagnosticsState}`",
            $"Parser preflight errors: `{result.ParserPreflightSnapshot.ErrorCount}`",
            $"Parser preflight copied diagnostics: `{result.ParserPreflightSnapshot.CopiedDiagnosticCount}`",
            $"Parser preflight identity support: `{result.ParserPreflightSnapshot.IdentityOperatorSupported}`",
            $"Parser model support state: `{result.ParserPreflightSnapshot.ModelSupportState}`",
            $"Parser model supported: `{result.ParserPreflightSnapshot.ModelSupported}`",
            $"Parser copied subgraphs: `{result.ParserPreflightSnapshot.CopiedSubgraphCount}`",
            $"ONNX refit attempted: `{result.RefitSnapshot.Attempted}`",
            $"ONNX refit succeeded: `{result.RefitSnapshot.Succeeded}`",
            $"ONNX refit state: `{result.RefitSnapshot.State}`",
            $"ONNX refit source: `{result.RefitSnapshot.SourcePath}`",
            $"ONNX refit source length bytes: `{result.RefitSnapshot.SourceLengthBytes}`",
            $"ONNX refit source SHA256: `{result.RefitSnapshot.SourceSha256}`",
            $"ONNX refit engine refittable before/after: `{result.RefitSnapshot.EngineRefittableBefore}/{result.RefitSnapshot.EngineRefittableAfter}`",
            $"ONNX refit parser returned: `{result.RefitSnapshot.ParserRefitReturned}`",
            $"ONNX refit engine commit returned: `{result.RefitSnapshot.EngineRefitReturned}`",
            $"ONNX refit missing weights before/after: `{result.RefitSnapshot.MissingWeightsBefore.Count}/{result.RefitSnapshot.MissingWeightsAfter.Count}`",
            $"ONNX refit all weights before/after: `{result.RefitSnapshot.AllWeightsBefore.Count}/{result.RefitSnapshot.AllWeightsAfter.Count}`",
            $"ONNX refit parser errors/copied diagnostics: `{result.RefitSnapshot.ParserErrorCount}/{result.RefitSnapshot.CopiedDiagnosticCount}`",
            $"ONNX refit context creation allowed: `{result.RefitSnapshot.ContextCreationAllowed}`",
            $"ONNX refit diagnostic summary: `{result.RefitSnapshot.DiagnosticSummary}`",
            $"ONNX refit evidence boundary: `{result.RefitSnapshot.EvidenceBoundary}`",
            $"Refitted plan persistence attempted: `{result.RefitPersistenceSnapshot.Attempted}`",
            $"Refitted plan persistence succeeded: `{result.RefitPersistenceSnapshot.Succeeded}`",
            $"Refitted plan persistence state: `{result.RefitPersistenceSnapshot.State}`",
            $"Refitted plan stripped artifact: `{result.RefitPersistenceSnapshot.StrippedPlanPath}`",
            $"Refitted plan stripped length/SHA256: `{result.RefitPersistenceSnapshot.StrippedPlanLengthBytes}/{result.RefitPersistenceSnapshot.StrippedPlanSha256}`",
            $"Refitted plan persisted artifact: `{result.RefitPersistenceSnapshot.PersistedPlanPath}`",
            $"Refitted plan persisted length/SHA256: `{result.RefitPersistenceSnapshot.PersistedPlanLengthBytes}/{result.RefitPersistenceSnapshot.PersistedPlanSha256}`",
            $"Refitted plan serialization flags before/after: `{result.RefitPersistenceSnapshot.SerializationFlagsBefore}/{result.RefitPersistenceSnapshot.SerializationFlagsAfter}`",
            $"Refittable weights included in serialization: `{result.RefitPersistenceSnapshot.RefittableWeightsIncludedInSerialization}`",
            $"Refitted artifact differs from stripped plan: `{result.RefitPersistenceSnapshot.ArtifactDiffersFromStrippedPlan}`",
            $"Original refitted engine disposed before reload: `{result.RefitPersistenceSnapshot.OriginalRefittedEngineDisposedBeforeReload}`",
            $"Refitted plan reload attempted/succeeded: `{result.RefitPersistenceSnapshot.ReloadAttempted}/{result.RefitPersistenceSnapshot.ReloadSucceeded}`",
            $"Reloaded engine refittable: `{result.RefitPersistenceSnapshot.ReloadEngineRefittable}`",
            $"Reloaded engine IO/layers/profiles: `{result.RefitPersistenceSnapshot.ReloadIOTensorCount}/{result.RefitPersistenceSnapshot.ReloadLayerCount}/{result.RefitPersistenceSnapshot.ReloadOptimizationProfileCount}`",
            $"Reloaded engine context creation allowed: `{result.RefitPersistenceSnapshot.ReloadContextCreationAllowed}`",
            $"Reloaded engine selected/inference ran: `{result.RefitPersistenceSnapshot.ReloadEngineSelectedForRuntime}/{result.RefitPersistenceSnapshot.InferenceRanFromReloadedEngine}`",
            $"Refitted plan persistence boundary: `{result.RefitPersistenceSnapshot.EvidenceBoundary}`",
            $"Loaded engine diagnostics attempted: `{result.LoadedEngineDiagnostics.Attempted}`",
            $"Loaded engine diagnostics succeeded: `{result.LoadedEngineDiagnostics.Succeeded}`",
            $"Loaded engine diagnostics state: `{result.LoadedEngineDiagnostics.DiagnosticsState}`",
            $"Loaded engine diagnostics failure: `{result.LoadedEngineDiagnostics.FailureReason}`",
            $"Loaded engine name: `{result.LoadedEngineDiagnostics.EngineName}`",
            $"Loaded engine IO tensor count: `{result.LoadedEngineDiagnostics.IOTensorCount}`",
            $"Loaded engine layer count: `{result.LoadedEngineDiagnostics.LayerCount}`",
            $"Loaded engine optimization profile count: `{result.LoadedEngineDiagnostics.OptimizationProfileCount}`",
            $"Loaded engine device memory bytes: `{result.LoadedEngineDiagnostics.DeviceMemorySizeInBytes}`",
            $"Loaded engine auxiliary streams: `{result.LoadedEngineDiagnostics.AuxiliaryStreamCount}`",
            $"Loaded engine capability: `{result.LoadedEngineDiagnostics.Capability}`",
            $"Loaded engine profiling verbosity: `{result.LoadedEngineDiagnostics.ProfilingVerbosity}`",
            $"Loaded engine inspector information length: `{result.LoadedEngineDiagnostics.InspectorInformationLength}`",
            $"Loaded engine readback fingerprint: `{result.LoadedEngineDiagnostics.ReadbackFingerprint}`",
            $"Loaded engine readback SHA256: `{result.LoadedEngineDiagnostics.ReadbackSha256}`",
            $"Loaded engine evidence boundary: `{result.LoadedEngineDiagnostics.EvidenceBoundary}`",
            $"Loaded engine IO tensors: `{string.Join("; ", result.LoadedEngineDiagnostics.IOTensorSummaries)}`",
            $"Timing cache state: `{result.TimingCacheArtifact.State}`",
            $"Timing cache input requested: `{result.TimingCacheArtifact.InputRequested}`",
            $"Timing cache input applied: `{result.TimingCacheArtifact.InputApplied}`",
            $"Timing cache input path: `{result.TimingCacheArtifact.InputPath}`",
            $"Timing cache input length bytes: `{result.TimingCacheArtifact.InputLengthBytes}`",
            $"Timing cache input SHA256: `{result.TimingCacheArtifact.InputSha256}`",
            $"Timing cache output requested: `{result.TimingCacheArtifact.OutputRequested}`",
            $"Timing cache output written: `{result.TimingCacheArtifact.OutputWritten}`",
            $"Timing cache output path: `{result.TimingCacheArtifact.OutputPath}`",
            $"Timing cache output length bytes: `{result.TimingCacheArtifact.OutputLengthBytes}`",
            $"Timing cache output SHA256: `{result.TimingCacheArtifact.OutputSha256}`",
            $"Timing cache evidence boundary: `{result.TimingCacheArtifact.EvidenceBoundary}`",
            $"Capability probe attempted: `{result.CapabilityProbe.Attempted}`",
            $"Capability probe state: `{result.CapabilityProbe.ProbeState}`",
            $"Capability probe TensorRT line: `{(int)result.CapabilityProbe.TensorRtLine}`",
            $"Capability probe TensorRT version: `{result.CapabilityProbe.TensorRtVersion}`",
            $"Capability probe CUDA toolkit version: `{result.CapabilityProbe.CudaToolkitVersion}`",
            $"Capability probe runtime available: `{result.CapabilityProbe.RuntimeAvailable}`",
            $"Capability probe builder available: `{result.CapabilityProbe.BuilderAvailable}`",
            $"Capability probe builder config available: `{result.CapabilityProbe.BuilderConfigAvailable}`",
            $"Capability probe engine inspector API available: `{result.CapabilityProbe.EngineInspectorApiAvailable}`",
            $"Capability probe FP8 requested: `{result.CapabilityProbe.Fp8FlagRequested}`",
            $"Capability probe FP8 flag known: `{result.CapabilityProbe.Fp8FlagKnown}`",
            $"Capability probe debug tensor requested: `{result.CapabilityProbe.DebugTensorOptionsRequested}`",
            $"Capability probe debug tensor API known: `{result.CapabilityProbe.DebugTensorApiKnown}`",
            $"Capability probe weight streaming requested: `{result.CapabilityProbe.WeightStreamingRequested}`",
            $"Capability probe weight streaming API known: `{result.CapabilityProbe.WeightStreamingApiKnown}`",
            $"Capability probe items: `{string.Join("; ", result.CapabilityProbe.ProbeItems)}`",
            $"Capability probe evidence boundary: `{result.CapabilityProbe.EvidenceBoundary}`",
            $"Dry run: `{string.Equals(result.ProofClassification, "precheck", StringComparison.Ordinal)}`",
            $"Inference ran: `{result.InferenceRan}`",
            $"Output match: `{result.OutputMatch}`",
            $"Profile index: `{result.ProfileIndex}`",
            $"Elapsed ms: `{result.ElapsedMilliseconds?.ToString("0.###") ?? ""}`",
            $"Skip reason: `{result.SkipReason}`",
            $"Normalized command line: `{result.NormalizedCommandLine}`",
            $"Normalized command SHA256: `{result.NormalizedCommandSha256}`",
            $"Builder optimization level: `{result.DeploymentOptions.BuilderOptimizationLevel}`",
            $"Max aux streams: `{result.DeploymentOptions.MaxAuxStreams?.ToString() ?? ""}`",
            $"Max tactics: `{result.DeploymentOptions.MaxNbTactics?.ToString() ?? ""}`",
            $"Tiling optimization level: `{result.DeploymentOptions.TilingOptimizationLevel?.ToString() ?? ""}`",
            $"L2 limit for tiling bytes: `{result.DeploymentOptions.L2LimitForTilingBytes?.ToString() ?? ""}`",
            $"Quantization flags: `{result.DeploymentOptions.QuantizationFlags?.ToString() ?? ""}`",
            $"Device: `{result.DeploymentOptions.DeviceOrdinal?.ToString() ?? ""}`",
            $"DLA core: `{result.DeploymentOptions.DlaCore?.ToString() ?? ""}`",
            $"Allow GPU fallback: `{result.DeploymentOptions.AllowGpuFallback}`",
            $"Tactic sources: `{result.DeploymentOptions.TacticSources}`",
            $"Memory pools: `{string.Join(", ", result.DeploymentOptions.MemoryPoolSizes.Select(static item => item.ToArgumentSegment()))}`",
            $"Input IO formats: `{result.DeploymentOptions.InputIOFormats}`",
            $"Output IO formats: `{result.DeploymentOptions.OutputIOFormats}`",
            $"Calibration cache: `{result.DeploymentOptions.CalibrationCacheFile}`",
            $"Direct IO: `{result.DeploymentOptions.DirectIO}`",
            $"Sparsity: `{result.DeploymentOptions.Sparsity}`",
            $"Strongly typed: `{result.DeploymentOptions.StronglyTyped}`",
            $"Min timing: `{result.DeploymentOptions.MinTiming?.ToString() ?? ""}`",
            $"Average timing: `{result.DeploymentOptions.AvgTiming?.ToString() ?? ""}`",
            $"Precision constraints: `{result.DeploymentOptions.PrecisionConstraints}`",
            $"Layer precisions: `{result.DeploymentOptions.LayerPrecisions}`",
            $"Layer output types: `{result.DeploymentOptions.LayerOutputTypes}`",
            $"FP8: `{result.DeploymentOptions.Fp8}`",
            $"Best precision shortcut: `{result.DeploymentOptions.Best}`",
            $"Dump refit: `{result.DeploymentOptions.DumpRefit}`",
            $"Allow weight streaming: `{result.DeploymentOptions.AllowWeightStreaming}`",
            $"Mark debug tensors: `{result.DeploymentOptions.MarkDebug}`",
            $"Dump debug tensors: `{result.DeploymentOptions.DumpDebugTensors}`",
            $"Version compatible: `{result.DeploymentOptions.VersionCompatible}`",
            $"Exclude lean runtime: `{result.DeploymentOptions.ExcludeLeanRuntime}`",
            $"Strip weights: `{result.DeploymentOptions.StripWeights}`",
            $"Refit: `{result.DeploymentOptions.Refit}`",
            $"Refit from ONNX: `{result.DeploymentOptions.RefitFromOnnxPath}`",
            $"Weight streaming budget bytes: `{result.DeploymentOptions.WeightStreamingBudgetBytes?.ToString() ?? ""}`",
            $"Export timing cache: `{result.DeploymentOptions.ExportTimingCachePath}`",
            $"Safe mode: `{result.DeploymentOptions.Safe}`",
            $"Consistency check: `{result.DeploymentOptions.Consistency}`",
            $"Builder cache: `{result.DeploymentOptions.BuilderCache}`",
            $"No builder cache: `{result.DeploymentOptions.NoBuilderCache}`",
            $"No data transfers: `{result.RuntimeOptions.NoDataTransfers}`",
            $"Use spin wait: `{result.RuntimeOptions.UseSpinWait}`",
            $"Threads: `{result.RuntimeOptions.Threads?.ToString() ?? ""}`",
            $"Average runs: `{result.RuntimeOptions.AvgRuns?.ToString() ?? ""}`",
            $"Percentile: `{result.RuntimeOptions.Percentile?.ToString() ?? ""}`",
            $"Sleep time ms: `{result.RuntimeOptions.SleepTimeMilliseconds?.ToString() ?? ""}`",
            $"Idle time ms: `{result.RuntimeOptions.IdleTimeMilliseconds?.ToString() ?? ""}`",
            $"Inference streams: `{result.RuntimeOptions.InfStreams?.ToString() ?? ""}`",
            $"Load inputs: `{result.RuntimeOptions.LoadInputs}`",
            $"Dump output: `{result.RuntimeOptions.DumpOutput}`",
            $"Dump raw bindings to file: `{result.RuntimeOptions.DumpRawBindingsToFile}`",
            $"Export output: `{result.RuntimeOptions.ExportOutputPath}`",
            $"Export times: `{result.RuntimeOptions.ExportTimesPath}`",
            $"Export profile: `{result.RuntimeOptions.ExportProfilePath}`",
            $"Save profile: `{result.RuntimeOptions.SaveProfilePath}`",
            string.Empty,
            "## Option Implementation Status",
            string.Empty,
            $"Evidence boundary: `{optionImplementationStatus.EvidenceBoundary}`",
            string.Empty,
            "Parsed options:",
            string.Empty,
            FormatOptionList(optionImplementationStatus.ParsedOptions),
            string.Empty,
            "Applied options:",
            string.Empty,
            FormatOptionList(optionImplementationStatus.AppliedOptions),
            string.Empty,
            "Parse-only options:",
            string.Empty,
            FormatOptionList(optionImplementationStatus.ParseOnlyOptions),
            string.Empty,
            $"Benchmark sample count: `{result.BenchmarkSummary.TimingSampleCount}`",
            $"Benchmark average ms: `{result.BenchmarkSummary.AverageElapsedMilliseconds?.ToString("0.###") ?? ""}`",
            $"Benchmark percentile requested: `{result.BenchmarkSummary.PercentileRequested?.ToString() ?? ""}`",
            $"Benchmark percentile ms: `{result.BenchmarkSummary.PercentileElapsedMilliseconds?.ToString("0.###") ?? ""}`",
            $"Benchmark avg runs requested: `{result.BenchmarkSummary.AvgRunsRequested?.ToString() ?? ""}`",
            $"Benchmark avg runs executed: `{result.BenchmarkSummary.AvgRunsExecuted}`",
            $"Benchmark averaged timing sample count: `{result.BenchmarkSummary.AveragedTimingSampleCount}`",
            $"Benchmark threads requested: `{result.BenchmarkSummary.ThreadsRequested?.ToString() ?? ""}`",
            $"Benchmark threads executed: `{result.BenchmarkSummary.ThreadsExecuted}`",
            $"Benchmark no data transfers requested: `{result.BenchmarkSummary.NoDataTransfersRequested}`",
            $"Benchmark no data transfers applied: `{result.BenchmarkSummary.NoDataTransfersApplied}`",
            $"Benchmark spin wait requested: `{result.BenchmarkSummary.UseSpinWaitRequested}`",
            $"Benchmark spin wait applied: `{result.BenchmarkSummary.UseSpinWaitApplied}`",
            $"Benchmark CUDA graph requested: `{result.BenchmarkSummary.UseCudaGraphRequested}`",
            $"Benchmark CUDA graph applied: `{result.BenchmarkSummary.UseCudaGraphApplied}`",
            $"Benchmark CUDA graph fallback reason: `{result.BenchmarkSummary.UseCudaGraphFallbackReason}`",
            $"Benchmark iterations requested: `{result.BenchmarkSummary.IterationsRequested}`",
            $"Benchmark measurement rounds executed: `{result.BenchmarkSummary.MeasurementRoundsExecuted}`",
            $"Benchmark measurement rounds per context: `{string.Join(",", result.BenchmarkSummary.MeasurementRoundsPerContext)}`",
            $"Benchmark inference iterations executed: `{result.BenchmarkSummary.InferenceIterationsExecuted}`",
            $"Benchmark warmup requested ms: `{result.BenchmarkSummary.WarmUpMillisecondsRequested}`",
            $"Benchmark warmup elapsed ms: `{result.BenchmarkSummary.WarmUpElapsedMilliseconds:0.###}`",
            $"Benchmark warmup iterations executed: `{result.BenchmarkSummary.WarmUpIterationsExecuted}`",
            $"Benchmark duration requested seconds: `{result.BenchmarkSummary.DurationSecondsRequested}`",
            $"Benchmark measurement elapsed ms: `{result.BenchmarkSummary.MeasurementElapsedMilliseconds:0.###}`",
            $"Benchmark streams requested: `{result.BenchmarkSummary.StreamsRequested}`",
            $"Benchmark inference streams requested: `{result.BenchmarkSummary.InfStreamsRequested?.ToString() ?? ""}`",
            $"Benchmark execution contexts created: `{result.BenchmarkSummary.ExecutionContextsCreated}`",
            $"Benchmark concurrent streams executed: `{result.BenchmarkSummary.ConcurrentStreamsExecuted}`",
            $"Benchmark idle time requested ms: `{result.BenchmarkSummary.IdleTimeMillisecondsRequested?.ToString() ?? ""}`",
            $"Benchmark idle time applied ms: `{result.BenchmarkSummary.IdleTimeMillisecondsApplied}`",
            $"Benchmark sleep time requested ms: `{result.BenchmarkSummary.SleepTimeMillisecondsRequested?.ToString() ?? ""}`",
            $"Benchmark sleep time applied ms: `{result.BenchmarkSummary.SleepTimeMillisecondsApplied}`",
            $"Benchmark boundary: `{result.BenchmarkSummary.BenchmarkBoundary}`",
            $"Proof classification: `{result.ProofClassification}`",
            $"Build evidence only: `{result.BuildEvidenceOnly}`",
            $"Runtime execution proof: `{result.IsRuntimeExecutionProof}`",
            $"Real model runtime proof: `{result.IsRealModelRuntimeProof}`",
            $"Package consumer runtime proof: `{result.IsPackageConsumerRuntimeProof}`",
            $"Stdout summary: `{result.StdoutSummary}`",
            $"Stderr summary: `{result.StderrSummary}`",
            $"Model source: `{result.ModelEvidence.ModelSource}`",
            $"Model SHA256: `{result.ModelEvidence.ModelSha256}`",
            $"Model license: `{result.ModelEvidence.ModelLicense}`",
            $"Input asset: `{result.ModelEvidence.InputAssetName}`",
            $"Input asset SHA256: `{result.ModelEvidence.InputAssetSha256}`",
            $"Evidence sidecar path: `{result.EvidenceSidecar.Path}`",
            $"Evidence sidecar proof classification: `{result.EvidenceSidecar.ProofClassification}`",
            $"Report boundary copied diagnostics: `{reportBoundary.CopiedDiagnosticsBoundary}`",
            $"Report boundary parser diagnostics evidence kind: `{reportBoundary.ParserDiagnosticsEvidenceKind}`",
            $"Report boundary parser-refitter diagnostics evidence kind: `{reportBoundary.ParserRefitterDiagnosticsEvidenceKind}`",
            $"Report boundary copied diagnostics runtime proof: `{reportBoundary.CanPromoteCopiedDiagnosticsToRuntimeProof}`",
            $"Report boundary parser diagnostics owner action: `{reportBoundary.ParserDiagnosticsOwnerAction}`",
            string.Empty,
            "This report is build/sample evidence only. precheck, build-only and dependency-probe-only are not runtime proof; synthetic-input-runtime is not real model proof; package-consumer-runtime is tracked by release proof records.",
            string.Empty,
            "## Evidence Classifications",
            string.Empty,
            string.Join(Environment.NewLine, result.EvidenceClassifications.Select(static item => "- `" + item + "`")),
            string.Empty,
            "## Evidence Sidecar Diagnostics",
            string.Empty,
            result.EvidenceSidecar.Diagnostics.Count == 0
                ? "- none"
                : string.Join(Environment.NewLine, result.EvidenceSidecar.Diagnostics.Select(static item => "- " + item)),
            string.Empty,
            "## Log",
            string.Empty,
            string.Join(Environment.NewLine, result.LogLines)
        });
    }

    private static OnnxEngineBuildOptionImplementationStatus CreateOptionImplementationStatus(OnnxEngineBuildResult result)
    {
        return new OnnxEngineBuildOptionImplementationStatus(
            BuildParsedOptions(result),
            BuildAppliedOptions(result),
            BuildParseOnlyOptions(result),
            "build reports distinguish parsed, applied, parse-only, and capability-probe-only evidence; bounded benchmark execution can apply iterations/warmUp/duration/streams/infStreams/idleTime without promoting tensor correctness or package-consumer proof; parse-only/build-only/capability-probe-only evidence cannot promote real-model-runtime or package-consumer-runtime proof.");
    }

    private static string[] BuildParsedOptions(OnnxEngineBuildResult result)
    {
        TrtexecLikeDeploymentOptions deploymentOptions = result.DeploymentOptions ?? TrtexecLikeDeploymentOptions.Default;
        TrtexecLikeRuntimeOptions runtimeOptions = result.RuntimeOptions ?? TrtexecLikeRuntimeOptions.Default;
        System.Collections.Generic.List<string> options = new System.Collections.Generic.List<string>
        {
            "--tensor-rt-line",
            "--workspace",
            "--iterations",
            "--warmUp",
            "--duration",
            "--streams"
        };

        AddIf(options, "--onnx", !string.IsNullOrWhiteSpace(result.ModelSource) && !string.Equals(result.ModelSource, "embedded-dynamic-identity", StringComparison.Ordinal));
        AddIf(options, "--saveEngine", !string.IsNullOrWhiteSpace(result.EnginePath));
        AddIf(options, "--loadEngine", result.State.Contains("load-engine", StringComparison.OrdinalIgnoreCase));
        AddIf(options, "--profilingVerbosity", result.NormalizedCommandLine.Contains("--profilingVerbosity", StringComparison.Ordinal));
        AddIf(options, "--dumpLayerInfo", result.NormalizedCommandLine.Contains("--dumpLayerInfo", StringComparison.Ordinal));
        AddIf(options, "--timingCacheFile", result.TimingCacheArtifact.InputRequested);
        AddIf(options, "--builderOptimizationLevel", true);
        AddIf(options, "--maxAuxStreams", deploymentOptions.MaxAuxStreams.HasValue);
        AddIf(options, "--maxNbTactics", deploymentOptions.MaxNbTactics.HasValue);
        AddIf(options, "--tilingOptimizationLevel", deploymentOptions.TilingOptimizationLevel.HasValue);
        AddIf(options, "--l2LimitForTiling", deploymentOptions.L2LimitForTilingBytes.HasValue);
        AddIf(options, "--quantizationFlags", deploymentOptions.QuantizationFlags.HasValue);
        AddIf(options, "--device", deploymentOptions.DeviceOrdinal.HasValue);
        AddIf(options, "--useDLACore", deploymentOptions.DlaCore.HasValue);
        AddIf(options, "--allowGPUFallback", deploymentOptions.AllowGpuFallback);
        AddIf(options, "--tacticSources", !string.IsNullOrWhiteSpace(deploymentOptions.TacticSources));
        AddIf(options, "--memPoolSize", deploymentOptions.MemoryPoolSizes.Count > 0);
        AddIf(options, "--inputIOFormats", !string.IsNullOrWhiteSpace(deploymentOptions.InputIOFormats));
        AddIf(options, "--outputIOFormats", !string.IsNullOrWhiteSpace(deploymentOptions.OutputIOFormats));
        AddIf(options, "--calib", !string.IsNullOrWhiteSpace(deploymentOptions.CalibrationCacheFile));
        AddIf(options, "--directIO", deploymentOptions.DirectIO);
        AddIf(options, "--sparsity", !string.IsNullOrWhiteSpace(deploymentOptions.Sparsity));
        AddIf(options, "--stronglyTyped", deploymentOptions.StronglyTyped);
        AddIf(options, "--minTiming", deploymentOptions.MinTiming.HasValue);
        AddIf(options, "--avgTiming", deploymentOptions.AvgTiming.HasValue);
        AddIf(options, "--precisionConstraints", !string.IsNullOrWhiteSpace(deploymentOptions.PrecisionConstraints));
        AddIf(options, "--layerPrecisions", !string.IsNullOrWhiteSpace(deploymentOptions.LayerPrecisions));
        AddIf(options, "--layerOutputTypes", !string.IsNullOrWhiteSpace(deploymentOptions.LayerOutputTypes));
        AddIf(options, "--fp8", deploymentOptions.Fp8);
        AddIf(options, "--best", deploymentOptions.Best);
        AddIf(options, "--dumpRefit", deploymentOptions.DumpRefit);
        AddIf(options, "--allowWeightStreaming", deploymentOptions.AllowWeightStreaming);
        AddIf(options, "--markDebug", !string.IsNullOrWhiteSpace(deploymentOptions.MarkDebug));
        AddIf(options, "--dumpDebugTensors", deploymentOptions.DumpDebugTensors);
        AddIf(options, "--versionCompatible", deploymentOptions.VersionCompatible);
        AddIf(options, "--excludeLeanRuntime", deploymentOptions.ExcludeLeanRuntime);
        AddIf(options, "--stripWeights", deploymentOptions.StripWeights);
        AddIf(options, "--refit", deploymentOptions.Refit);
        AddIf(options, "--refitFromOnnx", !string.IsNullOrWhiteSpace(deploymentOptions.RefitFromOnnxPath));
        AddIf(options, "--saveRefittedEngine", !string.IsNullOrWhiteSpace(deploymentOptions.SaveRefittedEnginePath));
        AddIf(options, "--weightStreamingBudget", deploymentOptions.WeightStreamingBudget.IsSpecified);
        AddIf(options, "--exportTimingCache", !string.IsNullOrWhiteSpace(deploymentOptions.ExportTimingCachePath));
        AddIf(options, "--safe", deploymentOptions.Safe);
        AddIf(options, "--consistency", deploymentOptions.Consistency);
        AddIf(options, "--builderCache", deploymentOptions.BuilderCache);
        AddIf(options, "--noBuilderCache", deploymentOptions.NoBuilderCache);
        AddIf(options, "--noDataTransfers", runtimeOptions.NoDataTransfers);
        AddIf(options, "--useSpinWait", runtimeOptions.UseSpinWait);
        AddIf(options, "--threads", runtimeOptions.Threads.HasValue);
        AddIf(options, "--avgRuns", runtimeOptions.AvgRuns.HasValue);
        AddIf(options, "--percentile", runtimeOptions.Percentile.HasValue);
        AddIf(options, "--sleepTime", runtimeOptions.SleepTimeMilliseconds.HasValue);
        AddIf(options, "--idleTime", runtimeOptions.IdleTimeMilliseconds.HasValue);
        AddIf(options, "--infStreams", runtimeOptions.InfStreams.HasValue);
        AddIf(options, "--loadInputs", !string.IsNullOrWhiteSpace(runtimeOptions.LoadInputs));
        AddIf(options, "--dumpOutput", runtimeOptions.DumpOutput);
        AddIf(options, "--dumpRawBindingsToFile", !string.IsNullOrWhiteSpace(runtimeOptions.DumpRawBindingsToFile));
        AddIf(options, "--exportOutput", !string.IsNullOrWhiteSpace(runtimeOptions.ExportOutputPath));
        AddIf(options, "--exportTimes", !string.IsNullOrWhiteSpace(runtimeOptions.ExportTimesPath));
        AddIf(options, "--dumpProfile", result.NormalizedCommandLine.Contains("--dumpProfile", StringComparison.Ordinal));
        AddIf(options, "--separateProfileRun", result.NormalizedCommandLine.Contains("--separateProfileRun", StringComparison.Ordinal));
        AddIf(options, "--exportProfile", !string.IsNullOrWhiteSpace(runtimeOptions.ExportProfilePath));
        AddIf(options, "--saveProfile", !string.IsNullOrWhiteSpace(runtimeOptions.SaveProfilePath));
        AddIf(options, "--useCudaGraph", result.NormalizedCommandLine.Contains("--useCudaGraph", StringComparison.Ordinal));
        AddIf(options, "--fp16", HasNormalizedOption(result, "--fp16"));
        AddIf(options, "--int8", HasNormalizedOption(result, "--int8"));
        AddIf(options, "--bf16", HasNormalizedOption(result, "--bf16"));
        AddIf(options, "--noTF32", HasNormalizedOption(result, "--noTF32"));

        return options.Distinct(StringComparer.Ordinal).ToArray();
    }

    private static string[] BuildAppliedOptions(OnnxEngineBuildResult result)
    {
        TrtexecLikeDeploymentOptions deploymentOptions = result.DeploymentOptions ?? TrtexecLikeDeploymentOptions.Default;
        TrtexecLikeRuntimeOptions runtimeOptions = result.RuntimeOptions ?? TrtexecLikeRuntimeOptions.Default;
        System.Collections.Generic.List<string> options = new System.Collections.Generic.List<string>
        {
            "--tensor-rt-line",
            "--workspace",
            "--builderOptimizationLevel"
        };

        AddIf(options, "--loadEngine", result.LoadedEngineDiagnostics.Attempted);
        AddIf(options, "--maxAuxStreams", deploymentOptions.MaxAuxStreams.HasValue);
        AddIf(options, "--maxNbTactics", HasAppliedBuilderScalar(result, "MaxNbTactics"));
        AddIf(options, "--tilingOptimizationLevel", HasAppliedBuilderScalar(result, "TilingOptimizationLevel"));
        AddIf(options, "--l2LimitForTiling", HasAppliedBuilderScalar(result, "L2LimitForTiling"));
        AddIf(options, "--quantizationFlags", HasAppliedBuilderScalar(result, "QuantizationFlags"));
        AddIf(options, "--device", HasAppliedDeploymentControl(result, "Device"));
        AddIf(options, "--useDLACore", HasAppliedDeploymentControl(result, "DlaCore"));
        AddIf(options, "--allowGPUFallback", HasAppliedDeploymentControl(result, "GpuFallback"));
        AddIf(options, "--tacticSources", HasAppliedDeploymentControl(result, "TacticSources"));
        AddIf(options, "--directIO", HasAppliedDeploymentControl(result, "DirectIO"));
        AddIf(options, "--sparsity", HasAppliedDeploymentControl(result, "Sparsity"));
        AddIf(options, "--stronglyTyped", HasAppliedDeploymentControl(result, "StronglyTyped"));
        AddIf(options, "--versionCompatible", HasAppliedDeploymentControl(result, "VersionCompatible"));
        AddIf(options, "--excludeLeanRuntime", HasAppliedDeploymentControl(result, "ExcludeLeanRuntime"));
        AddIf(options, "--stripWeights", HasAppliedDeploymentControl(result, "StripWeights"));
        AddIf(options, "--refit", HasAppliedDeploymentControl(result, "Refit"));
        AddIf(options, "--refitFromOnnx", result.RefitSnapshot.Succeeded && result.RefitSnapshot.ContextCreationAllowed);
        AddIf(options, "--saveRefittedEngine", result.RefitPersistenceSnapshot.Succeeded && result.RefitPersistenceSnapshot.ReloadContextCreationAllowed);
        AddIf(options, "--allowWeightStreaming", HasAppliedDeploymentControl(result, "WeightStreaming"));
        AddIf(options, "--weightStreamingBudget", HasAppliedDeploymentControl(result, "WeightStreamingBudget"));
        AddIf(options, "--inputIOFormats", HasAppliedBuildPolicy(result, "InputIOFormats"));
        AddIf(options, "--outputIOFormats", HasAppliedBuildPolicy(result, "OutputIOFormats"));
        AddIf(options, "--precisionConstraints", HasAppliedBuildPolicy(result, "PrecisionConstraints"));
        AddIf(options, "--layerPrecisions", HasAppliedBuildPolicy(result, "LayerPrecisions"));
        AddIf(options, "--layerOutputTypes", HasAppliedBuildPolicy(result, "LayerOutputTypes"));
        AddIf(options, "--fp16", HasAppliedBuildPolicy(result, "Fp16"));
        AddIf(options, "--int8", HasAppliedBuildPolicy(result, "Int8"));
        AddIf(options, "--bf16", HasAppliedBuildPolicy(result, "Bf16"));
        AddIf(options, "--noTF32", HasNormalizedOption(result, "--noTF32") && HasAppliedBuildPolicy(result, "Tf32"));
        AddIf(options, "--memPoolSize", deploymentOptions.MemoryPoolSizes.Count > 0 && (result.Parsed || result.EngineSaved));
        AddIf(options, "--minShapes/--optShapes/--maxShapes", result.Parsed || result.EngineSaved || result.InferenceRan);
        AddIf(options, "--avgTiming", deploymentOptions.AvgTiming.HasValue && (result.Parsed || result.EngineSaved));
        AddIf(options, "--minTiming", deploymentOptions.MinTiming.HasValue && result.TensorRtLine == TensorRtApiLine.TensorRt8 && (result.Parsed || result.EngineSaved));
        bool layerInfoRequested = result.NormalizedCommandLine.Contains("--dumpLayerInfo", StringComparison.Ordinal) ||
            result.NormalizedCommandLine.Contains("--exportLayerInfo", StringComparison.Ordinal);
        bool layerInfoCollected = result.LogLines.Any(static line => line.StartsWith("LayerInfo Collected=True", StringComparison.Ordinal));
        bool layerInfoExported = result.LogLines.Any(static line => line.StartsWith("LayerInfo ExportRequested=True Written=True", StringComparison.Ordinal));
        AddIf(options, "--dumpLayerInfo", layerInfoRequested && layerInfoCollected);
        AddIf(options, "--exportLayerInfo", layerInfoRequested && layerInfoExported);
        AddIf(options, "--saveEngine", result.EngineSaved);
        AddIf(options, "--timingCacheFile", result.TimingCacheArtifact.InputApplied);
        AddIf(options, "--exportTimingCache", result.TimingCacheArtifact.OutputWritten);
        AddIf(options, "--exportTimes", !string.IsNullOrWhiteSpace(runtimeOptions.ExportTimesPath));
        AddIf(options, "--exportProfile", !string.IsNullOrWhiteSpace(runtimeOptions.ExportProfilePath));
        bool outputReadbackAvailable = result.InferenceRan && !result.BenchmarkSummary.NoDataTransfersApplied;
        AddIf(options, "--dumpOutput", runtimeOptions.DumpOutput && outputReadbackAvailable);
        AddIf(options, "--exportOutput", !string.IsNullOrWhiteSpace(runtimeOptions.ExportOutputPath) && outputReadbackAvailable);
        AddIf(options, "--dumpRawBindingsToFile", !string.IsNullOrWhiteSpace(runtimeOptions.DumpRawBindingsToFile) && outputReadbackAvailable && result.OutputMatch);
        bool benchmarkExecuted = result.BenchmarkSummary.TimingSampleCount > 0;
        AddIf(options, "--iterations", benchmarkExecuted);
        AddIf(options, "--warmUp", benchmarkExecuted);
        AddIf(options, "--duration", benchmarkExecuted);
        AddIf(options, "--streams", benchmarkExecuted && !runtimeOptions.InfStreams.HasValue);
        AddIf(options, "--infStreams", benchmarkExecuted && runtimeOptions.InfStreams.HasValue && result.BenchmarkSummary.ExecutionContextsCreated == runtimeOptions.InfStreams.Value);
        AddIf(options, "--idleTime", benchmarkExecuted && runtimeOptions.IdleTimeMilliseconds.HasValue);
        AddIf(options, "--avgRuns", benchmarkExecuted && runtimeOptions.AvgRuns.HasValue && result.BenchmarkSummary.AveragedTimingSampleCount > 0);
        AddIf(options, "--percentile", benchmarkExecuted && runtimeOptions.Percentile.HasValue);
        AddIf(options, "--threads", benchmarkExecuted && runtimeOptions.UseThreads && result.BenchmarkSummary.ThreadsExecuted == result.BenchmarkSummary.ExecutionContextsCreated);
        AddIf(options, "--useSpinWait", benchmarkExecuted && runtimeOptions.UseSpinWait && result.BenchmarkSummary.UseSpinWaitApplied);
        AddIf(options, "--noDataTransfers", benchmarkExecuted && runtimeOptions.NoDataTransfers && result.BenchmarkSummary.NoDataTransfersApplied);
        AddIf(options, "--useCudaGraph", benchmarkExecuted && result.BenchmarkSummary.UseCudaGraphRequested && result.BenchmarkSummary.UseCudaGraphApplied);

        return options.Distinct(StringComparer.Ordinal).ToArray();
    }

    private static string[] BuildParseOnlyOptions(OnnxEngineBuildResult result)
    {
        TrtexecLikeDeploymentOptions deploymentOptions = result.DeploymentOptions ?? TrtexecLikeDeploymentOptions.Default;
        TrtexecLikeRuntimeOptions runtimeOptions = result.RuntimeOptions ?? TrtexecLikeRuntimeOptions.Default;
        System.Collections.Generic.List<string> options = new System.Collections.Generic.List<string>();

        AddIf(options, "--device", deploymentOptions.DeviceOrdinal.HasValue && !HasAppliedDeploymentControl(result, "Device"));
        AddIf(options, "--useDLACore", deploymentOptions.DlaCore.HasValue && !HasAppliedDeploymentControl(result, "DlaCore"));
        AddIf(options, "--allowGPUFallback", deploymentOptions.AllowGpuFallback && !HasAppliedDeploymentControl(result, "GpuFallback"));
        AddIf(options, "--tacticSources", !string.IsNullOrWhiteSpace(deploymentOptions.TacticSources) && !HasAppliedDeploymentControl(result, "TacticSources"));
        AddIf(options, "--memPoolSize", deploymentOptions.MemoryPoolSizes.Count > 0 && !(result.Parsed || result.EngineSaved));
        AddIf(options, "--inputIOFormats", !string.IsNullOrWhiteSpace(deploymentOptions.InputIOFormats) && !HasAppliedBuildPolicy(result, "InputIOFormats"));
        AddIf(options, "--outputIOFormats", !string.IsNullOrWhiteSpace(deploymentOptions.OutputIOFormats) && !HasAppliedBuildPolicy(result, "OutputIOFormats"));
        AddIf(options, "--calib", !string.IsNullOrWhiteSpace(deploymentOptions.CalibrationCacheFile));
        AddIf(options, "--directIO", deploymentOptions.DirectIO && !HasAppliedDeploymentControl(result, "DirectIO"));
        AddIf(options, "--sparsity", !string.IsNullOrWhiteSpace(deploymentOptions.Sparsity) && !HasAppliedDeploymentControl(result, "Sparsity"));
        AddIf(options, "--stronglyTyped", deploymentOptions.StronglyTyped && !HasAppliedDeploymentControl(result, "StronglyTyped"));
        AddIf(options, "--minTiming", deploymentOptions.MinTiming.HasValue && (result.TensorRtLine != TensorRtApiLine.TensorRt8 || !(result.Parsed || result.EngineSaved)));
        AddIf(options, "--avgTiming", deploymentOptions.AvgTiming.HasValue && !(result.Parsed || result.EngineSaved));
        AddIf(options, "--precisionConstraints", !string.IsNullOrWhiteSpace(deploymentOptions.PrecisionConstraints) && !HasAppliedBuildPolicy(result, "PrecisionConstraints"));
        AddIf(options, "--layerPrecisions", !string.IsNullOrWhiteSpace(deploymentOptions.LayerPrecisions) && !HasAppliedBuildPolicy(result, "LayerPrecisions"));
        AddIf(options, "--layerOutputTypes", !string.IsNullOrWhiteSpace(deploymentOptions.LayerOutputTypes) && !HasAppliedBuildPolicy(result, "LayerOutputTypes"));
        AddIf(options, "--fp16", HasNormalizedOption(result, "--fp16") && !HasAppliedBuildPolicy(result, "Fp16"));
        AddIf(options, "--int8", HasNormalizedOption(result, "--int8") && !HasAppliedBuildPolicy(result, "Int8"));
        AddIf(options, "--bf16", HasNormalizedOption(result, "--bf16") && !HasAppliedBuildPolicy(result, "Bf16"));
        AddIf(options, "--noTF32", HasNormalizedOption(result, "--noTF32") && !HasAppliedBuildPolicy(result, "Tf32"));
        AddIf(options, "--fp8", deploymentOptions.Fp8);
        AddIf(options, "--best", deploymentOptions.Best);
        AddIf(options, "--dumpRefit", deploymentOptions.DumpRefit);
        AddIf(options, "--allowWeightStreaming", deploymentOptions.AllowWeightStreaming && !HasAppliedDeploymentControl(result, "WeightStreaming"));
        AddIf(options, "--markDebug", !string.IsNullOrWhiteSpace(deploymentOptions.MarkDebug));
        AddIf(options, "--dumpDebugTensors", deploymentOptions.DumpDebugTensors);
        AddIf(options, "--versionCompatible", deploymentOptions.VersionCompatible && !HasAppliedDeploymentControl(result, "VersionCompatible"));
        AddIf(options, "--excludeLeanRuntime", deploymentOptions.ExcludeLeanRuntime && !HasAppliedDeploymentControl(result, "ExcludeLeanRuntime"));
        AddIf(options, "--stripWeights", deploymentOptions.StripWeights && !HasAppliedDeploymentControl(result, "StripWeights"));
        AddIf(options, "--refit", deploymentOptions.Refit && !HasAppliedDeploymentControl(result, "Refit"));
        AddIf(options, "--refitFromOnnx", !string.IsNullOrWhiteSpace(deploymentOptions.RefitFromOnnxPath) && !result.RefitSnapshot.Succeeded);
        AddIf(options, "--saveRefittedEngine", !string.IsNullOrWhiteSpace(deploymentOptions.SaveRefittedEnginePath) && !result.RefitPersistenceSnapshot.Succeeded);
        AddIf(options, "--weightStreamingBudget", deploymentOptions.WeightStreamingBudget.IsSpecified && !HasAppliedDeploymentControl(result, "WeightStreamingBudget"));
        AddIf(options, "--timingCacheFile", result.TimingCacheArtifact.InputRequested && !result.TimingCacheArtifact.InputApplied);
        AddIf(options, "--exportTimingCache", !string.IsNullOrWhiteSpace(deploymentOptions.ExportTimingCachePath) && !result.TimingCacheArtifact.OutputWritten);
        AddIf(options, "--maxNbTactics", deploymentOptions.MaxNbTactics.HasValue && !HasAppliedBuilderScalar(result, "MaxNbTactics"));
        AddIf(options, "--tilingOptimizationLevel", deploymentOptions.TilingOptimizationLevel.HasValue && !HasAppliedBuilderScalar(result, "TilingOptimizationLevel"));
        AddIf(options, "--l2LimitForTiling", deploymentOptions.L2LimitForTilingBytes.HasValue && !HasAppliedBuilderScalar(result, "L2LimitForTiling"));
        AddIf(options, "--quantizationFlags", deploymentOptions.QuantizationFlags.HasValue && !HasAppliedBuilderScalar(result, "QuantizationFlags"));
        bool layerInfoRequested = result.NormalizedCommandLine.Contains("--dumpLayerInfo", StringComparison.Ordinal) ||
            result.NormalizedCommandLine.Contains("--exportLayerInfo", StringComparison.Ordinal);
        bool layerInfoCollected = result.LogLines.Any(static line => line.StartsWith("LayerInfo Collected=True", StringComparison.Ordinal));
        bool layerInfoExported = result.LogLines.Any(static line => line.StartsWith("LayerInfo ExportRequested=True Written=True", StringComparison.Ordinal));
        AddIf(options, "--dumpLayerInfo", layerInfoRequested && !layerInfoCollected);
        AddIf(options, "--exportLayerInfo", layerInfoRequested && !layerInfoExported);
        AddIf(options, "--safe", deploymentOptions.Safe);
        AddIf(options, "--consistency", deploymentOptions.Consistency);
        AddIf(options, "--builderCache", deploymentOptions.BuilderCache);
        AddIf(options, "--noBuilderCache", deploymentOptions.NoBuilderCache);
        bool benchmarkExecuted = result.BenchmarkSummary.TimingSampleCount > 0;
        AddIf(options, "--iterations", !benchmarkExecuted);
        AddIf(options, "--warmUp", !benchmarkExecuted);
        AddIf(options, "--duration", !benchmarkExecuted);
        AddIf(options, "--streams", !benchmarkExecuted || runtimeOptions.InfStreams.HasValue);
        AddIf(options, "--infStreams", runtimeOptions.InfStreams.HasValue && !benchmarkExecuted);
        AddIf(options, "--noDataTransfers", runtimeOptions.NoDataTransfers && !result.BenchmarkSummary.NoDataTransfersApplied);
        AddIf(options, "--useSpinWait", runtimeOptions.UseSpinWait && !result.BenchmarkSummary.UseSpinWaitApplied);
        AddIf(options, "--threads", runtimeOptions.UseThreads && (!benchmarkExecuted || result.BenchmarkSummary.ThreadsExecuted != result.BenchmarkSummary.ExecutionContextsCreated));
        AddIf(options, "--avgRuns", runtimeOptions.AvgRuns.HasValue && !benchmarkExecuted);
        AddIf(options, "--percentile", runtimeOptions.Percentile.HasValue && !benchmarkExecuted);
        AddIf(options, "--sleepTime", runtimeOptions.SleepTimeMilliseconds.HasValue);
        AddIf(options, "--idleTime", runtimeOptions.IdleTimeMilliseconds.HasValue && !benchmarkExecuted);
        AddIf(options, "--useCudaGraph", result.NormalizedCommandLine.Contains("--useCudaGraph", StringComparison.Ordinal) && !result.BenchmarkSummary.UseCudaGraphApplied);
        AddIf(options, "--loadInputs", !string.IsNullOrWhiteSpace(runtimeOptions.LoadInputs) && !result.InferenceRan);
        bool outputReadbackUnavailable = !result.InferenceRan || result.BenchmarkSummary.NoDataTransfersApplied;
        AddIf(options, "--dumpOutput", runtimeOptions.DumpOutput && outputReadbackUnavailable);
        AddIf(options, "--dumpRawBindingsToFile", !string.IsNullOrWhiteSpace(runtimeOptions.DumpRawBindingsToFile) && (outputReadbackUnavailable || !result.OutputMatch));
        AddIf(options, "--exportOutput", !string.IsNullOrWhiteSpace(runtimeOptions.ExportOutputPath) && outputReadbackUnavailable);
        AddIf(options, "--dumpLayerInfo", result.NormalizedCommandLine.Contains("--dumpLayerInfo", StringComparison.Ordinal));
        AddIf(options, "--dumpProfile", result.NormalizedCommandLine.Contains("--dumpProfile", StringComparison.Ordinal));
        AddIf(options, "--separateProfileRun", result.NormalizedCommandLine.Contains("--separateProfileRun", StringComparison.Ordinal));
        AddIf(options, "capability-probe-only", result.CapabilityProbe.Attempted);

        return options.Distinct(StringComparer.Ordinal).ToArray();
    }

    private static bool HasAppliedBuilderScalar(OnnxEngineBuildResult result, string name)
    {
        string prefix = $"TrtexecBuilderScalar Name={name} Applied=True";
        return result.LogLines.Any(line =>
            line.StartsWith(prefix, StringComparison.Ordinal) &&
            line.Contains("ReadbackMatch=True", StringComparison.Ordinal));
    }

    private static bool HasAppliedDeploymentControl(OnnxEngineBuildResult result, string name)
    {
        string prefix = $"TrtexecDeploymentControl Name={name} Applied=True";
        return result.LogLines.Any(line =>
            line.StartsWith(prefix, StringComparison.Ordinal) &&
            line.Contains("ReadbackMatch=True", StringComparison.Ordinal));
    }

    private static bool HasAppliedBuildPolicy(OnnxEngineBuildResult result, string name)
    {
        string prefix = $"TrtexecBuildPolicy Name={name} Applied=True";
        return result.LogLines.Any(line =>
            line.StartsWith(prefix, StringComparison.Ordinal) &&
            line.Contains("ReadbackMatch=True", StringComparison.Ordinal));
    }

    private static bool HasNormalizedOption(OnnxEngineBuildResult result, string option)
    {
        string commandLine = result.NormalizedCommandLine;
        return string.Equals(commandLine, option, StringComparison.Ordinal) ||
            commandLine.StartsWith(option + " ", StringComparison.Ordinal) ||
            commandLine.Contains(" " + option + " ", StringComparison.Ordinal) ||
            commandLine.EndsWith(" " + option, StringComparison.Ordinal);
    }

    private static void AddIf(System.Collections.Generic.List<string> options, string option, bool condition)
    {
        if (condition)
        {
            options.Add(option);
        }
    }

    private static string FormatOptionList(System.Collections.Generic.IReadOnlyList<string> options)
    {
        return options.Count == 0
            ? "- none"
            : string.Join(Environment.NewLine, options.Select(static option => "- `" + option + "`"));
    }

    private static OnnxEngineBuildReportBoundary CreateReportBoundary(OnnxEngineBuildResult result)
    {
        return new OnnxEngineBuildReportBoundary(
            isRuntimeProof: false,
            isBuildOnly: result.BuildEvidenceOnly,
            forbiddenSubstituteReason: "TensorRtExec reports are diagnostic/build artifacts. They do not replace real-model-runtime, package-consumer-runtime, post-publish verification, or owner release-close evidence.",
            copiedDiagnosticsBoundary: "ONNX Parser and ParserRefitter copied diagnostics are troubleshooting and release-gate surface evidence only. They do not prove engine execution, model correctness, package-consumer-runtime, post-publish verification, or release close readiness.",
            parserDiagnosticsEvidenceKind: "copied-parser-diagnostics",
            parserRefitterDiagnosticsEvidenceKind: "copied-parser-refitter-diagnostics",
            canPromoteCopiedDiagnosticsToRuntimeProof: false,
            parserDiagnosticsOwnerAction: "Use copied parser/refitter diagnostics to repair ONNX export, shape/profile, or plugin plans; then collect real-model-runtime evidence with real inputs, output JSON, logs, hashes, host metadata, and owner review.",
            forbiddenSubstitutes: new[]
            {
                "build-only",
                "dry-run",
                "template",
                "local feed",
                "ProjectReference",
                "direct `.nupkg`",
                "TensorRtExec report",
                "YoloVision matrix",
                "OnnxToEngine report",
                "readonly diagnostics",
                "capability-probe-only",
                "ONNX Parser diagnostic snapshot",
                "ONNX ParserRefitter diagnostic snapshot",
                "copied-parser-diagnostics",
                "copied-parser-refitter-diagnostics"
            });
    }
}

public sealed class OnnxEngineBuildOptionImplementationStatus
{
    public OnnxEngineBuildOptionImplementationStatus(
        string[] parsedOptions,
        string[] appliedOptions,
        string[] parseOnlyOptions,
        string evidenceBoundary)
    {
        ParsedOptions = parsedOptions ?? Array.Empty<string>();
        AppliedOptions = appliedOptions ?? Array.Empty<string>();
        ParseOnlyOptions = parseOnlyOptions ?? Array.Empty<string>();
        EvidenceBoundary = evidenceBoundary ?? string.Empty;
    }

    public string[] ParsedOptions { get; }

    public string[] AppliedOptions { get; }

    public string[] ParseOnlyOptions { get; }

    public string EvidenceBoundary { get; }
}

public sealed class OnnxEngineBuildReportBoundary
{
    public OnnxEngineBuildReportBoundary(
        bool isRuntimeProof,
        bool isBuildOnly,
        string forbiddenSubstituteReason,
        string copiedDiagnosticsBoundary,
        string parserDiagnosticsEvidenceKind,
        string parserRefitterDiagnosticsEvidenceKind,
        bool canPromoteCopiedDiagnosticsToRuntimeProof,
        string parserDiagnosticsOwnerAction,
        IReadOnlyList<string> forbiddenSubstitutes)
    {
        IsRuntimeProof = isRuntimeProof;
        IsBuildOnly = isBuildOnly;
        ForbiddenSubstituteReason = forbiddenSubstituteReason ?? string.Empty;
        CopiedDiagnosticsBoundary = copiedDiagnosticsBoundary ?? string.Empty;
        ParserDiagnosticsEvidenceKind = parserDiagnosticsEvidenceKind ?? string.Empty;
        ParserRefitterDiagnosticsEvidenceKind = parserRefitterDiagnosticsEvidenceKind ?? string.Empty;
        CanPromoteCopiedDiagnosticsToRuntimeProof = canPromoteCopiedDiagnosticsToRuntimeProof;
        ParserDiagnosticsOwnerAction = parserDiagnosticsOwnerAction ?? string.Empty;
        ForbiddenSubstitutes = forbiddenSubstitutes ?? Array.Empty<string>();
    }

    public bool IsRuntimeProof { get; }

    public bool IsBuildOnly { get; }

    public string ForbiddenSubstituteReason { get; }

    public string CopiedDiagnosticsBoundary { get; }

    public string ParserDiagnosticsEvidenceKind { get; }

    public string ParserRefitterDiagnosticsEvidenceKind { get; }

    public bool CanPromoteCopiedDiagnosticsToRuntimeProof { get; }

    public string ParserDiagnosticsOwnerAction { get; }

    public IReadOnlyList<string> ForbiddenSubstitutes { get; }
}
