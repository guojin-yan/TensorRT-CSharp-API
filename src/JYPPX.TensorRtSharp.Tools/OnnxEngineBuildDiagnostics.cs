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
            $"Benchmark threads requested: `{result.BenchmarkSummary.ThreadsRequested?.ToString() ?? ""}`",
            $"Benchmark threads executed: `{result.BenchmarkSummary.ThreadsExecuted}`",
            $"Benchmark no data transfers requested: `{result.BenchmarkSummary.NoDataTransfersRequested}`",
            $"Benchmark no data transfers applied: `{result.BenchmarkSummary.NoDataTransfersApplied}`",
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
            "build reports distinguish parsed, applied, parse-only, and capability-probe-only evidence; parse-only/build-only/capability-probe-only evidence cannot promote real-model-runtime or package-consumer-runtime proof; load-engine readonly diagnostics are metadata-only and cannot promote runtime proof.");
    }

    private static string[] BuildParsedOptions(OnnxEngineBuildResult result)
    {
        TrtexecLikeDeploymentOptions deploymentOptions = result.DeploymentOptions ?? TrtexecLikeDeploymentOptions.Default;
        TrtexecLikeRuntimeOptions runtimeOptions = result.RuntimeOptions ?? TrtexecLikeRuntimeOptions.Default;
        System.Collections.Generic.List<string> options = new System.Collections.Generic.List<string>
        {
            "--tensor-rt-line",
            "--workspace"
        };

        AddIf(options, "--onnx", !string.IsNullOrWhiteSpace(result.ModelSource) && !string.Equals(result.ModelSource, "embedded-dynamic-identity", StringComparison.Ordinal));
        AddIf(options, "--saveEngine", !string.IsNullOrWhiteSpace(result.EnginePath));
        AddIf(options, "--loadEngine", result.State.Contains("load-engine", StringComparison.OrdinalIgnoreCase));
        AddIf(options, "--profilingVerbosity", result.NormalizedCommandLine.Contains("--profilingVerbosity", StringComparison.Ordinal));
        AddIf(options, "--dumpLayerInfo", result.NormalizedCommandLine.Contains("--dumpLayerInfo", StringComparison.Ordinal));
        AddIf(options, "--timingCacheFile", result.TimingCacheArtifact.InputRequested);
        AddIf(options, "--builderOptimizationLevel", true);
        AddIf(options, "--maxAuxStreams", deploymentOptions.MaxAuxStreams.HasValue);
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
        AddIf(options, "--weightStreamingBudget", deploymentOptions.WeightStreamingBudgetBytes.HasValue);
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
        AddIf(options, "--memPoolSize", deploymentOptions.MemoryPoolSizes.Count > 0 && (result.Parsed || result.EngineSaved));
        AddIf(options, "--fp16/--bf16/--noTF32", result.Parsed || result.EngineSaved || result.InferenceRan);
        AddIf(options, "--minShapes/--optShapes/--maxShapes", result.Parsed || result.EngineSaved || result.InferenceRan);
        AddIf(options, "--avgTiming", deploymentOptions.AvgTiming.HasValue && (result.Parsed || result.EngineSaved));
        AddIf(options, "--minTiming", deploymentOptions.MinTiming.HasValue && result.TensorRtLine == TensorRtApiLine.TensorRt8 && (result.Parsed || result.EngineSaved));
        AddIf(options, "--saveEngine", result.EngineSaved);
        AddIf(options, "--timingCacheFile", result.TimingCacheArtifact.InputApplied);
        AddIf(options, "--exportTimingCache", result.TimingCacheArtifact.OutputWritten);
        AddIf(options, "--exportTimes", !string.IsNullOrWhiteSpace(runtimeOptions.ExportTimesPath));
        AddIf(options, "--exportProfile", !string.IsNullOrWhiteSpace(runtimeOptions.ExportProfilePath));
        AddIf(options, "--exportOutput", !string.IsNullOrWhiteSpace(runtimeOptions.ExportOutputPath) && result.InferenceRan);
        AddIf(options, "--dumpRawBindingsToFile", !string.IsNullOrWhiteSpace(runtimeOptions.DumpRawBindingsToFile) && result.InferenceRan);

        return options.Distinct(StringComparer.Ordinal).ToArray();
    }

    private static string[] BuildParseOnlyOptions(OnnxEngineBuildResult result)
    {
        TrtexecLikeDeploymentOptions deploymentOptions = result.DeploymentOptions ?? TrtexecLikeDeploymentOptions.Default;
        TrtexecLikeRuntimeOptions runtimeOptions = result.RuntimeOptions ?? TrtexecLikeRuntimeOptions.Default;
        System.Collections.Generic.List<string> options = new System.Collections.Generic.List<string>();

        AddIf(options, "--device", deploymentOptions.DeviceOrdinal.HasValue);
        AddIf(options, "--useDLACore", deploymentOptions.DlaCore.HasValue);
        AddIf(options, "--allowGPUFallback", deploymentOptions.AllowGpuFallback);
        AddIf(options, "--tacticSources", !string.IsNullOrWhiteSpace(deploymentOptions.TacticSources));
        AddIf(options, "--memPoolSize", deploymentOptions.MemoryPoolSizes.Count > 0 && !(result.Parsed || result.EngineSaved));
        AddIf(options, "--inputIOFormats", !string.IsNullOrWhiteSpace(deploymentOptions.InputIOFormats));
        AddIf(options, "--outputIOFormats", !string.IsNullOrWhiteSpace(deploymentOptions.OutputIOFormats));
        AddIf(options, "--calib", !string.IsNullOrWhiteSpace(deploymentOptions.CalibrationCacheFile));
        AddIf(options, "--directIO", deploymentOptions.DirectIO);
        AddIf(options, "--sparsity", !string.IsNullOrWhiteSpace(deploymentOptions.Sparsity));
        AddIf(options, "--stronglyTyped", deploymentOptions.StronglyTyped);
        AddIf(options, "--minTiming", deploymentOptions.MinTiming.HasValue && (result.TensorRtLine != TensorRtApiLine.TensorRt8 || !(result.Parsed || result.EngineSaved)));
        AddIf(options, "--avgTiming", deploymentOptions.AvgTiming.HasValue && !(result.Parsed || result.EngineSaved));
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
        AddIf(options, "--weightStreamingBudget", deploymentOptions.WeightStreamingBudgetBytes.HasValue);
        AddIf(options, "--timingCacheFile", result.TimingCacheArtifact.InputRequested && !result.TimingCacheArtifact.InputApplied);
        AddIf(options, "--exportTimingCache", !string.IsNullOrWhiteSpace(deploymentOptions.ExportTimingCachePath) && !result.TimingCacheArtifact.OutputWritten);
        AddIf(options, "--safe", deploymentOptions.Safe);
        AddIf(options, "--consistency", deploymentOptions.Consistency);
        AddIf(options, "--builderCache", deploymentOptions.BuilderCache);
        AddIf(options, "--noBuilderCache", deploymentOptions.NoBuilderCache);
        AddIf(options, "--infStreams", runtimeOptions.InfStreams.HasValue);
        AddIf(options, "--noDataTransfers", runtimeOptions.NoDataTransfers);
        AddIf(options, "--useSpinWait", runtimeOptions.UseSpinWait);
        AddIf(options, "--threads", runtimeOptions.Threads.HasValue && !result.InferenceRan);
        AddIf(options, "--loadInputs", !string.IsNullOrWhiteSpace(runtimeOptions.LoadInputs) && !result.InferenceRan);
        AddIf(options, "--dumpOutput", runtimeOptions.DumpOutput && !result.InferenceRan);
        AddIf(options, "--dumpRawBindingsToFile", !string.IsNullOrWhiteSpace(runtimeOptions.DumpRawBindingsToFile) && !result.InferenceRan);
        AddIf(options, "--exportOutput", !string.IsNullOrWhiteSpace(runtimeOptions.ExportOutputPath) && !result.InferenceRan);
        AddIf(options, "--dumpLayerInfo", result.NormalizedCommandLine.Contains("--dumpLayerInfo", StringComparison.Ordinal));
        AddIf(options, "--dumpProfile", result.NormalizedCommandLine.Contains("--dumpProfile", StringComparison.Ordinal));
        AddIf(options, "--separateProfileRun", result.NormalizedCommandLine.Contains("--separateProfileRun", StringComparison.Ordinal));
        AddIf(options, "capability-probe-only", result.CapabilityProbe.Attempted);

        return options.Distinct(StringComparer.Ordinal).ToArray();
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
