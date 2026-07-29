using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Tools;

public static partial class OnnxEngineBuildDiagnostics
{
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
        AddIf(options, "--referenceOutputs", runtimeOptions.RequestsReferenceValidation);
        AddIf(options, "--referenceAbsTolerance", runtimeOptions.RequestsReferenceValidation);
        AddIf(options, "--referenceRelTolerance", runtimeOptions.RequestsReferenceValidation);
        AddIf(options, "--referenceNaNPolicy", runtimeOptions.RequestsReferenceValidation);
        AddIf(options, "--referenceInfinityPolicy", runtimeOptions.RequestsReferenceValidation);
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
        AddIf(options, "--loadInputs", !string.IsNullOrWhiteSpace(runtimeOptions.LoadInputs) && result.InferenceRan);
        AddIf(options, "--dumpOutput", runtimeOptions.DumpOutput && outputReadbackAvailable);
        AddIf(options, "--exportOutput", !string.IsNullOrWhiteSpace(runtimeOptions.ExportOutputPath) && outputReadbackAvailable);
        AddIf(options, "--dumpRawBindingsToFile", !string.IsNullOrWhiteSpace(runtimeOptions.DumpRawBindingsToFile) && outputReadbackAvailable);
        bool referenceValidationAttempted = result.LogLines.Any(static line => line.StartsWith("ReferenceOutputValidation Requested=True", StringComparison.Ordinal));
        AddIf(options, "--referenceOutputs", runtimeOptions.RequestsReferenceValidation && referenceValidationAttempted);
        AddIf(options, "--referenceAbsTolerance", runtimeOptions.RequestsReferenceValidation && referenceValidationAttempted);
        AddIf(options, "--referenceRelTolerance", runtimeOptions.RequestsReferenceValidation && referenceValidationAttempted);
        AddIf(options, "--referenceNaNPolicy", runtimeOptions.RequestsReferenceValidation && referenceValidationAttempted);
        AddIf(options, "--referenceInfinityPolicy", runtimeOptions.RequestsReferenceValidation && referenceValidationAttempted);
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
        AddIf(options, "--dumpRawBindingsToFile", !string.IsNullOrWhiteSpace(runtimeOptions.DumpRawBindingsToFile) && outputReadbackUnavailable);
        AddIf(options, "--exportOutput", !string.IsNullOrWhiteSpace(runtimeOptions.ExportOutputPath) && outputReadbackUnavailable);
        bool referenceValidationAttempted = result.LogLines.Any(static line => line.StartsWith("ReferenceOutputValidation Requested=True", StringComparison.Ordinal));
        AddIf(options, "--referenceOutputs", runtimeOptions.RequestsReferenceValidation && !referenceValidationAttempted);
        AddIf(options, "--referenceAbsTolerance", runtimeOptions.RequestsReferenceValidation && !referenceValidationAttempted);
        AddIf(options, "--referenceRelTolerance", runtimeOptions.RequestsReferenceValidation && !referenceValidationAttempted);
        AddIf(options, "--referenceNaNPolicy", runtimeOptions.RequestsReferenceValidation && !referenceValidationAttempted);
        AddIf(options, "--referenceInfinityPolicy", runtimeOptions.RequestsReferenceValidation && !referenceValidationAttempted);
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

}
