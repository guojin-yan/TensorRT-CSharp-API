using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace JYPPX.TensorRtSharp.Tools;

public static partial class OnnxEngineRuntimeArtifactWriter
{
    private static object CreateProfileArtifact(OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data)
    {
        RuntimeArtifactProofBoundary proofBoundary = CreateProofBoundary(result, data, "profile");
        return new
        {
            ArtifactKind = "trtexec-like-profile-boundary",
            ArtifactBoundary = proofBoundary.ArtifactProofBoundary,
            proofBoundary.ArtifactProofBoundary,
            proofBoundary.RuntimeProofClass,
            proofBoundary.HasTensorOutputProof,
            proofBoundary.HasRawBindingProof,
            proofBoundary.IsBuildOnlyEvidence,
            proofBoundary.IsDependencyProbeOnly,
            proofBoundary.IsSyntheticRuntime,
            proofBoundary.HasBenchmarkExecutionEvidence,
            proofBoundary.ModelSource,
            proofBoundary.EnginePath,
            proofBoundary.PreflightMetadata,
            result.State,
            result.ProofClassification,
            result.BuildEvidenceOnly,
            result.IsRuntimeExecutionProof,
            result.IsRealModelRuntimeProof,
            result.IsPackageConsumerRuntimeProof,
            result.NormalizedCommandSha256,
            result.ProfileIndex,
            result.ElapsedMilliseconds,
            TimingSampleCount = result.BenchmarkSummary.TimingSampleCount,
            AverageElapsedMilliseconds = result.BenchmarkSummary.AverageElapsedMilliseconds,
            LayerProfileAvailable = false,
            LayerProfileBoundary = "TensorRT per-layer profiler data is not collected by this generic sample path; no layer timings are fabricated.",
            data.ExecutionSummary
        };
    }

    private static string CreateProfileText(OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data)
    {
        RuntimeArtifactProofBoundary proofBoundary = CreateProofBoundary(result, data, "profile");
        return string.Join(Environment.NewLine, new[]
        {
            "TensorRtSharp trtexec-like profile boundary",
            $"ArtifactKind: trtexec-like-profile-boundary",
            $"ArtifactBoundary: {proofBoundary.ArtifactProofBoundary}",
            $"ArtifactProofBoundary: {proofBoundary.ArtifactProofBoundary}",
            $"RuntimeProofClass: {proofBoundary.RuntimeProofClass}",
            $"HasTensorOutputProof: {proofBoundary.HasTensorOutputProof}",
            $"HasRawBindingProof: {proofBoundary.HasRawBindingProof}",
            $"IsBuildOnlyEvidence: {proofBoundary.IsBuildOnlyEvidence}",
            $"IsDependencyProbeOnly: {proofBoundary.IsDependencyProbeOnly}",
            $"IsSyntheticRuntime: {proofBoundary.IsSyntheticRuntime}",
            $"HasBenchmarkExecutionEvidence: {proofBoundary.HasBenchmarkExecutionEvidence}",
            $"ModelSource: {proofBoundary.ModelSource}",
            $"EnginePath: {proofBoundary.EnginePath}",
            $"PreflightKind: {proofBoundary.PreflightMetadata.Kind}",
            $"PreflightPath: {proofBoundary.PreflightMetadata.Path}",
            $"PreflightEvidenceBoundary: {proofBoundary.PreflightMetadata.EvidenceBoundary}",
            $"State: {result.State}",
            $"ProofClassification: {result.ProofClassification}",
            $"BuildEvidenceOnly: {result.BuildEvidenceOnly}",
            $"IsRuntimeExecutionProof: {result.IsRuntimeExecutionProof}",
            $"IsRealModelRuntimeProof: {result.IsRealModelRuntimeProof}",
            $"IsPackageConsumerRuntimeProof: {result.IsPackageConsumerRuntimeProof}",
            $"NormalizedCommandSha256: {result.NormalizedCommandSha256}",
            $"ElapsedMilliseconds: {result.ElapsedMilliseconds?.ToString("0.###") ?? string.Empty}",
            $"TimingSampleCount: {result.BenchmarkSummary.TimingSampleCount}",
            $"AveragedTimingSampleCount: {result.BenchmarkSummary.AveragedTimingSampleCount}",
            $"AverageElapsedMilliseconds: {result.BenchmarkSummary.AverageElapsedMilliseconds?.ToString("0.###", CultureInfo.InvariantCulture) ?? string.Empty}",
            $"PercentileRequested: {result.BenchmarkSummary.PercentileRequested?.ToString(CultureInfo.InvariantCulture) ?? string.Empty}",
            $"PercentileElapsedMilliseconds: {result.BenchmarkSummary.PercentileElapsedMilliseconds?.ToString("0.###", CultureInfo.InvariantCulture) ?? string.Empty}",
            $"IterationsRequested: {result.BenchmarkSummary.IterationsRequested}",
            $"MeasurementRoundsExecuted: {result.BenchmarkSummary.MeasurementRoundsExecuted}",
            $"MeasurementRoundsPerContext: {string.Join(",", result.BenchmarkSummary.MeasurementRoundsPerContext)}",
            $"InferenceIterationsExecuted: {result.BenchmarkSummary.InferenceIterationsExecuted}",
            $"WarmUpMillisecondsRequested: {result.BenchmarkSummary.WarmUpMillisecondsRequested}",
            $"WarmUpElapsedMilliseconds: {result.BenchmarkSummary.WarmUpElapsedMilliseconds.ToString("0.###", CultureInfo.InvariantCulture)}",
            $"DurationSecondsRequested: {result.BenchmarkSummary.DurationSecondsRequested}",
            $"MeasurementElapsedMilliseconds: {result.BenchmarkSummary.MeasurementElapsedMilliseconds.ToString("0.###", CultureInfo.InvariantCulture)}",
            $"ExecutionContextsCreated: {result.BenchmarkSummary.ExecutionContextsCreated}",
            $"ThreadsExecuted: {result.BenchmarkSummary.ThreadsExecuted}",
            $"UseSpinWaitApplied: {result.BenchmarkSummary.UseSpinWaitApplied}",
            $"UseCudaGraphRequested: {result.BenchmarkSummary.UseCudaGraphRequested}",
            $"UseCudaGraphApplied: {result.BenchmarkSummary.UseCudaGraphApplied}",
            $"UseCudaGraphFallbackReason: {result.BenchmarkSummary.UseCudaGraphFallbackReason}",
            $"BenchmarkBoundary: {result.BenchmarkSummary.BenchmarkBoundary}",
            $"ExecutionSummary: {data.ExecutionSummary}",
            "LayerProfileAvailable: False",
            "LayerProfileBoundary: TensorRT per-layer profiler data is not collected by this generic sample path; no layer timings are fabricated."
        });
    }

}
