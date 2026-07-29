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
    private static object CreateTimesArtifact(OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data)
    {
        RuntimeArtifactProofBoundary proofBoundary = CreateProofBoundary(result, data, "timing");
        return new
        {
            ArtifactKind = "trtexec-like-times",
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
            TimingSamplesMilliseconds = result.BenchmarkSummary.TimingSamplesMilliseconds,
            TimingSampleCount = result.BenchmarkSummary.TimingSampleCount,
            AveragedTimingSamplesMilliseconds = result.BenchmarkSummary.AveragedTimingSamplesMilliseconds,
            AveragedTimingSampleCount = result.BenchmarkSummary.AveragedTimingSampleCount,
            AverageElapsedMilliseconds = result.BenchmarkSummary.AverageElapsedMilliseconds,
            MinElapsedMilliseconds = result.BenchmarkSummary.MinElapsedMilliseconds,
            MaxElapsedMilliseconds = result.BenchmarkSummary.MaxElapsedMilliseconds,
            PercentileRequested = result.BenchmarkSummary.PercentileRequested,
            PercentileElapsedMilliseconds = result.BenchmarkSummary.PercentileElapsedMilliseconds,
            AvgRunsRequested = result.BenchmarkSummary.AvgRunsRequested,
            AvgRunsExecuted = result.BenchmarkSummary.AvgRunsExecuted,
            ThreadsRequested = result.BenchmarkSummary.ThreadsRequested,
            ThreadsExecuted = result.BenchmarkSummary.ThreadsExecuted,
            NoDataTransfersRequested = result.BenchmarkSummary.NoDataTransfersRequested,
            NoDataTransfersApplied = result.BenchmarkSummary.NoDataTransfersApplied,
            UseSpinWaitRequested = result.BenchmarkSummary.UseSpinWaitRequested,
            UseSpinWaitApplied = result.BenchmarkSummary.UseSpinWaitApplied,
            UseCudaGraphRequested = result.BenchmarkSummary.UseCudaGraphRequested,
            UseCudaGraphApplied = result.BenchmarkSummary.UseCudaGraphApplied,
            UseCudaGraphFallbackReason = result.BenchmarkSummary.UseCudaGraphFallbackReason,
            SleepTimeMillisecondsRequested = result.BenchmarkSummary.SleepTimeMillisecondsRequested,
            SleepTimeMillisecondsApplied = result.BenchmarkSummary.SleepTimeMillisecondsApplied,
            IdleTimeMillisecondsRequested = result.BenchmarkSummary.IdleTimeMillisecondsRequested,
            IdleTimeMillisecondsApplied = result.BenchmarkSummary.IdleTimeMillisecondsApplied,
            IterationsRequested = result.BenchmarkSummary.IterationsRequested,
            MeasurementRoundsExecuted = result.BenchmarkSummary.MeasurementRoundsExecuted,
            MeasurementRoundsPerContext = result.BenchmarkSummary.MeasurementRoundsPerContext,
            InferenceIterationsExecuted = result.BenchmarkSummary.InferenceIterationsExecuted,
            WarmUpMillisecondsRequested = result.BenchmarkSummary.WarmUpMillisecondsRequested,
            WarmUpElapsedMilliseconds = result.BenchmarkSummary.WarmUpElapsedMilliseconds,
            WarmUpIterationsExecuted = result.BenchmarkSummary.WarmUpIterationsExecuted,
            DurationSecondsRequested = result.BenchmarkSummary.DurationSecondsRequested,
            MeasurementElapsedMilliseconds = result.BenchmarkSummary.MeasurementElapsedMilliseconds,
            StreamsRequested = result.BenchmarkSummary.StreamsRequested,
            InfStreamsRequested = result.BenchmarkSummary.InfStreamsRequested,
            ExecutionContextsCreated = result.BenchmarkSummary.ExecutionContextsCreated,
            ConcurrentStreamsExecuted = result.BenchmarkSummary.ConcurrentStreamsExecuted,
            BenchmarkBoundary = result.BenchmarkSummary.BenchmarkBoundary,
            RuntimeOptions = result.RuntimeOptions,
            data.ExecutionSummary
        };
    }

}
