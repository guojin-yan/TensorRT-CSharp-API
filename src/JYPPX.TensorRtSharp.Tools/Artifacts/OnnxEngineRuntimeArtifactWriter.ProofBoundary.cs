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
    private static RuntimeArtifactProofBoundary CreateProofBoundary(OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data, string artifactKind)
    {
        return new RuntimeArtifactProofBoundary(
            Boundary(result, data, artifactKind),
            result.ProofClassification,
            result.InferenceRan && result.OutputMatch && data.HasOutput,
            CanCaptureRawBindings(result, data) && result.OutputMatch,
            result.BuildEvidenceOnly,
            string.Equals(result.ProofClassification, "dependency-probe-only", StringComparison.Ordinal),
            string.Equals(result.ProofClassification, "synthetic-input-runtime", StringComparison.Ordinal),
            result.InferenceRan && result.BenchmarkSummary.TimingSampleCount > 0,
            result.ModelSource,
            result.EnginePath,
            result.PreflightMetadata);
    }

    private static string Boundary(OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data, string artifactKind)
    {
        if (result.InferenceRan && result.OutputValidated && data.OutputValidated && data.HasOutput)
        {
            return "runtime-reference-output-validated; every captured output matched a traceable structured reference under the recorded tolerances and special-value policies. Proof classification remains bounded by model and consumer evidence.";
        }

        if (result.InferenceRan && result.OutputMatch && data.HasOutput)
        {
            if (!string.Equals(result.ModelSource, "embedded-dynamic-identity", StringComparison.Ordinal))
            {
                return "runtime-executed-bounded-input; this is sample runtime evidence only, not real-model or package-consumer proof.";
            }

            return "runtime-executed-synthetic-input; this is sample runtime evidence only, not real-model or package-consumer proof.";
        }

        if (result.InferenceRan && data.HasOutput)
        {
            return "runtime-output-captured-unverified; TensorRT enqueue and output readback completed, but no reference output matched, so this is not runtime proof, real-model proof, or package-consumer proof.";
        }

        if (result.InferenceRan && result.BenchmarkSummary.NoDataTransfersApplied && result.BenchmarkSummary.TimingSampleCount > 0)
        {
            return "runtime-benchmark-executed-no-data-transfers; TensorRT enqueue timing is recorded, but input H2D and output D2H/readback were suppressed, so this is not tensor-correctness, real-model, or package-consumer proof.";
        }

        if (string.Equals(result.ProofClassification, "precheck", StringComparison.Ordinal))
        {
            return artifactKind + "-skipped-precheck; dry-run did not probe TensorRT runtime, parse ONNX, build engine, or run inference.";
        }

        if (string.Equals(result.ProofClassification, "build-only", StringComparison.Ordinal))
        {
            return artifactKind + "-skipped-build-only; engine build evidence does not include inference output, timing, or package-consumer runtime proof.";
        }

        if (string.Equals(result.ProofClassification, "dependency-probe-only", StringComparison.Ordinal))
        {
            return artifactKind + "-skipped-dependency-probe-only; runtime dependencies were probed but inference did not execute.";
        }

        return artifactKind + "-skipped; generic external model binding/output semantics are not inferred by this sample path.";
    }

    private sealed class RuntimeArtifactProofBoundary
    {
        public RuntimeArtifactProofBoundary(
            string artifactProofBoundary,
            string runtimeProofClass,
            bool hasTensorOutputProof,
            bool hasRawBindingProof,
            bool isBuildOnlyEvidence,
            bool isDependencyProbeOnly,
            bool isSyntheticRuntime,
            bool hasBenchmarkExecutionEvidence,
            string modelSource,
            string enginePath,
            OnnxEnginePreflightMetadata preflightMetadata)
        {
            ArtifactProofBoundary = artifactProofBoundary ?? string.Empty;
            RuntimeProofClass = runtimeProofClass ?? string.Empty;
            HasTensorOutputProof = hasTensorOutputProof;
            HasRawBindingProof = hasRawBindingProof;
            IsBuildOnlyEvidence = isBuildOnlyEvidence;
            IsDependencyProbeOnly = isDependencyProbeOnly;
            IsSyntheticRuntime = isSyntheticRuntime;
            HasBenchmarkExecutionEvidence = hasBenchmarkExecutionEvidence;
            ModelSource = modelSource ?? string.Empty;
            EnginePath = enginePath ?? string.Empty;
            PreflightMetadata = preflightMetadata ?? OnnxEnginePreflightMetadata.Empty;
        }

        public string ArtifactProofBoundary { get; }

        public string RuntimeProofClass { get; }

        public bool HasTensorOutputProof { get; }

        public bool HasRawBindingProof { get; }

        public bool IsBuildOnlyEvidence { get; }

        public bool IsDependencyProbeOnly { get; }

        public bool IsSyntheticRuntime { get; }

        public bool HasBenchmarkExecutionEvidence { get; }

        public string ModelSource { get; }

        public string EnginePath { get; }

        public OnnxEnginePreflightMetadata PreflightMetadata { get; }
    }
}
