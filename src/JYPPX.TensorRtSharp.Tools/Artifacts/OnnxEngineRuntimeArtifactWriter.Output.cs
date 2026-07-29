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
    private static object CreateOutputArtifact(OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data)
    {
        RuntimeArtifactProofBoundary proofBoundary = CreateProofBoundary(result, data, "output");
        return new
        {
            ArtifactKind = "trtexec-like-output-summary",
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
            result.InferenceRan,
            result.OutputMatch,
            result.IdentityOutputMatch,
            OutputCaptureAvailable = result.InferenceRan && data.HasOutput,
            OutputValidated = result.OutputValidated && data.OutputValidated,
            ReferenceValidation = data.ReferenceValidation,
            InputTensorCount = data.InputTensors.Count,
            InputTensors = data.InputTensors.Select(static input => new
            {
                input.TensorName,
                input.Shape,
                input.ElementCount,
                input.ByteLength,
                input.Preview,
                input.Sha256,
                input.SourceClassification,
                input.SourcePath
            }),
            OutputTensorCount = data.OutputTensors.Count,
            OutputTensors = data.OutputTensors.Select(static output => new
            {
                output.TensorName,
                output.Shape,
                output.ElementCount,
                output.Preview,
                output.ByteLength,
                output.Sha256
            }),
            TensorName = data.TensorName,
            Shape = data.Shape,
            data.InputElementCount,
            data.OutputElementCount,
            data.InputPreview,
            data.OutputPreview,
            OutputComparisonSample = ReadFloatSample(data.RawOutputBytes, 64),
            OutputByteLength = data.RawOutputBytes.LongLength,
            OutputSha256 = ComputeSha256(data.RawOutputBytes),
            Note = result.OutputValidated && data.OutputValidated
                ? "Output artifact contains bounded previews and hashes for every captured output tensor plus completed all-output reference comparison. Reference hashes alone do not establish correctness or promote the model/package proof classification."
                : "Output artifact contains bounded previews and hashes for every captured output tensor; it intentionally avoids large tensor dumps. Capture is not validation, and hashes do not promote the proof classification."
        };
    }

    private static IReadOnlyList<float> ReadFloatSample(byte[] bytes, int maximumValues)
    {
        if (bytes == null || bytes.Length == 0 || maximumValues <= 0)
        {
            return Array.Empty<float>();
        }

        int count = Math.Min(bytes.Length / sizeof(float), maximumValues);
        float[] values = new float[count];
        Buffer.BlockCopy(bytes, 0, values, 0, count * sizeof(float));
        return values;
    }

}
