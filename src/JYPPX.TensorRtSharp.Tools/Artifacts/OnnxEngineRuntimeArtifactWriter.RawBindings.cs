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
    private static void WriteRawBindingsOrBoundary(string path, OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data)
    {
        if (CanCaptureRawBindings(result, data))
        {
            WriteBytes(path, data.RawOutputBytes);
            WriteJson(path + ".manifest.json", CreateRawBindingsManifest(path, result, data));
            return;
        }

        RuntimeArtifactProofBoundary proofBoundary = CreateProofBoundary(result, data, "raw-bindings");
        WriteJson(path, new
        {
            ArtifactKind = "trtexec-like-raw-bindings-skipped",
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
            RawBindingBytesWritten = 0
        });
    }

    private static object CreateRawBindingsManifest(string path, OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data)
    {
        RuntimeArtifactProofBoundary proofBoundary = CreateProofBoundary(result, data, "raw-bindings");
        IReadOnlyList<RawBindingSegment> segments = CreateRawBindingSegments(data.OutputTensors);
        return new
        {
            ArtifactKind = "trtexec-like-raw-bindings-manifest",
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
            RawBindingCaptureAvailable = true,
            OutputValidated = result.OutputValidated && data.OutputValidated,
            ReferenceValidation = data.ReferenceValidation,
            RawBindingPath = Path.GetFullPath(path),
            RawBindingFormat = "contiguous IEEE 754 float32 tensors in engine output order",
            ByteOrder = BitConverter.IsLittleEndian ? "little-endian" : "big-endian",
            OutputTensorCount = segments.Count,
            RawBindingBytesWritten = data.RawOutputBytes.LongLength,
            RawBindingSha256 = ComputeSha256(data.RawOutputBytes),
            Segments = segments
        };
    }

    private static IReadOnlyList<RawBindingSegment> CreateRawBindingSegments(
        IReadOnlyList<OnnxEngineRuntimeOutputArtifact> outputTensors)
    {
        List<RawBindingSegment> segments = new List<RawBindingSegment>(outputTensors.Count);
        long offset = 0;
        foreach (OnnxEngineRuntimeOutputArtifact output in outputTensors)
        {
            segments.Add(new RawBindingSegment(
                output.TensorName,
                output.Shape,
                output.ElementCount,
                offset,
                output.ByteLength,
                output.Sha256));
            offset = checked(offset + output.ByteLength);
        }

        return segments;
    }

    private static bool CanCaptureRawBindings(OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data)
    {
        return result.InferenceRan &&
            data.HasRawOutput;
    }

    private sealed class RawBindingSegment
    {
        public RawBindingSegment(
            string tensorName,
            IReadOnlyList<int> shape,
            int elementCount,
            long byteOffset,
            long byteLength,
            string sha256)
        {
            TensorName = tensorName ?? string.Empty;
            Shape = shape ?? Array.Empty<int>();
            ElementCount = elementCount;
            ByteOffset = byteOffset;
            ByteLength = byteLength;
            Sha256 = sha256 ?? string.Empty;
        }

        public string TensorName { get; }

        public IReadOnlyList<int> Shape { get; }

        public int ElementCount { get; }

        public long ByteOffset { get; }

        public long ByteLength { get; }

        public string Sha256 { get; }
    }

}
