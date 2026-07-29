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
    private static object CreateEngineReadbackArtifact(OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData data)
    {
        RuntimeArtifactProofBoundary proofBoundary = CreateProofBoundary(result, data, "engine-readback");
        OnnxLoadedEngineDiagnostics loaded = result.LoadedEngineDiagnostics;
        bool hasReadback = loaded.Attempted &&
            loaded.Succeeded &&
            !string.IsNullOrWhiteSpace(loaded.ReadbackFingerprint) &&
            !string.IsNullOrWhiteSpace(loaded.ReadbackSha256);

        return new
        {
            ArtifactKind = hasReadback ? "trtexec-like-engine-readback" : "trtexec-like-engine-readback-skipped",
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
            IsRuntimeExecutionProof = false,
            IsRealModelRuntimeProof = false,
            IsPackageConsumerRuntimeProof = false,
            result.NormalizedCommandSha256,
            ReadbackAvailable = hasReadback,
            ReadbackFingerprint = hasReadback ? loaded.ReadbackFingerprint : string.Empty,
            ReadbackSha256 = hasReadback ? loaded.ReadbackSha256 : string.Empty,
            SkippedReason = hasReadback ? string.Empty : EngineReadbackSkippedReason(loaded),
            LoadedEngineDiagnostics = new
            {
                loaded.Attempted,
                loaded.Succeeded,
                loaded.DiagnosticsState,
                loaded.FailureReason,
                loaded.EngineName,
                loaded.IOTensorCount,
                loaded.LayerCount,
                loaded.OptimizationProfileCount,
                loaded.DeviceMemorySizeInBytes,
                loaded.AuxiliaryStreamCount,
                loaded.Capability,
                loaded.ProfilingVerbosity,
                loaded.InspectorInformationLength,
                loaded.IOTensorSummaries,
                loaded.EvidenceBoundary
            },
            Note = "Engine readback artifact is readonly metadata evidence only; it does not bind tensors, enqueue inference, validate outputs, or prove real-model/package-consumer runtime."
        };
    }

    private static string GetEngineReadbackArtifactPath(TrtexecLikeRuntimeOptions options)
    {
        string sourcePath = FirstNonEmpty(
            options.ExportProfilePath,
            options.ExportTimesPath,
            options.ExportOutputPath,
            options.SaveProfilePath,
            options.DumpRawBindingsToFile);

        if (string.IsNullOrWhiteSpace(sourcePath))
        {
            return string.Empty;
        }

        string extension = Path.GetExtension(sourcePath);
        if (string.Equals(extension, ".json", StringComparison.OrdinalIgnoreCase))
        {
            return Path.ChangeExtension(sourcePath, ".engine-readback.json");
        }

        return sourcePath + ".engine-readback.json";
    }

    private static string FirstNonEmpty(params string[] values)
    {
        foreach (string value in values)
        {
            if (!string.IsNullOrWhiteSpace(value))
            {
                return value;
            }
        }

        return string.Empty;
    }

    private static string EngineReadbackSkippedReason(OnnxLoadedEngineDiagnostics loaded)
    {
        if (!loaded.Attempted)
        {
            return string.IsNullOrWhiteSpace(loaded.DiagnosticsState)
                ? "readonly engine diagnostics were not attempted"
                : loaded.DiagnosticsState;
        }

        if (!loaded.Succeeded)
        {
            return string.IsNullOrWhiteSpace(loaded.FailureReason)
                ? loaded.DiagnosticsState
                : loaded.FailureReason;
        }

        return "readonly engine diagnostics did not provide a readback fingerprint and SHA256";
    }

}
