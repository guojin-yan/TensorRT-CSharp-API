using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp.Tools;

public static partial class OnnxEngineBuildDiagnostics
{
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
            result.OutputValidated,
            result.IdentityOutputMatch,
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

}
