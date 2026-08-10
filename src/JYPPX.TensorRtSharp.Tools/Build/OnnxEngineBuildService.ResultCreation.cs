using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class OnnxEngineBuildService
{
    private static OnnxEngineBuildResult CreateResult(
        bool success,
        bool skipped,
        string state,
        OnnxEngineBuildOptions options,
        string modelSource,
        string enginePath,
        bool parsed,
        bool engineSaved,
        bool engineFileRoundTrip,
        bool inferenceRan,
        bool outputMatch,
        int profileIndex,
        float? elapsedMilliseconds,
        string skipReason,
        IReadOnlyList<string> logLines,
        OnnxEngineBuildEvidenceSidecar evidenceSidecar,
        OnnxEngineBenchmarkSummary? benchmarkSummary = null,
        OnnxEnginePreflightMetadata? preflightMetadata = null,
        OnnxLoadedEngineDiagnostics? loadedEngineDiagnostics = null,
        OnnxEngineTimingCacheArtifact? timingCacheArtifact = null,
        TensorRtBuilderConfigDeploymentSnapshot? builderConfigDeploymentSnapshot = null,
        OnnxEngineParserPreflightSnapshot? parserPreflightSnapshot = null,
        OnnxEngineRefitSnapshot? refitSnapshot = null,
        OnnxEngineRefitPersistenceSnapshot? refitPersistenceSnapshot = null,
        bool outputValidated = false,
        bool identityOutputMatch = false,
        OnnxEngineBindingMetadata? bindingMetadata = null,
        OnnxEngineLayerInfoArtifact? layerInfoArtifact = null)
    {
        OnnxEngineCapabilityProbe capabilityProbe = ProbeCapabilities(options);
        logLines = AppendCapabilityProbeLog(logLines, capabilityProbe);

        return new OnnxEngineBuildResult(
            success,
            skipped,
            state,
            options.TensorRtLine,
            modelSource,
            enginePath,
            parsed,
            engineSaved,
            engineFileRoundTrip,
            inferenceRan,
            outputMatch,
            profileIndex,
            elapsedMilliseconds,
            skipReason,
            options.NormalizedCommandLine,
            options.DeploymentOptions,
            options.Diagnostics,
            logLines,
            evidenceSidecar,
            options.RuntimeOptions,
            benchmarkSummary,
            preflightMetadata,
            loadedEngineDiagnostics,
            timingCacheArtifact: timingCacheArtifact,
            capabilityProbe: capabilityProbe,
            workspaceBytes: options.WorkspaceBytes,
            builderConfigDeploymentSnapshot: builderConfigDeploymentSnapshot,
            parserPreflightSnapshot: parserPreflightSnapshot,
            refitSnapshot: refitSnapshot,
            refitPersistenceSnapshot: refitPersistenceSnapshot,
            outputValidated: outputValidated,
            identityOutputMatch: identityOutputMatch,
            bindingMetadata: bindingMetadata,
            layerInfoArtifact: layerInfoArtifact ?? loadedEngineDiagnostics?.LayerInfoArtifact ?? CreateLayerInformationBoundaryArtifact(options, "Result", state));
    }

    private static string RuntimeState(
        OnnxEngineRuntimeExecution? runtimeExecution,
        OnnxEngineBuildOptions options,
        string prefix)
    {
        if (runtimeExecution == null)
        {
            return prefix + "-runtime-unavailable";
        }
        if (runtimeExecution.OutputValidated)
        {
            return prefix + "-reference-validated-runtime";
        }
        if (options.RuntimeOptions.RequestsReferenceValidation)
        {
            return prefix + "-reference-validation-failed";
        }
        return runtimeExecution.IdentityOutputMatch
            ? prefix + "-identity-runtime"
            : prefix + "-runtime-output-unverified";
    }

}
