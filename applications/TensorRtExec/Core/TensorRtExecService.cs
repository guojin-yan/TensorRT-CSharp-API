using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Tools;

namespace TensorRtExecApp.Core;

public sealed class TensorRtExecService
{
    public TensorRtExecReport Execute(TensorRtExecOptions options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        OnnxEngineBuildOptions buildOptions = OnnxEngineBuildOptions.FromTrtexecLikeOptions(options.TrtexecOptions);

        OnnxEngineBuildResult result = new OnnxEngineBuildService().Execute(buildOptions);
        string summary = result.Skipped
            ? "TensorRT execution skipped: " + result.SkipReason
            : "TensorRT execution state: " + result.State;

        return new TensorRtExecReport(
            result.Success,
            result.State,
            summary,
            result.LogLines,
            result.EnginePath,
            result.Parsed,
            result.InferenceRan,
            buildOptions.ExportReportPath,
            result.ProofClassification,
            result.NormalizedCommandSha256,
            buildOptions.DryRun,
            result.BuildEvidenceOnly,
            result.LoadedEngineDiagnostics.DiagnosticsState,
            result.LoadedEngineDiagnostics.Attempted,
            result.LoadedEngineDiagnostics.Succeeded,
            result.LoadedEngineDiagnostics.EvidenceBoundary,
            result.WorkspaceBytes,
            result.BuilderConfigDeploymentSnapshot == null ? "unavailable" : "copied-readback",
            result.BuilderConfigDeploymentSnapshot?.Diagnostics.Count ?? 0);
    }

}
