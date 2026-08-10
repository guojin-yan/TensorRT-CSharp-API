using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using JYPPX.TensorRtSharp.Tools;

namespace TensorRtExecApp.Core;

public static class TensorRtExecReportFormatter
{
    public static IReadOnlyList<string> Format(TensorRtExecReport report)
    {
        if (report == null)
        {
            throw new ArgumentNullException(nameof(report));
        }

        List<string> lines = new List<string>(report.LogLines);
        if (!string.IsNullOrWhiteSpace(report.ReportPath))
        {
            lines.Add("TensorRtExec ReportPath=" + report.ReportPath);
        }

        if (!string.IsNullOrWhiteSpace(report.ProofClassification))
        {
            lines.Add(
                "TensorRtExec ProofClassification=" + report.ProofClassification +
                " BuildEvidenceOnly=" + FormatBoolean(report.BuildEvidenceOnly) +
                " DryRun=" + FormatBoolean(report.DryRun));
        }

        if (!string.IsNullOrWhiteSpace(report.NormalizedCommandSha256))
        {
            lines.Add("TensorRtExec NormalizedCommandSha256=" + report.NormalizedCommandSha256);
        }

        if (!string.IsNullOrWhiteSpace(report.LoadEngineDiagnosticsState))
        {
            lines.Add(
                "TensorRtExec LoadEngineDiagnosticsState=" + report.LoadEngineDiagnosticsState +
                " Attempted=" + FormatBoolean(report.LoadEngineDiagnosticsAttempted) +
                " Succeeded=" + FormatBoolean(report.LoadEngineDiagnosticsSucceeded));
            lines.Add("TensorRtExec LoadEngineDiagnosticsBoundary=" + report.LoadEngineDiagnosticsBoundary);
        }

        lines.Add("TensorRtExec WorkspaceBytes=" + report.WorkspaceBytes.ToString(CultureInfo.InvariantCulture));
        lines.Add(
            "TensorRtExec BuilderConfigDeploymentSnapshot=" + report.BuilderConfigDeploymentSnapshotState +
            " Diagnostics=" + report.BuilderConfigDeploymentDiagnosticCount.ToString(CultureInfo.InvariantCulture));
        lines.Add(
            "TensorRtExec ParserPreflightSnapshot=" + report.ParserPreflightSnapshotState +
            " Diagnostics=" + report.ParserPreflightDiagnosticCount.ToString(CultureInfo.InvariantCulture));
        AddBindingMetadata(lines, report.BindingMetadata);
        AddLayerInfoArtifact(lines, report.LayerInfoArtifact);
        lines.Add(
            "TensorRtExec RefitSnapshot=" + report.RefitState +
            " Attempted=" + FormatBoolean(report.RefitAttempted) +
            " Succeeded=" + FormatBoolean(report.RefitSucceeded));
        lines.Add(
            "TensorRtExec RefitPersistence=" + report.RefitPersistenceState +
            " Attempted=" + FormatBoolean(report.RefitPersistenceAttempted) +
            " Succeeded=" + FormatBoolean(report.RefitPersistenceSucceeded) +
            " Plan=" + report.PersistedRefittedEnginePath);
        if (!string.IsNullOrWhiteSpace(report.Summary))
        {
            lines.Add(report.Summary);
        }

        lines.Add("TensorRtExec State=" + report.State + " Success=" + FormatBoolean(report.Success));
        return lines;
    }

    public static bool IsInvalidArguments(Exception exception)
    {
        return exception is ArgumentException || exception is FileNotFoundException;
    }

    public static string FormatFailure(Exception exception)
    {
        if (exception == null)
        {
            throw new ArgumentNullException(nameof(exception));
        }

        return IsInvalidArguments(exception)
            ? "TensorRtExec=InvalidArguments Reason=" + exception.Message
            : "TensorRtExec=Failed Type=" + exception.GetType().Name + " Reason=" + exception.Message;
    }

    private static void AddBindingMetadata(List<string> lines, OnnxEngineBindingMetadata metadata)
    {
        lines.Add(
            "TensorRtExec BindingMetadataState=" + metadata.State +
            " Attempted=" + FormatBoolean(metadata.Attempted) +
            " Succeeded=" + FormatBoolean(metadata.Succeeded) +
            " Tensors=" + metadata.TensorCount.ToString(CultureInfo.InvariantCulture) +
            " Inputs=" + metadata.InputCount.ToString(CultureInfo.InvariantCulture) +
            " Outputs=" + metadata.OutputCount.ToString(CultureInfo.InvariantCulture) +
            " ContextReadinessAttached=" + FormatBoolean(metadata.ContextReadinessAttached) +
            " ReadyForEnqueue=" + FormatBoolean(metadata.IsReadyForEnqueue));
        foreach (OnnxEngineBindingTensorMetadata tensor in metadata.Tensors)
        {
            lines.Add(
                "TensorRtExec BindingMetadata" +
                " Index=" + tensor.Index.ToString(CultureInfo.InvariantCulture) +
                " Name=" + tensor.Name +
                " IOMode=" + tensor.IOMode +
                " DataType=" + tensor.DataType +
                " EngineShape=" + FormatShape(tensor.EngineShape) +
                " ProfileMin=" + FormatShape(tensor.ProfileMinShape) +
                " ProfileOpt=" + FormatShape(tensor.ProfileOptShape) +
                " ProfileMax=" + FormatShape(tensor.ProfileMaxShape) +
                " Location=" + tensor.Location +
                " Format=" + tensor.Format +
                " VectorizedDimension=" + tensor.VectorizedDimension.ToString(CultureInfo.InvariantCulture) +
                " UsesDataTypeSizeFallback=" + FormatBoolean(tensor.UsesDataTypeSizeFallback));
        }

        lines.Add("TensorRtExec BindingMetadataBoundary=" + metadata.EvidenceBoundary);
    }

    private static void AddLayerInfoArtifact(List<string> lines, OnnxEngineLayerInfoArtifact artifact)
    {
        lines.Add(
            "TensorRtExec LayerInfoArtifactState=" + artifact.State +
            " Requested=" + FormatBoolean(artifact.Requested) +
            " Collected=" + FormatBoolean(artifact.Collected) +
            " Source=" + artifact.Source +
            " Format=" + artifact.InformationFormat +
            " ContentKind=" + artifact.ContentKind +
            " Layers=" + artifact.LayerCount.ToString(CultureInfo.InvariantCulture) +
            " LengthBytes=" + artifact.LengthBytes.ToString(CultureInfo.InvariantCulture) +
            " Sha256=" + artifact.Sha256);
        lines.Add(
            "TensorRtExec LayerInfoArtifactExport Requested=" + FormatBoolean(artifact.ExportRequested) +
            " Written=" + FormatBoolean(artifact.ExportWritten) +
            " Path=" + artifact.ExportPath);
        lines.Add("TensorRtExec LayerInfoArtifactBoundary=" + artifact.EvidenceBoundary);
    }

    private static string FormatShape(IReadOnlyList<int>? shape)
    {
        return shape == null ? "unavailable" : string.Join("x", shape.Select(static value => value.ToString(CultureInfo.InvariantCulture)));
    }

    private static string FormatBoolean(bool value)
    {
        return value.ToString(CultureInfo.InvariantCulture);
    }
}
