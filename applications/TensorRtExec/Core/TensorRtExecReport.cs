using System;
using System.Collections.Generic;

namespace TensorRtExecApp.Core;

public sealed class TensorRtExecReport
{
    public TensorRtExecReport(bool success, string state, string summary, IReadOnlyList<string> logLines)
        : this(success, state, summary, logLines, string.Empty, false, false, string.Empty, string.Empty, string.Empty, false, false, string.Empty, false, false, string.Empty, 0)
    {
    }

    public TensorRtExecReport(bool success, string state, string summary, IReadOnlyList<string> logLines, string enginePath, bool parsed, bool inferenceRan)
        : this(success, state, summary, logLines, enginePath, parsed, inferenceRan, string.Empty, string.Empty, string.Empty, false, false, string.Empty, false, false, string.Empty, 0)
    {
    }

    public TensorRtExecReport(
        bool success,
        string state,
        string summary,
        IReadOnlyList<string> logLines,
        string enginePath,
        bool parsed,
        bool inferenceRan,
        string reportPath,
        string proofClassification,
        string normalizedCommandSha256,
        bool dryRun,
        bool buildEvidenceOnly,
        string loadEngineDiagnosticsState,
        bool loadEngineDiagnosticsAttempted,
        bool loadEngineDiagnosticsSucceeded,
        string loadEngineDiagnosticsBoundary,
        ulong workspaceBytes)
    {
        Success = success;
        State = state ?? string.Empty;
        Summary = summary ?? string.Empty;
        LogLines = logLines ?? Array.Empty<string>();
        EnginePath = enginePath ?? string.Empty;
        Parsed = parsed;
        InferenceRan = inferenceRan;
        ReportPath = reportPath ?? string.Empty;
        ProofClassification = proofClassification ?? string.Empty;
        NormalizedCommandSha256 = normalizedCommandSha256 ?? string.Empty;
        DryRun = dryRun;
        BuildEvidenceOnly = buildEvidenceOnly;
        LoadEngineDiagnosticsState = loadEngineDiagnosticsState ?? string.Empty;
        LoadEngineDiagnosticsAttempted = loadEngineDiagnosticsAttempted;
        LoadEngineDiagnosticsSucceeded = loadEngineDiagnosticsSucceeded;
        LoadEngineDiagnosticsBoundary = loadEngineDiagnosticsBoundary ?? string.Empty;
        WorkspaceBytes = workspaceBytes;
    }

    public bool Success { get; }

    public string State { get; }

    public string Summary { get; }

    public IReadOnlyList<string> LogLines { get; }

    public string EnginePath { get; }

    public bool Parsed { get; }

    public bool InferenceRan { get; }

    public string ReportPath { get; }

    public string ProofClassification { get; }

    public string NormalizedCommandSha256 { get; }

    public bool DryRun { get; }

    public bool BuildEvidenceOnly { get; }

    public string LoadEngineDiagnosticsState { get; }

    public bool LoadEngineDiagnosticsAttempted { get; }

    public bool LoadEngineDiagnosticsSucceeded { get; }

    public string LoadEngineDiagnosticsBoundary { get; }

    public ulong WorkspaceBytes { get; }
}
