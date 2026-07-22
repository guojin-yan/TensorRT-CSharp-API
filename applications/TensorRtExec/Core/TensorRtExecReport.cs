using System;
using System.Collections.Generic;

namespace TensorRtExecApp.Core;

public sealed class TensorRtExecReport
{
    public TensorRtExecReport(bool success, string state, string summary, IReadOnlyList<string> logLines)
        : this(success, state, summary, logLines, string.Empty, false, false, string.Empty, string.Empty, string.Empty, false, false, string.Empty, false, false, string.Empty, 0, "unavailable", 0, "not-attempted", 0)
    {
    }

    public TensorRtExecReport(bool success, string state, string summary, IReadOnlyList<string> logLines, string enginePath, bool parsed, bool inferenceRan)
        : this(success, state, summary, logLines, enginePath, parsed, inferenceRan, string.Empty, string.Empty, string.Empty, false, false, string.Empty, false, false, string.Empty, 0, "unavailable", 0, "not-attempted", 0)
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
        ulong workspaceBytes,
        string builderConfigDeploymentSnapshotState,
        int builderConfigDeploymentDiagnosticCount,
        string parserPreflightSnapshotState,
        int parserPreflightDiagnosticCount,
        bool refitAttempted = false,
        bool refitSucceeded = false,
        string refitState = "",
        bool refitPersistenceAttempted = false,
        bool refitPersistenceSucceeded = false,
        string refitPersistenceState = "",
        string persistedRefittedEnginePath = "")
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
        BuilderConfigDeploymentSnapshotState = builderConfigDeploymentSnapshotState ?? "unavailable";
        BuilderConfigDeploymentDiagnosticCount = builderConfigDeploymentDiagnosticCount < 0 ? 0 : builderConfigDeploymentDiagnosticCount;
        ParserPreflightSnapshotState = parserPreflightSnapshotState ?? "not-attempted";
        ParserPreflightDiagnosticCount = parserPreflightDiagnosticCount < 0 ? 0 : parserPreflightDiagnosticCount;
        RefitAttempted = refitAttempted;
        RefitSucceeded = refitSucceeded;
        RefitState = refitState ?? string.Empty;
        RefitPersistenceAttempted = refitPersistenceAttempted;
        RefitPersistenceSucceeded = refitPersistenceSucceeded;
        RefitPersistenceState = refitPersistenceState ?? string.Empty;
        PersistedRefittedEnginePath = persistedRefittedEnginePath ?? string.Empty;
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

    public string BuilderConfigDeploymentSnapshotState { get; }

    public int BuilderConfigDeploymentDiagnosticCount { get; }

    public string ParserPreflightSnapshotState { get; }

    public int ParserPreflightDiagnosticCount { get; }

    public bool RefitAttempted { get; }

    public bool RefitSucceeded { get; }

    public string RefitState { get; }

    public bool RefitPersistenceAttempted { get; }

    public bool RefitPersistenceSucceeded { get; }

    public string RefitPersistenceState { get; }

    public string PersistedRefittedEnginePath { get; }
}
