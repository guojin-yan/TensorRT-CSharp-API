using System;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Tools;

/// <summary>
/// Captures pointer-free ONNX parser diagnostics and support counters after a build parse attempt.
/// </summary>
public sealed class OnnxEngineParserPreflightSnapshot
{
    public OnnxEngineParserPreflightSnapshot(
        TensorRtApiLine line,
        bool parseAttempted,
        bool parseSucceeded,
        string diagnosticsState,
        int errorCount,
        int copiedDiagnosticCount,
        string diagnosticSummary,
        bool identityOperatorSupported,
        bool modelSupportAttempted,
        string modelSupportState,
        bool modelSupported,
        long supportedSubgraphCount,
        long unsupportedSubgraphCount,
        int copiedSubgraphCount,
        long copiedSupportedSubgraphCount,
        long copiedUnsupportedSubgraphCount,
        long copiedNodeCount)
    {
        Line = line;
        ParseAttempted = parseAttempted;
        ParseSucceeded = parseSucceeded;
        DiagnosticsState = diagnosticsState ?? string.Empty;
        ErrorCount = Math.Max(0, errorCount);
        CopiedDiagnosticCount = Math.Max(0, copiedDiagnosticCount);
        DiagnosticSummary = diagnosticSummary ?? string.Empty;
        IdentityOperatorSupported = identityOperatorSupported;
        ModelSupportAttempted = modelSupportAttempted;
        ModelSupportState = modelSupportState ?? string.Empty;
        ModelSupported = modelSupported;
        SupportedSubgraphCount = Math.Max(0, supportedSubgraphCount);
        UnsupportedSubgraphCount = Math.Max(0, unsupportedSubgraphCount);
        CopiedSubgraphCount = Math.Max(0, copiedSubgraphCount);
        CopiedSupportedSubgraphCount = Math.Max(0, copiedSupportedSubgraphCount);
        CopiedUnsupportedSubgraphCount = Math.Max(0, copiedUnsupportedSubgraphCount);
        CopiedNodeCount = Math.Max(0, copiedNodeCount);
    }

    public static OnnxEngineParserPreflightSnapshot Empty { get; } = new OnnxEngineParserPreflightSnapshot(
        TensorRtApiLine.TensorRt10, false, false, "not-attempted", 0, 0, string.Empty, false, false, "not-attempted", false, 0, 0, 0, 0, 0, 0);

    public TensorRtApiLine Line { get; }
    public bool ParseAttempted { get; }
    public bool ParseSucceeded { get; }
    public string DiagnosticsState { get; }
    public int ErrorCount { get; }
    public int CopiedDiagnosticCount { get; }
    public string DiagnosticSummary { get; }
    public bool IdentityOperatorSupported { get; }
    public bool ModelSupportAttempted { get; }
    public string ModelSupportState { get; }
    public bool ModelSupported { get; }
    public long SupportedSubgraphCount { get; }
    public long UnsupportedSubgraphCount { get; }
    public int CopiedSubgraphCount { get; }
    public long CopiedSupportedSubgraphCount { get; }
    public long CopiedUnsupportedSubgraphCount { get; }
    public long CopiedNodeCount { get; }
    public string EvidenceKind => "copied-parser-preflight";
    public string EvidenceBoundary => "copied parser diagnostics and support counters are build/preflight evidence only; they do not prove real-model-runtime or package-consumer-runtime.";
    public bool PointerFreeCopiedSnapshot => true;
    public bool CanPromoteRuntimeProof => false;
    public bool CanPromoteReleaseProof => false;
    public bool CanDeleteDeferredRecord => false;

    public override string ToString() => $"Line={(int)Line} ParseAttempted={ParseAttempted} ParseSucceeded={ParseSucceeded} Errors={ErrorCount} Diagnostics={CopiedDiagnosticCount} ModelSupport={ModelSupportState}:{ModelSupported} Subgraphs={CopiedSubgraphCount}";
}
