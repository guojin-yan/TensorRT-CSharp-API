using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class OnnxEngineBuildReportBoundary
{
    public OnnxEngineBuildReportBoundary(
        bool isRuntimeProof,
        bool isBuildOnly,
        string forbiddenSubstituteReason,
        string copiedDiagnosticsBoundary,
        string parserDiagnosticsEvidenceKind,
        string parserRefitterDiagnosticsEvidenceKind,
        bool canPromoteCopiedDiagnosticsToRuntimeProof,
        string parserDiagnosticsOwnerAction,
        IReadOnlyList<string> forbiddenSubstitutes)
    {
        IsRuntimeProof = isRuntimeProof;
        IsBuildOnly = isBuildOnly;
        ForbiddenSubstituteReason = forbiddenSubstituteReason ?? string.Empty;
        CopiedDiagnosticsBoundary = copiedDiagnosticsBoundary ?? string.Empty;
        ParserDiagnosticsEvidenceKind = parserDiagnosticsEvidenceKind ?? string.Empty;
        ParserRefitterDiagnosticsEvidenceKind = parserRefitterDiagnosticsEvidenceKind ?? string.Empty;
        CanPromoteCopiedDiagnosticsToRuntimeProof = canPromoteCopiedDiagnosticsToRuntimeProof;
        ParserDiagnosticsOwnerAction = parserDiagnosticsOwnerAction ?? string.Empty;
        ForbiddenSubstitutes = forbiddenSubstitutes ?? Array.Empty<string>();
    }

    public bool IsRuntimeProof { get; }

    public bool IsBuildOnly { get; }

    public string ForbiddenSubstituteReason { get; }

    public string CopiedDiagnosticsBoundary { get; }

    public string ParserDiagnosticsEvidenceKind { get; }

    public string ParserRefitterDiagnosticsEvidenceKind { get; }

    public bool CanPromoteCopiedDiagnosticsToRuntimeProof { get; }

    public string ParserDiagnosticsOwnerAction { get; }

    public IReadOnlyList<string> ForbiddenSubstitutes { get; }
}
