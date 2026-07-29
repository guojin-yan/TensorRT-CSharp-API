using System;
using System.Collections.Generic;

namespace JYPPX.TensorRtSharp.Tools;

/// <summary>
/// Represents copied diagnostics from an ONNX-driven stripped-plan refit lifecycle.
/// 表示 ONNX 驱动的 stripped-plan 重整生命周期中复制出的诊断信息。
/// </summary>
public sealed class OnnxEngineRefitSnapshot
{
    public OnnxEngineRefitSnapshot(
        bool attempted,
        bool succeeded,
        string state,
        string sourcePath,
        long sourceLengthBytes,
        string sourceSha256,
        bool engineRefittableBefore,
        bool engineRefittableAfter,
        bool parserRefitReturned,
        bool engineRefitReturned,
        IReadOnlyList<string> missingWeightsBefore,
        IReadOnlyList<string> allWeightsBefore,
        IReadOnlyList<string> missingWeightsAfter,
        IReadOnlyList<string> allWeightsAfter,
        int parserErrorCount,
        int copiedDiagnosticCount,
        string diagnosticSummary,
        bool contextCreationAllowed,
        string evidenceBoundary)
    {
        Attempted = attempted;
        Succeeded = succeeded;
        State = state ?? string.Empty;
        SourcePath = sourcePath ?? string.Empty;
        SourceLengthBytes = sourceLengthBytes;
        SourceSha256 = sourceSha256 ?? string.Empty;
        EngineRefittableBefore = engineRefittableBefore;
        EngineRefittableAfter = engineRefittableAfter;
        ParserRefitReturned = parserRefitReturned;
        EngineRefitReturned = engineRefitReturned;
        MissingWeightsBefore = missingWeightsBefore ?? Array.Empty<string>();
        AllWeightsBefore = allWeightsBefore ?? Array.Empty<string>();
        MissingWeightsAfter = missingWeightsAfter ?? Array.Empty<string>();
        AllWeightsAfter = allWeightsAfter ?? Array.Empty<string>();
        ParserErrorCount = parserErrorCount;
        CopiedDiagnosticCount = copiedDiagnosticCount;
        DiagnosticSummary = diagnosticSummary ?? string.Empty;
        ContextCreationAllowed = contextCreationAllowed;
        EvidenceBoundary = evidenceBoundary ?? string.Empty;
    }

    public static OnnxEngineRefitSnapshot Empty { get; } = new OnnxEngineRefitSnapshot(
        false, false, string.Empty, string.Empty, 0, string.Empty, false, false, false, false,
        Array.Empty<string>(), Array.Empty<string>(), Array.Empty<string>(), Array.Empty<string>(),
        0, 0, string.Empty, false, string.Empty);

    public bool Attempted { get; }
    public bool Succeeded { get; }
    public string State { get; }
    public string SourcePath { get; }
    public long SourceLengthBytes { get; }
    public string SourceSha256 { get; }
    public bool EngineRefittableBefore { get; }
    public bool EngineRefittableAfter { get; }
    public bool ParserRefitReturned { get; }
    public bool EngineRefitReturned { get; }
    public IReadOnlyList<string> MissingWeightsBefore { get; }
    public IReadOnlyList<string> AllWeightsBefore { get; }
    public IReadOnlyList<string> MissingWeightsAfter { get; }
    public IReadOnlyList<string> AllWeightsAfter { get; }
    public int ParserErrorCount { get; }
    public int CopiedDiagnosticCount { get; }
    public string DiagnosticSummary { get; }
    public bool ContextCreationAllowed { get; }
    public string EvidenceBoundary { get; }
}
