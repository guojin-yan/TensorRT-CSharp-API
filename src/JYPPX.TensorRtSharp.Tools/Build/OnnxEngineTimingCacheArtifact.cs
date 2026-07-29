using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class OnnxEngineTimingCacheArtifact
{
    public OnnxEngineTimingCacheArtifact(
        bool inputRequested,
        bool inputApplied,
        string inputPath,
        long inputLengthBytes,
        string inputSha256,
        bool outputRequested,
        bool outputWritten,
        string outputPath,
        long outputLengthBytes,
        string outputSha256,
        string state,
        string evidenceBoundary)
    {
        InputRequested = inputRequested;
        InputApplied = inputApplied;
        InputPath = inputPath ?? string.Empty;
        InputLengthBytes = inputLengthBytes;
        InputSha256 = inputSha256 ?? string.Empty;
        OutputRequested = outputRequested;
        OutputWritten = outputWritten;
        OutputPath = outputPath ?? string.Empty;
        OutputLengthBytes = outputLengthBytes;
        OutputSha256 = outputSha256 ?? string.Empty;
        State = state ?? string.Empty;
        EvidenceBoundary = evidenceBoundary ?? string.Empty;
    }

    public static OnnxEngineTimingCacheArtifact Empty { get; } = new OnnxEngineTimingCacheArtifact(
        inputRequested: false,
        inputApplied: false,
        inputPath: string.Empty,
        inputLengthBytes: 0,
        inputSha256: string.Empty,
        outputRequested: false,
        outputWritten: false,
        outputPath: string.Empty,
        outputLengthBytes: 0,
        outputSha256: string.Empty,
        state: string.Empty,
        evidenceBoundary: "timing-cache import/export evidence is build-cache lifecycle metadata only; no cache was requested.");

    public bool InputRequested { get; }

    public bool InputApplied { get; }

    public string InputPath { get; }

    public long InputLengthBytes { get; }

    public string InputSha256 { get; }

    public bool OutputRequested { get; }

    public bool OutputWritten { get; }

    public string OutputPath { get; }

    public long OutputLengthBytes { get; }

    public string OutputSha256 { get; }

    public string State { get; }

    public string EvidenceBoundary { get; }
}
