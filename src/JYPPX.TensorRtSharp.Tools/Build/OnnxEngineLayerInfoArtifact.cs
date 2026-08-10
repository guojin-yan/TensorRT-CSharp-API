using System;
using System.Collections.Generic;

namespace JYPPX.TensorRtSharp.Tools;

/// <summary>
/// Pointer-free metadata for an optional copied TensorRT engine-inspector layer export.
/// </summary>
public sealed class OnnxEngineLayerInfoArtifact
{
    public OnnxEngineLayerInfoArtifact(
        bool requested,
        bool collected,
        string source,
        string state,
        string informationFormat,
        string contentKind,
        string requestedProfilingVerbosity,
        int layerCount,
        bool dumpRequested,
        bool exportRequested,
        bool exportWritten,
        string exportPath,
        long lengthBytes,
        string sha256,
        IReadOnlyList<string> diagnostics,
        string evidenceBoundary)
    {
        Requested = requested;
        Collected = collected;
        Source = source ?? string.Empty;
        State = state ?? string.Empty;
        InformationFormat = informationFormat ?? string.Empty;
        ContentKind = contentKind ?? string.Empty;
        RequestedProfilingVerbosity = requestedProfilingVerbosity ?? string.Empty;
        LayerCount = layerCount < 0 ? 0 : layerCount;
        DumpRequested = dumpRequested;
        ExportRequested = exportRequested;
        ExportWritten = exportWritten;
        ExportPath = exportPath ?? string.Empty;
        LengthBytes = lengthBytes < 0 ? 0 : lengthBytes;
        Sha256 = sha256 ?? string.Empty;
        Diagnostics = diagnostics ?? Array.Empty<string>();
        EvidenceBoundary = evidenceBoundary ?? string.Empty;
    }

    public static OnnxEngineLayerInfoArtifact Empty { get; } = new OnnxEngineLayerInfoArtifact(
        requested: false,
        collected: false,
        source: string.Empty,
        state: "not-requested",
        informationFormat: string.Empty,
        contentKind: string.Empty,
        requestedProfilingVerbosity: string.Empty,
        layerCount: 0,
        dumpRequested: false,
        exportRequested: false,
        exportWritten: false,
        exportPath: string.Empty,
        lengthBytes: 0,
        sha256: string.Empty,
        diagnostics: Array.Empty<string>(),
        evidenceBoundary: "Layer inspector output is optional copied engine metadata. It cannot promote runtime, model, package-consumer, or release proof.");

    public bool Requested { get; }

    public bool Collected { get; }

    public string Source { get; }

    public string State { get; }

    public string InformationFormat { get; }

    public string ContentKind { get; }

    public string RequestedProfilingVerbosity { get; }

    public int LayerCount { get; }

    public bool DumpRequested { get; }

    public bool ExportRequested { get; }

    public bool ExportWritten { get; }

    public string ExportPath { get; }

    public long LengthBytes { get; }

    public string Sha256 { get; }

    public IReadOnlyList<string> Diagnostics { get; }

    public string EvidenceBoundary { get; }

    public bool PointerFreeCopiedSnapshot => true;

    public bool CanPromoteRuntimeProof => false;

    public bool CanPromoteReleaseProof => false;
}
