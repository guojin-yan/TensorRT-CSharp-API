using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

/// <summary>
/// Contains pointer-free engine binding metadata copied for TensorRtExec reports.
/// </summary>
public sealed class OnnxEngineBindingMetadata
{
    private const string Boundary =
        "Copied pointer-free engine binding metadata is deployment diagnostics only. It does not validate tensor semantics, output correctness, real-model-runtime, package-consumer-runtime, or release readiness.";

    private OnnxEngineBindingMetadata(
        bool attempted,
        bool succeeded,
        string state,
        string engineName,
        int profileIndex,
        bool contextReadinessAttached,
        bool isReadyForEnqueue,
        IReadOnlyList<OnnxEngineBindingTensorMetadata> tensors,
        IReadOnlyList<string> diagnostics)
    {
        Attempted = attempted;
        Succeeded = succeeded;
        State = state ?? string.Empty;
        EngineName = engineName ?? string.Empty;
        ProfileIndex = profileIndex;
        ContextReadinessAttached = contextReadinessAttached;
        IsReadyForEnqueue = isReadyForEnqueue;
        Tensors = tensors ?? Array.Empty<OnnxEngineBindingTensorMetadata>();
        InputCount = Tensors.Count(static tensor => string.Equals(tensor.IOMode, "Input", StringComparison.Ordinal));
        OutputCount = Tensors.Count(static tensor => string.Equals(tensor.IOMode, "Output", StringComparison.Ordinal));
        Diagnostics = diagnostics ?? Array.Empty<string>();
    }

    public static OnnxEngineBindingMetadata Empty { get; } = new OnnxEngineBindingMetadata(
        attempted: false,
        succeeded: false,
        state: "not-attempted",
        engineName: string.Empty,
        profileIndex: -1,
        contextReadinessAttached: false,
        isReadyForEnqueue: false,
        tensors: Array.Empty<OnnxEngineBindingTensorMetadata>(),
        diagnostics: Array.Empty<string>());

    public bool Attempted { get; }

    public bool Succeeded { get; }

    public string State { get; }

    public string EngineName { get; }

    public int ProfileIndex { get; }

    public bool ContextReadinessAttached { get; }

    public bool IsReadyForEnqueue { get; }

    public int TensorCount => Tensors.Count;

    public int InputCount { get; }

    public int OutputCount { get; }

    public IReadOnlyList<OnnxEngineBindingTensorMetadata> Tensors { get; }

    public IReadOnlyList<string> Diagnostics { get; }

    public string EvidenceKind => "copied-pointer-free-TensorRtEngineBindingReport";

    public string EvidenceBoundary => Boundary;

    public bool PointerFreeCopiedSnapshot => true;

    public bool CanPromoteRuntimeProof => false;

    public bool CanPromoteReleaseProof => false;

    public static OnnxEngineBindingMetadata FromBindingReport(
        TensorRtEngineBindingReport report,
        string state)
    {
        if (report == null)
        {
            throw new ArgumentNullException(nameof(report));
        }

        OnnxEngineBindingTensorMetadata[] tensors = report.Tensors
            .Select(static tensor => OnnxEngineBindingTensorMetadata.FromBinding(tensor))
            .ToArray();
        string[] diagnostics = tensors
            .SelectMany(static tensor => tensor.Diagnostics.Select(diagnostic => tensor.Name + ": " + diagnostic))
            .ToArray();

        return new OnnxEngineBindingMetadata(
            attempted: true,
            succeeded: true,
            state,
            report.EngineName,
            report.ProfileIndex,
            contextReadinessAttached: report.Readiness != null,
            report.IsReadyForEnqueue,
            tensors,
            diagnostics);
    }
}
