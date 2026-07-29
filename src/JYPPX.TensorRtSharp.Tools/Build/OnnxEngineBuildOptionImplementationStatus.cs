using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class OnnxEngineBuildOptionImplementationStatus
{
    public OnnxEngineBuildOptionImplementationStatus(
        string[] parsedOptions,
        string[] appliedOptions,
        string[] parseOnlyOptions,
        string evidenceBoundary)
    {
        ParsedOptions = parsedOptions ?? Array.Empty<string>();
        AppliedOptions = appliedOptions ?? Array.Empty<string>();
        ParseOnlyOptions = parseOnlyOptions ?? Array.Empty<string>();
        EvidenceBoundary = evidenceBoundary ?? string.Empty;
    }

    public string[] ParsedOptions { get; }

    public string[] AppliedOptions { get; }

    public string[] ParseOnlyOptions { get; }

    public string EvidenceBoundary { get; }
}
