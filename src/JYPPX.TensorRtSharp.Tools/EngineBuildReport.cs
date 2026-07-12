using System;
using System.Collections.Generic;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class EngineBuildReport
{
    public EngineBuildReport(string mode, string modelSource, string enginePath, bool parsed, bool engineFileRoundTrip, IReadOnlyList<string> diagnostics)
    {
        Mode = mode ?? string.Empty;
        ModelSource = modelSource ?? string.Empty;
        EnginePath = enginePath ?? string.Empty;
        Parsed = parsed;
        EngineFileRoundTrip = engineFileRoundTrip;
        Diagnostics = diagnostics ?? Array.Empty<string>();
    }

    public string Mode { get; }

    public string ModelSource { get; }

    public string EnginePath { get; }

    public bool Parsed { get; }

    public bool EngineFileRoundTrip { get; }

    public IReadOnlyList<string> Diagnostics { get; }
}
