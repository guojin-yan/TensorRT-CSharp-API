using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

internal static partial class TrtexecLikeBuildPolicy
{
    private static void ApplyPrecisionConstraints(
        TensorRtBuilderConfig config,
        string precisionConstraints,
        List<string> log)
    {
        if (string.IsNullOrWhiteSpace(precisionConstraints))
        {
            return;
        }

        if (config.Line == TensorRtApiLine.TensorRt11)
        {
            log.Add($"TrtexecBuildPolicy Name=PrecisionConstraints Applied=False Requested={precisionConstraints} VersionGuard=TRT11 Reason=precision-constraint-builder-flags-removed ReadbackMatch=False");
            return;
        }

        bool expectPrefer = string.Equals(precisionConstraints, "prefer", StringComparison.Ordinal);
        bool expectObey = string.Equals(precisionConstraints, "obey", StringComparison.Ordinal);
        config.SetFlag(TensorRtBuilderFlag.PreferPrecisionConstraints, expectPrefer);
        config.SetFlag(TensorRtBuilderFlag.ObeyPrecisionConstraints, expectObey);
        bool readbackPrefer = config.GetFlag(TensorRtBuilderFlag.PreferPrecisionConstraints);
        bool readbackObey = config.GetFlag(TensorRtBuilderFlag.ObeyPrecisionConstraints);
        bool readbackMatch = readbackPrefer == expectPrefer && readbackObey == expectObey;
        log.Add($"TrtexecBuildPolicy Name=PrecisionConstraints Applied={readbackMatch} Requested={precisionConstraints} ReadbackPrefer={readbackPrefer} ReadbackObey={readbackObey} ReadbackMatch={readbackMatch}");
        if (!readbackMatch)
        {
            throw new InvalidOperationException("Precision constraint flags did not match TensorRT builder-config readback.");
        }
    }
}
