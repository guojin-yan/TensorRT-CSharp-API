using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class TrtexecLikeDeploymentOptions
{
    public IReadOnlyList<string> ToArgumentSegments()
    {
        List<string> args = new List<string>();
        Add(args, "--builderOptimizationLevel", BuilderOptimizationLevel.ToString(CultureInfo.InvariantCulture));
        if (DeviceOrdinal.HasValue)
        {
            Add(args, "--device", DeviceOrdinal.Value.ToString(CultureInfo.InvariantCulture));
        }

        if (MaxAuxStreams.HasValue)
        {
            Add(args, "--maxAuxStreams", MaxAuxStreams.Value.ToString(CultureInfo.InvariantCulture));
        }

        if (DlaCore.HasValue)
        {
            Add(args, "--useDLACore", DlaCore.Value.ToString(CultureInfo.InvariantCulture));
        }

        AddSwitch(args, "--allowGPUFallback", AllowGpuFallback);
        Add(args, "--tacticSources", TacticSources);
        Add(args, "--memPoolSize", MemoryPoolSizesToArgument());
        Add(args, "--inputIOFormats", InputIOFormats);
        Add(args, "--outputIOFormats", OutputIOFormats);
        Add(args, "--calib", CalibrationCacheFile);
        AddSwitch(args, "--directIO", DirectIO);
        Add(args, "--sparsity", Sparsity);
        AddSwitch(args, "--stronglyTyped", StronglyTyped);
        Add(args, "--minTiming", FormatNullable(MinTiming));
        Add(args, "--avgTiming", FormatNullable(AvgTiming));
        Add(args, "--precisionConstraints", PrecisionConstraints);
        Add(args, "--layerPrecisions", LayerPrecisions);
        Add(args, "--layerOutputTypes", LayerOutputTypes);
        AddSwitch(args, "--fp8", Fp8);
        AddSwitch(args, "--best", Best);
        AddSwitch(args, "--dumpRefit", DumpRefit);
        AddSwitch(args, "--allowWeightStreaming", AllowWeightStreaming);
        Add(args, "--markDebug", MarkDebug);
        AddSwitch(args, "--dumpDebugTensors", DumpDebugTensors);
        AddSwitch(args, "--versionCompatible", VersionCompatible);
        AddSwitch(args, "--excludeLeanRuntime", ExcludeLeanRuntime);
        AddSwitch(args, "--stripWeights", StripWeights);
        AddSwitch(args, "--refit", Refit);
        Add(args, "--refitFromOnnx", RefitFromOnnxPath);
        Add(args, "--saveRefittedEngine", SaveRefittedEnginePath);
        Add(args, "--weightStreamingBudget", WeightStreamingBudget.ArgumentValue);
        Add(args, "--exportTimingCache", ExportTimingCachePath);
        AddSwitch(args, "--safe", Safe);
        AddSwitch(args, "--consistency", Consistency);
        AddSwitch(args, "--builderCache", BuilderCache);
        AddSwitch(args, "--noBuilderCache", NoBuilderCache);
        Add(args, "--maxNbTactics", MaxNbTactics?.ToString(CultureInfo.InvariantCulture) ?? string.Empty);
        Add(args, "--tilingOptimizationLevel", TilingOptimizationLevel?.ToString() ?? string.Empty);
        Add(args, "--l2LimitForTiling", FormatBytes(L2LimitForTilingBytes));
        Add(args, "--quantizationFlags", QuantizationFlags?.ToString() ?? string.Empty);
        return args;
    }
}
