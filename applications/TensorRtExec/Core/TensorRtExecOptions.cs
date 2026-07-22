using System;
using System.Collections.Generic;
using System.Globalization;
using JYPPX.TensorRtSharp.Tools;
using JYPPX.TensorRtSharp;

namespace TensorRtExecApp.Core;

public sealed class TensorRtExecOptions
{
    public TensorRtExecOptions(TrtexecLikeOptions trtexecOptions)
    {
        TrtexecOptions = trtexecOptions ?? throw new ArgumentNullException(nameof(trtexecOptions));
    }

    public TensorRtExecOptions(
        string onnxPath,
        string enginePath,
        string loadEnginePath,
        string tensorRtLine,
        bool fp16,
        bool int8,
        bool bf16,
        bool tf32,
        int workspaceMiB,
        bool buildOnly,
        bool skipInference,
        bool dryRun,
        int iterations,
        int warmUpMilliseconds,
        int durationSeconds,
        int streams,
        bool useCudaGraph,
        bool noDataTransfers,
        bool useSpinWait,
        int? threads,
        int? avgRuns,
        float? percentile,
        int? sleepTimeMilliseconds,
        int? idleTimeMilliseconds,
        int? infStreams,
        string loadInputs,
        bool dumpOutput,
        string dumpRawBindingsToFile,
        string exportOutputPath,
        string exportTimesPath,
        string exportProfilePath,
        string saveProfilePath,
        string minShapes,
        string optShapes,
        string maxShapes,
        IReadOnlyList<string> plugins,
        string timingCacheFile,
        string profilingVerbosity,
        int builderOptimizationLevel,
        int? maxAuxStreams,
        int? deviceOrdinal,
        int? dlaCore,
        bool allowGpuFallback,
        string tacticSources,
        string memoryPoolSizes,
        string inputIOFormats,
        string outputIOFormats,
        string calibrationCacheFile,
        bool directIO,
        string sparsity,
        bool stronglyTyped,
        int? minTiming,
        int? avgTiming,
        string precisionConstraints,
        string layerPrecisions,
        string layerOutputTypes,
        bool fp8,
        bool best,
        bool dumpRefit,
        bool allowWeightStreaming,
        string markDebug,
        bool dumpDebugTensors,
        bool versionCompatible,
        bool excludeLeanRuntime,
        bool stripWeights,
        bool refit,
        ulong? weightStreamingBudgetBytes,
        string exportTimingCachePath,
        bool safe,
        bool consistency,
        bool builderCache,
        bool noBuilderCache,
        bool dumpLayerInfo,
        bool dumpProfile,
        bool separateProfileRun,
        string exportLayerInfoPath,
        string exportReportPath,
        string evidenceSidecarPath,
        int? maxNbTactics = null,
        string tilingOptimizationLevel = "",
        ulong? l2LimitForTilingBytes = null,
        string quantizationFlags = "",
        string weightStreamingBudget = "")
    {
        List<string> args = new List<string>();
        Add(args, "--tensor-rt-line", string.IsNullOrWhiteSpace(tensorRtLine) ? "10" : tensorRtLine);
        Add(args, "--onnx", onnxPath);
        Add(args, "--saveEngine", enginePath);
        Add(args, "--loadEngine", loadEnginePath);
        Add(args, "--workspace", workspaceMiB.ToString(CultureInfo.InvariantCulture));
        Add(args, "--iterations", iterations.ToString(CultureInfo.InvariantCulture));
        Add(args, "--warmUp", warmUpMilliseconds.ToString(CultureInfo.InvariantCulture));
        Add(args, "--duration", durationSeconds.ToString(CultureInfo.InvariantCulture));
        Add(args, "--streams", streams.ToString(CultureInfo.InvariantCulture));
        Add(args, "--minShapes", minShapes);
        Add(args, "--optShapes", optShapes);
        Add(args, "--maxShapes", maxShapes);
        Add(args, "--timingCacheFile", timingCacheFile);
        Add(args, "--profilingVerbosity", profilingVerbosity);
        Add(args, "--builderOptimizationLevel", builderOptimizationLevel.ToString(CultureInfo.InvariantCulture));
        Add(args, "--maxAuxStreams", FormatNullable(maxAuxStreams));
        Add(args, "--device", FormatNullable(deviceOrdinal));
        Add(args, "--useDLACore", FormatNullable(dlaCore));
        Add(args, "--tacticSources", tacticSources);
        Add(args, "--memPoolSize", memoryPoolSizes);
        Add(args, "--inputIOFormats", inputIOFormats);
        Add(args, "--outputIOFormats", outputIOFormats);
        Add(args, "--calib", calibrationCacheFile);
        Add(args, "--sparsity", sparsity);
        Add(args, "--minTiming", FormatNullable(minTiming));
        Add(args, "--avgTiming", FormatNullable(avgTiming));
        Add(args, "--precisionConstraints", precisionConstraints);
        Add(args, "--layerPrecisions", layerPrecisions);
        Add(args, "--layerOutputTypes", layerOutputTypes);
        Add(args, "--markDebug", markDebug);
        Add(args, "--weightStreamingBudget", string.IsNullOrWhiteSpace(weightStreamingBudget)
            ? FormatBytesMiB(weightStreamingBudgetBytes)
            : weightStreamingBudget);
        Add(args, "--exportTimingCache", exportTimingCachePath);
        AddSwitch(args, "--safe", safe);
        AddSwitch(args, "--consistency", consistency);
        AddSwitch(args, "--builderCache", builderCache);
        AddSwitch(args, "--noBuilderCache", noBuilderCache);
        AddSwitch(args, "--dumpLayerInfo", dumpLayerInfo);
        AddSwitch(args, "--dumpProfile", dumpProfile);
        AddSwitch(args, "--separateProfileRun", separateProfileRun);
        Add(args, "--exportLayerInfo", exportLayerInfoPath);
        Add(args, "--exportReport", exportReportPath);
        Add(args, "--evidenceSidecar", evidenceSidecarPath);
        Add(args, "--maxNbTactics", FormatNullable(maxNbTactics));
        Add(args, "--tilingOptimizationLevel", tilingOptimizationLevel);
        Add(args, "--l2LimitForTiling", FormatBytes(l2LimitForTilingBytes));
        Add(args, "--quantizationFlags", quantizationFlags);
        Add(args, "--threads", FormatNullable(threads));
        Add(args, "--avgRuns", FormatNullable(avgRuns));
        Add(args, "--percentile", FormatNullable(percentile));
        Add(args, "--sleepTime", FormatNullable(sleepTimeMilliseconds));
        Add(args, "--idleTime", FormatNullable(idleTimeMilliseconds));
        Add(args, "--infStreams", FormatNullable(infStreams));
        Add(args, "--loadInputs", loadInputs);
        Add(args, "--dumpRawBindingsToFile", dumpRawBindingsToFile);
        Add(args, "--exportOutput", exportOutputPath);
        Add(args, "--exportTimes", exportTimesPath);
        Add(args, "--exportProfile", exportProfilePath);
        Add(args, "--saveProfile", saveProfilePath);
        if (plugins != null && plugins.Count > 0)
        {
            Add(args, "--plugins", string.Join(";", plugins));
        }

        AddSwitch(args, "--fp16", fp16);
        AddSwitch(args, "--int8", int8);
        AddSwitch(args, "--bf16", bf16);
        AddSwitch(args, "--fp8", fp8);
        AddSwitch(args, "--best", best);
        AddSwitch(args, "--noTF32", !tf32);
        AddSwitch(args, "--allowGPUFallback", allowGpuFallback);
        AddSwitch(args, "--directIO", directIO);
        AddSwitch(args, "--stronglyTyped", stronglyTyped);
        AddSwitch(args, "--versionCompatible", versionCompatible);
        AddSwitch(args, "--excludeLeanRuntime", excludeLeanRuntime);
        AddSwitch(args, "--stripWeights", stripWeights);
        AddSwitch(args, "--refit", refit);
        AddSwitch(args, "--dumpRefit", dumpRefit);
        AddSwitch(args, "--allowWeightStreaming", allowWeightStreaming);
        AddSwitch(args, "--dumpDebugTensors", dumpDebugTensors);
        AddSwitch(args, "--buildOnly", buildOnly);
        AddSwitch(args, "--skipInference", skipInference);
        AddSwitch(args, "--dryRun", dryRun);
        AddSwitch(args, "--useCudaGraph", useCudaGraph);
        AddSwitch(args, "--noDataTransfers", noDataTransfers);
        AddSwitch(args, "--useSpinWait", useSpinWait);
        AddSwitch(args, "--dumpOutput", dumpOutput);
        TrtexecOptions = TrtexecLikeParser.Parse(args.ToArray());
    }

    public TrtexecLikeOptions TrtexecOptions { get; }

    public string OnnxPath => TrtexecOptions.OnnxPath;

    public string EnginePath => TrtexecOptions.SaveEnginePath;

    public string LoadEnginePath => TrtexecOptions.LoadEnginePath;

    public string TensorRtLine => ((int)TrtexecOptions.TensorRtLine).ToString(CultureInfo.InvariantCulture);

    public bool Fp16 => TrtexecOptions.Fp16;

    public bool Int8 => TrtexecOptions.Int8;

    public bool Bf16 => TrtexecOptions.Bf16;

    public bool Tf32 => TrtexecOptions.Tf32;

    public int WorkspaceMiB => checked((int)(TrtexecOptions.WorkspaceBytes / (1024UL * 1024UL)));

    public bool BuildOnly => TrtexecOptions.BuildOnly;

    public bool SkipInference => TrtexecOptions.SkipInference;

    public bool DryRun => TrtexecOptions.DryRun;

    public int Iterations => TrtexecOptions.Iterations;

    public int WarmUpMilliseconds => TrtexecOptions.WarmUpMilliseconds;

    public int DurationSeconds => TrtexecOptions.DurationSeconds;

    public int Streams => TrtexecOptions.Streams;

    public bool UseCudaGraph => TrtexecOptions.UseCudaGraph;

    public string MinShapes => string.Join(",", TrtexecOptions.ShapeProfile.MinShapes);

    public string OptShapes => string.Join(",", TrtexecOptions.ShapeProfile.OptShapes);

    public string MaxShapes => string.Join(",", TrtexecOptions.ShapeProfile.MaxShapes);

    public IReadOnlyList<string> Plugins => TrtexecOptions.Plugins;

    public string TimingCacheFile => TrtexecOptions.TimingCacheFile;

    public string ProfilingVerbosity => TrtexecOptions.ProfilingVerbosity;

    public bool DumpLayerInfo => TrtexecOptions.DumpLayerInfo;

    public bool DumpProfile => TrtexecOptions.DumpProfile;

    public bool SeparateProfileRun => TrtexecOptions.SeparateProfileRun;

    public bool Fp8 => TrtexecOptions.DeploymentOptions.Fp8;

    public bool Best => TrtexecOptions.DeploymentOptions.Best;

    public bool DumpRefit => TrtexecOptions.DeploymentOptions.DumpRefit;

    public bool AllowWeightStreaming => TrtexecOptions.DeploymentOptions.AllowWeightStreaming;

    public string MarkDebug => TrtexecOptions.DeploymentOptions.MarkDebug;

    public bool DumpDebugTensors => TrtexecOptions.DeploymentOptions.DumpDebugTensors;

    public int BuilderOptimizationLevel => TrtexecOptions.DeploymentOptions.BuilderOptimizationLevel;

    public int? MaxAuxStreams => TrtexecOptions.DeploymentOptions.MaxAuxStreams;

    public int? DeviceOrdinal => TrtexecOptions.DeploymentOptions.DeviceOrdinal;

    public int? DlaCore => TrtexecOptions.DeploymentOptions.DlaCore;

    public bool AllowGpuFallback => TrtexecOptions.DeploymentOptions.AllowGpuFallback;

    public string TacticSources => TrtexecOptions.DeploymentOptions.TacticSources;

    public IReadOnlyList<TrtexecLikeMemoryPoolSize> MemoryPoolSizes => TrtexecOptions.DeploymentOptions.MemoryPoolSizes;

    public string InputIOFormats => TrtexecOptions.DeploymentOptions.InputIOFormats;

    public string OutputIOFormats => TrtexecOptions.DeploymentOptions.OutputIOFormats;

    public string CalibrationCacheFile => TrtexecOptions.DeploymentOptions.CalibrationCacheFile;

    public bool DirectIO => TrtexecOptions.DeploymentOptions.DirectIO;

    public string Sparsity => TrtexecOptions.DeploymentOptions.Sparsity;

    public bool StronglyTyped => TrtexecOptions.DeploymentOptions.StronglyTyped;

    public int? MinTiming => TrtexecOptions.DeploymentOptions.MinTiming;

    public int? AvgTiming => TrtexecOptions.DeploymentOptions.AvgTiming;

    public string PrecisionConstraints => TrtexecOptions.DeploymentOptions.PrecisionConstraints;

    public string LayerPrecisions => TrtexecOptions.DeploymentOptions.LayerPrecisions;

    public string LayerOutputTypes => TrtexecOptions.DeploymentOptions.LayerOutputTypes;

    public bool VersionCompatible => TrtexecOptions.DeploymentOptions.VersionCompatible;

    public bool ExcludeLeanRuntime => TrtexecOptions.DeploymentOptions.ExcludeLeanRuntime;

    public bool StripWeights => TrtexecOptions.DeploymentOptions.StripWeights;

    public bool Refit => TrtexecOptions.DeploymentOptions.Refit;

    public ulong? WeightStreamingBudgetBytes => TrtexecOptions.DeploymentOptions.WeightStreamingBudgetBytes;

    public TrtexecLikeWeightStreamingBudget WeightStreamingBudget => TrtexecOptions.DeploymentOptions.WeightStreamingBudget;

    public string ExportTimingCachePath => TrtexecOptions.DeploymentOptions.ExportTimingCachePath;

    public bool Safe => TrtexecOptions.DeploymentOptions.Safe;

    public bool Consistency => TrtexecOptions.DeploymentOptions.Consistency;

    public bool BuilderCache => TrtexecOptions.DeploymentOptions.BuilderCache;

    public bool NoBuilderCache => TrtexecOptions.DeploymentOptions.NoBuilderCache;

    public int? MaxNbTactics => TrtexecOptions.DeploymentOptions.MaxNbTactics;

    public TensorRtTilingOptimizationLevel? TilingOptimizationLevel => TrtexecOptions.DeploymentOptions.TilingOptimizationLevel;

    public long? L2LimitForTilingBytes => TrtexecOptions.DeploymentOptions.L2LimitForTilingBytes;

    public TensorRtQuantizationFlags? QuantizationFlags => TrtexecOptions.DeploymentOptions.QuantizationFlags;

    public string ExportLayerInfoPath => TrtexecOptions.ExportLayerInfoPath;

    public string ExportReportPath => TrtexecOptions.ExportReportPath;

    public string EvidenceSidecarPath => TrtexecOptions.EvidenceSidecarPath;

    public TrtexecLikeRuntimeOptions RuntimeOptions => TrtexecOptions.RuntimeOptions;

    public int? InfStreams => TrtexecOptions.RuntimeOptions.InfStreams;

    public static TensorRtExecOptions Parse(string[] args)
    {
        return new TensorRtExecOptions(TrtexecLikeParser.Parse(args));
    }

    public string ToArgumentLine()
    {
        return TrtexecOptions.ToArgumentLine();
    }

    private static void Add(List<string> args, string name, string value)
    {
        if (!string.IsNullOrWhiteSpace(value))
        {
            args.Add(name);
            args.Add(value);
        }
    }

    private static void AddSwitch(List<string> args, string name, bool enabled)
    {
        if (enabled)
        {
            args.Add(name);
        }
    }

    private static string FormatNullable(int? value)
    {
        return value.HasValue ? value.Value.ToString(CultureInfo.InvariantCulture) : string.Empty;
    }

    private static string FormatNullable(float? value)
    {
        return value.HasValue ? value.Value.ToString(CultureInfo.InvariantCulture) : string.Empty;
    }

    private static string FormatBytesMiB(ulong? value)
    {
        if (!value.HasValue)
        {
            return string.Empty;
        }

        const ulong mib = 1024UL * 1024UL;
        return value.Value % mib == 0
            ? (value.Value / mib).ToString(CultureInfo.InvariantCulture)
            : value.Value.ToString(CultureInfo.InvariantCulture) + "B";
    }

    private static string FormatBytes(ulong? value)
    {
        return value.HasValue ? value.Value.ToString(CultureInfo.InvariantCulture) + "B" : string.Empty;
    }

}
