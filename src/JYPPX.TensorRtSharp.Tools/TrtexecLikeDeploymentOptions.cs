using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class TrtexecLikeDeploymentOptions
{
    public TrtexecLikeDeploymentOptions(
        int? deviceOrdinal,
        int builderOptimizationLevel,
        int? maxAuxStreams,
        int? dlaCore,
        bool allowGpuFallback,
        string tacticSources,
        IReadOnlyList<TrtexecLikeMemoryPoolSize> memoryPoolSizes,
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
        int? maxNbTactics = null,
        TensorRtTilingOptimizationLevel? tilingOptimizationLevel = null,
        long? l2LimitForTilingBytes = null,
        TensorRtQuantizationFlags? quantizationFlags = null)
    {
        DeviceOrdinal = deviceOrdinal;
        BuilderOptimizationLevel = builderOptimizationLevel;
        MaxAuxStreams = maxAuxStreams;
        DlaCore = dlaCore;
        AllowGpuFallback = allowGpuFallback;
        TacticSources = tacticSources ?? string.Empty;
        MemoryPoolSizes = memoryPoolSizes ?? Array.Empty<TrtexecLikeMemoryPoolSize>();
        InputIOFormats = inputIOFormats ?? string.Empty;
        OutputIOFormats = outputIOFormats ?? string.Empty;
        CalibrationCacheFile = calibrationCacheFile ?? string.Empty;
        DirectIO = directIO;
        Sparsity = sparsity ?? string.Empty;
        StronglyTyped = stronglyTyped;
        MinTiming = minTiming;
        AvgTiming = avgTiming;
        PrecisionConstraints = precisionConstraints ?? string.Empty;
        LayerPrecisions = layerPrecisions ?? string.Empty;
        LayerOutputTypes = layerOutputTypes ?? string.Empty;
        Fp8 = fp8;
        Best = best;
        DumpRefit = dumpRefit;
        AllowWeightStreaming = allowWeightStreaming;
        MarkDebug = markDebug ?? string.Empty;
        DumpDebugTensors = dumpDebugTensors;
        VersionCompatible = versionCompatible;
        ExcludeLeanRuntime = excludeLeanRuntime;
        StripWeights = stripWeights;
        Refit = refit;
        WeightStreamingBudgetBytes = weightStreamingBudgetBytes;
        ExportTimingCachePath = exportTimingCachePath ?? string.Empty;
        Safe = safe;
        Consistency = consistency;
        BuilderCache = builderCache;
        NoBuilderCache = noBuilderCache;
        MaxNbTactics = maxNbTactics;
        TilingOptimizationLevel = tilingOptimizationLevel;
        L2LimitForTilingBytes = l2LimitForTilingBytes;
        QuantizationFlags = quantizationFlags;
    }

    public static TrtexecLikeDeploymentOptions Default { get; } = new TrtexecLikeDeploymentOptions(
        null,
        3,
        null,
        null,
        false,
        string.Empty,
        Array.Empty<TrtexecLikeMemoryPoolSize>(),
        string.Empty,
        string.Empty,
        string.Empty,
        false,
        string.Empty,
        false,
        null,
        null,
        string.Empty,
        string.Empty,
        string.Empty,
        false,
        false,
        false,
        false,
        string.Empty,
        false,
        false,
        false,
        false,
        false,
        null,
        string.Empty,
        false,
        false,
        false,
        false);

    public int? DeviceOrdinal { get; }

    public int BuilderOptimizationLevel { get; }

    public int? MaxAuxStreams { get; }

    public int? DlaCore { get; }

    public bool AllowGpuFallback { get; }

    public string TacticSources { get; }

    public IReadOnlyList<TrtexecLikeMemoryPoolSize> MemoryPoolSizes { get; }

    public string InputIOFormats { get; }

    public string OutputIOFormats { get; }

    public string CalibrationCacheFile { get; }

    public bool DirectIO { get; }

    public string Sparsity { get; }

    public bool StronglyTyped { get; }

    public int? MinTiming { get; }

    public int? AvgTiming { get; }

    public string PrecisionConstraints { get; }

    public string LayerPrecisions { get; }

    public string LayerOutputTypes { get; }

    public bool Fp8 { get; }

    public bool Best { get; }

    public bool DumpRefit { get; }

    public bool AllowWeightStreaming { get; }

    public string MarkDebug { get; }

    public bool DumpDebugTensors { get; }

    public bool VersionCompatible { get; }

    public bool ExcludeLeanRuntime { get; }

    public bool StripWeights { get; }

    public bool Refit { get; }

    public ulong? WeightStreamingBudgetBytes { get; }

    public string ExportTimingCachePath { get; }

    public bool Safe { get; }

    public bool Consistency { get; }

    public bool BuilderCache { get; }

    public bool NoBuilderCache { get; }

    public int? MaxNbTactics { get; }

    public TensorRtTilingOptimizationLevel? TilingOptimizationLevel { get; }

    public long? L2LimitForTilingBytes { get; }

    public TensorRtQuantizationFlags? QuantizationFlags { get; }

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
        Add(args, "--weightStreamingBudget", FormatBytesMiB(WeightStreamingBudgetBytes));
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

    public IReadOnlyList<string> ToDiagnostics()
    {
        List<string> diagnostics = new List<string>
        {
            "BuilderOptimizationLevel=" + BuilderOptimizationLevel + " is applied to TensorRT builder config."
        };

        if (MaxAuxStreams.HasValue)
        {
            diagnostics.Add("MaxAuxStreams=" + MaxAuxStreams.Value + " is applied to TensorRT builder config.");
        }

        if (DeviceOrdinal.HasValue)
        {
            diagnostics.Add("Device=" + DeviceOrdinal.Value + " is recorded for deployment diagnostics; this generic build service does not change the process CUDA device.");
        }

        if (DlaCore.HasValue || AllowGpuFallback)
        {
            diagnostics.Add("DLA options are parsed for deployment diagnostics; the TensorRT 11 build path applies parser capability validation and builder-config device placement, while older adapters still require a model-specific runtime stage.");
            if (DlaCore.HasValue)
            {
                diagnostics.Add("UseDLACore=" + DlaCore.Value);
            }

            if (AllowGpuFallback)
            {
                diagnostics.Add("AllowGPUFallback=True");
            }
        }

        if (!string.IsNullOrWhiteSpace(TacticSources))
        {
            diagnostics.Add("TacticSources=" + TacticSources + " is recorded; tactic-source enforcement remains a deployment boundary option.");
        }

        if (MemoryPoolSizes.Count > 0)
        {
            diagnostics.Add("MemPoolSize=" + MemoryPoolSizesToArgument() + " is applied during a real build through TensorRtBuilderConfig.SetMemoryPoolLimit and read back through GetMemoryPoolLimit; dry-run and load-engine remain parse-only.");
        }

        if (!string.IsNullOrWhiteSpace(InputIOFormats) || !string.IsNullOrWhiteSpace(OutputIOFormats) || DirectIO)
        {
            diagnostics.Add("IO format options are parsed for diagnostics; generic external-model inference still requires explicit binding semantics.");
            AddDiagnostic(diagnostics, "InputIOFormats", InputIOFormats);
            AddDiagnostic(diagnostics, "OutputIOFormats", OutputIOFormats);
            if (DirectIO)
            {
                diagnostics.Add("DirectIO=True");
            }
        }

        if (!string.IsNullOrWhiteSpace(CalibrationCacheFile))
        {
            diagnostics.Add("Calibration cache path is recorded; INT8 calibrator callbacks and cache lifecycle are not promoted by this build-report stage.");
            diagnostics.Add("CalibrationCacheFile=" + CalibrationCacheFile);
        }

        if (!string.IsNullOrWhiteSpace(Sparsity))
        {
            diagnostics.Add("Sparsity=" + Sparsity + " is parsed for deployment diagnostics.");
        }

        if (StronglyTyped)
        {
            diagnostics.Add("StronglyTyped=True is parsed for deployment diagnostics; network creation flags remain explicit-batch in this stage.");
        }

        if (MinTiming.HasValue || AvgTiming.HasValue)
        {
            diagnostics.Add("Timing iteration controls are parsed for trtexec alignment; --avgTiming and TRT8 --minTiming are applied/read back during a real build, while TRT10/11 --minTiming and full tactic timing policy remain parse-only.");
            AddDiagnostic(diagnostics, "MinTiming", MinTiming);
            AddDiagnostic(diagnostics, "AvgTiming", AvgTiming);
        }

        if (!string.IsNullOrWhiteSpace(PrecisionConstraints) ||
            !string.IsNullOrWhiteSpace(LayerPrecisions) ||
            !string.IsNullOrWhiteSpace(LayerOutputTypes) ||
            Fp8 ||
            Best)
        {
            diagnostics.Add("Layer precision and shortcut precision arguments are parse/report-only until model-specific precision routing is promoted.");
            AddDiagnostic(diagnostics, "PrecisionConstraints", PrecisionConstraints);
            AddDiagnostic(diagnostics, "LayerPrecisions", LayerPrecisions);
            AddDiagnostic(diagnostics, "LayerOutputTypes", LayerOutputTypes);
            AddDiagnostic(diagnostics, "Fp8", Fp8);
            AddDiagnostic(diagnostics, "Best", Best);
        }

        if (DumpRefit || AllowWeightStreaming || !string.IsNullOrWhiteSpace(MarkDebug) || DumpDebugTensors)
        {
            diagnostics.Add("Refit, weight-streaming, and debug tensor diagnostic arguments are parse/report-only until model-specific ownership and runtime proof are promoted.");
            AddDiagnostic(diagnostics, "DumpRefit", DumpRefit);
            AddDiagnostic(diagnostics, "AllowWeightStreaming", AllowWeightStreaming);
            AddDiagnostic(diagnostics, "MarkDebug", MarkDebug);
            AddDiagnostic(diagnostics, "DumpDebugTensors", DumpDebugTensors);
        }

        if (VersionCompatible || ExcludeLeanRuntime || StripWeights || Refit || WeightStreamingBudgetBytes.HasValue || AllowWeightStreaming)
        {
            diagnostics.Add("Advanced engine packaging/refit/weight-streaming arguments are parse/report-only in this application stage.");
            if (VersionCompatible)
            {
                diagnostics.Add("VersionCompatible=True");
            }

            if (ExcludeLeanRuntime)
            {
                diagnostics.Add("ExcludeLeanRuntime=True");
            }

            if (StripWeights)
            {
                diagnostics.Add("StripWeights=True");
            }

            if (Refit)
            {
                diagnostics.Add("Refit=True");
            }

            if (AllowWeightStreaming)
            {
                diagnostics.Add("AllowWeightStreaming=True");
            }

            if (WeightStreamingBudgetBytes.HasValue)
            {
                diagnostics.Add("WeightStreamingBudgetBytes=" + WeightStreamingBudgetBytes.Value.ToString(CultureInfo.InvariantCulture));
            }
        }

        if (!string.IsNullOrWhiteSpace(ExportTimingCachePath))
        {
            diagnostics.Add("ExportTimingCache=" + ExportTimingCachePath + " is written after a successful TensorRT build through the typed timing-cache owner; the artifact remains build-cache evidence only.");
        }

        if (Safe || Consistency)
        {
            diagnostics.Add("Safety/consistency arguments are parse/report-only; they require model-specific validation before promotion.");
            if (Safe)
            {
                diagnostics.Add("Safe=True");
            }

            if (Consistency)
            {
                diagnostics.Add("Consistency=True");
            }
        }

        if (BuilderCache || NoBuilderCache)
        {
            diagnostics.Add("Builder cache policy arguments are parse/report-only; builder cache lifecycle is not promoted by this application stage.");
            if (BuilderCache)
            {
                diagnostics.Add("BuilderCache=True");
            }

            if (NoBuilderCache)
            {
                diagnostics.Add("NoBuilderCache=True");
            }
        }

        if (MaxNbTactics.HasValue || TilingOptimizationLevel.HasValue || L2LimitForTilingBytes.HasValue || QuantizationFlags.HasValue)
        {
            diagnostics.Add("Builder scalar deployment controls are applied/read back during a real build when the selected TensorRT line exposes the corresponding vendor API; unsupported lines remain controlled diagnostics.");
            AddDiagnostic(diagnostics, "MaxNbTactics", MaxNbTactics);
            AddDiagnostic(diagnostics, "TilingOptimizationLevel", TilingOptimizationLevel?.ToString() ?? string.Empty);
            AddDiagnostic(diagnostics, "L2LimitForTilingBytes", L2LimitForTilingBytes);
            AddDiagnostic(diagnostics, "QuantizationFlags", QuantizationFlags?.ToString() ?? string.Empty);
        }

        return diagnostics;
    }

    public string ScalarControlSummary => $"MaxNbTactics={MaxNbTactics?.ToString(CultureInfo.InvariantCulture) ?? string.Empty};TilingOptimizationLevel={TilingOptimizationLevel?.ToString() ?? string.Empty};L2LimitForTilingBytes={L2LimitForTilingBytes?.ToString(CultureInfo.InvariantCulture) ?? string.Empty};QuantizationFlags={QuantizationFlags?.ToString() ?? string.Empty}";

    private string MemoryPoolSizesToArgument()
    {
        return MemoryPoolSizes.Count == 0
            ? string.Empty
            : string.Join(",", MemoryPoolSizes.Select(static item => item.ToArgumentSegment()));
    }

    private static void AddDiagnostic(List<string> diagnostics, string name, string value)
    {
        if (!string.IsNullOrWhiteSpace(value))
        {
            diagnostics.Add(name + "=" + value);
        }
    }

    private static void AddDiagnostic(List<string> diagnostics, string name, bool enabled)
    {
        if (enabled)
        {
            diagnostics.Add(name + "=True");
        }
    }

    private static void AddDiagnostic(List<string> diagnostics, string name, int? value)
    {
        if (value.HasValue)
        {
            diagnostics.Add(name + "=" + value.Value.ToString(CultureInfo.InvariantCulture));
        }
    }

    private static void AddDiagnostic(List<string> diagnostics, string name, long? value)
    {
        if (value.HasValue)
        {
            diagnostics.Add(name + "=" + value.Value.ToString(CultureInfo.InvariantCulture));
        }
    }

    private static void Add(List<string> args, string name, string value)
    {
        if (!string.IsNullOrWhiteSpace(value))
        {
            args.Add(name);
            args.Add(QuoteIfNeeded(value));
        }
    }

    private static void AddSwitch(List<string> args, string name, bool enabled)
    {
        if (enabled)
        {
            args.Add(name);
        }
    }

    private static string QuoteIfNeeded(string value)
    {
        return value.IndexOf(' ') >= 0 ? "\"" + value + "\"" : value;
    }

    private static string FormatNullable(int? value)
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

    private static string FormatBytes(long? value)
    {
        return value.HasValue
            ? value.Value.ToString(CultureInfo.InvariantCulture) + "B"
            : string.Empty;
    }
}

public sealed class TrtexecLikeMemoryPoolSize
{
    public TrtexecLikeMemoryPoolSize(string name, ulong sizeMiB)
    {
        Name = string.IsNullOrWhiteSpace(name) ? throw new ArgumentException("Memory pool name is required.", nameof(name)) : name.Trim();
        SizeMiB = sizeMiB;
        SizeBytes = checked(sizeMiB * 1024UL * 1024UL);
    }

    public string Name { get; }

    public ulong SizeMiB { get; }

    public ulong SizeBytes { get; }

    /// <summary>
    /// Resolves the trtexec pool token to the typed TensorRT memory-pool enum.
    /// 将 trtexec memory-pool token 解析为强类型 TensorRT memory-pool 枚举。
    /// </summary>
    public TensorRtMemoryPoolType ToTensorRtMemoryPoolType()
    {
        string normalized = Name
            .Replace("-", string.Empty, StringComparison.Ordinal)
            .Replace("_", string.Empty, StringComparison.Ordinal)
            .Replace(" ", string.Empty, StringComparison.Ordinal)
            .ToLowerInvariant();

        return normalized switch
        {
            "workspace" => TensorRtMemoryPoolType.Workspace,
            "dlasram" or "dlamanagedsram" => TensorRtMemoryPoolType.DlaManagedSram,
            "dlalocaldram" => TensorRtMemoryPoolType.DlaLocalDram,
            "dlaglobaldram" => TensorRtMemoryPoolType.DlaGlobalDram,
            "tacticdram" => TensorRtMemoryPoolType.TacticDram,
            "tacticsharedmem" or "tacticsharedmemory" => TensorRtMemoryPoolType.TacticSharedMemory,
            _ => throw new ArgumentException(
                $"Unsupported TensorRT memory pool '{Name}'. Supported pools: workspace, dlaSRAM, dlaLocalDRAM, dlaGlobalDRAM, tacticDRAM, tacticSharedMem.",
                nameof(Name))
        };
    }

    public string ToArgumentSegment()
    {
        return Name + ":" + SizeMiB.ToString(CultureInfo.InvariantCulture);
    }
}
