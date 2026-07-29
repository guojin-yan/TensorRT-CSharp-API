using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class TrtexecLikeDeploymentOptions
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
        TensorRtQuantizationFlags? quantizationFlags = null,
        TrtexecLikeWeightStreamingBudget? weightStreamingBudget = null,
        string refitFromOnnxPath = "",
        string saveRefittedEnginePath = "")
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
        WeightStreamingBudget = weightStreamingBudget ?? (weightStreamingBudgetBytes.HasValue
            ? TrtexecLikeWeightStreamingBudget.FromBytes(weightStreamingBudgetBytes.Value)
            : TrtexecLikeWeightStreamingBudget.Unspecified);
        WeightStreamingBudgetBytes = WeightStreamingBudget.Bytes;
        ExportTimingCachePath = exportTimingCachePath ?? string.Empty;
        Safe = safe;
        Consistency = consistency;
        BuilderCache = builderCache;
        NoBuilderCache = noBuilderCache;
        MaxNbTactics = maxNbTactics;
        TilingOptimizationLevel = tilingOptimizationLevel;
        L2LimitForTilingBytes = l2LimitForTilingBytes;
        QuantizationFlags = quantizationFlags;
        RefitFromOnnxPath = refitFromOnnxPath ?? string.Empty;
        SaveRefittedEnginePath = saveRefittedEnginePath ?? string.Empty;
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

    /// <summary>
    /// Gets the normalized trtexec weight-streaming budget specification.
    /// 获取规范化后的 trtexec 权重流式加载预算规范。
    /// </summary>
    public TrtexecLikeWeightStreamingBudget WeightStreamingBudget { get; }

    public string ExportTimingCachePath { get; }

    public bool Safe { get; }

    public bool Consistency { get; }

    public bool BuilderCache { get; }

    public bool NoBuilderCache { get; }

    public int? MaxNbTactics { get; }

    public TensorRtTilingOptimizationLevel? TilingOptimizationLevel { get; }

    public long? L2LimitForTilingBytes { get; }

    public TensorRtQuantizationFlags? QuantizationFlags { get; }

    /// <summary>
    /// Gets the explicit ONNX weight source used to refit a stripped plan before inference.
    /// 获取在推理前重整 stripped plan 使用的显式 ONNX 权重源。
    /// </summary>
    public string RefitFromOnnxPath { get; }

    /// <summary>
    /// Gets the explicit output path used to persist and independently reload a refitted engine.
    /// 获取用于持久化并独立重新加载 refitted engine 的显式输出路径。
    /// </summary>
    public string SaveRefittedEnginePath { get; }
}
