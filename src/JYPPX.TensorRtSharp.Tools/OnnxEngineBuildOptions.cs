using System;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class OnnxEngineBuildOptions
{
    public OnnxEngineBuildOptions(
        TensorRtApiLine tensorRtLine,
        string onnxPath,
        string saveEnginePath,
        string loadEnginePath,
        bool fp16,
        bool int8,
        bool bf16,
        bool tf32,
        ulong workspaceBytes,
        string timingCacheFile,
        bool buildOnly,
        bool skipInference,
        bool dryRun,
        int batch,
        EngineBuildProfile shapeProfile,
        string exportReportPath,
        string evidenceSidecarPath,
        string profilingVerbosity,
        TrtexecLikeDeploymentOptions deploymentOptions,
        TrtexecLikeRuntimeOptions runtimeOptions,
        string normalizedCommandLine,
        string[] diagnostics)
    {
        TensorRtLine = tensorRtLine;
        OnnxPath = onnxPath ?? string.Empty;
        SaveEnginePath = saveEnginePath ?? string.Empty;
        LoadEnginePath = loadEnginePath ?? string.Empty;
        Fp16 = fp16;
        Int8 = int8;
        Bf16 = bf16;
        Tf32 = tf32;
        WorkspaceBytes = workspaceBytes;
        TimingCacheFile = timingCacheFile ?? string.Empty;
        BuildOnly = buildOnly;
        SkipInference = skipInference;
        DryRun = dryRun;
        Batch = batch;
        ShapeProfile = shapeProfile ?? new EngineBuildProfile(Array.Empty<EngineBuildShape>(), Array.Empty<EngineBuildShape>(), Array.Empty<EngineBuildShape>());
        ExportReportPath = exportReportPath ?? string.Empty;
        EvidenceSidecarPath = evidenceSidecarPath ?? string.Empty;
        ProfilingVerbosity = profilingVerbosity ?? string.Empty;
        DeploymentOptions = deploymentOptions ?? TrtexecLikeDeploymentOptions.Default;
        RuntimeOptions = runtimeOptions ?? TrtexecLikeRuntimeOptions.Default;
        NormalizedCommandLine = normalizedCommandLine ?? string.Empty;
        Diagnostics = diagnostics ?? Array.Empty<string>();
    }

    public TensorRtApiLine TensorRtLine { get; }

    public string OnnxPath { get; }

    public string SaveEnginePath { get; }

    public string LoadEnginePath { get; }

    public bool Fp16 { get; }

    public bool Int8 { get; }

    public bool Bf16 { get; }

    public bool Tf32 { get; }

    public ulong WorkspaceBytes { get; }

    public string TimingCacheFile { get; }

    public bool BuildOnly { get; }

    public bool SkipInference { get; }

    public bool DryRun { get; }

    public int Batch { get; }

    public EngineBuildProfile ShapeProfile { get; }

    public string ExportReportPath { get; }

    public string EvidenceSidecarPath { get; }

    public string ProfilingVerbosity { get; }

    public TrtexecLikeDeploymentOptions DeploymentOptions { get; }

    public TrtexecLikeRuntimeOptions RuntimeOptions { get; }

    public string NormalizedCommandLine { get; }

    public string[] Diagnostics { get; }

    public bool UsesExternalOnnx => !string.IsNullOrWhiteSpace(OnnxPath);

    public bool LoadsExistingEngine => !string.IsNullOrWhiteSpace(LoadEnginePath);

    public static OnnxEngineBuildOptions FromTrtexecLikeOptions(TrtexecLikeOptions options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        string[] diagnostics = CreateDiagnostics(options);

        return new OnnxEngineBuildOptions(
            options.TensorRtLine,
            options.OnnxPath,
            options.SaveEnginePath,
            options.LoadEnginePath,
            options.Fp16,
            options.Int8,
            options.Bf16,
            options.Tf32,
            options.WorkspaceBytes,
            options.TimingCacheFile,
            options.BuildOnly,
            options.SkipInference,
            options.DryRun,
            options.Batch,
            options.ShapeProfile,
            options.ExportReportPath,
            options.EvidenceSidecarPath,
            options.ProfilingVerbosity,
            options.DeploymentOptions,
            options.RuntimeOptions,
            options.ToArgumentLine(),
            diagnostics);
    }

    private static string[] CreateDiagnostics(TrtexecLikeOptions options)
    {
        System.Collections.Generic.List<string> diagnostics = new System.Collections.Generic.List<string>(options.Diagnostics);
        diagnostics.AddRange(options.DeploymentOptions.ToDiagnostics());
        diagnostics.AddRange(options.RuntimeOptions.ToDiagnostics());
        if (options.Plugins.Count > 0)
        {
            diagnostics.Add("Plugin library arguments are recorded for deployment diagnostics; this safe stage does not load plugin libraries.");
            diagnostics.Add("Plugins=" + string.Join(";", options.Plugins));
        }

        if (!string.IsNullOrWhiteSpace(options.TimingCacheFile))
        {
            diagnostics.Add("Timing cache input is imported into the typed TensorRT builder-config timing-cache owner during a real build.");
            diagnostics.Add("TimingCacheFile=" + options.TimingCacheFile);
        }

        if (!string.IsNullOrWhiteSpace(options.ProfilingVerbosity))
        {
            diagnostics.Add("ProfilingVerbosity=" + options.ProfilingVerbosity);
        }

        if (options.DumpLayerInfo || !string.IsNullOrWhiteSpace(options.ExportLayerInfoPath))
        {
            diagnostics.Add("Layer-info export arguments are recorded for diagnostics; full layer-info export depends on TensorRT runtime support.");
            if (!string.IsNullOrWhiteSpace(options.ExportLayerInfoPath))
            {
                diagnostics.Add("ExportLayerInfo=" + options.ExportLayerInfoPath);
            }
        }

        if (options.DumpProfile || options.SeparateProfileRun)
        {
            diagnostics.Add("Profiling run arguments are parsed; benchmark/profile execution remains separate from build-only conversion proof.");
        }

        bool parseOnlyAdvancedOptions =
            (options.DeploymentOptions.MinTiming.HasValue && options.TensorRtLine != TensorRtApiLine.TensorRt8) ||
            !string.IsNullOrWhiteSpace(options.DeploymentOptions.PrecisionConstraints) ||
            !string.IsNullOrWhiteSpace(options.DeploymentOptions.LayerPrecisions) ||
            !string.IsNullOrWhiteSpace(options.DeploymentOptions.LayerOutputTypes) ||
            options.DeploymentOptions.Fp8 ||
            options.DeploymentOptions.Best ||
            options.DeploymentOptions.DumpRefit ||
            options.DeploymentOptions.AllowWeightStreaming ||
            !string.IsNullOrWhiteSpace(options.DeploymentOptions.MarkDebug) ||
            options.DeploymentOptions.DumpDebugTensors ||
            options.DeploymentOptions.VersionCompatible ||
            options.DeploymentOptions.ExcludeLeanRuntime ||
            options.DeploymentOptions.StripWeights ||
            options.DeploymentOptions.Refit ||
            options.DeploymentOptions.WeightStreamingBudgetBytes.HasValue ||
            options.DeploymentOptions.Safe ||
            options.DeploymentOptions.Consistency ||
            options.DeploymentOptions.BuilderCache ||
            options.DeploymentOptions.NoBuilderCache ||
            options.RuntimeOptions.InfStreams.HasValue;

        if (options.DeploymentOptions.MinTiming.HasValue ||
            options.DeploymentOptions.AvgTiming.HasValue ||
            parseOnlyAdvancedOptions)
        {
            if (options.DeploymentOptions.AvgTiming.HasValue)
            {
                diagnostics.Add("Average timing iterations are applied through TensorRtBuilderConfig and read back during a real build; the result remains builder-config evidence, not runtime proof.");
            }

            if (options.DeploymentOptions.MinTiming.HasValue)
            {
                diagnostics.Add(options.TensorRtLine == TensorRtApiLine.TensorRt8
                    ? "Minimum timing iterations use the TensorRT 8 legacy compatibility setter and are read back during a real build; TensorRT 10/11 keep this option parse-only."
                    : "Minimum timing iterations remain parse-only on TensorRT 10/11 because the legacy minimum setter is not available on those API lines.");
            }

            if (parseOnlyAdvancedOptions)
            {
                diagnostics.Add("TrtexecAlignmentStatus=parse-only for advanced precision, runtime stream, debug tensor, safety/consistency, engine packaging, builder cache, refit, and weight-streaming options in this stage.");
            }
        }

        if (!string.IsNullOrWhiteSpace(options.EvidenceSidecarPath))
        {
            diagnostics.Add("Evidence sidecar argument is recorded for model asset diagnostics; build reports cannot claim package-consumer-runtime.");
            diagnostics.Add("EvidenceSidecar=" + options.EvidenceSidecarPath);
        }

        return diagnostics.ToArray();
    }
}
