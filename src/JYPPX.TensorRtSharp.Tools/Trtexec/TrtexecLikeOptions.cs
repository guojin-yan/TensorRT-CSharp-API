using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class TrtexecLikeOptions
{
    public TrtexecLikeOptions(
        TensorRtApiLine tensorRtLine,
        string onnxPath,
        string saveEnginePath,
        string loadEnginePath,
        bool explicitBatch,
        bool fp16,
        bool int8,
        bool bf16,
        bool tf32,
        ulong workspaceBytes,
        string timingCacheFile,
        IReadOnlyList<string> plugins,
        string profilingVerbosity,
        bool dumpLayerInfo,
        string exportLayerInfoPath,
        bool dumpProfile,
        bool separateProfileRun,
        bool buildOnly,
        bool skipInference,
        bool dryRun,
        int iterations,
        int warmUpMilliseconds,
        int durationSeconds,
        int streams,
        bool useCudaGraph,
        int batch,
        EngineBuildProfile shapeProfile,
        string exportReportPath,
        string evidenceSidecarPath,
        TrtexecLikeDeploymentOptions deploymentOptions,
        TrtexecLikeRuntimeOptions runtimeOptions,
        IReadOnlyList<string> diagnostics)
    {
        TensorRtLine = tensorRtLine;
        OnnxPath = onnxPath ?? string.Empty;
        SaveEnginePath = saveEnginePath ?? string.Empty;
        LoadEnginePath = loadEnginePath ?? string.Empty;
        ExplicitBatch = explicitBatch;
        Fp16 = fp16;
        Int8 = int8;
        Bf16 = bf16;
        Tf32 = tf32;
        WorkspaceBytes = workspaceBytes;
        TimingCacheFile = timingCacheFile ?? string.Empty;
        Plugins = plugins ?? Array.Empty<string>();
        ProfilingVerbosity = profilingVerbosity ?? string.Empty;
        DumpLayerInfo = dumpLayerInfo;
        ExportLayerInfoPath = exportLayerInfoPath ?? string.Empty;
        DumpProfile = dumpProfile;
        SeparateProfileRun = separateProfileRun;
        BuildOnly = buildOnly;
        SkipInference = skipInference;
        DryRun = dryRun;
        Iterations = iterations;
        WarmUpMilliseconds = warmUpMilliseconds;
        DurationSeconds = durationSeconds;
        Streams = streams;
        UseCudaGraph = useCudaGraph;
        Batch = batch;
        ShapeProfile = shapeProfile ?? new EngineBuildProfile(Array.Empty<EngineBuildShape>(), Array.Empty<EngineBuildShape>(), Array.Empty<EngineBuildShape>());
        ExportReportPath = exportReportPath ?? string.Empty;
        EvidenceSidecarPath = evidenceSidecarPath ?? string.Empty;
        DeploymentOptions = deploymentOptions ?? TrtexecLikeDeploymentOptions.Default;
        RuntimeOptions = runtimeOptions ?? TrtexecLikeRuntimeOptions.Default;
        Diagnostics = diagnostics ?? Array.Empty<string>();
    }

    public TensorRtApiLine TensorRtLine { get; }

    public string OnnxPath { get; }

    public string SaveEnginePath { get; }

    public string LoadEnginePath { get; }

    public bool ExplicitBatch { get; }

    public bool Fp16 { get; }

    public bool Int8 { get; }

    public bool Bf16 { get; }

    public bool Tf32 { get; }

    public ulong WorkspaceBytes { get; }

    public string TimingCacheFile { get; }

    public IReadOnlyList<string> Plugins { get; }

    public string ProfilingVerbosity { get; }

    public bool DumpLayerInfo { get; }

    public string ExportLayerInfoPath { get; }

    public bool DumpProfile { get; }

    public bool SeparateProfileRun { get; }

    public bool BuildOnly { get; }

    public bool SkipInference { get; }

    public bool DryRun { get; }

    public int Iterations { get; }

    public int WarmUpMilliseconds { get; }

    public int DurationSeconds { get; }

    public int Streams { get; }

    public bool UseCudaGraph { get; }

    public int Batch { get; }

    public EngineBuildProfile ShapeProfile { get; }

    public string ExportReportPath { get; }

    public string EvidenceSidecarPath { get; }

    public TrtexecLikeDeploymentOptions DeploymentOptions { get; }

    public TrtexecLikeRuntimeOptions RuntimeOptions { get; }

    public IReadOnlyList<string> Diagnostics { get; }

    public bool UsesExternalOnnx => !string.IsNullOrWhiteSpace(OnnxPath);

    public bool LoadsExistingEngine => !string.IsNullOrWhiteSpace(LoadEnginePath);

    public string ToArgumentLine()
    {
        List<string> args = new List<string>();
        Add(args, "--tensor-rt-line", ((int)TensorRtLine).ToString(CultureInfo.InvariantCulture));
        Add(args, "--onnx", OnnxPath);
        Add(args, "--saveEngine", SaveEnginePath);
        Add(args, "--loadEngine", LoadEnginePath);
        Add(args, "--workspace", (WorkspaceBytes / (1024UL * 1024UL)).ToString(CultureInfo.InvariantCulture));
        Add(args, "--minShapes", ShapeListToString(ShapeProfile.MinShapes));
        Add(args, "--optShapes", ShapeListToString(ShapeProfile.OptShapes));
        Add(args, "--maxShapes", ShapeListToString(ShapeProfile.MaxShapes));
        Add(args, "--timingCacheFile", TimingCacheFile);
        Add(args, "--profilingVerbosity", ProfilingVerbosity);
        Add(args, "--exportLayerInfo", ExportLayerInfoPath);
        Add(args, "--exportReport", ExportReportPath);
        Add(args, "--evidenceSidecar", EvidenceSidecarPath);
        if (Plugins.Count > 0)
        {
            Add(args, "--plugins", string.Join(";", Plugins));
        }

        Add(args, "--batch", Batch.ToString(CultureInfo.InvariantCulture));
        Add(args, "--iterations", Iterations.ToString(CultureInfo.InvariantCulture));
        Add(args, "--warmUp", WarmUpMilliseconds.ToString(CultureInfo.InvariantCulture));
        Add(args, "--duration", DurationSeconds.ToString(CultureInfo.InvariantCulture));
        Add(args, "--streams", Streams.ToString(CultureInfo.InvariantCulture));
        args.AddRange(DeploymentOptions.ToArgumentSegments());
        args.AddRange(RuntimeOptions.ToArgumentSegments());
        AddSwitch(args, "--implicitBatch", !ExplicitBatch);
        AddSwitch(args, "--fp16", Fp16);
        AddSwitch(args, "--int8", Int8);
        AddSwitch(args, "--bf16", Bf16);
        AddSwitch(args, "--noTF32", !Tf32);
        AddSwitch(args, "--dumpLayerInfo", DumpLayerInfo);
        AddSwitch(args, "--dumpProfile", DumpProfile);
        AddSwitch(args, "--separateProfileRun", SeparateProfileRun);
        AddSwitch(args, "--buildOnly", BuildOnly);
        AddSwitch(args, "--skipInference", SkipInference);
        AddSwitch(args, "--dryRun", DryRun);
        AddSwitch(args, "--useCudaGraph", UseCudaGraph);
        return string.Join(" ", args);
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

    private static string ShapeListToString(IReadOnlyList<EngineBuildShape> shapes)
    {
        return shapes == null || shapes.Count == 0 ? string.Empty : string.Join(",", shapes.Select(static shape => shape.ToString()));
    }

    private static string QuoteIfNeeded(string value)
    {
        return value.IndexOf(' ') >= 0 ? "\"" + value + "\"" : value;
    }
}
