using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class OnnxEngineBuildResult
{
    public OnnxEngineBuildResult(
        bool success,
        bool skipped,
        string state,
        TensorRtApiLine tensorRtLine,
        string modelSource,
        string enginePath,
        bool parsed,
        bool engineSaved,
        bool engineFileRoundTrip,
        bool inferenceRan,
        bool outputMatch,
        int profileIndex,
        float? elapsedMilliseconds,
        string skipReason,
        string normalizedCommandLine,
        IReadOnlyList<string> diagnostics,
        IReadOnlyList<string> logLines,
        OnnxEngineBuildEvidenceSidecar? evidenceSidecar = null,
        TrtexecLikeRuntimeOptions? runtimeOptions = null,
        OnnxEngineBenchmarkSummary? benchmarkSummary = null,
        OnnxEnginePreflightMetadata? preflightMetadata = null,
        OnnxLoadedEngineDiagnostics? loadedEngineDiagnostics = null,
        OnnxEngineTimingCacheArtifact? timingCacheArtifact = null,
        OnnxEngineCapabilityProbe? capabilityProbe = null,
        ulong workspaceBytes = 0,
        TensorRtBuilderConfigDeploymentSnapshot? builderConfigDeploymentSnapshot = null,
        OnnxEngineParserPreflightSnapshot? parserPreflightSnapshot = null,
        OnnxEngineRefitSnapshot? refitSnapshot = null,
        OnnxEngineRefitPersistenceSnapshot? refitPersistenceSnapshot = null,
        bool outputValidated = false,
        bool identityOutputMatch = false)
        : this(
            success,
            skipped,
            state,
            tensorRtLine,
            modelSource,
            enginePath,
            parsed,
            engineSaved,
            engineFileRoundTrip,
            inferenceRan,
            outputMatch,
            profileIndex,
            elapsedMilliseconds,
            skipReason,
            normalizedCommandLine,
            TrtexecLikeDeploymentOptions.Default,
            diagnostics,
            logLines,
            evidenceSidecar,
            runtimeOptions,
            benchmarkSummary,
            preflightMetadata,
            loadedEngineDiagnostics,
            timingCacheArtifact,
            capabilityProbe,
            workspaceBytes,
            builderConfigDeploymentSnapshot,
            parserPreflightSnapshot,
            refitSnapshot,
            refitPersistenceSnapshot,
            outputValidated,
            identityOutputMatch)
    {
    }

    public OnnxEngineBuildResult(
        bool success,
        bool skipped,
        string state,
        TensorRtApiLine tensorRtLine,
        string modelSource,
        string enginePath,
        bool parsed,
        bool engineSaved,
        bool engineFileRoundTrip,
        bool inferenceRan,
        bool outputMatch,
        int profileIndex,
        float? elapsedMilliseconds,
        string skipReason,
        string normalizedCommandLine,
        TrtexecLikeDeploymentOptions deploymentOptions,
        IReadOnlyList<string> diagnostics,
        IReadOnlyList<string> logLines,
        OnnxEngineBuildEvidenceSidecar? evidenceSidecar = null,
        TrtexecLikeRuntimeOptions? runtimeOptions = null,
        OnnxEngineBenchmarkSummary? benchmarkSummary = null,
        OnnxEnginePreflightMetadata? preflightMetadata = null,
        OnnxLoadedEngineDiagnostics? loadedEngineDiagnostics = null,
        OnnxEngineTimingCacheArtifact? timingCacheArtifact = null,
        OnnxEngineCapabilityProbe? capabilityProbe = null,
        ulong workspaceBytes = 0,
        TensorRtBuilderConfigDeploymentSnapshot? builderConfigDeploymentSnapshot = null,
        OnnxEngineParserPreflightSnapshot? parserPreflightSnapshot = null,
        OnnxEngineRefitSnapshot? refitSnapshot = null,
        OnnxEngineRefitPersistenceSnapshot? refitPersistenceSnapshot = null,
        bool outputValidated = false,
        bool identityOutputMatch = false)
    {
        Success = success;
        Skipped = skipped;
        State = state ?? string.Empty;
        TensorRtLine = tensorRtLine;
        ModelSource = modelSource ?? string.Empty;
        EnginePath = enginePath ?? string.Empty;
        Parsed = parsed;
        EngineSaved = engineSaved;
        EngineFileRoundTrip = engineFileRoundTrip;
        InferenceRan = inferenceRan;
        OutputMatch = outputMatch;
        OutputValidated = outputValidated;
        IdentityOutputMatch = identityOutputMatch;
        ProfileIndex = profileIndex;
        ElapsedMilliseconds = elapsedMilliseconds;
        SkipReason = skipReason ?? string.Empty;
        NormalizedCommandLine = normalizedCommandLine ?? string.Empty;
        DeploymentOptions = deploymentOptions ?? TrtexecLikeDeploymentOptions.Default;
        RuntimeOptions = runtimeOptions ?? TrtexecLikeRuntimeOptions.Default;
        BenchmarkSummary = benchmarkSummary ?? OnnxEngineBenchmarkSummary.Empty;
        PreflightMetadata = preflightMetadata ?? OnnxEnginePreflightMetadata.Empty;
        LoadedEngineDiagnostics = loadedEngineDiagnostics ?? OnnxLoadedEngineDiagnostics.Empty;
        TimingCacheArtifact = timingCacheArtifact ?? OnnxEngineTimingCacheArtifact.Empty;
        CapabilityProbe = capabilityProbe ?? OnnxEngineCapabilityProbe.Empty;
        WorkspaceBytes = workspaceBytes;
        BuilderConfigDeploymentSnapshot = builderConfigDeploymentSnapshot;
        ParserPreflightSnapshot = parserPreflightSnapshot ?? OnnxEngineParserPreflightSnapshot.Empty;
        RefitSnapshot = refitSnapshot ?? OnnxEngineRefitSnapshot.Empty;
        RefitPersistenceSnapshot = refitPersistenceSnapshot ?? OnnxEngineRefitPersistenceSnapshot.Empty;
        Diagnostics = diagnostics ?? Array.Empty<string>();
        LogLines = logLines ?? Array.Empty<string>();
        EvidenceSidecar = evidenceSidecar ?? OnnxEngineBuildEvidenceSidecarReader.Empty;
    }

    public bool Success { get; }

    public bool Skipped { get; }

    public string State { get; }

    public TensorRtApiLine TensorRtLine { get; }

    public string ModelSource { get; }

    public string EnginePath { get; }

    public bool Parsed { get; }

    public bool EngineSaved { get; }

    public bool EngineFileRoundTrip { get; }

    public bool InferenceRan { get; }

    public bool OutputMatch { get; }

    public bool OutputValidated { get; }

    public bool IdentityOutputMatch { get; }

    public int ProfileIndex { get; }

    public float? ElapsedMilliseconds { get; }

    public string SkipReason { get; }

    public string NormalizedCommandLine { get; }

    public string NormalizedCommandSha256 => ComputeSha256(NormalizedCommandLine);

    public TrtexecLikeDeploymentOptions DeploymentOptions { get; }

    public TrtexecLikeRuntimeOptions RuntimeOptions { get; }

    public OnnxEngineBenchmarkSummary BenchmarkSummary { get; }

    public OnnxEnginePreflightMetadata PreflightMetadata { get; }

    public OnnxLoadedEngineDiagnostics LoadedEngineDiagnostics { get; }

    public OnnxEngineTimingCacheArtifact TimingCacheArtifact { get; }

    public OnnxEngineCapabilityProbe CapabilityProbe { get; }

    public ulong WorkspaceBytes { get; }

    /// <summary>
    /// Gets builder-config values copied after deployment options were applied.
    /// 获取应用部署选项后复制读回的 builder-config 实际值。
    /// </summary>
    /// <remarks>
    /// This is build/deployment diagnostics only. It is not model runtime or package-consumer proof.
    /// 该数据只属于 build/deployment 诊断，不是模型 runtime 或 package-consumer proof。
    /// </remarks>
    public TensorRtBuilderConfigDeploymentSnapshot? BuilderConfigDeploymentSnapshot { get; }

    public OnnxEngineParserPreflightSnapshot ParserPreflightSnapshot { get; }

    /// <summary>
    /// Gets copied diagnostics for the optional ONNX stripped-plan refit lifecycle.
    /// 获取可选 ONNX stripped-plan 重整生命周期的复制诊断。
    /// </summary>
    public OnnxEngineRefitSnapshot RefitSnapshot { get; }

    /// <summary>
    /// Gets copied evidence for the optional persisted-plan independent reload lifecycle.
    /// 获取可选持久化 plan 独立重新加载生命周期的复制证据。
    /// </summary>
    public OnnxEngineRefitPersistenceSnapshot RefitPersistenceSnapshot { get; }

    public bool IsRuntimeExecutionProof => InferenceRan && OutputMatch;

    public bool BuildEvidenceOnly => string.Equals(ProofClassification, "build-only", StringComparison.Ordinal) ||
        string.Equals(ProofClassification, "dependency-probe-only", StringComparison.Ordinal) ||
        string.Equals(ProofClassification, "precheck", StringComparison.Ordinal);

    public string ProofClassification
    {
        get
        {
            if (IsRuntimeExecutionProof)
            {
                return "synthetic-input-runtime";
            }

            if (Skipped || State.Contains("preflight", StringComparison.OrdinalIgnoreCase))
            {
                return "dependency-probe-only";
            }

            if (State.Contains("dry-run", StringComparison.OrdinalIgnoreCase) ||
                State.Contains("preview", StringComparison.OrdinalIgnoreCase) ||
                State.Contains("precheck", StringComparison.OrdinalIgnoreCase))
            {
                return "precheck";
            }

            return "build-only";
        }
    }

    public IReadOnlyList<string> EvidenceClassifications { get; } = new[]
    {
        "build-only",
        "dependency-probe-only",
        "precheck",
        "synthetic-input-runtime",
        "real-model-runtime",
        "package-consumer-runtime"
    };

    public bool IsRealModelRuntimeProof => string.Equals(ProofClassification, "real-model-runtime", StringComparison.Ordinal);

    public bool IsPackageConsumerRuntimeProof => string.Equals(ProofClassification, "package-consumer-runtime", StringComparison.Ordinal);

    public string StdoutSummary => string.IsNullOrWhiteSpace(EvidenceSidecar.StdoutSummary) ? CreateSummary(LogLines) : EvidenceSidecar.StdoutSummary;

    public string StderrSummary => EvidenceSidecar.StderrSummary;

    public OnnxEngineBuildModelEvidence ModelEvidence => EvidenceSidecar.HasModelEvidence
        ? EvidenceSidecar.ToModelEvidence(ModelSource)
        : new OnnxEngineBuildModelEvidence(ModelSource, string.Empty, string.Empty, string.Empty, string.Empty);

    public OnnxEngineBuildEvidenceSidecar EvidenceSidecar { get; }

    public IReadOnlyList<string> Diagnostics { get; }

    public IReadOnlyList<string> LogLines { get; }

    private static string CreateSummary(IReadOnlyList<string> lines)
    {
        if (lines == null || lines.Count == 0)
        {
            return string.Empty;
        }

        return string.Join(" | ", lines).Trim();
    }

    private static string ComputeSha256(string value)
    {
        using SHA256 sha256 = SHA256.Create();
        byte[] hash = sha256.ComputeHash(Encoding.UTF8.GetBytes(value ?? string.Empty));
        StringBuilder builder = new StringBuilder(hash.Length * 2);
        foreach (byte item in hash)
        {
            builder.Append(item.ToString("x2"));
        }

        return builder.ToString();
    }

}

public sealed class OnnxEngineTimingCacheArtifact
{
    public OnnxEngineTimingCacheArtifact(
        bool inputRequested,
        bool inputApplied,
        string inputPath,
        long inputLengthBytes,
        string inputSha256,
        bool outputRequested,
        bool outputWritten,
        string outputPath,
        long outputLengthBytes,
        string outputSha256,
        string state,
        string evidenceBoundary)
    {
        InputRequested = inputRequested;
        InputApplied = inputApplied;
        InputPath = inputPath ?? string.Empty;
        InputLengthBytes = inputLengthBytes;
        InputSha256 = inputSha256 ?? string.Empty;
        OutputRequested = outputRequested;
        OutputWritten = outputWritten;
        OutputPath = outputPath ?? string.Empty;
        OutputLengthBytes = outputLengthBytes;
        OutputSha256 = outputSha256 ?? string.Empty;
        State = state ?? string.Empty;
        EvidenceBoundary = evidenceBoundary ?? string.Empty;
    }

    public static OnnxEngineTimingCacheArtifact Empty { get; } = new OnnxEngineTimingCacheArtifact(
        inputRequested: false,
        inputApplied: false,
        inputPath: string.Empty,
        inputLengthBytes: 0,
        inputSha256: string.Empty,
        outputRequested: false,
        outputWritten: false,
        outputPath: string.Empty,
        outputLengthBytes: 0,
        outputSha256: string.Empty,
        state: string.Empty,
        evidenceBoundary: "timing-cache import/export evidence is build-cache lifecycle metadata only; no cache was requested.");

    public bool InputRequested { get; }

    public bool InputApplied { get; }

    public string InputPath { get; }

    public long InputLengthBytes { get; }

    public string InputSha256 { get; }

    public bool OutputRequested { get; }

    public bool OutputWritten { get; }

    public string OutputPath { get; }

    public long OutputLengthBytes { get; }

    public string OutputSha256 { get; }

    public string State { get; }

    public string EvidenceBoundary { get; }
}

public sealed class OnnxEngineCapabilityProbe
{
    public OnnxEngineCapabilityProbe(
        bool attempted,
        string probeState,
        TensorRtApiLine tensorRtLine,
        string tensorRtVersion,
        string cudaToolkitVersion,
        bool runtimeAvailable,
        bool builderAvailable,
        bool builderConfigAvailable,
        bool engineInspectorApiAvailable,
        bool fp8FlagRequested,
        bool fp8FlagKnown,
        bool debugTensorOptionsRequested,
        bool debugTensorApiKnown,
        bool weightStreamingRequested,
        bool weightStreamingApiKnown,
        IReadOnlyList<string> probeItems,
        string evidenceBoundary)
    {
        Attempted = attempted;
        ProbeState = probeState ?? string.Empty;
        TensorRtLine = tensorRtLine;
        TensorRtVersion = tensorRtVersion ?? string.Empty;
        CudaToolkitVersion = cudaToolkitVersion ?? string.Empty;
        RuntimeAvailable = runtimeAvailable;
        BuilderAvailable = builderAvailable;
        BuilderConfigAvailable = builderConfigAvailable;
        EngineInspectorApiAvailable = engineInspectorApiAvailable;
        Fp8FlagRequested = fp8FlagRequested;
        Fp8FlagKnown = fp8FlagKnown;
        DebugTensorOptionsRequested = debugTensorOptionsRequested;
        DebugTensorApiKnown = debugTensorApiKnown;
        WeightStreamingRequested = weightStreamingRequested;
        WeightStreamingApiKnown = weightStreamingApiKnown;
        ProbeItems = probeItems ?? Array.Empty<string>();
        EvidenceBoundary = evidenceBoundary ?? string.Empty;
    }

    public static OnnxEngineCapabilityProbe Empty { get; } = new OnnxEngineCapabilityProbe(
        attempted: false,
        probeState: string.Empty,
        tensorRtLine: TensorRtApiLine.TensorRt10,
        tensorRtVersion: string.Empty,
        cudaToolkitVersion: string.Empty,
        runtimeAvailable: false,
        builderAvailable: false,
        builderConfigAvailable: false,
        engineInspectorApiAvailable: false,
        fp8FlagRequested: false,
        fp8FlagKnown: false,
        debugTensorOptionsRequested: false,
        debugTensorApiKnown: false,
        weightStreamingRequested: false,
        weightStreamingApiKnown: false,
        probeItems: Array.Empty<string>(),
        evidenceBoundary: string.Empty);

    public bool Attempted { get; }

    public string ProbeState { get; }

    public TensorRtApiLine TensorRtLine { get; }

    public string TensorRtVersion { get; }

    public string CudaToolkitVersion { get; }

    public bool RuntimeAvailable { get; }

    public bool BuilderAvailable { get; }

    public bool BuilderConfigAvailable { get; }

    public bool EngineInspectorApiAvailable { get; }

    public bool Fp8FlagRequested { get; }

    public bool Fp8FlagKnown { get; }

    public bool DebugTensorOptionsRequested { get; }

    public bool DebugTensorApiKnown { get; }

    public bool WeightStreamingRequested { get; }

    public bool WeightStreamingApiKnown { get; }

    public IReadOnlyList<string> ProbeItems { get; }

    public string EvidenceBoundary { get; }
}

public sealed class OnnxLoadedEngineDiagnostics
{
    public OnnxLoadedEngineDiagnostics(
        bool attempted,
        bool succeeded,
        string diagnosticsState,
        string failureReason,
        string engineName,
        int ioTensorCount,
        int layerCount,
        int optimizationProfileCount,
        ulong deviceMemorySizeInBytes,
        int auxiliaryStreamCount,
        string capability,
        string profilingVerbosity,
        int inspectorInformationLength,
        IReadOnlyList<string> ioTensorSummaries,
        string readbackFingerprint,
        string readbackSha256,
        string evidenceBoundary)
    {
        Attempted = attempted;
        Succeeded = succeeded;
        DiagnosticsState = diagnosticsState ?? string.Empty;
        FailureReason = failureReason ?? string.Empty;
        EngineName = engineName ?? string.Empty;
        IOTensorCount = ioTensorCount;
        LayerCount = layerCount;
        OptimizationProfileCount = optimizationProfileCount;
        DeviceMemorySizeInBytes = deviceMemorySizeInBytes;
        AuxiliaryStreamCount = auxiliaryStreamCount;
        Capability = capability ?? string.Empty;
        ProfilingVerbosity = profilingVerbosity ?? string.Empty;
        InspectorInformationLength = inspectorInformationLength;
        IOTensorSummaries = ioTensorSummaries ?? Array.Empty<string>();
        ReadbackFingerprint = readbackFingerprint ?? string.Empty;
        ReadbackSha256 = readbackSha256 ?? string.Empty;
        EvidenceBoundary = evidenceBoundary ?? string.Empty;
    }

    public static OnnxLoadedEngineDiagnostics Empty { get; } = new OnnxLoadedEngineDiagnostics(
        attempted: false,
        succeeded: false,
        diagnosticsState: string.Empty,
        failureReason: string.Empty,
        engineName: string.Empty,
        ioTensorCount: 0,
        layerCount: 0,
        optimizationProfileCount: 0,
        deviceMemorySizeInBytes: 0,
        auxiliaryStreamCount: 0,
        capability: string.Empty,
        profilingVerbosity: string.Empty,
        inspectorInformationLength: 0,
        ioTensorSummaries: Array.Empty<string>(),
        readbackFingerprint: string.Empty,
        readbackSha256: string.Empty,
        evidenceBoundary: string.Empty);

    public bool Attempted { get; }

    public bool Succeeded { get; }

    public string DiagnosticsState { get; }

    public string FailureReason { get; }

    public string EngineName { get; }

    public int IOTensorCount { get; }

    public int LayerCount { get; }

    public int OptimizationProfileCount { get; }

    public ulong DeviceMemorySizeInBytes { get; }

    public int AuxiliaryStreamCount { get; }

    public string Capability { get; }

    public string ProfilingVerbosity { get; }

    public int InspectorInformationLength { get; }

    public IReadOnlyList<string> IOTensorSummaries { get; }

    public string ReadbackFingerprint { get; }

    public string ReadbackSha256 { get; }

    public string EvidenceBoundary { get; }
}

public sealed class OnnxEnginePreflightMetadata
{
    public OnnxEnginePreflightMetadata(
        string kind,
        string path,
        bool exists,
        long lengthBytes,
        string sha256,
        string preflightState,
        string proofClassification,
        string evidenceBoundary)
    {
        Kind = kind ?? string.Empty;
        Path = path ?? string.Empty;
        Exists = exists;
        LengthBytes = lengthBytes;
        Sha256 = sha256 ?? string.Empty;
        PreflightState = preflightState ?? string.Empty;
        ProofClassification = proofClassification ?? string.Empty;
        EvidenceBoundary = evidenceBoundary ?? string.Empty;
    }

    public static OnnxEnginePreflightMetadata Empty { get; } = new OnnxEnginePreflightMetadata(
        string.Empty,
        string.Empty,
        false,
        0,
        string.Empty,
        string.Empty,
        string.Empty,
        string.Empty);

    public string Kind { get; }

    public string Path { get; }

    public bool Exists { get; }

    public long LengthBytes { get; }

    public string Sha256 { get; }

    public string PreflightState { get; }

    public string ProofClassification { get; }

    public string EvidenceBoundary { get; }

    public static OnnxEnginePreflightMetadata FromExistingEngine(string enginePath)
    {
        string fullPath = string.IsNullOrWhiteSpace(enginePath) ? string.Empty : System.IO.Path.GetFullPath(enginePath);
        bool exists = !string.IsNullOrWhiteSpace(fullPath) && File.Exists(fullPath);
        long lengthBytes = exists ? new FileInfo(fullPath).Length : 0;
        string sha256 = exists ? ComputeFileSha256(fullPath) : string.Empty;
        return new OnnxEnginePreflightMetadata(
            "load-engine-preflight",
            fullPath,
            exists,
            lengthBytes,
            sha256,
            "dependency-probe-only",
            "dependency-probe-only",
            "load-engine preflight records file metadata only; it does not deserialize, bind tensors, enqueue inference, validate outputs, or prove package-consumer-runtime.");
    }

    private static string ComputeFileSha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        using SHA256 sha256 = SHA256.Create();
        byte[] hash = sha256.ComputeHash(stream);
        StringBuilder builder = new StringBuilder(hash.Length * 2);
        foreach (byte item in hash)
        {
            builder.Append(item.ToString("x2"));
        }

        return builder.ToString();
    }
}

public sealed class OnnxEngineBuildModelEvidence
{
    public OnnxEngineBuildModelEvidence(
        string modelSource,
        string modelSha256,
        string modelLicense,
        string inputAssetName,
        string inputAssetSha256)
    {
        ModelSource = modelSource ?? string.Empty;
        ModelSha256 = modelSha256 ?? string.Empty;
        ModelLicense = modelLicense ?? string.Empty;
        InputAssetName = inputAssetName ?? string.Empty;
        InputAssetSha256 = inputAssetSha256 ?? string.Empty;
    }

    public string ModelSource { get; }

    public string ModelSha256 { get; }

    public string ModelLicense { get; }

    public string InputAssetName { get; }

    public string InputAssetSha256 { get; }
}

public sealed class OnnxEngineBenchmarkSummary
{
    public OnnxEngineBenchmarkSummary(
        IReadOnlyList<float> timingSamplesMilliseconds,
        int? avgRunsRequested,
        int avgRunsExecuted,
        float? percentileRequested,
        float? percentileElapsedMilliseconds,
        int? threadsRequested,
        int threadsExecuted,
        bool noDataTransfersRequested,
        bool noDataTransfersApplied,
        bool useSpinWaitRequested,
        int? sleepTimeMillisecondsRequested,
        int sleepTimeMillisecondsApplied,
        int? idleTimeMillisecondsRequested,
        int idleTimeMillisecondsApplied,
        string benchmarkBoundary,
        int iterationsRequested = 0,
        int measurementRoundsExecuted = 0,
        int inferenceIterationsExecuted = 0,
        int warmUpMillisecondsRequested = 0,
        double warmUpElapsedMilliseconds = 0,
        int warmUpIterationsExecuted = 0,
        int durationSecondsRequested = 0,
        double measurementElapsedMilliseconds = 0,
        int streamsRequested = 0,
        int? infStreamsRequested = null,
        int executionContextsCreated = 0,
        int concurrentStreamsExecuted = 0,
        bool useSpinWaitApplied = false,
        bool useCudaGraphRequested = false,
        bool useCudaGraphApplied = false,
        string useCudaGraphFallbackReason = "",
        IReadOnlyList<int>? measurementRoundsPerContext = null)
    {
        TimingSamplesMilliseconds = timingSamplesMilliseconds ?? Array.Empty<float>();
        AveragedTimingSamplesMilliseconds = AverageWindows(TimingSamplesMilliseconds, avgRunsRequested);
        AvgRunsRequested = avgRunsRequested;
        AvgRunsExecuted = avgRunsExecuted;
        PercentileRequested = percentileRequested;
        PercentileElapsedMilliseconds = percentileElapsedMilliseconds;
        ThreadsRequested = threadsRequested;
        ThreadsExecuted = threadsExecuted;
        NoDataTransfersRequested = noDataTransfersRequested;
        NoDataTransfersApplied = noDataTransfersApplied;
        UseSpinWaitRequested = useSpinWaitRequested;
        SleepTimeMillisecondsRequested = sleepTimeMillisecondsRequested;
        SleepTimeMillisecondsApplied = sleepTimeMillisecondsApplied;
        IdleTimeMillisecondsRequested = idleTimeMillisecondsRequested;
        IdleTimeMillisecondsApplied = idleTimeMillisecondsApplied;
        BenchmarkBoundary = benchmarkBoundary ?? string.Empty;
        IterationsRequested = iterationsRequested;
        MeasurementRoundsExecuted = measurementRoundsExecuted;
        InferenceIterationsExecuted = inferenceIterationsExecuted;
        WarmUpMillisecondsRequested = warmUpMillisecondsRequested;
        WarmUpElapsedMilliseconds = warmUpElapsedMilliseconds;
        WarmUpIterationsExecuted = warmUpIterationsExecuted;
        DurationSecondsRequested = durationSecondsRequested;
        MeasurementElapsedMilliseconds = measurementElapsedMilliseconds;
        StreamsRequested = streamsRequested;
        InfStreamsRequested = infStreamsRequested;
        ExecutionContextsCreated = executionContextsCreated;
        ConcurrentStreamsExecuted = concurrentStreamsExecuted;
        UseSpinWaitApplied = useSpinWaitApplied;
        UseCudaGraphRequested = useCudaGraphRequested;
        UseCudaGraphApplied = useCudaGraphApplied;
        UseCudaGraphFallbackReason = useCudaGraphFallbackReason ?? string.Empty;
        MeasurementRoundsPerContext = measurementRoundsPerContext ?? Array.Empty<int>();
    }

    public static OnnxEngineBenchmarkSummary Empty { get; } = new OnnxEngineBenchmarkSummary(
        Array.Empty<float>(),
        avgRunsRequested: null,
        avgRunsExecuted: 0,
        percentileRequested: null,
        percentileElapsedMilliseconds: null,
        threadsRequested: null,
        threadsExecuted: 0,
        noDataTransfersRequested: false,
        noDataTransfersApplied: false,
        useSpinWaitRequested: false,
        sleepTimeMillisecondsRequested: null,
        sleepTimeMillisecondsApplied: 0,
        idleTimeMillisecondsRequested: null,
        idleTimeMillisecondsApplied: 0,
        benchmarkBoundary: "benchmark-skipped; no runtime timing samples were produced.");

    public IReadOnlyList<float> TimingSamplesMilliseconds { get; }

    public int TimingSampleCount => TimingSamplesMilliseconds.Count;

    public IReadOnlyList<float> AveragedTimingSamplesMilliseconds { get; }

    public int AveragedTimingSampleCount => AveragedTimingSamplesMilliseconds.Count;

    public float? AverageElapsedMilliseconds => TimingSampleCount == 0 ? null : TimingSamplesMilliseconds.Sum() / TimingSampleCount;

    public float? MinElapsedMilliseconds => TimingSampleCount == 0 ? null : TimingSamplesMilliseconds.Min();

    public float? MaxElapsedMilliseconds => TimingSampleCount == 0 ? null : TimingSamplesMilliseconds.Max();

    public int? AvgRunsRequested { get; }

    public int AvgRunsExecuted { get; }

    public float? PercentileRequested { get; }

    public float? PercentileElapsedMilliseconds { get; }

    public int? ThreadsRequested { get; }

    public int ThreadsExecuted { get; }

    public bool NoDataTransfersRequested { get; }

    public bool NoDataTransfersApplied { get; }

    public bool UseSpinWaitRequested { get; }

    /// <summary>Gets whether CUDA event polling was applied. 获取是否实际应用了 CUDA event 轮询。</summary>
    public bool UseSpinWaitApplied { get; }

    /// <summary>Gets whether CUDA graph execution was requested. 获取是否请求了 CUDA graph 执行。</summary>
    public bool UseCudaGraphRequested { get; }

    /// <summary>Gets whether every worker applied CUDA graph launch. 获取是否所有 worker 都应用了 CUDA graph launch。</summary>
    public bool UseCudaGraphApplied { get; }

    /// <summary>Gets the controlled CUDA graph fallback diagnostic. 获取 CUDA graph 受控回退诊断。</summary>
    public string UseCudaGraphFallbackReason { get; }

    public int? SleepTimeMillisecondsRequested { get; }

    public int SleepTimeMillisecondsApplied { get; }

    public int? IdleTimeMillisecondsRequested { get; }

    public int IdleTimeMillisecondsApplied { get; }

    public string BenchmarkBoundary { get; }

    public int IterationsRequested { get; }

    public int MeasurementRoundsExecuted { get; }

    public int InferenceIterationsExecuted { get; }

    public int WarmUpMillisecondsRequested { get; }

    public double WarmUpElapsedMilliseconds { get; }

    public int WarmUpIterationsExecuted { get; }

    public int DurationSecondsRequested { get; }

    public double MeasurementElapsedMilliseconds { get; }

    public int StreamsRequested { get; }

    public int? InfStreamsRequested { get; }

    public int ExecutionContextsCreated { get; }

    public int ConcurrentStreamsExecuted { get; }

    /// <summary>Gets executed measurement rounds for each context. 获取每个 context 实际执行的测量轮次。</summary>
    public IReadOnlyList<int> MeasurementRoundsPerContext { get; }

    public static OnnxEngineBenchmarkSummary Create(
        IReadOnlyList<float> timingSamplesMilliseconds,
        TrtexecLikeRuntimeOptions runtimeOptions,
        bool inferenceRan,
        bool outputMatch)
    {
        IReadOnlyList<float> samples = timingSamplesMilliseconds ?? Array.Empty<float>();
        int executedSamples = samples.Count;
        int threadsExecuted = executedSamples == 0 ? 0 : 1;
        string boundary = inferenceRan && outputMatch && executedSamples > 0
            ? "benchmark-executed-synthetic-input; avgRuns timing samples apply only to the embedded identity sample, threads are not parallelized, noDataTransfers does not suppress required sample copies, and the result is not real-model or package-consumer proof."
            : "benchmark-skipped; no runtime timing samples were produced.";

        return new OnnxEngineBenchmarkSummary(
            samples,
            runtimeOptions.AvgRuns,
            executedSamples,
            runtimeOptions.Percentile,
            PercentileOrNull(samples, runtimeOptions.Percentile),
            runtimeOptions.Threads,
            threadsExecuted,
            runtimeOptions.NoDataTransfers,
            noDataTransfersApplied: false,
            runtimeOptions.UseSpinWait,
            runtimeOptions.SleepTimeMilliseconds,
            sleepTimeMillisecondsApplied: 0,
            runtimeOptions.IdleTimeMilliseconds,
            idleTimeMillisecondsApplied: 0,
            boundary);
    }

    internal static OnnxEngineBenchmarkSummary CreateExecuted(
        IReadOnlyList<float> timingSamplesMilliseconds,
        OnnxEngineBuildOptions options,
        int measurementRoundsExecuted,
        int warmUpIterationsExecuted,
        double warmUpElapsedMilliseconds,
        double measurementElapsedMilliseconds,
        int executionContextsCreated,
        int threadsExecuted,
        bool useSpinWaitApplied,
        bool useCudaGraphApplied,
        string useCudaGraphFallbackReason,
        IReadOnlyList<int> measurementRoundsPerContext)
    {
        IReadOnlyList<float> samples = timingSamplesMilliseconds ?? Array.Empty<float>();
        TrtexecLikeRuntimeOptions runtimeOptions = options.RuntimeOptions;
        int idleApplied = measurementRoundsExecuted > 1
            ? runtimeOptions.IdleTimeMilliseconds ?? 0
            : 0;
        string graphBoundary = options.UseCudaGraph
            ? (useCudaGraphApplied
                ? "CUDA graph capture/instantiate/launch was applied."
                : $"CUDA graph capture fell back to direct enqueue ({useCudaGraphFallbackReason}).")
            : "CUDA graph was not requested.";
        string boundary =
            "benchmark-executed-bounded-runtime; iterations, warmUp, duration, streams/infStreams, avgRuns statistics, percentile, idleTime, requested host threads, spin-wait completion, and noDataTransfers are backed by actual scheduler behavior; " +
            graphBoundary +
            " sleepTime remains unapplied; noDataTransfers suppresses tensor readback and therefore cannot establish output correctness; tensor correctness and package-consumer proof require separate model-specific evidence.";

        return new OnnxEngineBenchmarkSummary(
            samples,
            runtimeOptions.AvgRuns,
            runtimeOptions.AvgRuns.HasValue ? Math.Min(runtimeOptions.AvgRuns.Value, samples.Count) : 0,
            runtimeOptions.Percentile,
            PercentileOrNull(samples, runtimeOptions.Percentile),
            runtimeOptions.Threads,
            threadsExecuted,
            runtimeOptions.NoDataTransfers,
            noDataTransfersApplied: runtimeOptions.NoDataTransfers,
            runtimeOptions.UseSpinWait,
            runtimeOptions.SleepTimeMilliseconds,
            sleepTimeMillisecondsApplied: 0,
            runtimeOptions.IdleTimeMilliseconds,
            idleTimeMillisecondsApplied: idleApplied,
            boundary,
            iterationsRequested: options.Iterations,
            measurementRoundsExecuted,
            inferenceIterationsExecuted: samples.Count,
            warmUpMillisecondsRequested: options.WarmUpMilliseconds,
            warmUpElapsedMilliseconds,
            warmUpIterationsExecuted,
            durationSecondsRequested: options.DurationSeconds,
            measurementElapsedMilliseconds,
            streamsRequested: options.Streams,
            infStreamsRequested: runtimeOptions.InfStreams,
            executionContextsCreated,
            concurrentStreamsExecuted: executionContextsCreated,
            useSpinWaitApplied,
            useCudaGraphRequested: options.UseCudaGraph,
            useCudaGraphApplied,
            useCudaGraphFallbackReason,
            measurementRoundsPerContext);
    }

    private static float? PercentileOrNull(IReadOnlyList<float> values, float? percentile)
    {
        if (values.Count == 0 || !percentile.HasValue)
        {
            return null;
        }

        float[] sorted = values.OrderBy(static value => value).ToArray();
        int index = (int)Math.Ceiling((percentile.Value / 100.0f) * sorted.Length) - 1;
        index = Math.Max(0, Math.Min(sorted.Length - 1, index));
        return sorted[index];
    }

    private static IReadOnlyList<float> AverageWindows(IReadOnlyList<float> values, int? windowSize)
    {
        if (values.Count == 0 || !windowSize.HasValue)
        {
            return Array.Empty<float>();
        }

        int size = Math.Max(1, windowSize.Value);
        List<float> averages = new List<float>((values.Count + size - 1) / size);
        for (int start = 0; start < values.Count; start += size)
        {
            int count = Math.Min(size, values.Count - start);
            float total = 0;
            for (int offset = 0; offset < count; offset++)
            {
                total += values[start + offset];
            }

            averages.Add(total / count);
        }

        return averages;
    }
}
