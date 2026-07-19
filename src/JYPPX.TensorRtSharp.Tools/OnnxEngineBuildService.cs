using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class OnnxEngineBuildService
{
    public OnnxEngineBuildResult Execute(OnnxEngineBuildOptions options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        List<string> log = new List<string>();
        OnnxEngineBuildEvidenceSidecar evidenceSidecar = OnnxEngineBuildEvidenceSidecarReader.Read(options.EvidenceSidecarPath);
        foreach (string diagnostic in options.Diagnostics)
        {
            log.Add($"Diagnostic={diagnostic}");
        }

        foreach (string diagnostic in evidenceSidecar.Diagnostics)
        {
            log.Add($"Diagnostic={diagnostic}");
        }

        if (options.DryRun)
        {
            log.Insert(0, $"OnnxToEngine TensorRtLine={(int)options.TensorRtLine} DryRun=True Source={DryRunModelSource(options)}");
            log.Insert(1, $"TrtexecLike DryRun=True BuildOnly={options.BuildOnly} SkipInference={options.SkipInference} Fp16={options.Fp16} Int8={options.Int8} Bf16={options.Bf16} Tf32={options.Tf32} WorkspaceBytes={options.WorkspaceBytes}");
            log.Insert(2, $"TrtexecDeployment BuilderOptimizationLevel={options.DeploymentOptions.BuilderOptimizationLevel} MaxAuxStreams={options.DeploymentOptions.MaxAuxStreams?.ToString() ?? ""} Device={options.DeploymentOptions.DeviceOrdinal?.ToString() ?? ""} DlaCore={options.DeploymentOptions.DlaCore?.ToString() ?? ""} DirectIO={options.DeploymentOptions.DirectIO} StronglyTyped={options.DeploymentOptions.StronglyTyped}");
            log.Insert(3, RuntimeOptionsLogLine(options));
            log.Add("OnnxToEngine DryRun=PrecheckOnly Note=TensorRT runtime probing, ONNX parsing, engine build, plugin loading, and inference were not executed.");
            OnnxEngineBuildResult dryRun = CreateResult(
                success: true,
                skipped: false,
                state: "dry-run-precheck",
                options,
                modelSource: DryRunModelSource(options),
                enginePath: string.Empty,
                parsed: false,
                engineSaved: false,
                engineFileRoundTrip: false,
                inferenceRan: false,
                outputMatch: false,
                profileIndex: -1,
                elapsedMilliseconds: null,
                skipReason: string.Empty,
                log,
                evidenceSidecar,
                timingCacheArtifact: CreateTimingCacheBoundaryArtifact(options, "precheck-not-executed"));
            OnnxEngineBuildDiagnostics.WriteReport(dryRun, options.ExportReportPath);
            OnnxEngineRuntimeArtifactWriter.WriteArtifacts(dryRun);
            return dryRun;
        }

        if (options.LoadsExistingEngine)
        {
            OnnxEnginePreflightMetadata preflightMetadata = OnnxEnginePreflightMetadata.FromExistingEngine(options.LoadEnginePath);
            log.Insert(0, $"OnnxToEngine TensorRtLine={(int)options.TensorRtLine} LoadEngine={options.LoadEnginePath}");
            log.Insert(1, $"LoadEnginePreflight Exists={preflightMetadata.Exists} LengthBytes={preflightMetadata.LengthBytes} Sha256={preflightMetadata.Sha256}");
            OnnxLoadedEngineDiagnostics loadedEngineDiagnostics = ProbeLoadedEngineDiagnostics(options, preflightMetadata, log);
            OnnxEngineRuntimeExecution? runtimeExecution = null;
            if (!options.BuildOnly && !options.SkipInference && loadedEngineDiagnostics.Succeeded)
            {
                runtimeExecution = TryRunGenericFloatEngineFromFile(options, options.LoadEnginePath, profileIndex: 0, log, statePrefix: "LoadEngine");
            }

            log.Add(runtimeExecution == null
                ? "OnnxToEngine LoadEngine=ReadonlyDiagnostics Note=Engine is deserialized for metadata only; generic bounded runtime did not execute."
                : $"OnnxToEngine LoadEngine=BoundedRuntime InferenceRan=True OutputMatch={runtimeExecution.OutputMatch}");
            OnnxEngineBuildResult loadResult = CreateResult(
                success: true,
                skipped: false,
                state: runtimeExecution != null
                    ? (runtimeExecution.OutputMatch ? "load-engine-identity-runtime" : "load-engine-runtime-output-unverified")
                    : (loadedEngineDiagnostics.Succeeded ? "load-engine-readonly-diagnostics" : "load-engine-preflight"),
                options,
                modelSource: options.LoadEnginePath,
                enginePath: options.LoadEnginePath,
                parsed: false,
                engineSaved: false,
                engineFileRoundTrip: loadedEngineDiagnostics.Succeeded,
                inferenceRan: runtimeExecution != null,
                outputMatch: runtimeExecution?.OutputMatch ?? false,
                profileIndex: runtimeExecution?.ProfileIndex ?? -1,
                elapsedMilliseconds: runtimeExecution?.ElapsedMilliseconds,
                skipReason: string.Empty,
                log,
                evidenceSidecar,
                benchmarkSummary: runtimeExecution?.BenchmarkSummary,
                preflightMetadata: preflightMetadata,
                loadedEngineDiagnostics: loadedEngineDiagnostics,
                timingCacheArtifact: CreateTimingCacheBoundaryArtifact(options, "not-applied-to-load-engine"));
            OnnxEngineBuildDiagnostics.WriteReport(loadResult, options.ExportReportPath);
            OnnxEngineRuntimeArtifactWriter.WriteArtifacts(loadResult, runtimeExecution?.ArtifactData);
            return loadResult;
        }

        byte[] model = options.UsesExternalOnnx
            ? File.ReadAllBytes(options.OnnxPath)
            : OnnxIdentityModel.CreateDynamicBatchModel();
        string modelSource = options.UsesExternalOnnx ? options.OnnxPath : "embedded-dynamic-identity";
        log.Insert(0, $"OnnxToEngine TensorRtLine={(int)options.TensorRtLine} ModelBytes={model.Length} Batch={options.Batch} Source={modelSource}");
        log.Insert(1, $"TrtexecLike BuildOnly={options.BuildOnly} SkipInference={options.SkipInference} Fp16={options.Fp16} Int8={options.Int8} Bf16={options.Bf16} Tf32={options.Tf32} WorkspaceBytes={options.WorkspaceBytes}");
        log.Insert(2, $"TrtexecDeployment BuilderOptimizationLevel={options.DeploymentOptions.BuilderOptimizationLevel} MaxAuxStreams={options.DeploymentOptions.MaxAuxStreams?.ToString() ?? ""} Device={options.DeploymentOptions.DeviceOrdinal?.ToString() ?? ""} DlaCore={options.DeploymentOptions.DlaCore?.ToString() ?? ""} DirectIO={options.DeploymentOptions.DirectIO} StronglyTyped={options.DeploymentOptions.StronglyTyped}");
        log.Insert(3, RuntimeOptionsLogLine(options));

        TensorRtEnvironmentSnapshot snapshot;
        try
        {
            snapshot = TensorRtEnvironmentProbe.GetCurrent();
        }
        catch (Exception exception) when (exception is TensorRtException ||
                                          exception is BridgeProbeException ||
                                          exception is InvalidOperationException ||
                                          exception is FileNotFoundException ||
                                          exception is CudaException ||
                                          exception is DllNotFoundException ||
                                          exception is BadImageFormatException)
        {
            string reason = "TensorRT native bridge dependency is unavailable: " + exception.Message;
            log.Add("OnnxToEngine=Skipped Reason=" + reason);
            OnnxEngineBuildResult dependencySkipped = CreateResult(
                success: true,
                skipped: true,
                state: "dependency-probe-only",
                options,
                modelSource,
                enginePath: string.Empty,
                parsed: false,
                engineSaved: false,
                engineFileRoundTrip: false,
                inferenceRan: false,
                outputMatch: false,
                profileIndex: -1,
                elapsedMilliseconds: null,
                skipReason: reason,
                log,
                evidenceSidecar,
                timingCacheArtifact: CreateTimingCacheBoundaryArtifact(options, "dependency-unavailable"));
            OnnxEngineBuildDiagnostics.WriteReport(dependencySkipped, options.ExportReportPath);
            OnnxEngineRuntimeArtifactWriter.WriteArtifacts(dependencySkipped);
            return dependencySkipped;
        }

        TensorRtAdapterInfo adapter = TensorRtToolSupport.SelectAdapter(snapshot, options.TensorRtLine);
        log.Add($"Preflight TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion} Runtime={adapter.RuntimeCreationSupported} Builder={adapter.BuilderCreationSupported}");
        if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
        {
            log.Add($"OnnxToEngine=Skipped Reason={adapter.StatusMessage}");
            OnnxEngineBuildResult skipped = CreateResult(
                success: true,
                skipped: true,
                state: "skipped",
                options,
                modelSource,
                enginePath: string.Empty,
                parsed: false,
                engineSaved: false,
                engineFileRoundTrip: false,
                inferenceRan: false,
                outputMatch: false,
                profileIndex: -1,
                elapsedMilliseconds: null,
                skipReason: adapter.StatusMessage,
                log,
                evidenceSidecar,
                timingCacheArtifact: CreateTimingCacheBoundaryArtifact(options, "dependency-unavailable"));
            OnnxEngineBuildDiagnostics.WriteReport(skipped, options.ExportReportPath);
            OnnxEngineRuntimeArtifactWriter.WriteArtifacts(skipped);
            return skipped;
        }

        try
        {
            using TensorRtLogger logger = new TensorRtLogger(options.TensorRtLine);
            using TensorRtRuntime runtime = new TensorRtRuntime(logger);
            using TensorRtBuilder builder = new TensorRtBuilder(logger);
            using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
            using CudaStream stream = new CudaStream();

        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, options.WorkspaceBytes);
        config.SetProfileStream(stream);
        config.SetOptimizationLevel(options.DeploymentOptions.BuilderOptimizationLevel);
        config.SetEngineCapability(TensorRtEngineCapability.Standard);
        ApplyDeploymentOptions(config, options, log);
        ApplyPrecisionFlags(config, options);
        using TimingCacheLease timingCache = CreateTimingCacheLease(config, options, log);

        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        using TensorRtOnnxParser parser = new TensorRtOnnxParser(logger, network);
        bool parserBuilderConfigAttached = false;
        if (options.TensorRtLine == TensorRtApiLine.TensorRt11)
        {
            parserBuilderConfigAttached = parser.SetBuilderConfig(config);
            if (!parserBuilderConfigAttached)
            {
                throw new InvalidOperationException("TensorRT 11 ONNX parser rejected the deployment builder configuration.");
            }

            if (options.DeploymentOptions.DlaCore.HasValue)
            {
                parser.SetFlag(TensorRtOnnxParserFlag.ReportCapabilityDla);
                parser.SetFlag(TensorRtOnnxParserFlag.AdjustForDla);
            }
        }

        log.Add($"ParserBuilderConfig Supported={options.TensorRtLine == TensorRtApiLine.TensorRt11} Attached={parserBuilderConfigAttached} ManagedLease={parser.HasBuilderConfigAttached} DlaCapabilityValidation={options.TensorRtLine == TensorRtApiLine.TensorRt11 && options.DeploymentOptions.DlaCore.HasValue}");
        if (!parser.Parse(model, options.UsesExternalOnnx ? Path.GetFileName(options.OnnxPath) : "sample-dynamic-identity.onnx"))
        {
            throw new InvalidOperationException(parser.GetErrorSummary());
        }

        int profileIndex = AddOptimizationProfile(builder, config, options);
        string enginePath = string.IsNullOrWhiteSpace(options.SaveEnginePath)
            ? Path.Combine(Path.GetTempPath(), $"jyppx-onnx-to-engine-{Guid.NewGuid():N}.plan")
            : options.SaveEnginePath;
        bool deleteEnginePath = string.IsNullOrWhiteSpace(options.SaveEnginePath);

        try
        {
            using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
            hostMemory.SaveToFile(enginePath);
            timingCache.Artifact = ExportTimingCache(timingCache, options, log);

            bool externalRuntimeRequested = options.UsesExternalOnnx && CanAttemptGenericExternalRuntime(options);
            if (options.BuildOnly || options.SkipInference || (options.UsesExternalOnnx && !externalRuntimeRequested))
            {
                log.Add($"Parsed=True ProfileIndex={profileIndex} EngineSaved={enginePath}");
                log.Add(options.UsesExternalOnnx
                    ? "OnnxToEngine ExternalOnnx=BuildOnlyOrSkipInference Note=Generic external-model inference requires explicit binding/output semantics. Supply --loadInputs tensor:file plus concrete shape semantics."
                    : "OnnxToEngine BuildOnly=True");
                log.Add("OnnxToEngine Passed=True");
                OnnxEngineBuildResult buildOnly = CreateResult(
                    success: true,
                    skipped: false,
                    state: options.UsesExternalOnnx ? "external-onnx-build-only" : "build-only",
                    options,
                    modelSource,
                    enginePath,
                    parsed: true,
                    engineSaved: true,
                    engineFileRoundTrip: false,
                    inferenceRan: false,
                    outputMatch: false,
                    profileIndex,
                    elapsedMilliseconds: null,
                    skipReason: string.Empty,
                    log,
                    evidenceSidecar,
                    timingCacheArtifact: timingCache.Artifact);
                OnnxEngineBuildDiagnostics.WriteReport(buildOnly, options.ExportReportPath);
                OnnxEngineRuntimeArtifactWriter.WriteArtifacts(buildOnly);
                return buildOnly;
            }

            if (externalRuntimeRequested)
            {
                OnnxEngineRuntimeExecution? runtimeExecution = TryRunGenericFloatEngineFromFile(options, enginePath, profileIndex, log, statePrefix: "ExternalOnnx");
                log.Add(runtimeExecution == null
                    ? "OnnxToEngine ExternalOnnx=RuntimeSkipped Note=Generic bounded runtime could not be executed."
                    : $"OnnxToEngine ExternalOnnx=BoundedRuntime InferenceRan=True OutputMatch={runtimeExecution.OutputMatch}");
                OnnxEngineBuildResult externalRuntime = CreateResult(
                    success: true,
                    skipped: false,
                    state: runtimeExecution != null && runtimeExecution.OutputMatch ? "external-onnx-identity-runtime" : "external-onnx-runtime-output-unverified",
                    options,
                    modelSource,
                    enginePath,
                    parsed: true,
                    engineSaved: true,
                    engineFileRoundTrip: true,
                    inferenceRan: runtimeExecution != null,
                    outputMatch: runtimeExecution?.OutputMatch ?? false,
                    profileIndex: runtimeExecution?.ProfileIndex ?? profileIndex,
                    elapsedMilliseconds: runtimeExecution?.ElapsedMilliseconds,
                    skipReason: string.Empty,
                    log,
                    evidenceSidecar,
                    benchmarkSummary: runtimeExecution?.BenchmarkSummary,
                    loadedEngineDiagnostics: ProbeLoadedEngineDiagnostics(options, OnnxEnginePreflightMetadata.FromExistingEngine(enginePath), log),
                    timingCacheArtifact: timingCache.Artifact);
                OnnxEngineBuildDiagnostics.WriteReport(externalRuntime, options.ExportReportPath);
                OnnxEngineRuntimeArtifactWriter.WriteArtifacts(externalRuntime, runtimeExecution?.ArtifactData);
                return externalRuntime;
            }

            using TensorRtEngine engine = runtime.DeserializeFromFile(enginePath);
            using TensorRtExecutionContext context = engine.CreateExecutionContext();

            TensorRtDims runtimeShape = new TensorRtDims(new[] { options.Batch, 4 });
            float[] inputValues = Enumerable.Range(0, options.Batch * 4).Select(index => index + 0.5f).ToArray();
            using TensorRtInferenceBindings bindings = new TensorRtInferenceBindings(engine, context, profileIndex);
            bindings.SetInputShape("input", runtimeShape)
                    .CopyInputFromHost("input", inputValues, runtimeShape);
            bindings.AllocateDeviceBuffer("output", runtimeShape, checked(inputValues.Length * sizeof(float)));
            bindings.BindAll();

            TensorRtInferenceExecutionSummary executionSummary = null!;
            float elapsedMilliseconds = stream.MeasureElapsedTime(cudaStream =>
            {
                executionSummary = bindings.EnqueueAsync(cudaStream, synchronize: false, runShapeInference: false);
            });
            List<float> timingSamplesMilliseconds = new List<float> { elapsedMilliseconds };

            float[] outputValues = bindings.ReadOutputSingles("output", inputValues.Length);
            if (!inputValues.SequenceEqual(outputValues))
            {
                throw new InvalidOperationException($"Output mismatch. Input=[{string.Join(", ", inputValues)}] Output=[{string.Join(", ", outputValues)}]");
            }

            int avgRuns = options.RuntimeOptions.AvgRuns ?? 1;
            for (int runIndex = 1; runIndex < avgRuns; runIndex++)
            {
                TensorRtInferenceExecutionSummary benchmarkSummary = null!;
                float benchmarkMilliseconds = stream.MeasureElapsedTime(cudaStream =>
                {
                    benchmarkSummary = bindings.EnqueueAsync(cudaStream, synchronize: false, runShapeInference: false);
                });
                executionSummary = benchmarkSummary;
                timingSamplesMilliseconds.Add(benchmarkMilliseconds);
            }

            log.Add($"Parsed=True ProfileIndex={profileIndex} EngineFileRoundTrip=True");
            log.Add($"BindingReport Ready={bindings.Report.IsReadyForEnqueue} Inputs={bindings.Report.GetInputs().Count} Outputs={bindings.Report.GetOutputs().Count}");
            log.Add($"Execution {executionSummary} ElapsedMs={elapsedMilliseconds:0.###} OutputMatch=True");
            log.Add($"RuntimeBenchmark Samples={timingSamplesMilliseconds.Count} AvgRunsRequested={options.RuntimeOptions.AvgRuns?.ToString() ?? ""} ThreadsRequested={options.RuntimeOptions.Threads?.ToString() ?? ""} ThreadsExecuted=1 NoDataTransfersRequested={options.RuntimeOptions.NoDataTransfers} NoDataTransfersApplied=False PercentileRequested={options.RuntimeOptions.Percentile?.ToString() ?? ""}");
            log.Add("OnnxToEngine Passed=True");

            OnnxEngineBuildResult roundTrip = CreateResult(
                success: true,
                skipped: false,
                state: "identity-roundtrip",
                options,
                modelSource,
                enginePath,
                parsed: true,
                engineSaved: true,
                engineFileRoundTrip: true,
                inferenceRan: true,
                outputMatch: true,
                profileIndex,
                elapsedMilliseconds,
                skipReason: string.Empty,
                log,
                evidenceSidecar,
                benchmarkSummary: OnnxEngineBenchmarkSummary.Create(
                    timingSamplesMilliseconds,
                    options.RuntimeOptions,
                    inferenceRan: true,
                    outputMatch: true),
                timingCacheArtifact: timingCache.Artifact);
            OnnxEngineBuildDiagnostics.WriteReport(roundTrip, options.ExportReportPath);
            OnnxEngineRuntimeArtifactWriter.WriteArtifacts(
                roundTrip,
                OnnxEngineRuntimeArtifactData.CreateIdentityOutput(
                    "output",
                    new[] { options.Batch, 4 },
                    inputValues,
                    outputValues,
                    executionSummary.ToString(),
                    timingSamplesMilliseconds));
            return roundTrip;
        }
        finally
        {
            if (deleteEnginePath && File.Exists(enginePath))
            {
                File.Delete(enginePath);
            }
        }
        }
        catch (Exception exception) when (exception is TensorRtException ||
                                          exception is BridgeProbeException ||
                                          exception is CudaException ||
                                          exception is DllNotFoundException ||
                                          exception is BadImageFormatException)
        {
            string reason = "TensorRT native build dependency or runtime is unavailable: " + exception.Message;
            log.Add("OnnxToEngine=Skipped Reason=" + reason);
            OnnxEngineBuildResult nativeSkipped = CreateResult(
                success: true,
                skipped: true,
                state: "dependency-probe-only",
                options,
                modelSource,
                enginePath: string.Empty,
                parsed: false,
                engineSaved: false,
                engineFileRoundTrip: false,
                inferenceRan: false,
                outputMatch: false,
                profileIndex: -1,
                elapsedMilliseconds: null,
                skipReason: reason,
                log,
                evidenceSidecar,
                timingCacheArtifact: CreateTimingCacheBoundaryArtifact(options, "dependency-unavailable"));
            OnnxEngineBuildDiagnostics.WriteReport(nativeSkipped, options.ExportReportPath);
            OnnxEngineRuntimeArtifactWriter.WriteArtifacts(nativeSkipped);
            return nativeSkipped;
        }
    }

    private static int AddOptimizationProfile(TensorRtBuilder builder, TensorRtBuilderConfig config, OnnxEngineBuildOptions options)
    {
        if (options.ShapeProfile.IsEmpty && options.UsesExternalOnnx)
        {
            return -1;
        }

        using TensorRtOptimizationProfile profile = builder.CreateOptimizationProfile();
        if (!options.ShapeProfile.IsEmpty)
        {
            foreach (EngineBuildShape minShape in options.ShapeProfile.MinShapes)
            {
                if (!options.ShapeProfile.TryGetShapeTriple(minShape.TensorName, out EngineBuildShape min, out EngineBuildShape opt, out EngineBuildShape max))
                {
                    throw new ArgumentException($"Missing complete shape profile for tensor '{minShape.TensorName}'.");
                }

                profile.SetShape(min.TensorName, new TensorRtDims(min.Dimensions), new TensorRtDims(opt.Dimensions), new TensorRtDims(max.Dimensions));
            }

            return config.AddOptimizationProfile(profile);
        }

        profile.SetShape(
            "input",
            new TensorRtDims(new[] { 1, 4 }),
            new TensorRtDims(new[] { 2, 4 }),
            new TensorRtDims(new[] { 4, 4 }));
        return config.AddOptimizationProfile(profile);
    }

    private static string DryRunModelSource(OnnxEngineBuildOptions options)
    {
        if (options.UsesExternalOnnx)
        {
            return options.OnnxPath;
        }

        if (options.LoadsExistingEngine)
        {
            return options.LoadEnginePath;
        }

        return "embedded-dynamic-identity";
    }

    private static void ApplyPrecisionFlags(TensorRtBuilderConfig config, OnnxEngineBuildOptions options)
    {
        if (options.Fp16)
        {
            config.SetFlag(TensorRtBuilderFlag.Fp16);
        }

        if (options.Int8)
        {
            config.SetFlag(TensorRtBuilderFlag.Int8);
        }

        if (options.Bf16)
        {
            config.SetFlag(TensorRtBuilderFlag.Bf16);
        }

        if (options.Tf32)
        {
            config.SetFlag(TensorRtBuilderFlag.Tf32);
        }
        else
        {
            config.ClearFlag(TensorRtBuilderFlag.Tf32);
        }
    }

    private const long MaxTimingCacheBytes = 512L * 1024L * 1024L;

    private static TimingCacheLease CreateTimingCacheLease(
        TensorRtBuilderConfig config,
        OnnxEngineBuildOptions options,
        List<string> log)
    {
        string inputPath = options.TimingCacheFile;
        string outputPath = options.DeploymentOptions.ExportTimingCachePath;
        bool inputRequested = !string.IsNullOrWhiteSpace(inputPath);
        bool outputRequested = !string.IsNullOrWhiteSpace(outputPath);
        if (!inputRequested && !outputRequested)
        {
            return new TimingCacheLease(null, OnnxEngineTimingCacheArtifact.Empty);
        }

        byte[]? inputBytes = null;
        long inputLengthBytes = 0;
        string inputSha256 = string.Empty;
        if (inputRequested)
        {
            FileInfo inputFile = new FileInfo(inputPath);
            if (!inputFile.Exists)
            {
                throw new FileNotFoundException("Timing cache file was not found.", inputPath);
            }

            if (inputFile.Length > MaxTimingCacheBytes)
            {
                throw new InvalidDataException($"Timing cache file exceeds the {MaxTimingCacheBytes} byte safety limit.");
            }

            inputBytes = File.ReadAllBytes(inputFile.FullName);
            inputLengthBytes = inputBytes.LongLength;
            inputSha256 = ComputeSha256(inputBytes);
        }

        TensorRtTimingCache cache = config.CreateTimingCache(inputBytes);
        try
        {
            config.SetTimingCache(cache, ignoreMismatch: false);
        }
        catch
        {
            cache.Dispose();
            throw;
        }

        log.Add($"TimingCache ImportRequested={inputRequested} Applied=True Path={inputPath} LengthBytes={inputLengthBytes} Sha256={inputSha256}");
        return new TimingCacheLease(
            cache,
            new OnnxEngineTimingCacheArtifact(
                inputRequested,
                inputApplied: true,
                inputPath,
                inputLengthBytes,
                inputSha256,
                outputRequested,
                outputWritten: false,
                outputPath,
                outputLengthBytes: 0,
                outputSha256: string.Empty,
                state: outputRequested ? "imported-export-pending" : "imported",
                evidenceBoundary: TimingCacheEvidenceBoundary));
    }

    private static OnnxEngineTimingCacheArtifact ExportTimingCache(
        TimingCacheLease lease,
        OnnxEngineBuildOptions options,
        List<string> log)
    {
        string outputPath = options.DeploymentOptions.ExportTimingCachePath;
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            return lease.Artifact;
        }

        if (lease.Cache == null)
        {
            log.Add("TimingCache ExportRequested=True Written=False Reason=timing cache owner was not created.");
            return CreateTimingCacheBoundaryArtifact(options, "export-not-applied");
        }

        using TensorRtHostMemory hostMemory = lease.Cache.Serialize();
        byte[] bytes = hostMemory.ToArray();
        string fullPath = Path.GetFullPath(outputPath);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllBytes(fullPath, bytes);
        string outputSha256 = ComputeSha256(bytes);
        log.Add($"TimingCache ExportRequested=True Written=True Path={fullPath} LengthBytes={bytes.LongLength} Sha256={outputSha256}");
        return new OnnxEngineTimingCacheArtifact(
            lease.Artifact.InputRequested,
            lease.Artifact.InputApplied,
            lease.Artifact.InputPath,
            lease.Artifact.InputLengthBytes,
            lease.Artifact.InputSha256,
            outputRequested: true,
            outputWritten: true,
            fullPath,
            bytes.LongLength,
            outputSha256,
            state: lease.Artifact.InputApplied ? "imported-and-exported" : "exported",
            evidenceBoundary: TimingCacheEvidenceBoundary);
    }

    private static OnnxEngineTimingCacheArtifact CreateTimingCacheBoundaryArtifact(
        OnnxEngineBuildOptions options,
        string state)
    {
        bool inputRequested = !string.IsNullOrWhiteSpace(options.TimingCacheFile);
        bool outputRequested = !string.IsNullOrWhiteSpace(options.DeploymentOptions.ExportTimingCachePath);
        if (!inputRequested && !outputRequested)
        {
            return OnnxEngineTimingCacheArtifact.Empty;
        }

        return new OnnxEngineTimingCacheArtifact(
            inputRequested,
            inputApplied: false,
            options.TimingCacheFile,
            inputLengthBytes: 0,
            inputSha256: string.Empty,
            outputRequested,
            outputWritten: false,
            options.DeploymentOptions.ExportTimingCachePath,
            outputLengthBytes: 0,
            outputSha256: string.Empty,
            state,
            TimingCacheEvidenceBoundary);
    }

    private const string TimingCacheEvidenceBoundary = "timing-cache import/export evidence is build-cache lifecycle metadata only; it is not model accuracy, runtime execution, real-model-runtime, or package-consumer-runtime proof.";

    private static string ComputeSha256(byte[] bytes)
    {
        using SHA256 sha256 = SHA256.Create();
        byte[] hash = sha256.ComputeHash(bytes ?? Array.Empty<byte>());
        StringBuilder builder = new StringBuilder(hash.Length * 2);
        foreach (byte item in hash)
        {
            builder.Append(item.ToString("x2"));
        }

        return builder.ToString();
    }

    private static OnnxEngineBuildResult CreateResult(
        bool success,
        bool skipped,
        string state,
        OnnxEngineBuildOptions options,
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
        IReadOnlyList<string> logLines,
        OnnxEngineBuildEvidenceSidecar evidenceSidecar,
        OnnxEngineBenchmarkSummary? benchmarkSummary = null,
        OnnxEnginePreflightMetadata? preflightMetadata = null,
        OnnxLoadedEngineDiagnostics? loadedEngineDiagnostics = null,
        OnnxEngineTimingCacheArtifact? timingCacheArtifact = null)
    {
        OnnxEngineCapabilityProbe capabilityProbe = ProbeCapabilities(options);
        logLines = AppendCapabilityProbeLog(logLines, capabilityProbe);

        return new OnnxEngineBuildResult(
            success,
            skipped,
            state,
            options.TensorRtLine,
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
            options.NormalizedCommandLine,
            options.DeploymentOptions,
            options.Diagnostics,
            logLines,
            evidenceSidecar,
            options.RuntimeOptions,
            benchmarkSummary,
            preflightMetadata,
            loadedEngineDiagnostics,
            timingCacheArtifact: timingCacheArtifact,
            capabilityProbe: capabilityProbe,
            workspaceBytes: options.WorkspaceBytes);
    }

    private static IReadOnlyList<string> AppendCapabilityProbeLog(IReadOnlyList<string> logLines, OnnxEngineCapabilityProbe capabilityProbe)
    {
        List<string> merged = new List<string>(logLines ?? Array.Empty<string>());
        if (capabilityProbe.Attempted)
        {
            merged.Add($"CapabilityProbe State={capabilityProbe.ProbeState} Runtime={capabilityProbe.RuntimeAvailable} Builder={capabilityProbe.BuilderAvailable} BuilderConfig={capabilityProbe.BuilderConfigAvailable} EngineInspectorApi={capabilityProbe.EngineInspectorApiAvailable}");
            merged.Add("CapabilityProbe Items=" + string.Join("; ", capabilityProbe.ProbeItems));
            merged.Add("CapabilityProbe Boundary=" + capabilityProbe.EvidenceBoundary);
        }

        return merged;
    }

    private static OnnxEngineCapabilityProbe ProbeCapabilities(OnnxEngineBuildOptions options)
    {
        const string boundary = "capability-probe-only records host/tool API availability and requested advanced options; it does not build a proof model, enqueue inference, validate outputs, or promote parse-only options to runtime proof.";
        TrtexecLikeDeploymentOptions deployment = options.DeploymentOptions ?? TrtexecLikeDeploymentOptions.Default;
        bool fp8Requested = deployment.Fp8 || deployment.Best;
        bool debugRequested = !string.IsNullOrWhiteSpace(deployment.MarkDebug) || deployment.DumpDebugTensors;
        bool weightStreamingRequested = deployment.AllowWeightStreaming || deployment.WeightStreamingBudgetBytes.HasValue;

        try
        {
            TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
            TensorRtAdapterInfo adapter = TensorRtToolSupport.SelectAdapter(snapshot, options.TensorRtLine);
            bool modernTensorRtLine = options.TensorRtLine == TensorRtApiLine.TensorRt10 || options.TensorRtLine == TensorRtApiLine.TensorRt11;
            List<string> probeItems = new List<string>
            {
                "runtime:create:" + adapter.RuntimeCreationSupported,
                "builder:create:" + adapter.BuilderCreationSupported,
                "builder-config:create:" + adapter.BuilderCreationSupported,
                "engine-inspector:managed-api:" + modernTensorRtLine,
                "fp8-builder-flag:requested:" + fp8Requested + ":known:" + modernTensorRtLine,
                "debug-tensor-options:requested:" + debugRequested + ":known:" + modernTensorRtLine,
                "weight-streaming-options:requested:" + weightStreamingRequested + ":known:" + modernTensorRtLine
            };

            return new OnnxEngineCapabilityProbe(
                attempted: true,
                probeState: adapter.RuntimeCreationSupported || adapter.BuilderCreationSupported ? "capability-probe-only" : "dependency-unavailable",
                tensorRtLine: options.TensorRtLine,
                tensorRtVersion: snapshot.BuildInfo.TensorRtVersion,
                cudaToolkitVersion: snapshot.BuildInfo.CudaToolkitVersion,
                runtimeAvailable: adapter.RuntimeCreationSupported,
                builderAvailable: adapter.BuilderCreationSupported,
                builderConfigAvailable: adapter.BuilderCreationSupported,
                engineInspectorApiAvailable: modernTensorRtLine,
                fp8FlagRequested: fp8Requested,
                fp8FlagKnown: modernTensorRtLine,
                debugTensorOptionsRequested: debugRequested,
                debugTensorApiKnown: modernTensorRtLine,
                weightStreamingRequested: weightStreamingRequested,
                weightStreamingApiKnown: modernTensorRtLine,
                probeItems: probeItems,
                evidenceBoundary: boundary);
        }
        catch (Exception exception) when (exception is TensorRtException || exception is BridgeProbeException || exception is InvalidOperationException || exception is DllNotFoundException || exception is BadImageFormatException)
        {
            return new OnnxEngineCapabilityProbe(
                attempted: true,
                probeState: "capability-probe-failed",
                tensorRtLine: options.TensorRtLine,
                tensorRtVersion: string.Empty,
                cudaToolkitVersion: string.Empty,
                runtimeAvailable: false,
                builderAvailable: false,
                builderConfigAvailable: false,
                engineInspectorApiAvailable: options.TensorRtLine == TensorRtApiLine.TensorRt10 || options.TensorRtLine == TensorRtApiLine.TensorRt11,
                fp8FlagRequested: fp8Requested,
                fp8FlagKnown: false,
                debugTensorOptionsRequested: debugRequested,
                debugTensorApiKnown: false,
                weightStreamingRequested: weightStreamingRequested,
                weightStreamingApiKnown: false,
                probeItems: new[]
                {
                    "probe-error:" + exception.GetType().Name,
                    "runtime:create:false",
                    "builder:create:false"
                },
                evidenceBoundary: boundary);
        }
    }

    private static OnnxLoadedEngineDiagnostics ProbeLoadedEngineDiagnostics(
        OnnxEngineBuildOptions options,
        OnnxEnginePreflightMetadata preflightMetadata,
        List<string> log)
    {
        const string boundary = "load-engine readonly diagnostics may deserialize the engine and copy metadata, but it does not create execution bindings, enqueue inference, validate outputs, or prove package-consumer-runtime.";
        if (!preflightMetadata.Exists)
        {
            log.Add("LoadEngineReadonlyDiagnostics Attempted=False Succeeded=False Reason=engine file does not exist.");
            return new OnnxLoadedEngineDiagnostics(
                attempted: false,
                succeeded: false,
                diagnosticsState: "missing-engine-file",
                failureReason: "Serialized TensorRT engine file was not found.",
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
                evidenceBoundary: boundary);
        }

        try
        {
            TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
            TensorRtAdapterInfo adapter = TensorRtToolSupport.SelectAdapter(snapshot, options.TensorRtLine);
            log.Add($"LoadEngineReadonlyDiagnostics Preflight TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion} Runtime={adapter.RuntimeCreationSupported}");
            if (!adapter.RuntimeCreationSupported)
            {
                log.Add($"LoadEngineReadonlyDiagnostics Attempted=False Succeeded=False Reason={adapter.StatusMessage}");
                return new OnnxLoadedEngineDiagnostics(
                    attempted: false,
                    succeeded: false,
                    diagnosticsState: "runtime-unavailable",
                    failureReason: adapter.StatusMessage,
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
                    evidenceBoundary: boundary);
            }

            using TensorRtLogger logger = new TensorRtLogger(options.TensorRtLine);
            using TensorRtRuntime runtime = new TensorRtRuntime(logger);
            using TensorRtEngine engine = runtime.DeserializeFromFile(options.LoadEnginePath);
            using TensorRtEngineInspector inspector = engine.CreateInspector();
            IReadOnlyList<TensorRtTensorInfo> tensors = engine.GetIOTensors();
            string inspectorInformation = inspector.GetEngineInformation(TensorRtLayerInformationFormat.Oneline);
            string[] tensorSummaries = tensors
                .Select(static tensor => $"{tensor.Index}:{tensor.Name}:{tensor.IOMode}:{tensor.DataType}:{tensor.Shape}")
                .ToArray();
            string readbackFingerprint = CreateLoadedEngineReadbackFingerprint(engine, inspectorInformation, tensorSummaries);
            string readbackSha256 = ComputeSha256(readbackFingerprint);
            log.Add($"LoadEngineReadonlyDiagnostics Attempted=True Succeeded=True IOTensors={engine.IOTensorCount} Layers={engine.LayerCount} Profiles={engine.OptimizationProfileCount} InspectorBytes={inspectorInformation.Length} ReadbackSha256={readbackSha256}");
            log.Add("LoadEngineReadonlyDiagnostics Tensors=" + string.Join("; ", tensorSummaries));
            return new OnnxLoadedEngineDiagnostics(
                attempted: true,
                succeeded: true,
                diagnosticsState: "readonly-deserialize-succeeded",
                failureReason: string.Empty,
                engineName: engine.Name,
                ioTensorCount: engine.IOTensorCount,
                layerCount: engine.LayerCount,
                optimizationProfileCount: engine.OptimizationProfileCount,
                deviceMemorySizeInBytes: engine.DeviceMemorySizeInBytes,
                auxiliaryStreamCount: engine.AuxiliaryStreamCount,
                capability: engine.Capability.ToString(),
                profilingVerbosity: engine.ProfilingVerbosity.ToString(),
                inspectorInformationLength: inspectorInformation.Length,
                ioTensorSummaries: tensorSummaries,
                readbackFingerprint: readbackFingerprint,
                readbackSha256: readbackSha256,
                evidenceBoundary: boundary);
        }
        catch (Exception exception) when (exception is TensorRtException || exception is BridgeProbeException || exception is InvalidOperationException || exception is FileNotFoundException || exception is DllNotFoundException || exception is BadImageFormatException)
        {
            log.Add($"LoadEngineReadonlyDiagnostics Attempted=True Succeeded=False Reason={exception.GetType().Name}:{exception.Message}");
            return new OnnxLoadedEngineDiagnostics(
                attempted: true,
                succeeded: false,
                diagnosticsState: "readonly-deserialize-failed",
                failureReason: exception.GetType().Name + ": " + exception.Message,
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
                evidenceBoundary: boundary);
        }
    }

    private static string CreateLoadedEngineReadbackFingerprint(
        TensorRtEngine engine,
        string inspectorInformation,
        IReadOnlyList<string> tensorSummaries)
    {
        return string.Join("|", new[]
        {
            "load-engine-readonly-diagnostics",
            "name=" + engine.Name,
            "io=" + engine.IOTensorCount.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "layers=" + engine.LayerCount.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "profiles=" + engine.OptimizationProfileCount.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "deviceMemory=" + engine.DeviceMemorySizeInBytes.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "auxStreams=" + engine.AuxiliaryStreamCount.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "capability=" + engine.Capability,
            "profilingVerbosity=" + engine.ProfilingVerbosity,
            "inspectorLength=" + (inspectorInformation ?? string.Empty).Length.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "tensors=" + string.Join(";", tensorSummaries ?? Array.Empty<string>())
        });
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

    private static bool CanAttemptGenericExternalRuntime(OnnxEngineBuildOptions options)
    {
        return !string.IsNullOrWhiteSpace(options.RuntimeOptions.LoadInputs);
    }

    private static OnnxEngineRuntimeExecution? TryRunGenericFloatEngineFromFile(
        OnnxEngineBuildOptions options,
        string enginePath,
        int profileIndex,
        List<string> log,
        string statePrefix)
    {
        try
        {
            using TensorRtLogger logger = new TensorRtLogger(options.TensorRtLine);
            using TensorRtRuntime runtime = new TensorRtRuntime(logger);
            using TensorRtEngine engine = runtime.DeserializeFromFile(enginePath);
            return TryRunGenericFloatEngine(engine, options, profileIndex, log, statePrefix);
        }
        catch (Exception exception) when (exception is NotSupportedException || exception is ArgumentException || exception is InvalidOperationException || exception is TensorRtException || exception is BridgeProbeException || exception is DllNotFoundException || exception is BadImageFormatException || exception is FileNotFoundException)
        {
            log.Add($"{statePrefix}BoundedRuntime Attempted=True Succeeded=False Reason={exception.GetType().Name}:{exception.Message}");
            return null;
        }
    }

    private static OnnxEngineRuntimeExecution TryRunGenericFloatEngine(
        TensorRtEngine engine,
        OnnxEngineBuildOptions options,
        int profileIndex,
        List<string> log,
        string statePrefix)
    {
        int safeProfileIndex = Math.Max(0, profileIndex);
        using CudaStream stream = new CudaStream(CudaStreamCreationFlags.NonBlocking);
        using TensorRtExecutionContext context = engine.CreateExecutionContext();
        using TensorRtInferenceBindings bindings = new TensorRtInferenceBindings(engine, context, safeProfileIndex);

        IReadOnlyList<TensorRtEngineTensorBinding> inputs = bindings.Report.GetInputs();
        IReadOnlyList<TensorRtEngineTensorBinding> outputs = bindings.Report.GetOutputs();
        if (inputs.Count != 1)
        {
            throw new NotSupportedException($"Generic bounded runtime supports exactly one input tensor. Engine input count: {inputs.Count}.");
        }

        if (outputs.Count == 0)
        {
            throw new NotSupportedException("Generic bounded runtime requires at least one output tensor.");
        }

        TensorRtEngineTensorBinding input = inputs[0];
        if (input.DataType != TensorRtDataType.Float)
        {
            throw new NotSupportedException($"Generic bounded runtime supports float input tensors only. Input '{input.Name}' is {input.DataType}.");
        }

        foreach (TensorRtEngineTensorBinding output in outputs)
        {
            if (output.DataType != TensorRtDataType.Float)
            {
                throw new NotSupportedException($"Generic bounded runtime supports float output tensors only. Output '{output.Name}' is {output.DataType}.");
            }
        }

        TensorRtDims runtimeShape = ResolveRuntimeInputShape(input, options);
        if (ShouldSetInputShape(input, runtimeShape, options))
        {
            bindings.SetInputShape(input.Name, runtimeShape);
        }

        float[] inputValues = CreateRuntimeInputValues(input.Name, CountElements(runtimeShape), options.RuntimeOptions.LoadInputs);
        bindings.CopyInputFromHost(input.Name, inputValues, runtimeShape);
        _ = bindings.GetReadiness(runShapeInference: true);

        Dictionary<string, TensorRtInferenceBuffer> outputBuffers = new Dictionary<string, TensorRtInferenceBuffer>(StringComparer.Ordinal);
        foreach (TensorRtEngineTensorBinding output in outputs)
        {
            outputBuffers[output.Name] = bindings.AllocateDeviceBuffer(output.Name);
        }

        TensorRtInferenceExecutionSummary executionSummary = null!;
        float elapsedMilliseconds = stream.MeasureElapsedTime(cudaStream =>
        {
            executionSummary = bindings.EnqueueAsync(cudaStream, synchronize: false, runShapeInference: true);
        });
        List<float> timingSamplesMilliseconds = new List<float> { elapsedMilliseconds };
        int avgRuns = options.RuntimeOptions.AvgRuns ?? 1;
        for (int runIndex = 1; runIndex < avgRuns; runIndex++)
        {
            TensorRtInferenceExecutionSummary benchmarkSummary = null!;
            float benchmarkMilliseconds = stream.MeasureElapsedTime(cudaStream =>
            {
                benchmarkSummary = bindings.EnqueueAsync(cudaStream, synchronize: false, runShapeInference: true);
            });
            executionSummary = benchmarkSummary;
            timingSamplesMilliseconds.Add(benchmarkMilliseconds);
        }

        List<OnnxEngineRuntimeOutputTensor> capturedOutputs = new List<OnnxEngineRuntimeOutputTensor>(outputs.Count);
        foreach (TensorRtEngineTensorBinding output in outputs)
        {
            TensorRtInferenceBuffer outputBuffer = outputBuffers[output.Name];
            TensorRtDims outputShape = outputBuffer.RuntimeShape ?? throw new InvalidOperationException($"Output '{output.Name}' does not have a concrete runtime shape.");
            int outputElementCount = CountElements(outputShape);
            float[] outputValues = bindings.ReadOutputSingles(output.Name, outputElementCount);
            capturedOutputs.Add(new OnnxEngineRuntimeOutputTensor(output.Name, outputShape.Values, outputValues));
        }

        OnnxEngineRuntimeOutputTensor primaryOutput = capturedOutputs[0];
        bool identityOutputMatch = capturedOutputs.Count == 1 &&
            primaryOutput.Values.Length == inputValues.Length &&
            ValuesEqual(inputValues, primaryOutput.Values);
        log.Add($"{statePrefix}BoundedRuntime Attempted=True Succeeded=True Input={input.Name}:{runtimeShape} Outputs={capturedOutputs.Count} PrimaryOutput={primaryOutput.Name}:{FormatShape(primaryOutput.Shape)} ElapsedMs={elapsedMilliseconds:0.###} IdentityOutputMatch={identityOutputMatch}");
        log.Add($"{statePrefix}BoundedRuntime OutputTensors=" + string.Join("; ", capturedOutputs.Select(static item => $"{item.Name}:{FormatShape(item.Shape)}:{item.Values.Length}")));

        OnnxEngineRuntimeArtifactData artifactData = identityOutputMatch
            ? OnnxEngineRuntimeArtifactData.CreateIdentityOutput(
                primaryOutput.Name,
                primaryOutput.Shape,
                inputValues,
                primaryOutput.Values,
                executionSummary.ToString(),
                timingSamplesMilliseconds)
            : OnnxEngineRuntimeArtifactData.CreateOutputSummary(
                primaryOutput.Name,
                primaryOutput.Shape,
                inputValues.Length,
                primaryOutput.Values,
                executionSummary.ToString(),
                timingSamplesMilliseconds);

        return new OnnxEngineRuntimeExecution(
            inferenceRan: true,
            outputMatch: identityOutputMatch,
            safeProfileIndex,
            elapsedMilliseconds,
            OnnxEngineBenchmarkSummary.Create(
                timingSamplesMilliseconds,
                options.RuntimeOptions,
                inferenceRan: true,
                outputMatch: identityOutputMatch),
            artifactData);
    }

    private static TensorRtDims ResolveRuntimeInputShape(TensorRtEngineTensorBinding input, OnnxEngineBuildOptions options)
    {
        if (options.ShapeProfile.TryGetShapeTriple(input.Name, out _, out EngineBuildShape opt, out _))
        {
            return new TensorRtDims(opt.Dimensions);
        }

        if (input.ProfileOptShape != null && CanEstimate(input.ProfileOptShape))
        {
            return input.ProfileOptShape;
        }

        if (input.EngineShape != null && CanEstimate(input.EngineShape))
        {
            return input.EngineShape;
        }

        throw new NotSupportedException($"Input '{input.Name}' does not have a concrete runtime shape. Provide --shapes or --optShapes for generic bounded runtime.");
    }

    private static bool ShouldSetInputShape(TensorRtEngineTensorBinding input, TensorRtDims runtimeShape, OnnxEngineBuildOptions options)
    {
        return !CanEstimate(input.EngineShape) ||
            input.ProfileOptShape != null ||
            !options.ShapeProfile.IsEmpty ||
            !ShapesEqual(input.EngineShape, runtimeShape);
    }

    private static float[] CreateRuntimeInputValues(string inputName, int expectedCount, string loadInputs)
    {
        Dictionary<string, string> inputs = ParseLoadInputs(loadInputs);
        if (inputs.Count == 0)
        {
            float[] generated = new float[expectedCount];
            for (int index = 0; index < generated.Length; index++)
            {
                generated[index] = index + 0.5f;
            }

            return generated;
        }

        if (!inputs.TryGetValue(inputName, out string? path))
        {
            throw new ArgumentException($"--loadInputs must include a mapping for input tensor '{inputName}'.");
        }

        return ReadFloatInputData(expectedCount, path);
    }

    private static Dictionary<string, string> ParseLoadInputs(string value)
    {
        Dictionary<string, string> result = new Dictionary<string, string>(StringComparer.Ordinal);
        if (string.IsNullOrWhiteSpace(value))
        {
            return result;
        }

        foreach (string segment in value.Split(new[] { ';', ',' }, StringSplitOptions.RemoveEmptyEntries))
        {
            int separator = segment.IndexOf(':');
            if (separator <= 0 || separator == segment.Length - 1)
            {
                throw new ArgumentException("--loadInputs entries must use tensor:path format.");
            }

            string tensorName = segment.Substring(0, separator).Trim();
            string path = Path.GetFullPath(segment.Substring(separator + 1).Trim().Trim('"'));
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("Input tensor file was not found.", path);
            }

            result[tensorName] = path;
        }

        return result;
    }

    private static float[] ReadFloatInputData(int expectedCount, string path)
    {
        string extension = Path.GetExtension(path);
        if (string.Equals(extension, ".bin", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(extension, ".raw", StringComparison.OrdinalIgnoreCase))
        {
            byte[] bytes = File.ReadAllBytes(path);
            if (bytes.Length % sizeof(float) != 0)
            {
                throw new ArgumentException($"Float input file byte length must be divisible by {sizeof(float)}.");
            }

            int actualCount = bytes.Length / sizeof(float);
            if (actualCount != expectedCount)
            {
                throw new ArgumentException($"Float input file has {actualCount} elements, expected {expectedCount}.");
            }

            float[] values = new float[actualCount];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }

        string text = File.ReadAllText(path);
        float[] parsed = text
            .Split(new[] { ',', ';', ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(static item => float.Parse(item, System.Globalization.CultureInfo.InvariantCulture))
            .ToArray();
        if (parsed.Length != expectedCount)
        {
            throw new ArgumentException($"Text input file has {parsed.Length} float values, expected {expectedCount}.");
        }

        return parsed;
    }

    private static int CountElements(TensorRtDims shape)
    {
        if (shape == null)
        {
            throw new ArgumentNullException(nameof(shape));
        }

        int result = 1;
        foreach (int value in shape.Values)
        {
            if (value <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(shape), "Shape must be concrete and positive.");
            }

            result = checked(result * value);
        }

        return result;
    }

    private static bool CanEstimate(TensorRtDims? shape)
    {
        if (shape == null || shape.Values.Length == 0)
        {
            return false;
        }

        foreach (int value in shape.Values)
        {
            if (value <= 0)
            {
                return false;
            }
        }

        return true;
    }

    private static bool ShapesEqual(TensorRtDims? left, TensorRtDims right)
    {
        return left != null && left.Values.SequenceEqual(right.Values);
    }

    private static bool ValuesEqual(IReadOnlyList<float> left, IReadOnlyList<float> right)
    {
        if (left.Count != right.Count)
        {
            return false;
        }

        for (int index = 0; index < left.Count; index++)
        {
            if (Math.Abs(left[index] - right[index]) > 1e-5f)
            {
                return false;
            }
        }

        return true;
    }

    private static string FormatShape(IReadOnlyList<int> values)
    {
        return string.Join("x", values);
    }

    private sealed class TimingCacheLease : IDisposable
    {
        public TimingCacheLease(TensorRtTimingCache? cache, OnnxEngineTimingCacheArtifact artifact)
        {
            Cache = cache;
            Artifact = artifact ?? OnnxEngineTimingCacheArtifact.Empty;
        }

        public TensorRtTimingCache? Cache { get; }

        public OnnxEngineTimingCacheArtifact Artifact { get; set; }

        public void Dispose()
        {
            Cache?.Dispose();
        }
    }

    private sealed class OnnxEngineRuntimeExecution
    {
        public OnnxEngineRuntimeExecution(
            bool inferenceRan,
            bool outputMatch,
            int profileIndex,
            float elapsedMilliseconds,
            OnnxEngineBenchmarkSummary benchmarkSummary,
            OnnxEngineRuntimeArtifactData artifactData)
        {
            InferenceRan = inferenceRan;
            OutputMatch = outputMatch;
            ProfileIndex = profileIndex;
            ElapsedMilliseconds = elapsedMilliseconds;
            BenchmarkSummary = benchmarkSummary ?? OnnxEngineBenchmarkSummary.Empty;
            ArtifactData = artifactData ?? OnnxEngineRuntimeArtifactData.Empty;
        }

        public bool InferenceRan { get; }

        public bool OutputMatch { get; }

        public int ProfileIndex { get; }

        public float ElapsedMilliseconds { get; }

        public OnnxEngineBenchmarkSummary BenchmarkSummary { get; }

        public OnnxEngineRuntimeArtifactData ArtifactData { get; }
    }

    private sealed class OnnxEngineRuntimeOutputTensor
    {
        public OnnxEngineRuntimeOutputTensor(string name, IReadOnlyList<int> shape, float[] values)
        {
            Name = name ?? string.Empty;
            Shape = shape ?? Array.Empty<int>();
            Values = values ?? Array.Empty<float>();
        }

        public string Name { get; }

        public IReadOnlyList<int> Shape { get; }

        public float[] Values { get; }
    }

    private static void ApplyDeploymentOptions(TensorRtBuilderConfig config, OnnxEngineBuildOptions options, List<string> log)
    {
        foreach (TrtexecLikeMemoryPoolSize memoryPool in options.DeploymentOptions.MemoryPoolSizes)
        {
            TensorRtMemoryPoolType pool = memoryPool.ToTensorRtMemoryPoolType();
            config.SetMemoryPoolLimit(pool, memoryPool.SizeBytes);
            ulong readbackBytes = config.GetMemoryPoolLimit(pool);
            log.Add(
                $"TrtexecMemoryPool Applied=True Name={memoryPool.Name} Pool={pool} " +
                $"RequestedBytes={memoryPool.SizeBytes} ReadbackBytes={readbackBytes} " +
                $"ReadbackMatch={readbackBytes == memoryPool.SizeBytes}");
        }

        if (options.DeploymentOptions.MaxAuxStreams.HasValue)
        {
            config.SetMaxAuxStreams(options.DeploymentOptions.MaxAuxStreams.Value);
        }

        if (options.DeploymentOptions.AvgTiming.HasValue)
        {
            int requestedIterations = options.DeploymentOptions.AvgTiming.Value;
            config.SetAverageTimingIterations(requestedIterations);
            int readbackIterations = config.GetAverageTimingIterations();
            log.Add(
                $"TrtexecTiming AverageApplied=True RequestedIterations={requestedIterations} " +
                $"ReadbackIterations={readbackIterations} ReadbackMatch={readbackIterations == requestedIterations} " +
                "EvidenceBoundary=builder-config-readback-only");
        }

        if (options.DeploymentOptions.MinTiming.HasValue)
        {
            if (options.TensorRtLine == TensorRtApiLine.TensorRt8)
            {
                int requestedIterations = options.DeploymentOptions.MinTiming.Value;
                config.SetMinTimingIterationsCompatibility(requestedIterations);
                int readbackIterations = config.MinTimingIterationsCompatibility;
                log.Add(
                    $"TrtexecTiming MinimumApplied=True VersionGuard=TRT8 RequestedIterations={requestedIterations} " +
                    $"ReadbackIterations={readbackIterations} ReadbackMatch={readbackIterations == requestedIterations} " +
                    "EvidenceBoundary=builder-config-readback-only");
            }
            else
            {
                log.Add(
                    $"TrtexecTiming MinimumApplied=False VersionGuard=TRT8 RequestedIterations={options.DeploymentOptions.MinTiming.Value} " +
                    "Reason=TensorRT 10/11 use average timing iterations; legacy minimum setter is not available on this API line.");
            }
        }

        if (!string.IsNullOrWhiteSpace(options.ProfilingVerbosity))
        {
            config.SetProfilingVerbosity(ParseProfilingVerbosity(options.ProfilingVerbosity));
        }

        if (options.TensorRtLine == TensorRtApiLine.TensorRt11 && options.DeploymentOptions.DlaCore.HasValue)
        {
            config.SetDefaultDeviceType(TensorRtDeviceType.Dla);
            config.SetDlaCore(options.DeploymentOptions.DlaCore.Value);
        }

        if (options.TensorRtLine == TensorRtApiLine.TensorRt11 && options.DeploymentOptions.AllowGpuFallback)
        {
            config.SetFlag(TensorRtBuilderFlag.GpuFallback, true);
        }
    }

    private static TensorRtProfilingVerbosity ParseProfilingVerbosity(string value)
    {
        if (string.Equals(value, "none", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtProfilingVerbosity.None;
        }

        if (string.Equals(value, "detailed", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtProfilingVerbosity.Detailed;
        }

        return TensorRtProfilingVerbosity.LayerNamesOnly;
    }

    private static string RuntimeOptionsLogLine(OnnxEngineBuildOptions options)
    {
        return $"TrtexecRuntime NoDataTransfers={options.RuntimeOptions.NoDataTransfers} UseSpinWait={options.RuntimeOptions.UseSpinWait} Threads={options.RuntimeOptions.Threads?.ToString() ?? ""} AvgRuns={options.RuntimeOptions.AvgRuns?.ToString() ?? ""} Percentile={options.RuntimeOptions.Percentile?.ToString() ?? ""} DumpOutput={options.RuntimeOptions.DumpOutput} ExportTimes={options.RuntimeOptions.ExportTimesPath} ExportProfile={options.RuntimeOptions.ExportProfilePath}";
    }
}
