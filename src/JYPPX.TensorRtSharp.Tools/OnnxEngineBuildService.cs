using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
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

        if (!options.DryRun && options.DeploymentOptions.DeviceOrdinal.HasValue)
        {
            OnnxEngineBuildResult? selectedDeviceResult = null;
            Exception? selectedDeviceFailure = null;
            Thread executionThread = new Thread(() =>
            {
                try
                {
                    selectedDeviceResult = ExecuteCore(options);
                }
                catch (Exception exception)
                {
                    selectedDeviceFailure = exception;
                }
            })
            {
                IsBackground = true,
                Name = "TensorRtExec-device-" + options.DeploymentOptions.DeviceOrdinal.Value
            };
            executionThread.Start();
            executionThread.Join();
            if (selectedDeviceFailure != null)
            {
                ExceptionDispatchInfo.Capture(selectedDeviceFailure).Throw();
            }

            return selectedDeviceResult ?? throw new InvalidOperationException("The selected-device TensorRtExec thread completed without a result.");
        }

        return ExecuteCore(options);
    }

    private static OnnxEngineBuildResult ExecuteCore(OnnxEngineBuildOptions options)
    {
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

        if (options.DeploymentOptions.DeviceOrdinal.HasValue)
        {
            int requestedDevice = options.DeploymentOptions.DeviceOrdinal.Value;
            int deviceCount = CudaDevice.Count;
            if (requestedDevice >= deviceCount)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(options),
                    requestedDevice,
                    $"--device requested CUDA device {requestedDevice}, but this host reports {deviceCount} device(s).");
            }

            CudaDevice.SetCurrent(requestedDevice);
            int selectedDevice = CudaDevice.Current;
            log.Add(
                $"TrtexecDeploymentControl Name=Device Applied=True Requested={requestedDevice} " +
                $"Readback={selectedDevice} ReadbackMatch={selectedDevice == requestedDevice} " +
                $"DeviceCount={deviceCount} ExecutionThread={Thread.CurrentThread.ManagedThreadId}");
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
            ConfigureRuntimeForEnginePolicies(runtime, options, log, "Build");
            using TensorRtBuilder builder = new TensorRtBuilder(logger);
            using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
            using CudaStream stream = new CudaStream();

            config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, options.WorkspaceBytes);
            config.SetProfileStream(stream);
            config.SetOptimizationLevel(options.DeploymentOptions.BuilderOptimizationLevel);
            config.SetEngineCapability(TensorRtEngineCapability.Standard);
            ApplyDeploymentOptions(builder, config, options, log);
            ApplyPrecisionFlags(config, options, log);
            using TimingCacheLease timingCache = CreateTimingCacheLease(config, options, log);

            bool stronglyTypedApplied = ShouldCreateStronglyTypedNetwork(options, log);
            using TensorRtNetworkDefinition network = stronglyTypedApplied
                ? builder.CreateNetwork(stronglyTyped: true)
                : builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
            if (stronglyTypedApplied)
            {
                string readback = options.TensorRtLine == TensorRtApiLine.TensorRt11
                    ? "tensor-rt-11-always-strongly-typed"
                    : "network-created-with-tensor-rt-10-strongly-typed-flag";
                log.Add($"TrtexecDeploymentControl Name=StronglyTyped Applied=True Requested=True Readback={readback} ReadbackMatch=True");
            }
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
            string parserModelPath = options.UsesExternalOnnx ? Path.GetFileName(options.OnnxPath) : "sample-dynamic-identity.onnx";
            bool parsed = parser.Parse(model, parserModelPath);
            OnnxEngineParserPreflightSnapshot parserPreflightSnapshot = CaptureParserPreflightSnapshot(parser, model, parserModelPath, parsed, log);
            if (!parsed)
            {
                throw new InvalidOperationException(parser.GetErrorSummary());
            }

            TrtexecLikeBuildPolicy.Apply(config, network, options.DeploymentOptions, log);
            int profileIndex = AddOptimizationProfile(builder, config, options);
            TensorRtBuilderConfigDeploymentSnapshot? builderConfigDeploymentSnapshot = TryGetBuilderConfigDeploymentSnapshot(config, log);
            string enginePath = string.IsNullOrWhiteSpace(options.SaveEnginePath)
                ? Path.Combine(Path.GetTempPath(), $"jyppx-onnx-to-engine-{Guid.NewGuid():N}.plan")
                : options.SaveEnginePath;
            bool deleteEnginePath = string.IsNullOrWhiteSpace(options.SaveEnginePath);
            TensorRtEngine? refittedEngine = null;

            try
            {
                using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
                hostMemory.SaveToFile(enginePath);
                timingCache.Artifact = ExportTimingCache(timingCache, options, log);
                TryCollectLayerInformationFromSerializedEngine(runtime, enginePath, options, log, "Build");

                OnnxEngineRefitSnapshot refitSnapshot = OnnxEngineRefitSnapshot.Empty;
                OnnxEngineRefitPersistenceSnapshot refitPersistenceSnapshot = OnnxEngineRefitPersistenceSnapshot.Empty;
                if (!string.IsNullOrWhiteSpace(options.DeploymentOptions.RefitFromOnnxPath))
                {
                    refittedEngine = runtime.DeserializeFromFile(enginePath);
                    refitSnapshot = RefitStrippedEngineFromOnnx(
                        refittedEngine,
                        logger,
                        options.DeploymentOptions.RefitFromOnnxPath,
                        log);
                    if (!refitSnapshot.Succeeded || !refitSnapshot.ContextCreationAllowed)
                    {
                        throw new InvalidOperationException(
                            "ONNX stripped-plan refit did not reach the context-creation gate: " + refitSnapshot.DiagnosticSummary);
                    }

                    if (!string.IsNullOrWhiteSpace(options.DeploymentOptions.SaveRefittedEnginePath))
                    {
                        TensorRtEngine committedEngine = refittedEngine;
                        refittedEngine = null;
                        (refittedEngine, refitPersistenceSnapshot) = PersistAndReloadRefittedEngine(
                            runtime,
                            committedEngine,
                            enginePath,
                            options.DeploymentOptions.SaveRefittedEnginePath,
                            log);
                        if (!refitPersistenceSnapshot.Succeeded || !refitPersistenceSnapshot.ReloadContextCreationAllowed)
                        {
                            throw new InvalidOperationException("Refitted-plan persistence did not reach the independent reload gate.");
                        }
                    }
                }

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
                        state: refitPersistenceSnapshot.Succeeded
                            ? "external-onnx-refit-persisted-reload-build-only"
                            : refitSnapshot.Succeeded
                            ? "external-onnx-refit-complete-build-only"
                            : (options.UsesExternalOnnx ? "external-onnx-build-only" : "build-only"),
                        options,
                        modelSource,
                        enginePath,
                        parsed: true,
                        engineSaved: true,
                        engineFileRoundTrip: refitSnapshot.Succeeded,
                        inferenceRan: false,
                        outputMatch: false,
                        profileIndex,
                        elapsedMilliseconds: null,
                        skipReason: string.Empty,
                        log,
                        evidenceSidecar,
                        timingCacheArtifact: timingCache.Artifact,
                        builderConfigDeploymentSnapshot: builderConfigDeploymentSnapshot,
                        parserPreflightSnapshot: parserPreflightSnapshot,
                        refitSnapshot: refitSnapshot,
                        refitPersistenceSnapshot: refitPersistenceSnapshot);
                    OnnxEngineBuildDiagnostics.WriteReport(buildOnly, options.ExportReportPath);
                    OnnxEngineRuntimeArtifactWriter.WriteArtifacts(buildOnly);
                    return buildOnly;
                }

                if (externalRuntimeRequested)
                {
                    OnnxEngineRuntimeExecution? runtimeExecution = refittedEngine != null
                        ? TryRunGenericFloatEngine(
                            refittedEngine,
                            options,
                            profileIndex,
                            log,
                            statePrefix: refitPersistenceSnapshot.Succeeded ? "ExternalOnnxRefitReload" : "ExternalOnnxRefit",
                            validateRefittableState: !refitPersistenceSnapshot.Succeeded)
                        : TryRunGenericFloatEngineFromFile(options, enginePath, profileIndex, log, statePrefix: "ExternalOnnx");
                    if (refitPersistenceSnapshot.Succeeded)
                    {
                        refitPersistenceSnapshot = refitPersistenceSnapshot.WithRuntimeOutcome(
                            selectedForRuntime: true,
                            inferenceRan: runtimeExecution != null);
                    }
                    log.Add(runtimeExecution == null
                        ? "OnnxToEngine ExternalOnnx=RuntimeSkipped Note=Generic bounded runtime could not be executed."
                        : $"OnnxToEngine ExternalOnnx=BoundedRuntime InferenceRan=True OutputMatch={runtimeExecution.OutputMatch}");
                    OnnxEngineBuildResult externalRuntime = CreateResult(
                        success: true,
                        skipped: false,
                        state: refitPersistenceSnapshot.Succeeded
                            ? (runtimeExecution != null && runtimeExecution.OutputMatch ? "external-onnx-refit-reload-identity-runtime" : "external-onnx-refit-reload-runtime-output-unverified")
                            : refitSnapshot.Succeeded
                            ? (runtimeExecution != null && runtimeExecution.OutputMatch ? "external-onnx-refit-identity-runtime" : "external-onnx-refit-runtime-output-unverified")
                            : (runtimeExecution != null && runtimeExecution.OutputMatch ? "external-onnx-identity-runtime" : "external-onnx-runtime-output-unverified"),
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
                        loadedEngineDiagnostics: ProbeLoadedEngineDiagnostics(
                            options,
                            OnnxEnginePreflightMetadata.FromExistingEngine(
                                refitPersistenceSnapshot.Succeeded ? refitPersistenceSnapshot.PersistedPlanPath : enginePath),
                            log),
                        timingCacheArtifact: timingCache.Artifact,
                        builderConfigDeploymentSnapshot: builderConfigDeploymentSnapshot,
                        parserPreflightSnapshot: parserPreflightSnapshot,
                        refitSnapshot: refitSnapshot,
                        refitPersistenceSnapshot: refitPersistenceSnapshot);
                    OnnxEngineBuildDiagnostics.WriteReport(externalRuntime, options.ExportReportPath);
                    OnnxEngineRuntimeArtifactWriter.WriteArtifacts(externalRuntime, runtimeExecution?.ArtifactData);
                    return externalRuntime;
                }

                using TensorRtEngine engine = runtime.DeserializeFromFile(enginePath);
                OnnxEngineRuntimeExecution identityRuntimeExecution = TryRunGenericFloatEngine(
                    engine,
                    options,
                    profileIndex,
                    log,
                    statePrefix: "Identity");
                if (!options.RuntimeOptions.NoDataTransfers && !identityRuntimeExecution.OutputMatch)
                {
                    throw new InvalidOperationException("Embedded identity runtime output did not match the generated input values.");
                }

                log.Add($"Parsed=True ProfileIndex={profileIndex} EngineFileRoundTrip=True");
                log.Add($"Execution ElapsedMs={identityRuntimeExecution.ElapsedMilliseconds:0.###} OutputMatch={identityRuntimeExecution.OutputMatch} NoDataTransfers={options.RuntimeOptions.NoDataTransfers}");
                log.Add("OnnxToEngine Passed=True");

                OnnxEngineBuildResult roundTrip = CreateResult(
                    success: true,
                    skipped: false,
                    state: options.RuntimeOptions.NoDataTransfers ? "identity-no-data-transfer-benchmark" : "identity-roundtrip",
                    options,
                    modelSource,
                    enginePath,
                    parsed: true,
                    engineSaved: true,
                    engineFileRoundTrip: true,
                    inferenceRan: true,
                    outputMatch: identityRuntimeExecution.OutputMatch,
                    profileIndex,
                    identityRuntimeExecution.ElapsedMilliseconds,
                    skipReason: string.Empty,
                    log,
                    evidenceSidecar,
                    benchmarkSummary: identityRuntimeExecution.BenchmarkSummary,
                    timingCacheArtifact: timingCache.Artifact,
                    builderConfigDeploymentSnapshot: builderConfigDeploymentSnapshot,
                    parserPreflightSnapshot: parserPreflightSnapshot);
                OnnxEngineBuildDiagnostics.WriteReport(roundTrip, options.ExportReportPath);
                OnnxEngineRuntimeArtifactWriter.WriteArtifacts(roundTrip, identityRuntimeExecution.ArtifactData);
                return roundTrip;
            }
            finally
            {
                refittedEngine?.Dispose();
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

    private static void ApplyPrecisionFlags(TensorRtBuilderConfig config, OnnxEngineBuildOptions options, List<string> log)
    {
        if (options.Fp16)
        {
            ApplyGlobalPrecisionFlag(config, TensorRtBuilderFlag.Fp16, "Fp16", options.TensorRtLine != TensorRtApiLine.TensorRt11, log);
        }

        if (options.Int8)
        {
            ApplyGlobalPrecisionFlag(config, TensorRtBuilderFlag.Int8, "Int8", options.TensorRtLine != TensorRtApiLine.TensorRt11, log);
        }

        if (options.Bf16)
        {
            ApplyGlobalPrecisionFlag(config, TensorRtBuilderFlag.Bf16, "Bf16", options.TensorRtLine == TensorRtApiLine.TensorRt10, log);
        }

        config.SetFlag(TensorRtBuilderFlag.Tf32, options.Tf32);
        bool tf32Readback = config.GetFlag(TensorRtBuilderFlag.Tf32);
        bool tf32Match = tf32Readback == options.Tf32;
        log.Add($"TrtexecBuildPolicy Name=Tf32 Applied={tf32Match} Requested={options.Tf32} Readback={tf32Readback} ReadbackMatch={tf32Match}");
        if (!tf32Match)
        {
            throw new InvalidOperationException("TF32 builder flag did not match TensorRT readback.");
        }
    }

    private static void ApplyGlobalPrecisionFlag(
        TensorRtBuilderConfig config,
        TensorRtBuilderFlag flag,
        string name,
        bool supported,
        List<string> log)
    {
        if (!supported)
        {
            log.Add($"TrtexecBuildPolicy Name={name} Applied=False Requested=True VersionGuard={config.Line} Reason=builder-precision-flag-not-supported ReadbackMatch=False");
            return;
        }

        config.SetFlag(flag, true);
        bool readback = config.GetFlag(flag);
        log.Add($"TrtexecBuildPolicy Name={name} Applied={readback} Requested=True Readback={readback} ReadbackMatch={readback}");
        if (!readback)
        {
            throw new InvalidOperationException(name + " builder flag did not match TensorRT readback.");
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
        OnnxEngineTimingCacheArtifact? timingCacheArtifact = null,
        TensorRtBuilderConfigDeploymentSnapshot? builderConfigDeploymentSnapshot = null,
        OnnxEngineParserPreflightSnapshot? parserPreflightSnapshot = null,
        OnnxEngineRefitSnapshot? refitSnapshot = null,
        OnnxEngineRefitPersistenceSnapshot? refitPersistenceSnapshot = null)
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
            workspaceBytes: options.WorkspaceBytes,
            builderConfigDeploymentSnapshot: builderConfigDeploymentSnapshot,
            parserPreflightSnapshot: parserPreflightSnapshot,
            refitSnapshot: refitSnapshot,
            refitPersistenceSnapshot: refitPersistenceSnapshot);
    }

    private static (TensorRtEngine Engine, OnnxEngineRefitPersistenceSnapshot Snapshot) PersistAndReloadRefittedEngine(
        TensorRtRuntime runtime,
        TensorRtEngine refittedEngine,
        string strippedPlanPath,
        string persistedPlanPath,
        List<string> log)
    {
        const string boundary = "The snapshot proves that this committed engine was serialized, the original engine was disposed, and a distinct managed engine reloaded the persisted bytes. Runtime output still requires an explicit enqueue/baseline comparison and this is not package-consumer or public-release proof.";
        byte[] strippedPlan = File.ReadAllBytes(strippedPlanPath);
        byte[] persistedPlan;
        TensorRtSerializationFlags serializationFlagsBefore = TensorRtSerializationFlags.None;
        TensorRtSerializationFlags serializationFlagsAfter = TensorRtSerializationFlags.None;
        bool refittableWeightsIncluded = false;
        bool originalDisposed = false;
        try
        {
            using TensorRtSerializationConfig serializationConfig = refittedEngine.CreateSerializationConfig();
            serializationFlagsBefore = serializationConfig.Flags;
            bool excludeWeightsCleared = serializationConfig.ClearFlag(TensorRtSerializationFlag.ExcludeWeights);
            serializationFlagsAfter = serializationConfig.Flags;
            refittableWeightsIncluded = excludeWeightsCleared &&
                (serializationFlagsAfter & TensorRtSerializationFlags.ExcludeWeights) == 0;
            if (!refittableWeightsIncluded)
            {
                throw new InvalidOperationException("TensorRT did not clear ExcludeWeights before refitted-engine serialization.");
            }

            using TensorRtHostMemory serialized = refittedEngine.Serialize(serializationConfig);
            persistedPlan = serialized.ToArray();
            string? directory = Path.GetDirectoryName(persistedPlanPath);
            if (!string.IsNullOrWhiteSpace(directory))
            {
                Directory.CreateDirectory(directory);
            }

            File.WriteAllBytes(persistedPlanPath, persistedPlan);
        }
        finally
        {
            refittedEngine.Dispose();
            originalDisposed = true;
        }

        string strippedSha256 = ComputeSha256(strippedPlan);
        string persistedSha256 = ComputeSha256(persistedPlan);
        bool differsFromStripped = strippedPlan.LongLength != persistedPlan.LongLength ||
            !string.Equals(strippedSha256, persistedSha256, StringComparison.Ordinal);
        TensorRtEngine? reloadedEngine = null;
        try
        {
            reloadedEngine = runtime.DeserializeFromFile(persistedPlanPath);
            bool reloadRefittable = reloadedEngine.IsRefittable;
            int ioTensorCount = reloadedEngine.IOTensorCount;
            int layerCount = reloadedEngine.LayerCount;
            int profileCount = reloadedEngine.OptimizationProfileCount;
            bool reloadGate = ioTensorCount > 0 && layerCount > 0 && profileCount > 0;
            bool succeeded = persistedPlan.LongLength > 0 &&
                refittableWeightsIncluded &&
                originalDisposed &&
                differsFromStripped &&
                reloadGate;
            OnnxEngineRefitPersistenceSnapshot snapshot = new OnnxEngineRefitPersistenceSnapshot(
                attempted: true,
                succeeded,
                state: succeeded ? "refitted-plan-persisted-and-reloaded" : "refitted-plan-persistence-incomplete",
                strippedPlanPath,
                strippedPlanLengthBytes: strippedPlan.LongLength,
                strippedPlanSha256: strippedSha256,
                persistedPlanPath,
                persistedPlanLengthBytes: persistedPlan.LongLength,
                persistedPlanSha256: persistedSha256,
                serializationFlagsBefore,
                serializationFlagsAfter,
                refittableWeightsIncludedInSerialization: refittableWeightsIncluded,
                artifactDiffersFromStrippedPlan: differsFromStripped,
                originalRefittedEngineDisposedBeforeReload: originalDisposed,
                reloadAttempted: true,
                reloadSucceeded: true,
                reloadEngineRefittable: reloadRefittable,
                reloadIoTensorCount: ioTensorCount,
                reloadLayerCount: layerCount,
                reloadOptimizationProfileCount: profileCount,
                reloadContextCreationAllowed: succeeded,
                reloadEngineSelectedForRuntime: false,
                inferenceRanFromReloadedEngine: false,
                evidenceBoundary: boundary);
            log.Add(
                $"OnnxRefitPersistence Attempted=True Succeeded={succeeded} StrippedPlan={strippedPlanPath} StrippedLengthBytes={strippedPlan.LongLength} StrippedSha256={strippedSha256} " +
                $"PersistedPlan={persistedPlanPath} PersistedLengthBytes={persistedPlan.LongLength} PersistedSha256={persistedSha256} ArtifactDiffers={differsFromStripped} " +
                $"SerializationFlagsBefore={serializationFlagsBefore} SerializationFlagsAfter={serializationFlagsAfter} RefittableWeightsIncluded={refittableWeightsIncluded} " +
                $"OriginalDisposedBeforeReload={originalDisposed} ReloadSucceeded=True ReloadRefittable={reloadRefittable} IOTensors={ioTensorCount} Layers={layerCount} Profiles={profileCount} ContextCreationAllowed={snapshot.ReloadContextCreationAllowed}");
            if (!succeeded)
            {
                throw new InvalidOperationException("The refitted plan was written but did not pass the independent reload gate.");
            }

            TensorRtEngine result = reloadedEngine;
            reloadedEngine = null;
            return (result, snapshot);
        }
        finally
        {
            reloadedEngine?.Dispose();
        }
    }

    private static OnnxEngineRefitSnapshot RefitStrippedEngineFromOnnx(
        TensorRtEngine engine,
        TensorRtLogger logger,
        string sourcePath,
        List<string> log)
    {
        const string boundary = "Copied refit inventory and parser diagnostics prove only this in-memory engine lifecycle; they do not prove model accuracy, persistence of refitted weights in the stripped plan, package-consumer runtime, or public release readiness.";
        if (engine.Line == TensorRtApiLine.TensorRt8)
        {
            throw new NotSupportedException("ONNX parser-refitter is available only for TensorRT 10 and TensorRT 11.");
        }

        byte[] sourceBytes = File.ReadAllBytes(sourcePath);
        bool refittableBefore = engine.IsRefittable;
        if (!refittableBefore)
        {
            throw new InvalidOperationException("The deserialized stripped plan is not refittable.");
        }

        using TensorRtRefitter refitter = engine.CreateRefitter(logger);
        IReadOnlyList<string> missingBefore = CopyRefitEntries(refitter.GetMissingEntries());
        IReadOnlyList<string> allBefore = CopyRefitEntries(refitter.GetAllEntries());
        using TensorRtOnnxParserRefitter parserRefitter = refitter.CreateOnnxParserRefitter(logger);
        parserRefitter.ClearErrors();
        bool parserRefitReturned = parserRefitter.RefitFromFile(sourcePath);
        TensorRtOnnxParserRefitterDiagnosticSnapshot parserSnapshot = parserRefitter.GetDiagnosticSnapshot();
        bool engineRefitReturned = parserRefitReturned && parserSnapshot.ErrorCount == 0 && refitter.RefitCudaEngine();
        IReadOnlyList<string> missingAfter = CopyRefitEntries(refitter.GetMissingEntries());
        IReadOnlyList<string> allAfter = CopyRefitEntries(refitter.GetAllEntries());
        bool refittableAfter = engine.IsRefittable;
        bool succeeded = parserRefitReturned &&
            engineRefitReturned &&
            parserSnapshot.ErrorCount == 0 &&
            missingAfter.Count == 0 &&
            refittableAfter;

        OnnxEngineRefitSnapshot snapshot = new OnnxEngineRefitSnapshot(
            attempted: true,
            succeeded,
            state: succeeded ? "onnx-refit-complete" : "onnx-refit-incomplete",
            sourcePath,
            sourceLengthBytes: sourceBytes.LongLength,
            sourceSha256: ComputeSha256(sourceBytes),
            engineRefittableBefore: refittableBefore,
            engineRefittableAfter: refittableAfter,
            parserRefitReturned,
            engineRefitReturned,
            missingWeightsBefore: missingBefore,
            allWeightsBefore: allBefore,
            missingWeightsAfter: missingAfter,
            allWeightsAfter: allAfter,
            parserErrorCount: parserSnapshot.ErrorCount,
            copiedDiagnosticCount: parserSnapshot.Diagnostics.Count,
            diagnosticSummary: parserSnapshot.DiagnosticSummary,
            contextCreationAllowed: succeeded,
            evidenceBoundary: boundary);

        log.Add(
            $"OnnxRefitLifecycle Attempted=True Succeeded={succeeded} Source={sourcePath} SourceLengthBytes={sourceBytes.LongLength} " +
            $"SourceSha256={snapshot.SourceSha256} EngineRefittableBefore={refittableBefore} EngineRefittableAfter={refittableAfter} " +
            $"ParserRefitReturned={parserRefitReturned} EngineRefitReturned={engineRefitReturned} ParserErrors={parserSnapshot.ErrorCount} CopiedDiagnostics={parserSnapshot.Diagnostics.Count} " +
            $"MissingBefore={missingBefore.Count} AllBefore={allBefore.Count} MissingAfter={missingAfter.Count} AllAfter={allAfter.Count} " +
            $"ContextCreationAllowed={snapshot.ContextCreationAllowed}");
        log.Add(
            $"OnnxRefitInventory MissingBeforeSha256={ComputeSha256(string.Join("\n", missingBefore))} " +
            $"AllBeforeSha256={ComputeSha256(string.Join("\n", allBefore))} " +
            $"MissingAfterSha256={ComputeSha256(string.Join("\n", missingAfter))} " +
            $"AllAfterSha256={ComputeSha256(string.Join("\n", allAfter))}");
        return snapshot;
    }

    private static IReadOnlyList<string> CopyRefitEntries(IReadOnlyList<TensorRtRefitEntry> entries)
    {
        return entries.Select(static entry => entry.ToString()).ToArray();
    }

    private static OnnxEngineParserPreflightSnapshot CaptureParserPreflightSnapshot(
        TensorRtOnnxParser parser,
        byte[] model,
        string modelPath,
        bool parsed,
        List<string> log)
    {
        TensorRtOnnxParserDiagnosticSnapshot? diagnostics = null;
        try
        {
            diagnostics = parser.GetDiagnosticSnapshot();
        }
        catch (Exception exception) when (exception is TensorRtException || exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            log.Add($"ParserDiagnostics State=unavailable Reason={exception.GetType().Name}:{exception.Message}");
        }
        bool modelSupportAttempted = false;
        string modelSupportState = "not-attempted";
        bool modelSupported = false;
        long supportedSubgraphCount = 0;
        long unsupportedSubgraphCount = 0;
        int copiedSubgraphCount = 0;
        long copiedSupportedSubgraphCount = 0;
        long copiedUnsupportedSubgraphCount = 0;
        long copiedNodeCount = 0;

        try
        {
            TensorRtOnnxModelSupportSummary summary = parser.CheckModelSupport(model, modelPath).ToSummary();
            modelSupportAttempted = true;
            modelSupportState = "copied-readback";
            modelSupported = summary.IsSupported;
            supportedSubgraphCount = summary.ReportedSupportedSubgraphCount;
            unsupportedSubgraphCount = summary.ReportedUnsupportedSubgraphCount;
            copiedSubgraphCount = summary.CopiedSubgraphCount;
            copiedSupportedSubgraphCount = summary.CopiedSupportedSubgraphCount;
            copiedUnsupportedSubgraphCount = summary.CopiedUnsupportedSubgraphCount;
            copiedNodeCount = summary.CopiedNodeCount;
        }
        catch (Exception exception) when (exception is TensorRtException || exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            modelSupportState = "unavailable";
            log.Add($"ParserModelSupport State=unavailable Reason={exception.GetType().Name}:{exception.Message}");
        }

        OnnxEngineParserPreflightSnapshot snapshot = new OnnxEngineParserPreflightSnapshot(
            parser.Line, true, parsed, diagnostics == null ? "unavailable" : "copied-readback", diagnostics?.ErrorCount ?? 0, diagnostics?.Diagnostics.Count ?? 0,
            diagnostics?.DiagnosticSummary ?? string.Empty, diagnostics?.IdentityOperatorSupported ?? false, modelSupportAttempted,
            modelSupportState, modelSupported, supportedSubgraphCount, unsupportedSubgraphCount,
            copiedSubgraphCount, copiedSupportedSubgraphCount, copiedUnsupportedSubgraphCount, copiedNodeCount);
        log.Add($"ParserPreflightSnapshot State={snapshot.DiagnosticsState} Errors={snapshot.ErrorCount} Diagnostics={snapshot.CopiedDiagnosticCount} Identity={snapshot.IdentityOperatorSupported} ModelSupport={snapshot.ModelSupportState}:{snapshot.ModelSupported} Subgraphs={snapshot.CopiedSubgraphCount}");
        return snapshot;
    }

    private static TensorRtBuilderConfigDeploymentSnapshot? TryGetBuilderConfigDeploymentSnapshot(
        TensorRtBuilderConfig config,
        List<string> log)
    {
        try
        {
            TensorRtBuilderConfigDeploymentSnapshot snapshot = config.GetDeploymentSnapshot();
            log.Add($"BuilderConfigDeploymentSnapshot State=copied-readback Diagnostics={snapshot.Diagnostics.Count} Summary={snapshot}");
            return snapshot;
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            log.Add($"BuilderConfigDeploymentSnapshot State=unavailable Reason={exception.GetType().Name}:{exception.Message}");
            return null;
        }
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
        bool weightStreamingRequested = deployment.AllowWeightStreaming || deployment.WeightStreamingBudget.IsSpecified;

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
            ConfigureRuntimeForEnginePolicies(runtime, options, log, "LoadEngineDiagnostics");
            using TensorRtEngine engine = runtime.DeserializeFromFile(preflightMetadata.Path);
            using TensorRtEngineInspector inspector = engine.CreateInspector();
            IReadOnlyList<TensorRtTensorInfo> tensors = engine.GetIOTensors();
            string inspectorInformation = inspector.GetEngineInformation(TensorRtLayerInformationFormat.Oneline);
            TryCollectLayerInformation(inspector, engine.LayerCount, options, log, "LoadEngine");
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

    private static void TryCollectLayerInformationFromSerializedEngine(
        TensorRtRuntime runtime,
        string enginePath,
        OnnxEngineBuildOptions options,
        List<string> log,
        string source)
    {
        if (!options.DumpLayerInfo && string.IsNullOrWhiteSpace(options.ExportLayerInfoPath))
        {
            return;
        }

        try
        {
            using TensorRtEngine engine = runtime.DeserializeFromFile(enginePath);
            using TensorRtEngineInspector inspector = engine.CreateInspector();
            TryCollectLayerInformation(inspector, engine.LayerCount, options, log, source);
        }
        catch (Exception exception) when (exception is TensorRtException ||
                                          exception is BridgeProbeException ||
                                          exception is CudaException ||
                                          exception is DllNotFoundException ||
                                          exception is BadImageFormatException ||
                                          exception is FileNotFoundException)
        {
            log.Add($"LayerInfo Collected=False Source={source} Reason={exception.GetType().Name}:{exception.Message} EvidenceBoundary=copied-engine-inspector-diagnostics-only");
        }
    }

    private static void TryCollectLayerInformation(
        TensorRtEngineInspector inspector,
        int layerCount,
        OnnxEngineBuildOptions options,
        List<string> log,
        string source)
    {
        if (!options.DumpLayerInfo && string.IsNullOrWhiteSpace(options.ExportLayerInfoPath))
        {
            return;
        }

        string[] layerInformation;
        try
        {
            layerInformation = Enumerable.Range(0, Math.Max(0, layerCount))
                .Select(index => $"Layer[{index}] {inspector.GetLayerInformation(index, TensorRtLayerInformationFormat.Oneline)}")
                .ToArray();
        }
        catch (Exception exception) when (exception is TensorRtException ||
                                          exception is BridgeProbeException ||
                                          exception is CudaException ||
                                          exception is DllNotFoundException ||
                                          exception is BadImageFormatException)
        {
            log.Add($"LayerInfo Collected=False Source={source} Reason={exception.GetType().Name}:{exception.Message} EvidenceBoundary=copied-engine-inspector-diagnostics-only");
            return;
        }

        string content = layerInformation.Length == 0
            ? string.Empty
            : string.Join(Environment.NewLine, layerInformation) + Environment.NewLine;
        Encoding utf8NoBom = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
        int byteCount = utf8NoBom.GetByteCount(content);
        string sha256 = ComputeSha256(content);
        log.Add($"LayerInfo Collected=True Source={source} Layers={layerInformation.Length} Bytes={byteCount} Sha256={sha256} EvidenceBoundary=copied-engine-inspector-diagnostics-only");

        if (options.DumpLayerInfo)
        {
            foreach (string line in layerInformation)
            {
                log.Add("LayerInfo " + line);
            }
        }

        if (!string.IsNullOrWhiteSpace(options.ExportLayerInfoPath))
        {
            string fullPath = Path.GetFullPath(options.ExportLayerInfoPath);
            string? directory = Path.GetDirectoryName(fullPath);
            if (!string.IsNullOrWhiteSpace(directory))
            {
                Directory.CreateDirectory(directory);
            }

            File.WriteAllText(fullPath, content, utf8NoBom);
            log.Add($"LayerInfo ExportRequested=True Written=True Path={fullPath} LengthBytes={byteCount} Sha256={sha256} EvidenceBoundary=copied-engine-inspector-diagnostics-only");
        }
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
        return options.RuntimeOptions.NoDataTransfers ||
            !string.IsNullOrWhiteSpace(options.RuntimeOptions.LoadInputs);
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
            ConfigureRuntimeForEnginePolicies(runtime, options, log, statePrefix + "Runtime");
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
        string statePrefix,
        bool validateRefittableState = true)
    {
        ApplyEngineRuntimePolicies(engine, options, log, validateRefittableState);
        int safeProfileIndex = Math.Max(0, profileIndex);
        int executionContextCount = options.RuntimeOptions.InfStreams ?? options.Streams;
        List<OnnxEngineBenchmarkWorker> workers = new List<OnnxEngineBenchmarkWorker>(executionContextCount);
        try
        {
            OnnxEngineBenchmarkWorker firstWorker = new OnnxEngineBenchmarkWorker(engine, safeProfileIndex, options.RuntimeOptions.UseSpinWait);
            workers.Add(firstWorker);

            IReadOnlyList<TensorRtEngineTensorBinding> inputs = firstWorker.Bindings.Report.GetInputs();
            IReadOnlyList<TensorRtEngineTensorBinding> outputs = firstWorker.Bindings.Report.GetOutputs();
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
            int inputElementCount = CountElements(runtimeShape);
            float[] inputValues = options.RuntimeOptions.NoDataTransfers
                ? Array.Empty<float>()
                : CreateRuntimeInputValues(input.Name, inputElementCount, options.RuntimeOptions.LoadInputs);
            bool setInputShape = ShouldSetInputShape(input, runtimeShape, options);
            firstWorker.Configure(input, outputs, runtimeShape, inputValues, setInputShape, options.RuntimeOptions.NoDataTransfers);
            for (int workerIndex = 1; workerIndex < executionContextCount; workerIndex++)
            {
                OnnxEngineBenchmarkWorker worker = new OnnxEngineBenchmarkWorker(engine, safeProfileIndex, options.RuntimeOptions.UseSpinWait);
                workers.Add(worker);
                worker.Configure(input, outputs, runtimeShape, inputValues, setInputShape, options.RuntimeOptions.NoDataTransfers);
            }

            OnnxEngineBenchmarkRun benchmark = RunBoundedBenchmark(workers, options);
            float elapsedMilliseconds = benchmark.TimingSamplesMilliseconds[0];

            List<OnnxEngineRuntimeOutputTensor> capturedOutputs = new List<OnnxEngineRuntimeOutputTensor>(outputs.Count);
            if (!options.RuntimeOptions.NoDataTransfers)
            {
                foreach (TensorRtEngineTensorBinding output in outputs)
                {
                    TensorRtInferenceBuffer outputBuffer = firstWorker.Bindings.Buffers[output.Name];
                    TensorRtDims outputShape = outputBuffer.RuntimeShape ?? throw new InvalidOperationException($"Output '{output.Name}' does not have a concrete runtime shape.");
                    int outputElementCount = CountElements(outputShape);
                    float[] outputValues = firstWorker.Bindings.ReadOutputSingles(output.Name, outputElementCount);
                    capturedOutputs.Add(new OnnxEngineRuntimeOutputTensor(output.Name, outputShape.Values, outputValues));
                }
            }

            OnnxEngineRuntimeOutputTensor? primaryOutput = capturedOutputs.Count == 0 ? null : capturedOutputs[0];
            bool identityOutputMatch = primaryOutput != null &&
                capturedOutputs.Count == 1 &&
                primaryOutput.Values.Length == inputValues.Length &&
                ValuesEqual(inputValues, primaryOutput.Values);
            string primaryOutputSummary = primaryOutput == null
                ? "not-read-back"
                : $"{primaryOutput.Name}:{FormatShape(primaryOutput.Shape)}";
            log.Add($"{statePrefix}BoundedRuntime Attempted=True Succeeded=True Input={input.Name}:{runtimeShape} Outputs={capturedOutputs.Count} PrimaryOutput={primaryOutputSummary} ElapsedMs={elapsedMilliseconds:0.###} IdentityOutputMatch={identityOutputMatch} NoDataTransfers={options.RuntimeOptions.NoDataTransfers}");
            log.Add(options.RuntimeOptions.NoDataTransfers
                ? $"{statePrefix}BoundedRuntime OutputReadback=False Reason=noDataTransfers"
                : $"{statePrefix}BoundedRuntime OutputTensors=" + string.Join("; ", capturedOutputs.Select(static item => $"{item.Name}:{FormatShape(item.Shape)}:{item.Values.Length}")));
            log.Add(
                $"RuntimeBenchmark Contexts={workers.Count} MeasurementRounds={benchmark.MeasurementRoundsExecuted} " +
                $"InferenceIterations={benchmark.TimingSamplesMilliseconds.Count} WarmUpIterations={benchmark.WarmUpIterationsExecuted} " +
                $"WarmUpElapsedMs={benchmark.WarmUpElapsedMilliseconds:0.###} MeasurementElapsedMs={benchmark.MeasurementElapsedMilliseconds:0.###} " +
                $"IterationsRequested={options.Iterations} DurationSecondsRequested={options.DurationSeconds} " +
                $"StreamsRequested={options.Streams} InfStreamsRequested={options.RuntimeOptions.InfStreams?.ToString() ?? ""} " +
                $"ThreadsApplied={benchmark.ThreadsExecuted} SpinWaitApplied={benchmark.UseSpinWaitApplied} " +
                $"NoDataTransfersApplied={options.RuntimeOptions.NoDataTransfers} CudaGraphApplied={benchmark.UseCudaGraphApplied} " +
                $"IdleTimeApplied={benchmark.IdleTimeMillisecondsApplied} SleepTimeApplied=0");
            if (!string.IsNullOrWhiteSpace(benchmark.UseCudaGraphFallbackReason))
            {
                log.Add($"RuntimeBenchmark CudaGraphFallbackReason={benchmark.UseCudaGraphFallbackReason}");
            }

            OnnxEngineRuntimeArtifactData artifactData = options.RuntimeOptions.NoDataTransfers
                ? OnnxEngineRuntimeArtifactData.CreateBenchmarkOnly(
                    inputElementCount,
                    benchmark.LastExecutionSummary.ToString(),
                    benchmark.TimingSamplesMilliseconds)
                : identityOutputMatch
                ? OnnxEngineRuntimeArtifactData.CreateIdentityOutput(
                    primaryOutput!.Name,
                    primaryOutput.Shape,
                    inputValues,
                    primaryOutput.Values,
                    benchmark.LastExecutionSummary.ToString(),
                    benchmark.TimingSamplesMilliseconds)
                : OnnxEngineRuntimeArtifactData.CreateOutputSummary(
                    primaryOutput!.Name,
                    primaryOutput.Shape,
                    inputValues.Length,
                    primaryOutput.Values,
                    benchmark.LastExecutionSummary.ToString(),
                    benchmark.TimingSamplesMilliseconds);

            return new OnnxEngineRuntimeExecution(
                inferenceRan: true,
                outputMatch: identityOutputMatch,
                safeProfileIndex,
                elapsedMilliseconds,
                OnnxEngineBenchmarkSummary.CreateExecuted(
                    benchmark.TimingSamplesMilliseconds,
                    options,
                    benchmark.MeasurementRoundsExecuted,
                    benchmark.WarmUpIterationsExecuted,
                    benchmark.WarmUpElapsedMilliseconds,
                    benchmark.MeasurementElapsedMilliseconds,
                    workers.Count,
                    benchmark.ThreadsExecuted,
                    benchmark.UseSpinWaitApplied,
                    benchmark.UseCudaGraphApplied,
                    benchmark.UseCudaGraphFallbackReason,
                    benchmark.MeasurementRoundsPerContext),
                artifactData);
        }
        finally
        {
            for (int index = workers.Count - 1; index >= 0; index--)
            {
                workers[index].Dispose();
            }
        }
    }

    private static void ApplyEngineRuntimePolicies(
        TensorRtEngine engine,
        OnnxEngineBuildOptions options,
        List<string> log,
        bool validateRefittableState)
    {
        TrtexecLikeDeploymentOptions deployment = options.DeploymentOptions;
        if (deployment.Refit && validateRefittableState)
        {
            bool refittable = engine.IsRefittable;
            log.Add($"TrtexecEnginePolicy Name=Refit Applied=True Requested=True Readback={refittable} ReadbackMatch={refittable}");
            if (!refittable)
            {
                throw new InvalidOperationException("TensorRT built an engine that is not refittable after --refit was applied.");
            }
        }
        else if (deployment.Refit)
        {
            log.Add("TrtexecEnginePolicy Name=Refit RuntimeRevalidation=False PersistenceCommitted=True Reason=full-weight-reload-does-not-require-refittable-state");
        }

        if (!deployment.WeightStreamingBudget.IsSpecified)
        {
            return;
        }

        if (options.TensorRtLine == TensorRtApiLine.TensorRt8)
        {
            log.Add(
                $"TrtexecDeploymentControl Name=WeightStreamingBudget Applied=False Requested={deployment.WeightStreamingBudget.ArgumentValue} " +
                "VersionGuard=TRT8 Reason=weight-streaming-runtime-api-is-not-available");
            return;
        }

        long streamableWeights = engine.StreamableWeightsSizeInBytes;
        long automaticBudget = engine.WeightStreamingAutomaticBudgetInBytes;
        long requestedBudget = deployment.WeightStreamingBudget.ResolveBytes(streamableWeights, automaticBudget);
        bool accepted = engine.SetWeightStreamingBudgetV2(requestedBudget);
        long readbackBudget = engine.WeightStreamingBudgetV2InBytes;
        long scratchBytes = engine.WeightStreamingScratchMemorySizeInBytes;
        bool readbackMatch = accepted && readbackBudget == requestedBudget;
        log.Add(
            $"TrtexecDeploymentControl Name=WeightStreamingBudget Applied={accepted} " +
            $"Requested={deployment.WeightStreamingBudget.ArgumentValue} Mode={deployment.WeightStreamingBudget.Kind} " +
            $"ResolvedBytes={requestedBudget} StreamableWeightsBytes={streamableWeights} AutomaticBudgetBytes={automaticBudget} " +
            $"Readback={readbackBudget} ScratchBytes={scratchBytes} ReadbackMatch={readbackMatch}");
        if (!readbackMatch)
        {
            throw new InvalidOperationException(
                $"TensorRT rejected or changed the requested weight-streaming budget. Requested={requestedBudget}, Readback={readbackBudget}.");
        }
    }

    private static void ConfigureRuntimeForEnginePolicies(
        TensorRtRuntime runtime,
        OnnxEngineBuildOptions options,
        List<string> log,
        string source)
    {
        if (!options.DeploymentOptions.VersionCompatible)
        {
            return;
        }

        runtime.EngineHostCodeAllowed = true;
        bool readback = runtime.EngineHostCodeAllowed;
        log.Add(
            $"TrtexecRuntimePolicy Name=EngineHostCodeAllowed Applied=True Requested=True " +
            $"Readback={readback} ReadbackMatch={readback} Source={source}");
        if (!readback)
        {
            throw new InvalidOperationException("TensorRT runtime did not enable host code for a version-compatible engine.");
        }
    }

    private static OnnxEngineBenchmarkRun RunBoundedBenchmark(
        IReadOnlyList<OnnxEngineBenchmarkWorker> workers,
        OnnxEngineBuildOptions options)
    {
        bool useCudaGraphApplied = TryEnableCudaGraphs(workers, options.UseCudaGraph, out string cudaGraphFallbackReason);
        return options.RuntimeOptions.UseThreads
            ? RunThreadedBoundedBenchmark(workers, options, useCudaGraphApplied, cudaGraphFallbackReason)
            : RunSingleThreadBoundedBenchmark(workers, options, useCudaGraphApplied, cudaGraphFallbackReason);
    }

    private static bool TryEnableCudaGraphs(
        IReadOnlyList<OnnxEngineBenchmarkWorker> workers,
        bool requested,
        out string fallbackReason)
    {
        fallbackReason = string.Empty;
        if (!requested)
        {
            return false;
        }

        for (int index = 0; index < workers.Count; index++)
        {
            if (workers[index].TryEnableCudaGraph(out string workerReason))
            {
                continue;
            }

            foreach (OnnxEngineBenchmarkWorker worker in workers)
            {
                worker.DisableCudaGraph();
            }

            fallbackReason = $"worker-{index}:{workerReason}";
            return false;
        }

        return true;
    }

    private static OnnxEngineBenchmarkRun RunSingleThreadBoundedBenchmark(
        IReadOnlyList<OnnxEngineBenchmarkWorker> workers,
        OnnxEngineBuildOptions options,
        bool useCudaGraphApplied,
        string cudaGraphFallbackReason)
    {
        int warmUpIterations = 0;
        Stopwatch warmUpStopwatch = Stopwatch.StartNew();
        while (warmUpStopwatch.ElapsedMilliseconds < options.WarmUpMilliseconds)
        {
            foreach (OnnxEngineBenchmarkWorker worker in workers)
            {
                worker.Enqueue();
            }

            foreach (OnnxEngineBenchmarkWorker worker in workers)
            {
                worker.RecordCompletion();
            }

            foreach (OnnxEngineBenchmarkWorker worker in workers)
            {
                worker.WaitForCompletion(options.RuntimeOptions.UseSpinWait);
            }

            warmUpIterations += workers.Count;
        }
        warmUpStopwatch.Stop();

        List<float> timingSamples = new List<float>();
        int measurementRounds = 0;
        TensorRtInferenceExecutionSummary? lastExecutionSummary = null;
        Stopwatch measurementStopwatch = Stopwatch.StartNew();
        TimeSpan minimumDuration = TimeSpan.FromSeconds(options.DurationSeconds);
        while (measurementRounds < options.Iterations || measurementStopwatch.Elapsed < minimumDuration)
        {
            if (measurementRounds > 0 && options.RuntimeOptions.IdleTimeMilliseconds.GetValueOrDefault() > 0)
            {
                Thread.Sleep(options.RuntimeOptions.IdleTimeMilliseconds!.Value);
            }

            foreach (OnnxEngineBenchmarkWorker worker in workers)
            {
                worker.StartTiming();
                lastExecutionSummary = worker.Enqueue();
                worker.StopTiming();
            }

            foreach (OnnxEngineBenchmarkWorker worker in workers)
            {
                timingSamples.Add(worker.CompleteTiming(options.RuntimeOptions.UseSpinWait));
            }

            measurementRounds++;
        }
        measurementStopwatch.Stop();

        if (lastExecutionSummary == null || timingSamples.Count == 0)
        {
            throw new InvalidOperationException("Bounded benchmark did not execute any inference iterations.");
        }

        return new OnnxEngineBenchmarkRun(
            timingSamples,
            lastExecutionSummary,
            measurementRounds,
            warmUpIterations,
            warmUpStopwatch.Elapsed.TotalMilliseconds,
            measurementStopwatch.Elapsed.TotalMilliseconds,
            measurementRounds > 1 ? options.RuntimeOptions.IdleTimeMilliseconds.GetValueOrDefault() : 0,
            threadsExecuted: 1,
            useSpinWaitApplied: options.RuntimeOptions.UseSpinWait,
            useCudaGraphApplied,
            cudaGraphFallbackReason,
            Enumerable.Repeat(measurementRounds, workers.Count).ToArray());
    }

    private static OnnxEngineBenchmarkRun RunThreadedBoundedBenchmark(
        IReadOnlyList<OnnxEngineBenchmarkWorker> workers,
        OnnxEngineBuildOptions options,
        bool useCudaGraphApplied,
        string cudaGraphFallbackReason)
    {
        int deviceOrdinal = CudaDevice.Current;
        OnnxEngineWorkerRun?[] runs = new OnnxEngineWorkerRun?[workers.Count];
        Exception?[] failures = new Exception?[workers.Count];
        Thread[] threads = new Thread[workers.Count];
        using CountdownEvent ready = new CountdownEvent(workers.Count);
        using CountdownEvent warmUpCompleted = new CountdownEvent(workers.Count);
        using ManualResetEventSlim start = new ManualResetEventSlim(false);
        using ManualResetEventSlim measurementStart = new ManualResetEventSlim(false);

        for (int index = 0; index < workers.Count; index++)
        {
            int workerIndex = index;
            threads[index] = new Thread(() =>
            {
                bool readySignaled = false;
                bool warmUpSignaled = false;
                try
                {
                    CudaDevice.SetCurrent(deviceOrdinal);
                    ready.Signal();
                    readySignaled = true;
                    start.Wait();

                    OnnxEngineWorkerWarmUp warmUp = RunWorkerWarmUp(workers[workerIndex], options);
                    warmUpCompleted.Signal();
                    warmUpSignaled = true;
                    measurementStart.Wait();
                    runs[workerIndex] = RunWorkerMeasurement(workers[workerIndex], options, warmUp);
                }
                catch (Exception exception)
                {
                    failures[workerIndex] = exception;
                }
                finally
                {
                    if (!readySignaled)
                    {
                        ready.Signal();
                    }

                    if (!warmUpSignaled)
                    {
                        warmUpCompleted.Signal();
                    }
                }
            })
            {
                IsBackground = true,
                Name = $"TensorRtExec-worker-{index}"
            };
            threads[index].Start();
        }

        ready.Wait();
        start.Set();
        warmUpCompleted.Wait();
        measurementStart.Set();
        foreach (Thread thread in threads)
        {
            thread.Join();
        }

        Exception? failure = failures.FirstOrDefault(static item => item != null);
        if (failure != null)
        {
            ExceptionDispatchInfo.Capture(failure).Throw();
        }

        OnnxEngineWorkerRun[] completedRuns = runs.Select(static item => item ?? throw new InvalidOperationException("A TensorRtExec benchmark worker completed without a result.")).ToArray();
        float[] timingSamples = completedRuns.SelectMany(static item => item.TimingSamplesMilliseconds).ToArray();
        if (timingSamples.Length == 0)
        {
            throw new InvalidOperationException("Bounded benchmark did not execute any inference iterations.");
        }

        return new OnnxEngineBenchmarkRun(
            timingSamples,
            completedRuns[completedRuns.Length - 1].LastExecutionSummary,
            completedRuns.Min(static item => item.MeasurementRoundsExecuted),
            completedRuns.Sum(static item => item.WarmUpIterationsExecuted),
            completedRuns.Max(static item => item.WarmUpElapsedMilliseconds),
            completedRuns.Max(static item => item.MeasurementElapsedMilliseconds),
            completedRuns.Any(static item => item.IdleTimeMillisecondsApplied > 0)
                ? options.RuntimeOptions.IdleTimeMilliseconds.GetValueOrDefault()
                : 0,
            threadsExecuted: workers.Count,
            useSpinWaitApplied: options.RuntimeOptions.UseSpinWait,
            useCudaGraphApplied,
            cudaGraphFallbackReason,
            completedRuns.Select(static item => item.MeasurementRoundsExecuted).ToArray());
    }

    private static OnnxEngineWorkerWarmUp RunWorkerWarmUp(
        OnnxEngineBenchmarkWorker worker,
        OnnxEngineBuildOptions options)
    {
        int iterations = 0;
        Stopwatch stopwatch = Stopwatch.StartNew();
        while (stopwatch.ElapsedMilliseconds < options.WarmUpMilliseconds)
        {
            worker.Enqueue();
            worker.RecordCompletion();
            worker.WaitForCompletion(options.RuntimeOptions.UseSpinWait);
            iterations++;
        }

        stopwatch.Stop();
        return new OnnxEngineWorkerWarmUp(iterations, stopwatch.Elapsed.TotalMilliseconds);
    }

    private static OnnxEngineWorkerRun RunWorkerMeasurement(
        OnnxEngineBenchmarkWorker worker,
        OnnxEngineBuildOptions options,
        OnnxEngineWorkerWarmUp warmUp)
    {
        List<float> timingSamples = new List<float>();
        int measurementRounds = 0;
        TensorRtInferenceExecutionSummary? lastExecutionSummary = null;
        Stopwatch stopwatch = Stopwatch.StartNew();
        TimeSpan minimumDuration = TimeSpan.FromSeconds(options.DurationSeconds);
        while (measurementRounds < options.Iterations || stopwatch.Elapsed < minimumDuration)
        {
            if (measurementRounds > 0 && options.RuntimeOptions.IdleTimeMilliseconds.GetValueOrDefault() > 0)
            {
                Thread.Sleep(options.RuntimeOptions.IdleTimeMilliseconds!.Value);
            }

            worker.StartTiming();
            lastExecutionSummary = worker.Enqueue();
            worker.StopTiming();
            timingSamples.Add(worker.CompleteTiming(options.RuntimeOptions.UseSpinWait));
            measurementRounds++;
        }

        stopwatch.Stop();
        if (lastExecutionSummary == null || timingSamples.Count == 0)
        {
            throw new InvalidOperationException("Bounded benchmark worker did not execute any inference iterations.");
        }

        return new OnnxEngineWorkerRun(
            timingSamples,
            lastExecutionSummary,
            measurementRounds,
            warmUp.IterationsExecuted,
            warmUp.ElapsedMilliseconds,
            stopwatch.Elapsed.TotalMilliseconds,
            measurementRounds > 1 ? options.RuntimeOptions.IdleTimeMilliseconds.GetValueOrDefault() : 0);
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

    private sealed class OnnxEngineBenchmarkWorker : IDisposable
    {
        private readonly TensorRtExecutionContext _context;
        private readonly CudaEvent _startEvent;
        private readonly CudaEvent _stopEvent;
        private CudaGraph? _cudaGraph;
        private CudaGraphExec? _cudaGraphExec;
        private TensorRtInferenceExecutionSummary? _cudaGraphExecutionSummary;

        public OnnxEngineBenchmarkWorker(TensorRtEngine engine, int profileIndex, bool useSpinWait)
        {
            CudaStream? stream = null;
            TensorRtExecutionContext? context = null;
            TensorRtInferenceBindings? bindings = null;
            CudaEvent? startEvent = null;
            try
            {
                stream = new CudaStream(CudaStreamCreationFlags.NonBlocking);
                context = engine.CreateExecutionContext();
                bindings = new TensorRtInferenceBindings(engine, context, profileIndex);
                CudaEventCreationFlags eventFlags = useSpinWait
                    ? CudaEventCreationFlags.Default
                    : CudaEventCreationFlags.BlockingSync;
                startEvent = new CudaEvent(eventFlags);
                CudaEvent stopEvent = new CudaEvent(eventFlags);
                Stream = stream;
                _context = context;
                Bindings = bindings;
                _startEvent = startEvent;
                _stopEvent = stopEvent;
            }
            catch
            {
                startEvent?.Dispose();
                bindings?.Dispose();
                context?.Dispose();
                stream?.Dispose();
                throw;
            }
        }

        public CudaStream Stream { get; }

        public TensorRtInferenceBindings Bindings { get; }

        public void Configure(
            TensorRtEngineTensorBinding input,
            IReadOnlyList<TensorRtEngineTensorBinding> outputs,
            TensorRtDims runtimeShape,
            float[] inputValues,
            bool setInputShape,
            bool noDataTransfers)
        {
            if (setInputShape)
            {
                Bindings.SetInputShape(input.Name, runtimeShape);
            }

            if (noDataTransfers)
            {
                Bindings.AllocateDeviceBuffer(input.Name, runtimeShape);
            }
            else
            {
                Bindings.CopyInputFromHost(input.Name, inputValues, runtimeShape);
            }

            _ = Bindings.GetReadiness(runShapeInference: true);
            foreach (TensorRtEngineTensorBinding output in outputs)
            {
                Bindings.AllocateDeviceBuffer(output.Name);
            }

            Bindings.BindAll();
        }

        public TensorRtInferenceExecutionSummary Enqueue()
        {
            if (_cudaGraphExec != null)
            {
                _cudaGraphExec.Launch(Stream);
                return _cudaGraphExecutionSummary ?? throw new InvalidOperationException("CUDA graph execution summary is unavailable.");
            }

            return Bindings.EnqueueAsync(Stream, synchronize: false, runShapeInference: false);
        }

        public bool TryEnableCudaGraph(out string fallbackReason)
        {
            CudaGraph? graph = null;
            CudaGraphExec? graphExec = null;
            bool captureActive = false;
            try
            {
                _ = Bindings.EnqueueAsync(Stream, synchronize: false, runShapeInference: false);
                Stream.Synchronize();

                Stream.BeginCapture(CudaStreamCaptureMode.ThreadLocal);
                captureActive = true;
                TensorRtInferenceExecutionSummary executionSummary = Bindings.EnqueueAsync(Stream, synchronize: false, runShapeInference: false);
                graph = Stream.EndCapture();
                captureActive = false;
                graphExec = graph.Instantiate();

                _cudaGraph = graph;
                _cudaGraphExec = graphExec;
                _cudaGraphExecutionSummary = executionSummary;
                fallbackReason = string.Empty;
                return true;
            }
            catch (Exception exception) when (IsCudaGraphFallbackException(exception))
            {
                if (captureActive)
                {
                    try
                    {
                        using CudaGraph abandonedGraph = Stream.EndCapture();
                    }
                    catch (Exception cleanupException) when (IsCudaGraphFallbackException(cleanupException))
                    {
                    }
                }

                graphExec?.Dispose();
                graph?.Dispose();
                if (Stream.CaptureStatus != CudaStreamCaptureStatus.None)
                {
                    throw new InvalidOperationException("CUDA graph capture failed and the worker stream did not return to a reusable state.", exception);
                }

                fallbackReason = $"{exception.GetType().Name}:{SanitizeDiagnostic(exception.Message)}";
                return false;
            }
        }

        public void DisableCudaGraph()
        {
            _cudaGraphExec?.Dispose();
            _cudaGraphExec = null;
            _cudaGraph?.Dispose();
            _cudaGraph = null;
            _cudaGraphExecutionSummary = null;
        }

        public void StartTiming()
        {
            _startEvent.Record(Stream);
        }

        public void StopTiming()
        {
            _stopEvent.Record(Stream);
        }

        public void RecordCompletion()
        {
            _stopEvent.Record(Stream);
        }

        public void WaitForCompletion(bool useSpinWait)
        {
            if (useSpinWait)
            {
                while (!_stopEvent.IsReady())
                {
                    Thread.SpinWait(64);
                }

                return;
            }

            _stopEvent.Synchronize();
        }

        public float CompleteTiming(bool useSpinWait)
        {
            WaitForCompletion(useSpinWait);
            return _stopEvent.ElapsedTimeSince(_startEvent);
        }

        public void Dispose()
        {
            try
            {
                Stream.Synchronize();
            }
            catch (CudaException)
            {
            }

            DisableCudaGraph();
            _stopEvent.Dispose();
            _startEvent.Dispose();
            Bindings.Dispose();
            _context.Dispose();
            Stream.Dispose();
        }

        private static bool IsCudaGraphFallbackException(Exception exception)
        {
            return exception is CudaException ||
                exception is TensorRtException ||
                exception is NotSupportedException ||
                exception is InvalidOperationException ||
                exception is BridgeProbeException ||
                exception is DllNotFoundException ||
                exception is BadImageFormatException ||
                exception is FileNotFoundException;
        }

        private static string SanitizeDiagnostic(string value)
        {
            return (value ?? string.Empty).Replace('\r', ' ').Replace('\n', ' ').Trim();
        }
    }

    private sealed class OnnxEngineBenchmarkRun
    {
        public OnnxEngineBenchmarkRun(
            IReadOnlyList<float> timingSamplesMilliseconds,
            TensorRtInferenceExecutionSummary lastExecutionSummary,
            int measurementRoundsExecuted,
            int warmUpIterationsExecuted,
            double warmUpElapsedMilliseconds,
            double measurementElapsedMilliseconds,
            int idleTimeMillisecondsApplied,
            int threadsExecuted,
            bool useSpinWaitApplied,
            bool useCudaGraphApplied,
            string useCudaGraphFallbackReason,
            IReadOnlyList<int> measurementRoundsPerContext)
        {
            TimingSamplesMilliseconds = timingSamplesMilliseconds;
            LastExecutionSummary = lastExecutionSummary;
            MeasurementRoundsExecuted = measurementRoundsExecuted;
            WarmUpIterationsExecuted = warmUpIterationsExecuted;
            WarmUpElapsedMilliseconds = warmUpElapsedMilliseconds;
            MeasurementElapsedMilliseconds = measurementElapsedMilliseconds;
            IdleTimeMillisecondsApplied = idleTimeMillisecondsApplied;
            ThreadsExecuted = threadsExecuted;
            UseSpinWaitApplied = useSpinWaitApplied;
            UseCudaGraphApplied = useCudaGraphApplied;
            UseCudaGraphFallbackReason = useCudaGraphFallbackReason ?? string.Empty;
            MeasurementRoundsPerContext = measurementRoundsPerContext ?? Array.Empty<int>();
        }

        public IReadOnlyList<float> TimingSamplesMilliseconds { get; }

        public TensorRtInferenceExecutionSummary LastExecutionSummary { get; }

        public int MeasurementRoundsExecuted { get; }

        public int WarmUpIterationsExecuted { get; }

        public double WarmUpElapsedMilliseconds { get; }

        public double MeasurementElapsedMilliseconds { get; }

        public int IdleTimeMillisecondsApplied { get; }

        public int ThreadsExecuted { get; }

        public bool UseSpinWaitApplied { get; }

        public bool UseCudaGraphApplied { get; }

        public string UseCudaGraphFallbackReason { get; }

        public IReadOnlyList<int> MeasurementRoundsPerContext { get; }
    }

    private sealed class OnnxEngineWorkerWarmUp
    {
        public OnnxEngineWorkerWarmUp(int iterationsExecuted, double elapsedMilliseconds)
        {
            IterationsExecuted = iterationsExecuted;
            ElapsedMilliseconds = elapsedMilliseconds;
        }

        public int IterationsExecuted { get; }

        public double ElapsedMilliseconds { get; }
    }

    private sealed class OnnxEngineWorkerRun
    {
        public OnnxEngineWorkerRun(
            IReadOnlyList<float> timingSamplesMilliseconds,
            TensorRtInferenceExecutionSummary lastExecutionSummary,
            int measurementRoundsExecuted,
            int warmUpIterationsExecuted,
            double warmUpElapsedMilliseconds,
            double measurementElapsedMilliseconds,
            int idleTimeMillisecondsApplied)
        {
            TimingSamplesMilliseconds = timingSamplesMilliseconds;
            LastExecutionSummary = lastExecutionSummary;
            MeasurementRoundsExecuted = measurementRoundsExecuted;
            WarmUpIterationsExecuted = warmUpIterationsExecuted;
            WarmUpElapsedMilliseconds = warmUpElapsedMilliseconds;
            MeasurementElapsedMilliseconds = measurementElapsedMilliseconds;
            IdleTimeMillisecondsApplied = idleTimeMillisecondsApplied;
        }

        public IReadOnlyList<float> TimingSamplesMilliseconds { get; }

        public TensorRtInferenceExecutionSummary LastExecutionSummary { get; }

        public int MeasurementRoundsExecuted { get; }

        public int WarmUpIterationsExecuted { get; }

        public double WarmUpElapsedMilliseconds { get; }

        public double MeasurementElapsedMilliseconds { get; }

        public int IdleTimeMillisecondsApplied { get; }
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

    private static void ApplyDeploymentOptions(TensorRtBuilder builder, TensorRtBuilderConfig config, OnnxEngineBuildOptions options, List<string> log)
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

        ApplyBuilderScalarDeploymentControls(config, options, log);

        if (!string.IsNullOrWhiteSpace(options.ProfilingVerbosity))
        {
            config.SetProfilingVerbosity(ParseProfilingVerbosity(options.ProfilingVerbosity));
        }

        TrtexecLikeDeploymentOptions deployment = options.DeploymentOptions;
        if (deployment.DlaCore.HasValue)
        {
            int requestedCore = deployment.DlaCore.Value;
            int dlaCoreCount = builder.DlaCoreCount;
            if (requestedCore >= dlaCoreCount)
            {
                throw new InvalidOperationException($"--useDLACore requested core {requestedCore}, but TensorRT reports {dlaCoreCount} DLA core(s).");
            }

            config.SetDefaultDeviceType(TensorRtDeviceType.Dla);
            config.SetDlaCore(requestedCore);
            TensorRtDeviceType deviceReadback = config.GetDefaultDeviceType();
            int coreReadback = config.GetDlaCore();
            log.Add(
                $"TrtexecDeploymentControl Name=DlaCore Applied=True Requested={requestedCore} " +
                $"Readback={coreReadback} DeviceReadback={deviceReadback} DlaCoreCount={dlaCoreCount} " +
                $"ReadbackMatch={coreReadback == requestedCore && deviceReadback == TensorRtDeviceType.Dla}");
        }

        if (deployment.AllowGpuFallback)
        {
            config.SetFlag(TensorRtBuilderFlag.GpuFallback, true);
            bool readback = config.GetFlag(TensorRtBuilderFlag.GpuFallback);
            log.Add($"TrtexecDeploymentControl Name=GpuFallback Applied=True Requested=True Readback={readback} ReadbackMatch={readback}");
        }

        if (!string.IsNullOrWhiteSpace(deployment.TacticSources))
        {
            TensorRtTacticSources defaultSources = config.GetTacticSources();
            TensorRtTacticSources requestedSources = deployment.ResolveTacticSources(defaultSources);
            config.SetTacticSources(requestedSources);
            TensorRtTacticSources readbackSources = config.GetTacticSources();
            log.Add(
                $"TrtexecDeploymentControl Name=TacticSources Applied=True Requested={requestedSources} " +
                $"Readback={readbackSources} Default={defaultSources} ReadbackMatch={readbackSources == requestedSources}");
        }

        if (deployment.DirectIO)
        {
            config.SetFlag(TensorRtBuilderFlag.DirectIO, true);
            bool readback = config.GetFlag(TensorRtBuilderFlag.DirectIO);
            log.Add($"TrtexecDeploymentControl Name=DirectIO Applied=True Requested=True Readback={readback} ReadbackMatch={readback}");
        }

        if (string.Equals(deployment.Sparsity, "enable", StringComparison.Ordinal) ||
            string.Equals(deployment.Sparsity, "disable", StringComparison.Ordinal))
        {
            bool requested = string.Equals(deployment.Sparsity, "enable", StringComparison.Ordinal);
            config.SetFlag(TensorRtBuilderFlag.SparseWeights, requested);
            bool readback = config.GetFlag(TensorRtBuilderFlag.SparseWeights);
            log.Add($"TrtexecDeploymentControl Name=Sparsity Applied=True Requested={deployment.Sparsity} Readback={readback} ReadbackMatch={readback == requested}");
        }
        else if (string.Equals(deployment.Sparsity, "force", StringComparison.Ordinal))
        {
            log.Add("TrtexecDeploymentControl Name=Sparsity Applied=False Requested=force Reason=official-force-mode-rewrites-model-weights-and-is-not-implemented");
        }

        ApplyEnginePackagingOptions(config, options, log);
    }

    private static void ApplyEnginePackagingOptions(
        TensorRtBuilderConfig config,
        OnnxEngineBuildOptions options,
        List<string> log)
    {
        TrtexecLikeDeploymentOptions deployment = options.DeploymentOptions;
        ApplyBuilderFlagWithReadback(config, TensorRtBuilderFlag.VersionCompatible, deployment.VersionCompatible, "VersionCompatible", log);
        ApplyBuilderFlagWithReadback(config, TensorRtBuilderFlag.ExcludeLeanRuntime, deployment.ExcludeLeanRuntime, "ExcludeLeanRuntime", log);
        if (deployment.Refit && options.TensorRtLine == TensorRtApiLine.TensorRt8 && deployment.VersionCompatible)
        {
            log.Add("TrtexecDeploymentControl Name=Refit Applied=False Requested=True VersionGuard=TRT8 Reason=version-compatible-refit-vendor-readback-conflict");
        }
        else
        {
            ApplyBuilderFlagWithReadback(config, TensorRtBuilderFlag.Refit, deployment.Refit, "Refit", log);
        }

        if (deployment.StripWeights)
        {
            if (options.TensorRtLine == TensorRtApiLine.TensorRt8)
            {
                log.Add("TrtexecDeploymentControl Name=StripWeights Applied=False Requested=True VersionGuard=TRT8 Reason=strip-plan-and-refit-identical-flags-are-not-available");
            }
            else
            {
                TensorRtBuilderFlag refitMode = deployment.Refit
                    ? TensorRtBuilderFlag.Refit
                    : TensorRtBuilderFlag.RefitIdentical;
                config.SetFlag(refitMode, true);
                config.SetFlag(TensorRtBuilderFlag.StripPlan, true);
                bool stripReadback = config.GetFlag(TensorRtBuilderFlag.StripPlan);
                bool refitReadback = config.GetFlag(refitMode);
                log.Add(
                    $"TrtexecDeploymentControl Name=StripWeights Applied=True Requested=True Readback={stripReadback} " +
                    $"RefitMode={refitMode} RefitReadback={refitReadback} ReadbackMatch={stripReadback && refitReadback}");
            }
        }

        if (deployment.AllowWeightStreaming)
        {
            if (options.TensorRtLine == TensorRtApiLine.TensorRt8)
            {
                log.Add("TrtexecDeploymentControl Name=WeightStreaming Applied=False Requested=True VersionGuard=TRT8 Reason=weight-streaming-builder-flag-is-not-available");
            }
            else
            {
                ApplyBuilderFlagWithReadback(config, TensorRtBuilderFlag.WeightStreaming, true, "WeightStreaming", log);
            }
        }
    }

    private static void ApplyBuilderFlagWithReadback(
        TensorRtBuilderConfig config,
        TensorRtBuilderFlag flag,
        bool requested,
        string name,
        List<string> log)
    {
        if (!requested)
        {
            return;
        }

        config.SetFlag(flag, true);
        bool readback = config.GetFlag(flag);
        log.Add($"TrtexecDeploymentControl Name={name} Applied={readback} Requested=True Readback={readback} ReadbackMatch={readback}");
    }

    private static bool ShouldCreateStronglyTypedNetwork(OnnxEngineBuildOptions options, List<string> log)
    {
        if (!options.DeploymentOptions.StronglyTyped)
        {
            return false;
        }

        if (options.TensorRtLine == TensorRtApiLine.TensorRt8)
        {
            log.Add("TrtexecDeploymentControl Name=StronglyTyped Applied=False Requested=True VersionGuard=TRT8 Reason=strongly-typed-network-creation-is-not-exposed-on-this-api-line");
            return false;
        }

        return true;
    }

    private static void ApplyBuilderScalarDeploymentControls(TensorRtBuilderConfig config, OnnxEngineBuildOptions options, List<string> log)
    {
        TrtexecLikeDeploymentOptions deployment = options.DeploymentOptions;
        if (deployment.MaxNbTactics.HasValue)
        {
            if (options.TensorRtLine == TensorRtApiLine.TensorRt8)
            {
                log.Add($"TrtexecBuilderScalar Name=MaxNbTactics Applied=False Requested={deployment.MaxNbTactics.Value} Reason=TensorRT8Unsupported");
            }
            else
            {
                config.SetMaxTactics(deployment.MaxNbTactics.Value);
                int readback = config.GetMaxTactics();
                log.Add($"TrtexecBuilderScalar Name=MaxNbTactics Applied=True Requested={deployment.MaxNbTactics.Value} Readback={readback} ReadbackMatch={readback == deployment.MaxNbTactics.Value}");
            }
        }

        if (deployment.TilingOptimizationLevel.HasValue)
        {
            if (options.TensorRtLine == TensorRtApiLine.TensorRt8)
            {
                log.Add($"TrtexecBuilderScalar Name=TilingOptimizationLevel Applied=False Requested={deployment.TilingOptimizationLevel.Value} Reason=TensorRT8Unsupported");
            }
            else
            {
                bool accepted = config.SetTilingOptimizationLevel(deployment.TilingOptimizationLevel.Value);
                TensorRtTilingOptimizationLevel readback = config.GetTilingOptimizationLevel();
                log.Add($"TrtexecBuilderScalar Name=TilingOptimizationLevel Applied={accepted} Requested={deployment.TilingOptimizationLevel.Value} Readback={readback} ReadbackMatch={readback == deployment.TilingOptimizationLevel.Value}");
            }
        }

        if (deployment.L2LimitForTilingBytes.HasValue)
        {
            if (options.TensorRtLine == TensorRtApiLine.TensorRt8)
            {
                log.Add($"TrtexecBuilderScalar Name=L2LimitForTiling Applied=False RequestedBytes={deployment.L2LimitForTilingBytes.Value} Reason=TensorRT8Unsupported");
            }
            else
            {
                bool accepted = config.SetL2LimitForTiling(deployment.L2LimitForTilingBytes.Value);
                long readback = config.GetL2LimitForTiling();
                log.Add($"TrtexecBuilderScalar Name=L2LimitForTiling Applied={accepted} RequestedBytes={deployment.L2LimitForTilingBytes.Value} ReadbackBytes={readback} ReadbackMatch={readback == deployment.L2LimitForTilingBytes.Value}");
            }
        }

        if (deployment.QuantizationFlags.HasValue)
        {
            if (options.TensorRtLine == TensorRtApiLine.TensorRt11)
            {
                log.Add($"TrtexecBuilderScalar Name=QuantizationFlags Applied=False Requested={deployment.QuantizationFlags.Value} Reason=RemovedByTensorRT11");
            }
            else
            {
                config.SetQuantizationFlags(deployment.QuantizationFlags.Value);
                TensorRtQuantizationFlags readback = config.GetQuantizationFlags();
                log.Add($"TrtexecBuilderScalar Name=QuantizationFlags Applied=True Requested={deployment.QuantizationFlags.Value} Readback={readback} ReadbackMatch={readback == deployment.QuantizationFlags.Value}");
            }
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
        return $"TrtexecRuntime Iterations={options.Iterations} WarmUpMs={options.WarmUpMilliseconds} DurationSeconds={options.DurationSeconds} Streams={options.Streams} InfStreams={options.RuntimeOptions.InfStreams?.ToString() ?? ""} NoDataTransfers={options.RuntimeOptions.NoDataTransfers} UseSpinWait={options.RuntimeOptions.UseSpinWait} Threads={options.RuntimeOptions.Threads?.ToString() ?? ""} AvgRuns={options.RuntimeOptions.AvgRuns?.ToString() ?? ""} Percentile={options.RuntimeOptions.Percentile?.ToString() ?? ""} IdleTimeMs={options.RuntimeOptions.IdleTimeMilliseconds?.ToString() ?? ""} SleepTimeMs={options.RuntimeOptions.SleepTimeMilliseconds?.ToString() ?? ""} DumpOutput={options.RuntimeOptions.DumpOutput} ExportTimes={options.RuntimeOptions.ExportTimesPath} ExportProfile={options.RuntimeOptions.ExportProfilePath}";
    }
}
