using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class OnnxEngineBuildService
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
                : $"OnnxToEngine LoadEngine=BoundedRuntime InferenceRan=True OutputMatch={runtimeExecution.OutputMatch} OutputValidated={runtimeExecution.OutputValidated}");
            OnnxEngineBuildResult loadResult = CreateResult(
                success: !options.RuntimeOptions.RequestsReferenceValidation || (runtimeExecution?.OutputValidated ?? false),
                skipped: false,
                state: runtimeExecution != null
                    ? (runtimeExecution.OutputValidated
                        ? "load-engine-reference-validated-runtime"
                        : runtimeExecution.IdentityOutputMatch && !options.RuntimeOptions.RequestsReferenceValidation
                            ? "load-engine-identity-runtime"
                            : options.RuntimeOptions.RequestsReferenceValidation
                                ? "load-engine-reference-validation-failed"
                                : "load-engine-runtime-output-unverified")
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
                timingCacheArtifact: CreateTimingCacheBoundaryArtifact(options, "not-applied-to-load-engine"),
                outputValidated: runtimeExecution?.OutputValidated ?? false,
                identityOutputMatch: runtimeExecution?.IdentityOutputMatch ?? false);
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
                        : $"OnnxToEngine ExternalOnnx=BoundedRuntime InferenceRan=True OutputMatch={runtimeExecution.OutputMatch} OutputValidated={runtimeExecution.OutputValidated}");
                    OnnxEngineBuildResult externalRuntime = CreateResult(
                        success: !options.RuntimeOptions.RequestsReferenceValidation || (runtimeExecution?.OutputValidated ?? false),
                        skipped: false,
                        state: refitPersistenceSnapshot.Succeeded
                            ? RuntimeState(runtimeExecution, options, "external-onnx-refit-reload")
                            : refitSnapshot.Succeeded
                            ? RuntimeState(runtimeExecution, options, "external-onnx-refit")
                            : RuntimeState(runtimeExecution, options, "external-onnx"),
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
                        refitPersistenceSnapshot: refitPersistenceSnapshot,
                        outputValidated: runtimeExecution?.OutputValidated ?? false,
                        identityOutputMatch: runtimeExecution?.IdentityOutputMatch ?? false);
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
                if (!options.RuntimeOptions.NoDataTransfers && !identityRuntimeExecution.IdentityOutputMatch)
                {
                    throw new InvalidOperationException("Embedded identity runtime output did not match the generated input values.");
                }

                log.Add($"Parsed=True ProfileIndex={profileIndex} EngineFileRoundTrip=True");
                log.Add($"Execution ElapsedMs={identityRuntimeExecution.ElapsedMilliseconds:0.###} OutputMatch={identityRuntimeExecution.OutputMatch} NoDataTransfers={options.RuntimeOptions.NoDataTransfers}");
                log.Add("OnnxToEngine Passed=True");

                OnnxEngineBuildResult roundTrip = CreateResult(
                    success: !options.RuntimeOptions.RequestsReferenceValidation || identityRuntimeExecution.OutputValidated,
                    skipped: false,
                    state: options.RuntimeOptions.NoDataTransfers
                        ? "identity-no-data-transfer-benchmark"
                        : identityRuntimeExecution.OutputValidated
                            ? "identity-reference-validated-runtime"
                            : options.RuntimeOptions.RequestsReferenceValidation
                                ? "identity-reference-validation-failed"
                                : "identity-roundtrip",
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
                    parserPreflightSnapshot: parserPreflightSnapshot,
                    outputValidated: identityRuntimeExecution.OutputValidated,
                    identityOutputMatch: identityRuntimeExecution.IdentityOutputMatch);
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

}
