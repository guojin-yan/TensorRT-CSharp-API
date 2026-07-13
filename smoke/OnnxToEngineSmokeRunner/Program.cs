using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

internal static class Program
{
    private static void Main(string[] args)
    {
        TensorRtApiLine line = ResolveLine(JYPPX.SampleSupport.SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "10"));
        int batch = JYPPX.SampleSupport.SampleCommandLine.GetIntArgument(args, "--batch", 2);
        bool dependencyProbeOnly = JYPPX.SampleSupport.SampleCommandLine.HasSwitch(args, "--dependency-probe-only");
        if (batch < 1 || batch > 4)
        {
            throw new ArgumentOutOfRangeException(nameof(batch), "Batch must be in the optimization profile range [1, 4].");
        }

        byte[] model = OnnxIdentityModel.CreateDynamicBatchModel();
        Console.WriteLine($"OnnxToEngineSmokeRunner TensorRtLine={(int)line} ModelBytes={model.Length} Batch={batch}");
        TensorRtDependencyProbeReport dependencyProbe = TensorRtEnvironmentProbe.ProbeNativeDependencies(line);
        Console.WriteLine($"TensorRtDependencyProbe {FormatDependencyProbe(dependencyProbe)}");
        if (dependencyProbeOnly)
        {
            Console.WriteLine("Skipped=True Reason=DependencyProbeOnly");
            return;
        }

        TensorRtEnvironmentSnapshot environment;
        try
        {
            environment = TensorRtEnvironmentProbe.GetCurrent();
        }
        catch (Exception exception) when (IsSkippableEnvironmentException(exception))
        {
            Console.WriteLine($"Skipped=True Reason=EnvironmentProbe:{exception.GetType().Name}:{exception.Message}");
            return;
        }

        TensorRtAdapterInfo adapter = GetAdapterInfo(environment, line);
        Console.WriteLine($"TensorRtPreflight BridgeTrt={environment.BuildInfo.TensorRtVersion} Cuda={environment.BuildInfo.CudaToolkitVersion} RuntimeTensorRtAvailable={environment.RuntimeInfo.TensorRtAvailable} AdapterVendor={adapter.VendorDependencyAvailable} AdapterRuntime={adapter.RuntimeCreationSupported} AdapterBuilder={adapter.BuilderCreationSupported} Status={adapter.StatusMessage}");
        TensorRtRuntimeProbeReport runtimeProbe = TensorRtEnvironmentProbe.ProbeRuntime(line);
        Console.WriteLine($"TensorRtRuntimeProbe {FormatRuntimeProbe(runtimeProbe)}");
        bool runtimePreflightOk = runtimeProbe.RuntimeCreationSucceeded;
        string runtimePreflightMessage = GetRuntimeStageMessage(runtimeProbe);
        Console.WriteLine($"TensorRtRuntimePreflight Ok={runtimePreflightOk} Message={runtimePreflightMessage}");
        if (!runtimePreflightOk)
        {
            Console.WriteLine($"Skipped=True Reason=RuntimePreflightFailed:{runtimePreflightMessage}");
            return;
        }

        using TensorRtLogger logger = new TensorRtLogger(line);
        using TensorRtRuntime runtime = new TensorRtRuntime(logger);
        using TensorRtBuilder builder = new TensorRtBuilder(logger);
        string builderCaps = $"FastFp16={builder.PlatformHasFastFp16} FastInt8={builder.PlatformHasFastInt8} Tf32={builder.PlatformHasTf32} DlaCores={builder.DlaCoreCount}";
        string runtimeControls = ProbeRuntimeControls(runtime);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        using TensorRtOnnxConfig onnxConfig = new TensorRtOnnxConfig(line);
        onnxConfig.ModelDataType = TensorRtDataType.Float;
        onnxConfig.VerbosityLevel = 1;
        onnxConfig.ModelFileName = "generated-dynamic-identity.onnx";
        onnxConfig.TextFileName = "generated-dynamic-identity.layers.txt";
        onnxConfig.FullTextFileName = "generated-dynamic-identity.full.txt";
        onnxConfig.PrintLayerInfo = true;
        TensorRtOnnxConfigSnapshot onnxConfigSnapshot = onnxConfig.ToSnapshot();
        TensorRtOnnxConfigSummary onnxConfigSummary = onnxConfigSnapshot.ToSummary();
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
        ulong workspaceLimit = config.GetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace);
        using CudaStream stream = new CudaStream();
        config.SetProfileStream(stream);
        bool profileStreamSet = config.IsProfileStreamSet;
        config.SetFlag(TensorRtBuilderFlag.Tf32, true);
        config.SetFlag(TensorRtBuilderFlag.Refit, true);
        bool tf32Enabled = config.GetFlag(TensorRtBuilderFlag.Tf32);
        bool refitEnabled = config.GetFlag(TensorRtBuilderFlag.Refit);
        config.SetEngineCapability(TensorRtEngineCapability.Standard);
        config.SetHardwareCompatibilityLevel(TensorRtHardwareCompatibilityLevel.None);
        string builderConfigDeploymentState = ProbeBuilderConfigDeploymentState(config, line);
        using TensorRtTimingCache timingCache = config.CreateTimingCache();
        config.SetTimingCache(timingCache, ignoreMismatch: false);
        (TensorRtOnnxModelSupportReport parserModelSupport, string parserModelSupportFirstSubgraph) = ProbeModelSupport(builder, logger, model);
        TensorRtOnnxModelSupportSummary parserModelSupportSummary = parserModelSupport.ToSummary();
        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        TensorRtOnnxParser parser;
        try
        {
            parser = new TensorRtOnnxParser(logger, network);
        }
        catch (BridgeProbeException exception) when (exception.StatusCode == BridgeStatusCode.NotSupported)
        {
            Console.WriteLine($"Skipped=True Reason={exception.Message}");
            return;
        }

        string engineFile;
        using (parser)
        {
        uint parserFlagsBefore = (uint)parser.Flags;
        bool parserSupportsIdentity = parser.SupportsOperator("Identity");
        bool nativeInstanceNormalizationFlag = parser.GetFlag(TensorRtOnnxParserFlag.NativeInstanceNormalization);
        parser.SetFlag(TensorRtOnnxParserFlag.NativeInstanceNormalization);
        bool nativeInstanceNormalizationAfterSet = parser.GetFlag(TensorRtOnnxParserFlag.NativeInstanceNormalization);

        using MemoryStream parserModelStream = new MemoryStream(model, writable: false);
        bool parsed = parser.Parse(parserModelStream, "generated-dynamic-identity.onnx");
        if (!parsed)
        {
            throw new InvalidOperationException($"ONNX parser failed. {parser.GetErrorSummary()}");
        }

        TensorRtOnnxParserDiagnosticSnapshot parserDiagnosticSnapshot = parser.GetDiagnosticSnapshot();
        TensorRtOnnxParserDiagnosticSummary parserDiagnosticSummary = parserDiagnosticSnapshot.ToSummary();
        IReadOnlyList<string> parserUsedVCPluginLibraries = parserDiagnosticSnapshot.UsedVCPluginLibraries;
        string parserLayerOutputState = ProbeLayerOutputTensor(parser, "identity");
        string diagnosticProbe = ProbeParserDiagnostics(builder, logger, line);

        using TensorRtOptimizationProfile profile = builder.CreateOptimizationProfile();
        profile.SetShape(
            "input",
            new TensorRtDims(new[] { 1, 4 }),
            new TensorRtDims(new[] { 2, 4 }),
            new TensorRtDims(new[] { 4, 4 }));
        TensorRtOptimizationProfileShapeRange configuredProfileRange = profile.GetShapeRange("input");
        bool configuredProfileValid = profile.IsValid;
        float profileExtraMemoryTarget = profile.ExtraMemoryTarget;
        int inputShapeValueCount = profile.GetShapeValueCount("input");
        int profileIndex = config.AddOptimizationProfile(profile);
        string calibrationProfileState = ProbeCalibrationProfile(config, profile);
        int configProfileCount = config.OptimizationProfileCount;
        bool pluginInventoryOk = config.TryGetPluginsToSerialize(out IReadOnlyList<string> pluginInventory, out string pluginInventoryDiagnostic);
        bool pluginRegistryOk = builder.TryGetPluginRegistryInventory(out TensorRtPluginRegistryInventory pluginRegistryInventory, out string pluginRegistryDiagnostic);
        string pluginRegistryFirst = FormatPluginRegistryFirst(pluginRegistryInventory);
        string globalPluginRegistryLookupState = ProbeGlobalPluginRegistryLookup(runtimeProbe);

        using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
        using TensorRtHostMemory serializedTimingCache = timingCache.Serialize();
        byte[] engineBytes = hostMemory.ToArray();
        using MemoryStream copiedEngineStream = new MemoryStream();
        hostMemory.CopyTo(copiedEngineStream);
        byte[] streamEngineBytes = copiedEngineStream.ToArray();
        if (!engineBytes.SequenceEqual(streamEngineBytes))
        {
            throw new InvalidOperationException("TensorRT host-memory stream copy did not match the byte-array copy.");
        }

        copiedEngineStream.Position = 0;
        engineFile = Path.Combine(Path.GetTempPath(), $"jyppx-trt-engine-{Guid.NewGuid():N}.plan");
        hostMemory.SaveToFile(engineFile);
        using TensorRtEngine engine = runtime.Deserialize(copiedEngineStream);
        using TensorRtEngineInspector inspector = engine.CreateInspector();
        using TensorRtExecutionContext context = engine.CreateExecutionContext();
        inspector.SetExecutionContext(context);
        TensorRtTensorInfo compatibilityBinding = engine.GetCompatibilityBindingInfo(0);

        TensorRtDims minShape = engine.GetProfileShape("input", profileIndex, TensorRtOptimizationProfileSelector.Min);
        TensorRtDims optShape = engine.GetProfileShape("input", profileIndex, TensorRtOptimizationProfileSelector.Opt);
        TensorRtDims maxShape = engine.GetProfileShape("input", profileIndex, TensorRtOptimizationProfileSelector.Max);
        AssertDims(minShape, 1, 4);
        AssertDims(optShape, 2, 4);
        AssertDims(maxShape, 4, 4);
        int activeProfileBefore = context.OptimizationProfileIndex;
        context.SetOptimizationProfileAsync(profileIndex, stream);
        stream.Synchronize();
        int activeProfileAfter = context.OptimizationProfileIndex;
        bool originalEnqueueEmitsProfile = context.EnqueueEmitsProfile;
        context.EnqueueEmitsProfile = false;
        bool enqueueEmitsProfileAfterToggle = context.EnqueueEmitsProfile;
        context.EnqueueEmitsProfile = originalEnqueueEmitsProfile;
        context.SetInputShape("input", new TensorRtDims(new[] { batch, 4 }));
        int missingShapeInferenceCount = context.InferShapes();
        long maxOutputSize = context.GetMaxOutputSize("output");
        string tensorDebugState = ProbeTensorDebugState(context, "output");

        string refitterState;
        if (engine.IsRefittable)
        {
            using TensorRtRefitter refitter = engine.CreateRefitter(logger);
            string refitterControls = ProbeRefitterControls(refitter);
            string parserRefitterControls = ProbeParserRefitterControls(refitter, logger);
            bool refitted = refitter.RefitCudaEngine();
            refitterState = $"Created All={refitter.AllRefittableWeightCount} Missing={refitter.MissingWeightCount} Refit={refitted} Controls=[{refitterControls}] ParserRefitter=[{parserRefitterControls}]";
        }
        else
        {
            refitterState = "Skipped NonRefittable";
        }

        float[] inputValues = Enumerable.Range(0, batch * 4).Select(value => value + 0.25f).ToArray();
        TensorRtDims runtimeShape = new TensorRtDims(new[] { batch, 4 });
        using TensorRtInferenceBindings bindings = new TensorRtInferenceBindings(engine, context, profileIndex);
        bindings.SetInputShape("input", runtimeShape)
                .CopyInputFromHost("input", inputValues, runtimeShape);
        bindings.AllocateDeviceBuffer("output", runtimeShape, checked(inputValues.Length * sizeof(float)));
        bindings.BindAll();

        TensorRtExecutionContextReadiness readiness = bindings.GetReadiness();
        TensorRtEngineBindingReport bindingReport = bindings.Report;
        TensorRtInferenceExecutionSummary? bindingExecution = null;
        float bindingElapsedMilliseconds = stream.MeasureElapsedTime(cudaStream =>
        {
            bindingExecution = bindings.EnqueueAsync(cudaStream, synchronize: false, runShapeInference: false);
        });

        if (bindingExecution == null)
        {
            throw new InvalidOperationException("TensorRT binding execution did not produce a summary.");
        }

        float[] outputValues = bindings.ReadOutputSingles("output", inputValues.Length);
        bool match = inputValues.SequenceEqual(outputValues);
        if (!match)
        {
            throw new InvalidOperationException($"ONNX identity output mismatch. Input=[{string.Join(", ", inputValues)}] Output=[{string.Join(", ", outputValues)}]");
        }

        string inspectorText = inspector.GetEngineInformation(TensorRtLayerInformationFormat.Oneline);
        string bindingSummary = string.Join("; ", bindingReport.Tensors.Select(tensor => $"{tensor.Index}:{tensor.Name}:{tensor.IOMode}:{tensor.Format}:{tensor.VectorizedDimension}:{tensor.FormatDescription}"));
        Console.WriteLine($"Parsed=True ParserErrors={parser.ErrorCount} ProfileIndex={profileIndex} HostMemory={hostMemory.SizeInBytes}/{hostMemory.DataType}");
        Console.WriteLine($"BuilderCaps {builderCaps}");
        Console.WriteLine($"RuntimeControls {runtimeControls}");
        Console.WriteLine($"BuilderPluginRegistry Ok={pluginRegistryOk} Source={pluginRegistryInventory.Source} Creators={pluginRegistryInventory.CreatorCount} Recursive={pluginRegistryInventory.RecursiveCreatorCount?.ToString() ?? "n/a"} ParentSearch={pluginRegistryInventory.ParentSearchEnabled} ErrorRecorder={pluginRegistryInventory.HasErrorRecorder} Diagnostic={pluginRegistryDiagnostic} First={pluginRegistryFirst}");
        Console.WriteLine($"GlobalPluginRegistryLookup {globalPluginRegistryLookupState}");
        Console.WriteLine($"BuilderConfig WorkspaceLimit={workspaceLimit} Tf32={tf32Enabled} Refit={refitEnabled} ProfileStream={profileStreamSet} ProfileCount={configProfileCount} CalibrationProfile={calibrationProfileState} PluginInventory={pluginInventoryOk}:{pluginInventory.Count}:{pluginInventoryDiagnostic} {builderConfigDeploymentState} TimingCacheBytes={serializedTimingCache.SizeInBytes}/{serializedTimingCache.DataType}");
        Console.WriteLine($"EngineFileRoundTrip=True EngineBytes={engineBytes.Length} EngineFileBytes={new FileInfo(engineFile).Length} StreamRoundTrip=True StreamBytes={streamEngineBytes.Length}");
        Console.WriteLine($"ProfileShapes ConfiguredMin={configuredProfileRange.Min} ConfiguredOpt={configuredProfileRange.Opt} ConfiguredMax={configuredProfileRange.Max} Valid={configuredProfileValid} ExtraMemoryTarget={profileExtraMemoryTarget} ShapeValueCount={inputShapeValueCount} Min={minShape} Opt={optShape} Max={maxShape} ActiveBefore={activeProfileBefore} ActiveAfter={activeProfileAfter} EnqueueEmitsProfileToggle={enqueueEmitsProfileAfterToggle}");
        Console.WriteLine($"ParserFlags Before={parserFlagsBefore} NativeInstanceNorm={nativeInstanceNormalizationFlag}->{nativeInstanceNormalizationAfterSet} SupportsIdentity={parserSupportsIdentity} ParseStream={parsed}");
        Console.WriteLine($"ParserModelSupport {parserModelSupport} FirstSubgraph={parserModelSupportFirstSubgraph}");
        Console.WriteLine($"ParserModelSupportSummary={parserModelSupportSummary} RuntimeEvidenceKind={parserModelSupportSummary.RuntimeEvidenceKind} RuntimeProof={parserModelSupportSummary.IsRuntimeExecutionProof} ReleaseProof={parserModelSupportSummary.CanPromoteReleaseProof} DeleteDeferred={parserModelSupportSummary.CanDeleteDeferredRecord}");
        Console.WriteLine($"ParserUsedVCPluginLibraries Count={parserUsedVCPluginLibraries.Count} First={FormatFirst(parserUsedVCPluginLibraries)} LayerOutputIdentity={parserLayerOutputState}");
        Console.WriteLine($"OnnxConfigSnapshot={onnxConfigSnapshot}");
        Console.WriteLine($"OnnxConfigSummary={onnxConfigSummary}");
        Console.WriteLine($"ParserDiagnosticSnapshot={parserDiagnosticSnapshot}");
        Console.WriteLine($"ParserDiagnosticSummary={parserDiagnosticSummary}");
        Console.WriteLine($"CompatibilityBindings Count={engine.CompatibilityBindingCount} First={compatibilityBinding.Index}:{compatibilityBinding.Name}:{compatibilityBinding.IOMode}:{compatibilityBinding.Shape}");
        Console.WriteLine($"InferShapes MissingCount={missingShapeInferenceCount} MaxOutputSize={maxOutputSize} Readiness=Ready:{readiness.IsReadyForEnqueue}/Bound:{readiness.AllTensorAddressesBound}/Profile:{readiness.ActiveOptimizationProfile}/Tensors:{readiness.Tensors.Count} TensorDebug={tensorDebugState} Refitter={refitterState}");
        Console.WriteLine($"BindingReport Ready={bindingReport.IsReadyForEnqueue} Profile={bindingReport.ProfileIndex} Inputs={bindingReport.GetInputs().Count} Outputs={bindingReport.GetOutputs().Count} Tensors={bindingReport.Tensors.Count} Formats=[{bindingSummary}]");
        Console.WriteLine($"InferenceBindings Execution={bindingExecution} ElapsedMs={bindingElapsedMilliseconds:0.###}");
        Console.WriteLine($"ParserErrorSummary={parser.GetErrorSummary()}");
        Console.WriteLine($"ParserDiagnosticProbe={diagnosticProbe}");
        Console.WriteLine($"IOTensors={engine.IOTensorCount} InspectorContext=True InspectorBytes={inspectorText.Length} Enqueue=True OutputMatch=True");
        parser.ClearErrors();
    }

    File.Delete(engineFile);
    }

    static TensorRtApiLine ResolveLine(string value)
    {
        if (string.Equals(value, "8", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, "trt8", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt8;
        }

        if (string.Equals(value, "10", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, "trt10", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt10;
        }

        if (string.Equals(value, "11", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, "trt11", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt11;
        }

        throw new ArgumentException("TensorRT line must be 8, 10, or 11.", nameof(value));
    }

    static TensorRtAdapterInfo GetAdapterInfo(TensorRtEnvironmentSnapshot environment, TensorRtApiLine line)
    {
        return line switch
        {
            TensorRtApiLine.TensorRt8 => environment.TensorRt8,
            TensorRtApiLine.TensorRt10 => environment.TensorRt10,
            TensorRtApiLine.TensorRt11 => environment.TensorRt11,
            _ => throw new ArgumentException("Unsupported TensorRT API line.", nameof(line))
        };
    }

    static bool IsSkippableEnvironmentException(Exception exception)
    {
        if (exception is DllNotFoundException || exception is BadImageFormatException)
        {
            return true;
        }

        if (exception is BridgeProbeException bridgeProbe)
        {
            return bridgeProbe.StatusCode == BridgeStatusCode.DependencyMissing ||
                bridgeProbe.StatusCode == BridgeStatusCode.NotSupported ||
                bridgeProbe.StatusCode == BridgeStatusCode.InvalidState ||
                bridgeProbe.StatusCode == BridgeStatusCode.RuntimeError;
        }

        return false;
    }

    static string FormatRuntimeProbe(TensorRtRuntimeProbeReport report)
    {
        string version = report.GlobalVersion != null
            ? $"{report.GlobalVersion.Major}.{report.GlobalVersion.Minor}.{report.GlobalVersion.Patch}.{report.GlobalVersion.Build}/Packed={report.GlobalVersion.InferLibVersion}/Onnx={report.GlobalVersion.OnnxParserVersion}/GlobalLogger={report.GlobalVersion.HasGlobalLogger}"
            : "n/a";
        string registry = report.GlobalPluginRegistry != null
            ? $"{report.GlobalPluginRegistry.CreatorCount}/{report.GlobalPluginRegistry.RecursiveCreatorCount?.ToString() ?? "n/a"}/ParentSearch={report.GlobalPluginRegistry.ParentSearchEnabled}/ErrorRecorder={report.GlobalPluginRegistry.HasErrorRecorder}"
            : "n/a";
        string stages = string.Join(",", report.Stages.Select(stage => $"{stage.Name}={stage.Succeeded}"));
        string firstFailure = report.FirstFailure != null ? $"{report.FirstFailure.Name}:{report.FirstFailure.Message}" : "None";
        string failures = string.Join("|", report.Stages.Where(stage => !stage.Succeeded).Select(stage => $"{stage.Name}:{stage.Message}"));
        if (string.IsNullOrEmpty(failures))
        {
            failures = "None";
        }

        return $"Version={version} GlobalRegistry={registry} RuntimeCreate={report.RuntimeCreationSucceeded} FirstFailure={firstFailure} Failures={failures} Stages={stages}";
    }

    static string GetRuntimeStageMessage(TensorRtRuntimeProbeReport report)
    {
        TensorRtRuntimeProbeStage? stage = report.Stages.FirstOrDefault(item => string.Equals(item.Name, "RuntimeCreate", StringComparison.Ordinal));
        return stage?.Message ?? report.FirstFailure?.Message ?? "Runtime creation stage did not run.";
    }

    static string ProbeRuntimeControls(TensorRtRuntime runtime)
    {
        try
        {
            int dlaCore = runtime.DlaCore;
            int dlaCount = runtime.DlaCoreCount;
            int maxThreads = runtime.MaxThreads;
            bool hostCodeAllowed = runtime.EngineHostCodeAllowed;
            TensorRtTempfileControlFlags tempfileFlags = runtime.TempfileControlFlags;
            string temporaryDirectory = runtime.GetTemporaryDirectory();
            bool hasErrorRecorder = runtime.HasErrorRecorder;
            TensorRtRuntimeDiagnosticSnapshot diagnosticSnapshot = runtime.GetDiagnosticSnapshot();
            TensorRtRuntimeDiagnosticSummary diagnosticSummary = diagnosticSnapshot.ToSummary();
            string directory = string.IsNullOrEmpty(temporaryDirectory) ? "Default" : temporaryDirectory;
            return $"Dla={dlaCore}/{dlaCount} MaxThreads={maxThreads} HostCode={hostCodeAllowed} TempFlags={tempfileFlags} TempDir={directory} ErrorRecorder={hasErrorRecorder} RuntimeDiagnosticSnapshot={diagnosticSnapshot.HasLogger}/{diagnosticSnapshot.HasErrorRecorder}/{diagnosticSnapshot.ErrorRecorder.ErrorCount}/{diagnosticSnapshot.Diagnostics.Count} RuntimeDiagnosticSummary={diagnosticSummary.HasLogger}/{diagnosticSummary.HasErrorRecorder}/{diagnosticSummary.ErrorCount}/{diagnosticSummary.CopiedErrorRecordCount}/{diagnosticSummary.DiagnosticCount}";
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeRefitterControls(TensorRtRefitter refitter)
    {
        int originalMaxThreads = refitter.MaxThreads;
        refitter.MaxThreads = Math.Max(1, originalMaxThreads);
        TensorRtRefitterDiagnosticSnapshot diagnosticSnapshot = refitter.GetDiagnosticSnapshot();
        TensorRtRefitterDiagnosticSummary diagnosticSummary = diagnosticSnapshot.ToSummary();

        List<string> parts = new()
        {
            $"MaxThreads={refitter.MaxThreads}",
            $"ErrorRecorder={refitter.HasErrorRecorder}",
            $"RefitterDiagnosticSnapshot={diagnosticSnapshot.HasLogger}/{diagnosticSnapshot.HasErrorRecorder}/{diagnosticSnapshot.MissingNamedWeightCount}/{diagnosticSnapshot.AllNamedWeightCount}/{diagnosticSnapshot.Diagnostics.Count}",
            $"RefitterDiagnosticSummary={diagnosticSummary.HasLogger}/{diagnosticSummary.HasErrorRecorder}/{diagnosticSummary.MissingNamedWeightCount}/{diagnosticSummary.CopiedMissingNamedWeightCount}/{diagnosticSummary.AllNamedWeightCount}/{diagnosticSummary.CopiedAllNamedWeightCount}/{diagnosticSummary.DiagnosticCount}"
        };

        if (refitter.Line == TensorRtApiLine.TensorRt8 || refitter.Line == TensorRtApiLine.TensorRt10)
        {
            IReadOnlyList<string> dynamicRangeTensors = refitter.GetDynamicRangeTensorNames();
            IReadOnlyList<string> missingNamedWeights = refitter.GetMissingNamedWeights();
            IReadOnlyList<string> allNamedWeights = refitter.GetAllNamedWeights();
            parts.Add($"Logger={refitter.HasLogger}");
            parts.Add($"DynamicRanges={refitter.DynamicRangeTensorCount}/{dynamicRangeTensors.Count}");
            parts.Add($"NamedWeights={refitter.MissingNamedWeightCount}/{refitter.AllNamedWeightCount}/{missingNamedWeights.Count}/{allNamedWeights.Count}");
        }

        if (refitter.Line == TensorRtApiLine.TensorRt10 || refitter.Line == TensorRtApiLine.TensorRt11)
        {
            bool validation = refitter.WeightsValidation;
            refitter.WeightsValidation = validation;
            parts.Add($"Validation={refitter.WeightsValidation}");
        }

        return string.Join(" ", parts);
    }

    static string ProbeParserRefitterControls(TensorRtRefitter refitter, TensorRtLogger logger)
    {
        if (refitter.Line == TensorRtApiLine.TensorRt8)
        {
            return "Skipped:TensorRT8";
        }

        try
        {
            using TensorRtOnnxParserRefitter parserRefitter = refitter.CreateOnnxParserRefitter(logger);
            TensorRtOnnxParserRefitterDiagnosticSnapshot snapshot = parserRefitter.GetDiagnosticSnapshot();
            TensorRtOnnxParserRefitterDiagnosticSummary summarySnapshot = snapshot.ToSummary();
            string summary = snapshot.DiagnosticSummary.Length == 0 ? "Empty" : snapshot.DiagnosticSummary.Substring(0, Math.Min(snapshot.DiagnosticSummary.Length, 32)).Replace(' ', '_');
            return $"ParserRefitterDiagnosticSnapshot={snapshot.ErrorCount}/{snapshot.Diagnostics.Count}/{snapshot.DiagnosticSummary.Length}:{summary} ParserRefitterDiagnosticSummary={summarySnapshot.ErrorCount}/{summarySnapshot.CopiedDiagnosticCount}/{summarySnapshot.DiagnosticSummaryLength}";
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string FormatDependencyProbe(TensorRtDependencyProbeReport report)
    {
        string firstBridge = FormatDependencyEntry(report.NativeBridgeCandidates.FirstOrDefault(item => item.Exists));
        string loaded = FormatDependencyEntries(report.LoadedModules);
        string candidates = FormatDependencyEntries(report.SearchPathCandidates);
        string diagnostic = report.Diagnostics.Count == 0 ? "None" : report.Diagnostics[0];
        return $"BridgeInit={report.BridgeInitialized} BridgeCandidates={report.NativeBridgeCandidates.Count} FirstBridge={firstBridge} Loaded={report.LoadedModuleCount}:{loaded} PathCandidates={report.SearchPathCandidateCount}:{candidates} Diagnostics={report.Diagnostics.Count}:{diagnostic} Message={report.BridgeDiagnostic}";
    }

    static string FormatDependencyEntries(IReadOnlyList<TensorRtNativeDependencyInfo> entries)
    {
        if (entries.Count == 0)
        {
            return "None";
        }

        return string.Join(";", entries.Take(6).Select(FormatDependencyEntry));
    }

    static string FormatDependencyEntry(TensorRtNativeDependencyInfo? entry)
    {
        if (entry == null)
        {
            return "None";
        }

        string version = !string.IsNullOrWhiteSpace(entry.FileVersion) ? entry.FileVersion : entry.ProductVersion;
        string directory = string.IsNullOrWhiteSpace(entry.Path) ? "n/a" : Path.GetDirectoryName(entry.Path) ?? "n/a";
        return $"{entry.Name}@{directory}#{(string.IsNullOrWhiteSpace(version) ? "n/a" : version)}";
    }


    static void AssertDims(TensorRtDims actual, params int[] expected)
    {
        if (!actual.Values.SequenceEqual(expected))
        {
            throw new InvalidOperationException($"Unexpected TensorRT dims. Expected=[{string.Join(", ", expected)}] Actual={actual}");
        }
    }

    static string ProbeTensorDebugState(TensorRtExecutionContext context, string tensorName)
    {
        if (!context.SupportsTensorDebugState)
        {
            return "Unsupported";
        }

        try
        {
            context.SetAllTensorsDebugState(false);
            bool before = context.GetTensorDebugState(tensorName);
            context.SetTensorDebugState(tensorName, false);
            bool after = context.GetTensorDebugState(tensorName);
            return $"{tensorName}:{before}->{after}";
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.Message}";
        }
    }

    static (TensorRtOnnxModelSupportReport Report, string FirstSubgraph) ProbeModelSupport(TensorRtBuilder builder, TensorRtLogger logger, byte[] model)
    {
        try
        {
            using TensorRtNetworkDefinition supportNetwork = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
            using TensorRtOnnxParser supportParser = new TensorRtOnnxParser(logger, supportNetwork);
            using MemoryStream modelStream = new MemoryStream(model, writable: false);
            TensorRtOnnxModelSupportReport report = supportParser.CheckModelSupport(modelStream, "generated-dynamic-identity.onnx");
            return (report, FormatFirstSubgraph(report));
        }
        catch (Exception exception)
        {
            TensorRtOnnxModelSupportReport report = new TensorRtOnnxModelSupportReport(false, 0, 0, Array.Empty<TensorRtOnnxSubgraphSupportInfo>());
            return (report, $"Skipped:{exception.Message}");
        }
    }

    static string ProbeParserDiagnostics(TensorRtBuilder builder, TensorRtLogger logger, TensorRtApiLine line)
    {
        try
        {
            using TensorRtNetworkDefinition diagnosticNetwork = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
            using TensorRtOnnxParser diagnosticParser = new TensorRtOnnxParser(logger, diagnosticNetwork);
            using MemoryStream invalidModelStream = new MemoryStream(new byte[] { 0xFF, 0xFF, 0xFF, 0xFF }, writable: false);
            bool parsed = diagnosticParser.TryParse(invalidModelStream, $"jyppx-invalid-diagnostic-trt{(int)line}.onnx", out IReadOnlyList<TensorRtOnnxParserDiagnostic> diagnostics);
            string first = diagnostics.Count == 0
                ? "None"
                : $"{diagnostics[0].Code}:{diagnostics[0].Line}:{diagnostics[0].Node}:{diagnostics[0].Description.Length}:{diagnostics[0].NodeName}:{diagnostics[0].NodeOperator}:{diagnostics[0].LocalFunctionStack.Count}";
            diagnosticParser.ClearErrors();
            return $"Parsed={parsed} Count={diagnostics.Count} First={first}";
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.Message}";
        }
    }

    static string ProbeLayerOutputTensor(TensorRtOnnxParser parser, string layerName)
    {
        try
        {
            return parser.LayerOutputTensorExists(layerName).ToString();
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.Message}";
        }
    }

    static string FormatFirstSubgraph(TensorRtOnnxModelSupportReport report)
    {
        if (report.Subgraphs.Count == 0)
        {
            return "None";
        }

        TensorRtOnnxSubgraphSupportInfo subgraph = report.Subgraphs[0];
        string firstNode = subgraph.Nodes.Count == 0 ? "None" : subgraph.Nodes[0].ToString();
        return $"{subgraph.Index}:{subgraph.IsSupported}:Nodes={subgraph.Nodes.Count}:FirstNode={firstNode}";
    }

    static string FormatFirst(IReadOnlyList<string> values)
    {
        return values.Count == 0 ? "None" : values[0];
    }

    static string ProbeCalibrationProfile(TensorRtBuilderConfig config, TensorRtOptimizationProfile profile)
    {
        try
        {
            config.SetCalibrationProfile(profile);
            return $"Set={config.HasCalibrationProfile}";
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.Message}";
        }
    }

    static string FormatPluginRegistryFirst(TensorRtPluginRegistryInventory inventory)
    {
        if (inventory.CreatorCount == 0)
        {
            return "None";
        }

        TensorRtPluginCreatorInfo creator = inventory.Creators[0];
        string firstField = creator.Fields.Count == 0 ? "NoFields" : creator.Fields[0].ToString();
        return $"{creator.Index}:{creator.Name}:{creator.Version}:{creator.Namespace}:{creator.InterfaceKind}:{creator.InterfaceMajor}.{creator.InterfaceMinor}:ApiLanguage={creator.ApiLanguage}:Fields={creator.Fields.Count}:FirstField={firstField}";
    }

    static string ProbeGlobalPluginRegistryLookup(TensorRtRuntimeProbeReport runtimeProbe)
    {
        TensorRtPluginRegistryInventory? inventory = runtimeProbe.GlobalPluginRegistry;
        if (inventory == null)
        {
            return "Ok=False Diagnostic=Global registry inventory was not collected.";
        }

        if (inventory.CreatorCount == 0)
        {
            return $"Ok=True Source={inventory.Source} Creators=0 LookupFirst=n/a";
        }

        TensorRtPluginCreatorInfo creator = inventory.Creators[0];
        try
        {
            bool lookupFound = TensorRtEnvironmentProbe.IsGlobalPluginCreatorRegistered(inventory.Line, creator.Name, creator.Version, creator.Namespace);
            return $"Ok=True Source={inventory.Source} Creators={inventory.CreatorCount} LookupFirst={lookupFound} First={FormatPluginCreator(creator)}";
        }
        catch (Exception exception)
        {
            return $"Ok=False Source={inventory.Source} Creators={inventory.CreatorCount} Diagnostic={exception.Message}";
        }
    }

    static string FormatPluginCreator(TensorRtPluginCreatorInfo creator)
    {
        string firstField = creator.Fields.Count == 0 ? "NoFields" : creator.Fields[0].ToString();
        return $"{creator.Index}:{creator.Name}:{creator.Version}:{creator.Namespace}:{creator.InterfaceKind}:{creator.InterfaceMajor}.{creator.InterfaceMinor}:ApiLanguage={creator.ApiLanguage}:Fields={creator.Fields.Count}:FirstField={firstField}";
    }

    static string ProbeBuilderConfigDeploymentState(TensorRtBuilderConfig config, TensorRtApiLine line)
    {
        TensorRtEngineCapability capability = config.GetEngineCapability();
        TensorRtHardwareCompatibilityLevel hardwareCompatibility = config.GetHardwareCompatibilityLevel();
        TensorRtPreviewFeature previewFeature = line == TensorRtApiLine.TensorRt10
            ? TensorRtPreviewFeature.ProfileSharing0806Trt10
            : TensorRtPreviewFeature.FasterDynamicShapes0805;
        bool previewEnabled = config.GetPreviewFeature(previewFeature);
        string runtimePlatform = "Unsupported";
        if (line == TensorRtApiLine.TensorRt10)
        {
            config.SetRuntimePlatform(TensorRtRuntimePlatform.SameAsBuild);
            runtimePlatform = config.GetRuntimePlatform().ToString();
        }

        TensorRtBuilderConfigDeploymentSnapshot deploymentSnapshot = config.GetDeploymentSnapshot();
        return $"Capability={capability} HardwareCompatibility={hardwareCompatibility} PreviewFeature={previewFeature}:{previewEnabled} RuntimePlatform={runtimePlatform} DeploymentSnapshot={deploymentSnapshot.PluginToSerializeCount}/{deploymentSnapshot.SerializedPluginSnapshot.Count}/{deploymentSnapshot.SerializedPluginSnapshot.PluginLibraryPaths.Count}/{deploymentSnapshot.Diagnostics.Count}";
    }

    internal static class OnnxIdentityModel
    {
        public static byte[] CreateDynamicBatchModel()
        {
            ProtoWriter model = new ProtoWriter();
            model.Int64(1, 8);
            model.String(2, "JYPPX.TensorRtSharp");
            model.Message(7, CreateGraph());
            model.Message(8, CreateOpsetImport(13));
            return model.ToArray();
        }

        private static byte[] CreateGraph()
        {
            ProtoWriter graph = new ProtoWriter();
            graph.Message(1, CreateIdentityNode());
            graph.String(2, "jyppx_dynamic_identity_graph");
            graph.Message(11, CreateValueInfo("input"));
            graph.Message(12, CreateValueInfo("output"));
            return graph.ToArray();
        }

        private static byte[] CreateIdentityNode()
        {
            ProtoWriter node = new ProtoWriter();
            node.String(1, "input");
            node.String(2, "output");
            node.String(3, "identity");
            node.String(4, "Identity");
            return node.ToArray();
        }

        private static byte[] CreateValueInfo(string name)
        {
            ProtoWriter valueInfo = new ProtoWriter();
            valueInfo.String(1, name);
            valueInfo.Message(2, CreateTensorFloatType());
            return valueInfo.ToArray();
        }

        private static byte[] CreateTensorFloatType()
        {
            ProtoWriter tensorType = new ProtoWriter();
            tensorType.Message(1, CreateTensorType());
            return tensorType.ToArray();
        }

        private static byte[] CreateTensorType()
        {
            ProtoWriter type = new ProtoWriter();
            type.UInt64(1, 1);
            type.Message(2, CreateShape());
            return type.ToArray();
        }

        private static byte[] CreateShape()
        {
            ProtoWriter shape = new ProtoWriter();
            shape.Message(1, CreateDimension("batch"));
            shape.Message(1, CreateDimension(4));
            return shape.ToArray();
        }

        private static byte[] CreateDimension(string parameterName)
        {
            ProtoWriter dimension = new ProtoWriter();
            dimension.String(2, parameterName);
            return dimension.ToArray();
        }

        private static byte[] CreateDimension(long value)
        {
            ProtoWriter dimension = new ProtoWriter();
            dimension.Int64(1, value);
            return dimension.ToArray();
        }

        private static byte[] CreateOpsetImport(long version)
        {
            ProtoWriter opset = new ProtoWriter();
            opset.Int64(2, version);
            return opset.ToArray();
        }
    }

    internal sealed class ProtoWriter
    {
        private readonly MemoryStream _stream = new MemoryStream();

        public void Int64(int fieldNumber, long value)
        {
            WriteTag(fieldNumber, 0);
            WriteVarint(unchecked((ulong)value));
        }

        public void UInt64(int fieldNumber, ulong value)
        {
            WriteTag(fieldNumber, 0);
            WriteVarint(value);
        }

        public void String(int fieldNumber, string value)
        {
            byte[] bytes = System.Text.Encoding.UTF8.GetBytes(value);
            Bytes(fieldNumber, bytes);
        }

        public void Message(int fieldNumber, byte[] value)
        {
            Bytes(fieldNumber, value);
        }

        public byte[] ToArray()
        {
            return _stream.ToArray();
        }

        private void Bytes(int fieldNumber, byte[] value)
        {
            WriteTag(fieldNumber, 2);
            WriteVarint((ulong)value.Length);
            _stream.Write(value, 0, value.Length);
        }

        private void WriteTag(int fieldNumber, int wireType)
        {
            WriteVarint((ulong)((fieldNumber << 3) | wireType));
        }

        private void WriteVarint(ulong value)
        {
            while (value >= 0x80)
            {
                _stream.WriteByte((byte)(value | 0x80));
                value >>= 7;
            }

            _stream.WriteByte((byte)value);
        }
    }
}
