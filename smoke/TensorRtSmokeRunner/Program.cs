using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

internal static class Program
{
    private static void Main(string[] args)
    {
        string requestedLine = GetStringArgument(args, "--tensor-rt-line", "auto");
        bool dependencyProbeOnly = HasSwitch(args, "--dependency-probe-only");
        Console.WriteLine($"TensorRtSmokeRunner TensorRtLineRequest={requestedLine} DependencyProbeOnly={dependencyProbeOnly}");
        if (dependencyProbeOnly)
        {
            TensorRtApiLine probeLine = ResolveProbeLine(requestedLine);
            PrintDependencyProbe(probeLine);
            Console.WriteLine("Skipped=True Reason=DependencyProbeOnly");
            return;
        }

        TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
        Console.WriteLine($"Bridge={snapshot.BuildInfo.BridgeName} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");
        Console.WriteLine($"TRT8 Vendor={snapshot.TensorRt8.VendorDependencyAvailable} Runtime={snapshot.TensorRt8.RuntimeCreationSupported} Message={snapshot.TensorRt8.StatusMessage}");
        Console.WriteLine($"TRT10 Vendor={snapshot.TensorRt10.VendorDependencyAvailable} Runtime={snapshot.TensorRt10.RuntimeCreationSupported} Builder={snapshot.TensorRt10.BuilderCreationSupported} Message={snapshot.TensorRt10.StatusMessage}");
        Console.WriteLine($"TRT11 Vendor={snapshot.TensorRt11.VendorDependencyAvailable} Runtime={snapshot.TensorRt11.RuntimeCreationSupported} Builder={snapshot.TensorRt11.BuilderCreationSupported} Message={snapshot.TensorRt11.StatusMessage}");

        if (snapshot.TensorRt8.RuntimeCreationSupported)
        {
            Console.WriteLine($"TryCreateRuntime8={TensorRtEnvironmentProbe.TryCreateRuntime(TensorRtApiLine.TensorRt8, out string runtime8Message)}:{runtime8Message}");
            Console.WriteLine($"TryCreateBuilder8={TensorRtEnvironmentProbe.TryCreateBuilder(TensorRtApiLine.TensorRt8, out string builder8Message)}:{builder8Message}");
            Console.WriteLine($"TryBuildSerializedNetwork8={TensorRtEnvironmentProbe.TryBuildTensorRt8SerializedNetworkOnly(out string build8Message)}:{build8Message}");
            Console.WriteLine($"TryRunMinimalBuildChain8={TensorRtEnvironmentProbe.TryRunTensorRt8MinimalBuildChain(out string chain8Message)}:{chain8Message}");
            Console.WriteLine($"HighLevelChain8={RunHighLevelChain(TensorRtApiLine.TensorRt8, out string highLevel8Message)}:{highLevel8Message}");
        }
        else
        {
            Console.WriteLine($"TryCreateRuntime8=False:{snapshot.TensorRt8.StatusMessage}");
            Console.WriteLine($"TryCreateBuilder8=False:{snapshot.TensorRt8.StatusMessage}");
            Console.WriteLine($"TryBuildSerializedNetwork8=False:{snapshot.TensorRt8.StatusMessage}");
            Console.WriteLine($"TryRunMinimalBuildChain8=False:{snapshot.TensorRt8.StatusMessage}");
            Console.WriteLine($"HighLevelChain8=False:{snapshot.TensorRt8.StatusMessage}");
        }

        if (snapshot.TensorRt10.RuntimeCreationSupported && snapshot.TensorRt10.BuilderCreationSupported)
        {
            Console.WriteLine($"TryCreateRuntime10={TensorRtEnvironmentProbe.TryCreateRuntime(TensorRtApiLine.TensorRt10, out string runtimeMessage)}:{runtimeMessage}");
            Console.WriteLine($"TryCreateBuilder10={TensorRtEnvironmentProbe.TryCreateBuilder(TensorRtApiLine.TensorRt10, out string builder10Message)}:{builder10Message}");
            Console.WriteLine($"TryBuildSerializedNetwork10={TensorRtEnvironmentProbe.TryBuildTensorRt10SerializedNetworkOnly(out string build10Message)}:{build10Message}");
            Console.WriteLine($"TryRunMinimalBuildChain10={TensorRtEnvironmentProbe.TryRunTensorRt10MinimalBuildChain(out string chainMessage)}:{chainMessage}");
            Console.WriteLine($"HighLevelChain10={RunHighLevelChain(TensorRtApiLine.TensorRt10, out string highLevelMessage)}:{highLevelMessage}");
        }
        else
        {
            Console.WriteLine($"TryCreateRuntime10=False:{snapshot.TensorRt10.StatusMessage}");
            Console.WriteLine($"TryCreateBuilder10=False:{snapshot.TensorRt10.StatusMessage}");
            Console.WriteLine($"TryBuildSerializedNetwork10=False:{snapshot.TensorRt10.StatusMessage}");
            Console.WriteLine($"TryRunMinimalBuildChain10=False:{snapshot.TensorRt10.StatusMessage}");
            Console.WriteLine($"HighLevelChain10=False:{snapshot.TensorRt10.StatusMessage}");
        }

        if (snapshot.TensorRt11.RuntimeCreationSupported && snapshot.TensorRt11.BuilderCreationSupported)
        {
            Console.WriteLine($"TryCreateRuntime11={TensorRtEnvironmentProbe.TryCreateRuntime(TensorRtApiLine.TensorRt11, out string runtime11Message)}:{runtime11Message}");
            Console.WriteLine($"TryCreateBuilder11={TensorRtEnvironmentProbe.TryCreateBuilder(TensorRtApiLine.TensorRt11, out string builder11Message)}:{builder11Message}");
            Console.WriteLine($"TryBuildSerializedNetwork11={TensorRtEnvironmentProbe.TryBuildTensorRt11SerializedNetworkOnly(out string build11Message)}:{build11Message}");
            Console.WriteLine($"TryRunMinimalBuildChain11={TensorRtEnvironmentProbe.TryRunTensorRt11MinimalBuildChain(out string chain11Message)}:{chain11Message}");
            Console.WriteLine($"HighLevelChain11={RunTrt11HighLevelChain(out string highLevel11Message)}:{highLevel11Message}");
        }
        else
        {
            Console.WriteLine($"TryCreateRuntime11=False:{snapshot.TensorRt11.StatusMessage}");
            Console.WriteLine($"TryCreateBuilder11=False:{snapshot.TensorRt11.StatusMessage}");
            Console.WriteLine($"TryBuildSerializedNetwork11=False:{snapshot.TensorRt11.StatusMessage}");
            Console.WriteLine($"TryRunMinimalBuildChain11=False:{snapshot.TensorRt11.StatusMessage}");
            Console.WriteLine($"HighLevelChain11=False:{snapshot.TensorRt11.StatusMessage}");
        }

    }

    static bool RunHighLevelChain(TensorRtApiLine line, out string message)
    {
        try
        {
            using TensorRtLogger logger = new TensorRtLogger(line);
            using TensorRtRuntime runtime = new TensorRtRuntime(logger);
            using TensorRtBuilder builder = new TensorRtBuilder(logger);
            string builderCaps = $"FastFp16={builder.PlatformHasFastFp16} FastInt8={builder.PlatformHasFastInt8} Tf32={builder.PlatformHasTf32} DlaCores={builder.DlaCoreCount}";
            using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
            config.SetFlag(TensorRtBuilderFlag.Refit, true);
            config.SetEngineCapability(TensorRtEngineCapability.Standard);
            config.SetHardwareCompatibilityLevel(TensorRtHardwareCompatibilityLevel.None);
            using CudaStream profileStream = new CudaStream();
            config.SetProfileStream(profileStream);
            bool profileStreamSet = config.IsProfileStreamSet;
            string builderConfigDeploymentState = ProbeBuilderConfigDeploymentState(config, line);
            using TensorRtNetworkDefinition network = builder.CreateNetwork();
            string parserState = ProbeOnnxParser(logger, network);
            using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
            using TensorRtEngine engine = runtime.Deserialize(hostMemory);
            using TensorRtHostMemory serializedEngine = engine.Serialize();
            using TensorRtEngineInspector inspector = engine.CreateInspector();
            using TensorRtExecutionContext context = engine.CreateExecutionContext();
            inspector.SetExecutionContext(context);
            using CudaStream stream = new CudaStream();

            IReadOnlyList<TensorRtTensorInfo> ioTensors = engine.GetIOTensors();
            List<CudaMemory> buffers = new List<CudaMemory>();
            try
            {
                foreach (TensorRtTensorInfo tensor in ioTensors)
                {
                    if (tensor.IOMode == TensorRtIOMode.Input)
                    {
                        context.SetInputShape(tensor.Name, tensor.Shape);
                    }

                    int byteCount = EstimateTensorBytes(tensor);
                    CudaMemory buffer = new CudaMemory(byteCount);
                    buffers.Add(buffer);
                    if (tensor.IOMode == TensorRtIOMode.Input)
                    {
                        buffer.Fill(0, byteCount);
                    }

                    context.SetTensorAddress(tensor.Name, buffer);
                }

                int missingShapeInferenceCount = context.InferShapes();
                TensorRtExecutionContextReadiness readiness = context.GetReadiness(engine);
                string outputSizingState = ProbeOutputSizing(context, ioTensors);
                string tensorDebugState = ProbeTensorDebugState(context, ioTensors);
                context.EnqueueAsync(stream);
                stream.Synchronize();
                string refitterState = "Skipped NonRefittable";
                if (engine.IsRefittable)
                {
                    using TensorRtRefitter refitter = engine.CreateRefitter(logger);
                    IReadOnlyList<TensorRtRefitEntry> allEntries = refitter.GetAllEntries();
                    IReadOnlyList<TensorRtRefitEntry> missingEntries = refitter.GetMissingEntries();
                    string allPreview = string.Join(",", allEntries.Take(4));
                    string missingPreview = string.Join(",", missingEntries.Take(4));
                    refitterState = $"Created All={refitter.AllRefittableWeightCount}/{allEntries.Count} Missing={refitter.MissingWeightCount}/{missingEntries.Count} AllPreview=[{allPreview}] MissingPreview=[{missingPreview}] Refit={refitter.RefitCudaEngine()}";
                }

                string trt10RuntimeSerializationState = ProbeTrt10RuntimeSerializationControls(line, engine);
                message = $"HostMemory={hostMemory.SizeInBytes}/{hostMemory.DataType} EngineSerialized={serializedEngine.SizeInBytes}/{serializedEngine.DataType} RuntimeSerialization=[{trt10RuntimeSerializationState}] Parser={parserState} BuilderCaps=[{builderCaps}] BuilderConfig=[ProfileStream={profileStreamSet} ProfileCount={config.OptimizationProfileCount} {builderConfigDeploymentState}] InspectorContext=True InspectorBytes={{INSPECTOR_BYTES}} InspectorLayer={{INSPECTOR_LAYER_STATE}} Enqueue=True InferShapesMissing={missingShapeInferenceCount} Readiness=[Ready={readiness.IsReadyForEnqueue} Profile={readiness.ActiveOptimizationProfile} Bound={readiness.AllTensorAddressesBound} Tensors={readiness.Tensors.Count}] OutputSizing=[{outputSizingState}] TensorDebug={tensorDebugState} Refitter={refitterState} IOTensors={{IO_TENSOR_COUNT}} [{{TENSORS}}]";
            }
            finally
            {
                foreach (CudaMemory buffer in buffers)
                {
                    buffer.Dispose();
                }
            }

            string inspectorText = inspector.GetEngineInformation(TensorRtLayerInformationFormat.Oneline);
            string inspectorLayerState = ProbeInspectorLayerInformation(inspector);
            string tensors = string.Join(
                "; ",
                ioTensors.Select(tensor => $"{tensor.Index}:{tensor.Name}:{tensor.IOMode}:{tensor.DataType}:{tensor.Shape}"));
            message = message
                .Replace("{INSPECTOR_BYTES}", inspectorText.Length.ToString())
                .Replace("{INSPECTOR_LAYER_STATE}", inspectorLayerState)
                .Replace("{IO_TENSOR_COUNT}", engine.IOTensorCount.ToString())
                .Replace("{TENSORS}", tensors);
            return true;
        }
        catch (Exception exception)
        {
            message = exception.ToString().Replace(Environment.NewLine, " | ");
            return false;
        }
    }

    static string ProbeInspectorLayerInformation(TensorRtEngineInspector inspector)
    {
        try
        {
            string layerText = inspector.GetLayerInformation(0, TensorRtLayerInformationFormat.Oneline);
            return $"Bytes={layerText.Length}";
        }
        catch (Exception exception)
        {
            return $"Blocked:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeTrt10RuntimeSerializationControls(TensorRtApiLine line, TensorRtEngine engine)
    {
        if (line != TensorRtApiLine.TensorRt10)
        {
            return $"Skipped:{line}";
        }

        try
        {
            using TensorRtSerializationConfig serializationConfig = engine.CreateSerializationConfig();
            TensorRtSerializationFlags flagsBefore = serializationConfig.Flags;
            using TensorRtHostMemory configuredPlan = engine.Serialize(serializationConfig);

            using TensorRtRuntimeConfig runtimeConfig = engine.CreateRuntimeConfig();
            TensorRtExecutionContextAllocationStrategy strategyBefore = runtimeConfig.AllocationStrategy;
            runtimeConfig.AllocationStrategy = TensorRtExecutionContextAllocationStrategy.Static;
            TensorRtExecutionContextAllocationStrategy strategyAfter = runtimeConfig.AllocationStrategy;
            using TensorRtExecutionContext runtimeContext = engine.CreateExecutionContext(runtimeConfig);

            long streamableWeights = engine.StreamableWeightsSizeInBytes;
            long minimumBudget = engine.MinimumWeightStreamingBudgetInBytes;
            long budget = engine.WeightStreamingBudgetV2InBytes;
            long automaticBudget = engine.WeightStreamingAutomaticBudgetInBytes;
            long scratch = engine.WeightStreamingScratchMemorySizeInBytes;
            TensorRtHardwareCompatibilityLevel hardware = engine.EngineHardwareCompatibilityLevel;

            return $"ConfigPlan={configuredPlan.SizeInBytes}/{configuredPlan.DataType} Flags={flagsBefore} RuntimeStrategy={strategyBefore}->{strategyAfter} RuntimeContext={runtimeContext != null} Streamable={streamableWeights} MinimumBudget={minimumBudget} Budget={budget}/Auto={automaticBudget} Scratch={scratch} Hardware={hardware}";
        }
        catch (Exception exception)
        {
            return $"Blocked:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static bool RunTrt11HighLevelChain(out string message)
    {
        try
        {
            using TensorRtLogger logger = new TensorRtLogger(TensorRtApiLine.TensorRt11);
            using TensorRtRuntime runtime = new TensorRtRuntime(logger);
            using TensorRtBuilder builder = new TensorRtBuilder(logger);
            string trt11BuilderBoundary = ProbeTrt11BuilderBoundary(builder);
            using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
            string trt11RuntimeControls = ProbeTrt11RuntimeControls(runtime);
            config.SetFlag(TensorRtBuilderFlag.Refit, true);
            config.SetEngineCapability(TensorRtEngineCapability.Standard);
            config.SetHardwareCompatibilityLevel(TensorRtHardwareCompatibilityLevel.None);
            config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 1UL << 20);
            config.SetOptimizationLevel(0);
            config.SetProfilingVerbosity(TensorRtProfilingVerbosity.LayerNamesOnly);
            config.SetMaxAuxStreams(0);
            config.SetAverageTimingIterations(1);
            string trt11ConfigRuntimeControls = ProbeTrt11BuilderConfigRuntimeControls(config);
            TensorRtBuilderConfigDeploymentSnapshot configSnapshot = config.GetDeploymentSnapshot();
            string trt11ConfigSnapshot = configSnapshot.ToString();
            string trt11ConfigSerializedPluginSnapshot = $"{configSnapshot.SerializedPluginSnapshot.Count}/{configSnapshot.SerializedPluginSnapshot.PluginLibraryPaths.Count}/{configSnapshot.SerializedPluginSnapshot.HasPathInventory}";
            string trt11FourteenthBatchBuildOutputs = ProbeTrt11FourteenthBatchBuildOutputs(builder, runtime);

            using CudaStream stream = new CudaStream();
            config.SetProfileStream(stream);

            using TensorRtNetworkDefinition network = builder.CreateNetwork();
            network.Name = "trt11_smoke_network";
            using TensorRtTensor removableTensor = network.AddInput("scratch_remove", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1 }));
            network.RemoveTensor(removableTensor);
            using TensorRtTensor input = network.AddInput("input", TensorRtDataType.Float, new TensorRtDims(new[] { -1, 4 }));
            input.SetDimensionName(0, "batch");
            using TensorRtLayer topKV2Probe = network.AddTopKV2(input, TensorRtTopKOperation.Max, 1, 1u << 1, TensorRtDataType.Int32);
            topKV2Probe.Name = "topk_v2_probe";
            using TensorRtLayer identity = network.AddIdentity(input);
            identity.Name = "identity_output";
            using TensorRtTensor output = identity.GetOutput(0);
            output.Name = "output";
            output.SetDimensionName(0, "batch");
            network.MarkOutput(output);
            TensorRtLayerTensorMetadata identityInputMetadata = identity.GetInputTensorMetadata(0);
            TensorRtLayerTensorMetadata identityOutputMetadata = identity.GetOutputTensorMetadata(0);
            string identityInputSummary = identity.GetInputTensorSummary(0);
            string identityOutputSummary = identity.GetOutputTensorSummary(0);
            string trt11LayerTensorMetadata = $"IdentityInput=[{identityInputMetadata}] IdentityOutput=[{identityOutputMetadata}] Summaries=[{identityInputSummary}|{identityOutputSummary}] DirectRoles=Input:{input.IsNetworkInput}/{input.IsNetworkOutput} Output:{output.IsNetworkInput}/{output.IsNetworkOutput}";
            bool debugMarked = network.MarkDebugTensor(output);
            bool debugTensor = network.IsDebugTensor(output);
            bool unfusedDebugMarked = network.MarkUnfusedTensorsAsDebugTensors();
            bool unfusedDebugUnmarked = network.UnmarkUnfusedTensorsAsDebugTensors();
            bool canRunOnDla = config.CanRunOnDla(identity);
            bool networkHasErrorRecorderBefore = network.HasErrorRecorder;
            network.ClearErrorRecorder();
            bool networkHasErrorRecorderAfter = network.HasErrorRecorder;
            string trt11NetworkDebug = $"DebugMarked={debugMarked} DebugTensor={debugTensor} UnfusedMark={unfusedDebugMarked}/{unfusedDebugUnmarked} CanDla={canRunOnDla} ErrorRecorder={networkHasErrorRecorderBefore}->{networkHasErrorRecorderAfter} RemoveTensor=True TopKV2=True";

            using TensorRtOptimizationProfile profile = builder.CreateOptimizationProfile();
            profile.SetShape(
                "input",
                new TensorRtDims(new[] { 1, 4 }),
                new TensorRtDims(new[] { 1, 4 }),
                new TensorRtDims(new[] { 4, 4 }));
            TensorRtOptimizationProfileShapeRange64 configuredProfileRange64 = profile.GetShapeRange64("input");
            int shapeValueCountV2 = profile.GetShapeValueCountV2("input");
            int shapeValueReadCountV2 = profile.GetShapeValuesV2("input", TensorRtOptimizationProfileSelector.Min).Count;
            int profileIndex = config.AddOptimizationProfile(profile);
            bool networkSupported = builder.IsNetworkSupported(network, config);

            string parserState = ProbeOnnxParser(logger, network);
            using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
            byte[] serialized = hostMemory.ToArray();
            using TensorRtEngine engine = runtime.Deserialize(hostMemory);
            string trt11EngineBoundary = ProbeTrt11EngineBoundary(engine);
            string trt8LegacyShapeBindingBoundary = ProbeTrt8LegacyShapeBindingBoundary(engine);
            string trt11SerializationRuntimeConfig = ProbeTrt11SerializationRuntimeConfig(engine);
            string trt11RefitterControls = ProbeTrt11RefitterControls(engine, logger, stream);
            using TensorRtEngineInspector inspector = engine.CreateInspector();
            using TensorRtExecutionContext context = engine.CreateExecutionContext();
            string trt11ContextBoundary = ProbeTrt11ContextBoundary(context);
            string trt8LegacyContextShapeBindingBoundary = ProbeTrt8LegacyContextShapeBindingBoundary(context);
            inspector.SetExecutionContext(context);
            context.SetOptimizationProfileAsync(0, stream);
            IReadOnlyList<TensorRtTensorInfo> ioTensors = engine.GetIOTensors();
            List<CudaMemory> buffers = new List<CudaMemory>();

            try
            {
                foreach (TensorRtTensorInfo tensor in ioTensors)
                {
                    if (tensor.IOMode == TensorRtIOMode.Input)
                    {
                        context.SetInputShape(tensor.Name, new TensorRtDims(new[] { 1, 4 }));
                    }

                    int byteCount = EstimateTensorBytes(tensor);
                    CudaMemory buffer = new CudaMemory(byteCount);
                    buffers.Add(buffer);
                    buffer.Fill(0, byteCount);
                    context.SetTensorAddress(tensor.Name, buffer);
                }

                int missingShapes = context.InferShapes();
                TensorRtExecutionContextReadiness readiness = context.GetReadiness(engine, runShapeInference: false);
                TensorRtEngineBindingReport bindingReport = engine.GetBindingReport(context, 0, runShapeInference: false);
                string trt11EngineContextRuntimeControls = ProbeTrt11EngineContextRuntimeControls(engine, context, ioTensors);
                TensorRtEngineDeploymentSnapshot engineSnapshot = engine.GetDeploymentSnapshot(0);
                TensorRtExecutionContextDeploymentSnapshot contextSnapshot = context.GetDeploymentSnapshot(engine);
                TensorRtEngineDeploymentSummary engineDeploymentSummary = engineSnapshot.ToSummary();
                TensorRtExecutionContextDeploymentSummary contextDeploymentSummary = contextSnapshot.ToSummary();
                string deploymentSnapshotEvidence = $"EngineProfileTensorValues={engineSnapshot.ProfileTensorValues.Count} ContextRuntimeDiagnostics={contextSnapshot.RuntimeDiagnostics.Count} EngineDeploymentSummary=[{engineDeploymentSummary}] ExecutionContextDeploymentSummary=[{contextDeploymentSummary}]";
                string profileTensorValuesV2 = ProbeTrt11EngineProfileTensorValuesV2(engine, ioTensors);
                string profileTensorSnapshots = ProbeTrt11EngineProfileTensorValueSnapshots(engine, ioTensors);
                string dims64Evidence = ProbeTrt11Dims64(network, input, output, identity, configuredProfileRange64, engine, context, ioTensors, profileIndex);
                context.EnqueueAsync(stream);
                stream.Synchronize();

                using TensorRtEngine engineFromBytes = runtime.Deserialize(serialized);
                int bytePathTensorCount = engineFromBytes.IOTensorCount;
                bool inspectorHasContextBefore = inspector.HasExecutionContext;
                string layerInformation = inspector.GetLayerInformation(0, TensorRtLayerInformationFormat.Oneline);
                bool inspectorHasErrorRecorder = inspector.HasErrorRecorder;
                inspector.ClearErrorRecorder();
                bool inspectorHasErrorRecorderAfterClear = inspector.HasErrorRecorder;
                string inspectorText = inspector.GetEngineInformation(TensorRtLayerInformationFormat.Oneline);
                inspector.ClearExecutionContext();
                bool inspectorHasContextAfter = inspector.HasExecutionContext;

                string tensors = string.Join(
                    "; ",
                    ioTensors.Select(tensor => $"{tensor.Index}:{tensor.Name}:{tensor.IOMode}:{tensor.DataType}:{tensor.Shape}:Ctx={context.GetTensorShape(tensor.Name)}"));
                message = $"HostMemory={hostMemory.SizeInBytes}/{hostMemory.DataType} SerializedBytes={serialized.Length} ProfileIndex={profileIndex} ProfileValid={profile.IsValid} ProfileCount={config.OptimizationProfileCount} ProfileShapeValuesV2={shapeValueCountV2}/{shapeValueReadCountV2} EngineProfileTensorValuesV2=[{profileTensorValuesV2}] EngineProfileTensorSnapshot=[{profileTensorSnapshots}] DeploymentSnapshotEvidence=[{deploymentSnapshotEvidence}] Dims64=[{dims64Evidence}] ProfileStream={config.IsProfileStreamSet} NetworkSupported={networkSupported} Parser={parserState} Network={network.Name}:I{network.InputCount}:O{network.OutputCount}:L{network.LayerCount} NetworkDebug=[{trt11NetworkDebug}] LayerTensorMetadata=[{trt11LayerTensorMetadata}] BuilderBoundary=[{trt11BuilderBoundary}] Runtime=[{trt11RuntimeControls}] Config=Capability:{config.GetEngineCapability()}:Hardware:{config.GetHardwareCompatibilityLevel()}:Opt:{config.GetOptimizationLevel()}:Verbosity:{config.GetProfilingVerbosity()}:Aux:{config.GetMaxAuxStreams()}:Timing:{config.GetAverageTimingIterations()} Workspace={config.GetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace)} ConfigSnapshot=[{trt11ConfigSnapshot}] ConfigSerializedPlugins=[{trt11ConfigSerializedPluginSnapshot}] Trt11ConfigRuntime=[{trt11ConfigRuntimeControls}] Trt11BuildOutputs=[{trt11FourteenthBatchBuildOutputs}] EngineBoundary=[{trt11EngineBoundary}] Trt8LegacyShapeBinding=[{trt8LegacyShapeBindingBoundary}] EngineSnapshot=[{engineSnapshot}] EngineDeploymentSummary=[{engineDeploymentSummary}] ContextBoundary=[{trt11ContextBoundary}] Trt8LegacyContextShapeBinding=[{trt8LegacyContextShapeBindingBoundary}] ContextSnapshot=[{contextSnapshot}] ExecutionContextDeploymentSummary=[{contextDeploymentSummary}] SerializationRuntimeConfig=[{trt11SerializationRuntimeConfig}] Refitter=[{trt11RefitterControls}] Trt11EngineContextRuntime=[{trt11EngineContextRuntimeControls}] Context=True ActiveProfile={context.OptimizationProfileIndex} MissingShapes={missingShapes} Ready={readiness.IsReadyForEnqueue}/{bindingReport.IsReadyForEnqueue} InspectorBytes={inspectorText.Length} InspectorLayerBytes={layerInformation.Length} InspectorContext={inspectorHasContextBefore}->{inspectorHasContextAfter} InspectorErrorRecorder={inspectorHasErrorRecorder}->{inspectorHasErrorRecorderAfterClear} ByteDeserializeIOTensors={bytePathTensorCount} Enqueue=True IOTensors={ioTensors.Count} [{tensors}]";
            }
            finally
            {
                foreach (CudaMemory buffer in buffers)
                {
                    buffer.Dispose();
                }
            }

            return true;
        }
        catch (Exception exception)
        {
            message = exception.ToString().Replace(Environment.NewLine, " | ");
            return false;
        }
    }

    static string ProbeOnnxParser(TensorRtLogger logger, TensorRtNetworkDefinition network)
    {
        try
        {
            using TensorRtOnnxParser parser = new TensorRtOnnxParser(logger, network);
            bool supportsIdentity = parser.SupportsOperator("Identity");
            bool nativeInstanceNorm = parser.GetFlag(TensorRtOnnxParserFlag.NativeInstanceNormalization);
            parser.ClearErrors();
            return $"Available ErrorCount={parser.ErrorCount} SupportsIdentity={supportsIdentity} NativeInstanceNorm={nativeInstanceNorm}";
        }
        catch (BridgeProbeException exception) when (exception.StatusCode == BridgeStatusCode.DependencyMissing || exception.StatusCode == BridgeStatusCode.NotSupported)
        {
            return $"Unavailable {exception.StatusCode}";
        }
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
        TensorRtBuilderConfigDeploymentSummary deploymentSummary = deploymentSnapshot.ToSummary();
        return $"Capability={capability} HardwareCompatibility={hardwareCompatibility} PreviewFeature={previewFeature}:{previewEnabled} RuntimePlatform={runtimePlatform} DeploymentSnapshot={deploymentSnapshot.PluginToSerializeCount}/{deploymentSnapshot.SerializedPluginSnapshot.Count}/{deploymentSnapshot.SerializedPluginSnapshot.PluginLibraryPaths.Count}/{deploymentSnapshot.Diagnostics.Count} BuilderConfigDeploymentSummary=[{deploymentSummary}]";
    }

    static string ProbeTrt11BuilderBoundary(TensorRtBuilder builder)
    {
        try
        {
            int maxDlaBatchSize = builder.MaxDlaBatchSize;
            int maxThreadsBefore = builder.MaxThreads;
            bool setMaxThreads = builder.SetMaxThreads(Math.Max(1, maxThreadsBefore));
            int maxThreadsAfter = builder.MaxThreads;
            bool hasErrorRecorderBefore = builder.HasErrorRecorder;
            builder.ClearErrorRecorder();
            bool hasErrorRecorderAfter = builder.HasErrorRecorder;
            builder.ClearGpuAllocator();
            builder.Reset();
            return $"MaxDlaBatch={maxDlaBatchSize} MaxThreads={maxThreadsBefore}->{maxThreadsAfter}/Set={setMaxThreads} ErrorRecorder={hasErrorRecorderBefore}->{hasErrorRecorderAfter} ClearGpuAllocator=True Reset=True";
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeTrt11RuntimeControls(TensorRtRuntime runtime)
    {
        try
        {
            int dlaCore = runtime.DlaCore;
            runtime.DlaCore = dlaCore;
            int dlaCoreCount = runtime.DlaCoreCount;
            int maxThreads = runtime.MaxThreads;
            if (maxThreads > 0)
            {
                runtime.MaxThreads = maxThreads;
            }

            TensorRtTempfileControlFlags tempfileFlags = runtime.TempfileControlFlags;
            runtime.TempfileControlFlags = tempfileFlags;
            bool hostCodeAllowed = runtime.EngineHostCodeAllowed;
            runtime.EngineHostCodeAllowed = hostCodeAllowed;
            bool hasErrorRecorderBefore = runtime.HasErrorRecorder;
            runtime.ClearErrorRecorder();
            bool hasErrorRecorderAfter = runtime.HasErrorRecorder;
            string tempDirectoryBefore = runtime.GetTemporaryDirectory();
            runtime.SetTemporaryDirectory(System.IO.Path.GetTempPath());
            string tempDirectoryAfterSet = runtime.GetTemporaryDirectory();
            runtime.ClearTemporaryDirectory();
            string tempDirectoryAfterClear = runtime.GetTemporaryDirectory();

            return $"Dla={dlaCore}/{dlaCoreCount} MaxThreads={maxThreads} Tempfile={tempfileFlags} HostCode={hostCodeAllowed} ErrorRecorder={hasErrorRecorderBefore}->{hasErrorRecorderAfter} TempDirChars={tempDirectoryBefore.Length}->{tempDirectoryAfterSet.Length}->{tempDirectoryAfterClear.Length}";
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeTrt11EngineBoundary(TensorRtEngine engine)
    {
        try
        {
            bool hasErrorRecorderBefore = engine.HasErrorRecorder;
            engine.ClearErrorRecorder();
            bool hasErrorRecorderAfter = engine.HasErrorRecorder;
            string outputAlias = string.Empty;
            TensorRtTensorInfo? output = engine.GetIOTensors().FirstOrDefault(static tensor => tensor.IOMode == TensorRtIOMode.Output);
            if (!string.IsNullOrWhiteSpace(output?.Name))
            {
                outputAlias = engine.GetAliasedInputTensorName(output.Name);
            }

            return $"ErrorRecorder={hasErrorRecorderBefore}->{hasErrorRecorderAfter} AliasChars={outputAlias.Length}";
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeTrt8LegacyShapeBindingBoundary(TensorRtEngine engine)
    {
        try
        {
            int[] values = engine.GetProfileShapeValues(0, 0, TensorRtOptimizationProfileSelector.Min);
            return $"Values={values.Length}";
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeTrt11ContextBoundary(TensorRtExecutionContext context)
    {
        try
        {
            bool hasErrorRecorderBefore = context.HasErrorRecorder;
            context.ClearErrorRecorder();
            bool hasErrorRecorderAfter = context.HasErrorRecorder;
            return $"ErrorRecorder={hasErrorRecorderBefore}->{hasErrorRecorderAfter}";
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeTrt11SerializationRuntimeConfig(TensorRtEngine engine)
    {
        try
        {
            using TensorRtHostMemory serializedDefault = engine.Serialize();
            using TensorRtSerializationConfig serializationConfig = engine.CreateSerializationConfig();
            TensorRtSerializationFlags flagsBefore = serializationConfig.Flags;
            serializationConfig.Flags = flagsBefore;
            TensorRtSerializationConfigSummary serializationSummary = serializationConfig.ToSummary();
            bool includeRefitBefore = serializationConfig.GetFlag(TensorRtSerializationFlag.IncludeRefit);
            bool includeRefitSet = serializationConfig.SetFlag(TensorRtSerializationFlag.IncludeRefit);
            bool includeRefitAfterSet = serializationConfig.GetFlag(TensorRtSerializationFlag.IncludeRefit);
            bool includeRefitCleared = serializationConfig.ClearFlag(TensorRtSerializationFlag.IncludeRefit);
            bool includeRefitAfterClear = serializationConfig.GetFlag(TensorRtSerializationFlag.IncludeRefit);
            using TensorRtHostMemory serializedWithConfig = engine.Serialize(serializationConfig);

            using TensorRtRuntimeConfig runtimeConfig = engine.CreateRuntimeConfig();
            TensorRtExecutionContextAllocationStrategy strategyBefore = runtimeConfig.AllocationStrategy;
            runtimeConfig.AllocationStrategy = TensorRtExecutionContextAllocationStrategy.Static;
            TensorRtExecutionContextAllocationStrategy strategyAfter = runtimeConfig.AllocationStrategy;
            TensorRtRuntimeConfigSummary runtimeSummary = runtimeConfig.ToSummary();
            using TensorRtExecutionContext contextByStrategy = engine.CreateExecutionContext(TensorRtExecutionContextAllocationStrategy.Static);
            using TensorRtExecutionContext contextByConfig = engine.CreateExecutionContext(runtimeConfig);

            return $"Serialized={serializedDefault.SizeInBytes}/{serializedDefault.DataType}->{serializedWithConfig.SizeInBytes}/{serializedWithConfig.DataType} Flags={flagsBefore} SerializationConfigSummary=[{serializationSummary}] IncludeRefit={includeRefitBefore}->{includeRefitSet}/{includeRefitAfterSet}->{includeRefitCleared}/{includeRefitAfterClear} RuntimeStrategy={strategyBefore}->{strategyAfter} RuntimeConfigSummary=[{runtimeSummary}] Contexts=True/True";
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeTrt11RefitterControls(TensorRtEngine engine, TensorRtLogger logger, CudaStream stream)
    {
        try
        {
            if (!engine.IsRefittable)
            {
                return "Skipped NonRefittable";
            }

            using TensorRtRefitter refitter = engine.CreateRefitter(logger);
            int maxThreads = refitter.MaxThreads;
            if (maxThreads > 0)
            {
                refitter.MaxThreads = maxThreads;
            }

            bool validationBefore = refitter.WeightsValidation;
            refitter.WeightsValidation = validationBefore;
            bool validationAfter = refitter.WeightsValidation;
            bool hasErrorRecorderBefore = refitter.HasErrorRecorder;
            refitter.ClearErrorRecorder();
            bool hasErrorRecorderAfter = refitter.HasErrorRecorder;
            IReadOnlyList<TensorRtRefitEntry> entries = refitter.GetAllEntries();
            bool asyncRefit = refitter.RefitCudaEngineAsync(stream);
            stream.Synchronize();

            string entryState = "NoNamedWeights";
            TensorRtRefitEntry firstEntry = entries.FirstOrDefault();
            if (!string.IsNullOrWhiteSpace(firstEntry.LayerName))
            {
                string weightsName = firstEntry.LayerName;
                TensorRtTensorLocation location = refitter.GetWeightsLocation(weightsName);
                TensorRtWeightsInfo current = refitter.GetNamedWeightsInfo(weightsName);
                TensorRtWeightsInfo prototype = refitter.GetWeightsPrototypeInfo(weightsName);
                bool unset = refitter.UnsetNamedWeights(weightsName);
                entryState = $"{weightsName}:Location={location}:Current={current.DataType}/{current.ElementCount}/{current.HasValues}:Prototype={prototype.DataType}/{prototype.ElementCount}/{prototype.HasValues}:Unset={unset}";
            }

            return $"MaxThreads={maxThreads} Validation={validationBefore}->{validationAfter} ErrorRecorder={hasErrorRecorderBefore}->{hasErrorRecorderAfter} Entries={entries.Count} AsyncRefit={asyncRefit} Entry={entryState}";
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeTrt11BuilderConfigRuntimeControls(TensorRtBuilderConfig config)
    {
        try
        {
            TensorRtBuilderFlags beforeFlags = config.GetFlags();
            config.SetFlags(beforeFlags);
            TensorRtBuilderFlags afterFlags = config.GetFlags();
            TensorRtDeviceType defaultDevice = config.GetDefaultDeviceType();
            int dlaCore = config.GetDlaCore();
            bool tilingSet = config.SetTilingOptimizationLevel(TensorRtTilingOptimizationLevel.None);
            TensorRtTilingOptimizationLevel tilingLevel = config.GetTilingOptimizationLevel();
            bool l2Set = config.SetL2LimitForTiling(0);
            long l2Limit = config.GetL2LimitForTiling();
            int maxTactics = config.GetMaxTactics();
            string remoteAutoTuning = config.GetRemoteAutoTuningConfig();
            bool hasTimingCache = config.HasTimingCache;
            config.ClearPluginsToSerialize();
            int pluginCount = config.PluginToSerializeCount;
            bool progressMonitorBefore = config.HasProgressMonitor;
            config.ClearProgressMonitor();
            bool progressMonitorAfter = config.HasProgressMonitor;
            return $"Flags={beforeFlags}->{afterFlags} DefaultDevice={defaultDevice} DlaCore={dlaCore} Tiling={tilingLevel}/Set={tilingSet} L2={l2Limit}/Set={l2Set} MaxTactics={maxTactics} TimingCache={hasTimingCache} Plugins={pluginCount} ProgressMonitor={progressMonitorBefore}->{progressMonitorAfter} RemoteAutoTuningChars={remoteAutoTuning.Length}";
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeTrt11FourteenthBatchBuildOutputs(TensorRtBuilder builder, TensorRtRuntime runtime)
    {
        try
        {
            List<string> states = new List<string>();
            using TensorRtBuilderConfig directEngineConfig = builder.CreateBuilderConfig();
            directEngineConfig.SetOptimizationLevel(0);
            directEngineConfig.SetPluginsToSerialize(Array.Empty<string>());
            directEngineConfig.ClearFlag(TensorRtBuilderFlag.Fp16);
            using TensorRtNetworkDefinition directEngineNetwork = builder.CreateNetwork();
            using TensorRtEngine directEngine = builder.BuildEngineWithConfig(directEngineNetwork, directEngineConfig);
            states.Add($"DirectEngineIOTensors={directEngine.IOTensorCount}");
            states.Add("PluginsToSerialize=True");

            using TensorRtBuilderConfig kernelTextConfig = builder.CreateBuilderConfig();
            kernelTextConfig.SetOptimizationLevel(0);
            using TensorRtNetworkDefinition kernelTextNetwork = builder.CreateNetwork();
            try
            {
                using TensorRtSerializedNetworkWithKernelText serialized = builder.BuildSerializedNetworkWithKernelText(kernelTextNetwork, kernelTextConfig);
                using TensorRtEngine deserialized = runtime.Deserialize(serialized.Plan);
                ulong kernelTextBytes = serialized.KernelText?.SizeInBytes ?? 0UL;
                states.Add($"Plan={serialized.Plan.SizeInBytes}/{serialized.Plan.DataType}");
                states.Add($"KernelTextBytes={kernelTextBytes}");
                states.Add($"DeserializedIOTensors={deserialized.IOTensorCount}");
            }
            catch (Exception exception)
            {
                states.Add($"KernelText=Blocked:{exception.GetType().Name}:{exception.Message}");
            }

            return string.Join(" ", states);
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeTrt11EngineContextRuntimeControls(TensorRtEngine engine, TensorRtExecutionContext context, IReadOnlyList<TensorRtTensorInfo> ioTensors)
    {
        try
        {
            long streamableWeights = engine.StreamableWeightsSizeInBytes;
            long budget = engine.WeightStreamingBudgetV2InBytes;
            long automaticBudget = engine.WeightStreamingAutomaticBudgetInBytes;
            long scratch = engine.WeightStreamingScratchMemorySizeInBytes;
            long totalWeights = engine.GetEngineStat(TensorRtEngineStat.TotalWeightsSize);
            long strippedWeights = engine.GetEngineStat(TensorRtEngineStat.StrippedWeightsSize);
            bool budgetSet = automaticBudget >= 0 && engine.SetWeightStreamingBudgetV2(automaticBudget);
            TensorRtHardwareCompatibilityLevel hardware = engine.EngineHardwareCompatibilityLevel;
            bool inputConsumedEventSet = context.IsInputConsumedEventSet;
            ulong inputConsumedEventAddress = context.InputConsumedEventAddressValue;
            string contextEngineName = context.EngineName;
            int contextEngineIOTensors = context.EngineIOTensorCount;
            int contextEngineLayers = context.EngineLayerCount;
            int contextEngineProfiles = context.EngineOptimizationProfileCount;
            bool temporaryAllocator = context.HasTemporaryStorageAllocator;
            bool temporaryAllocatorCleared = context.ClearTemporaryStorageAllocator();
            TensorRtProfilingVerbosity nvtxBefore = context.GetNvtxVerbosity();
            bool nvtxSet = context.SetNvtxVerbosity(nvtxBefore);
            TensorRtProfilingVerbosity nvtxAfter = context.GetNvtxVerbosity();
            context.ClearAuxStreams();
            bool profilerBefore = context.HasNativeProfiler;
            context.ClearProfiler();
            bool profilerAfter = context.HasNativeProfiler;
            bool managedProfilerAfter = context.HasProfiler;
            bool debugListenerBefore = context.HasDebugListener;
            bool debugListenerCleared = context.ClearDebugListener();
            bool debugListenerAfter = context.HasDebugListener;
            bool runtimeConfig = context.HasRuntimeConfig;
            string runtimeConfigStrategy = runtimeConfig ? context.RuntimeConfigAllocationStrategy.ToString() : "None";
            bool unfusedSet = context.SetUnfusedTensorsDebugState(false);
            bool unfusedState = context.GetUnfusedTensorsDebugState();
            TensorRtTensorInfo? firstTensor = ioTensors.FirstOrDefault();
            string addressState = firstTensor == null
                ? "NoTensor"
                : $"{firstTensor.Name}:0x{context.GetTensorAddressValue(firstTensor.Name):X}";
            TensorRtTensorInfo? output = ioTensors.FirstOrDefault(static tensor => tensor.IOMode == TensorRtIOMode.Output);
            string outputState = output == null
                ? "NoOutput"
                : $"{output.Name}:Address={context.IsOutputTensorAddressSet(output.Name)}:AddressValue=0x{context.GetOutputTensorAddressValue(output.Name):X}:Allocator={context.HasOutputAllocator(output.Name)}:ClearAllocator={context.ClearOutputAllocator(output.Name)}";

            using TensorRtExecutionContext clearContext = engine.CreateExecutionContext();
            TensorRtTensorInfo? input = ioTensors.FirstOrDefault(static tensor => tensor.IOMode == TensorRtIOMode.Input);
            bool clearAnyAddress = firstTensor != null && clearContext.ClearTensorAddress(firstTensor.Name);
            bool clearInputAddress = input != null && clearContext.ClearInputTensorAddress(input.Name);
            bool clearOutputAddress = output != null && clearContext.ClearOutputTensorAddress(output.Name);
            clearContext.ClearDeviceMemory();
            bool clearInputConsumedEvent = clearContext.ClearInputConsumedEvent();
            clearContext.SetAuxStreams(Array.Empty<CudaStream>());

            return $"Streamable={streamableWeights} Budget={budget}/Auto={automaticBudget}/Set={budgetSet} Scratch={scratch} Stats={totalWeights}/{strippedWeights} Hardware={hardware} ContextEngine={contextEngineName}:IO{contextEngineIOTensors}:L{contextEngineLayers}:P{contextEngineProfiles} InputConsumedEvent={inputConsumedEventSet}:0x{inputConsumedEventAddress:X}->{clearInputConsumedEvent} TempAllocator={temporaryAllocator}->{temporaryAllocatorCleared} Nvtx={nvtxBefore}->{nvtxAfter}/Set={nvtxSet} ProfilerNative={profilerBefore}->{profilerAfter}/Managed={managedProfilerAfter} DebugListener={debugListenerBefore}->{debugListenerAfter}/Clear={debugListenerCleared} RuntimeConfig={runtimeConfig}:{runtimeConfigStrategy} UnfusedDebug={unfusedState}/Set={unfusedSet} Address={addressState} Output={outputState} Clears=Any:{clearAnyAddress}/Input:{clearInputAddress}/Output:{clearOutputAddress}/DeviceMemory:True/AuxStreams:True";
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeTrt11EngineProfileTensorValuesV2(TensorRtEngine engine, IReadOnlyList<TensorRtTensorInfo> ioTensors)
    {
        try
        {
            List<string> states = new List<string>();
            foreach (TensorRtTensorInfo tensor in ioTensors)
            {
                int valueCount = Math.Max(1, tensor.Shape.Rank);
                long[] values = engine.GetProfileTensorValuesV2(tensor.Name, 0, TensorRtOptimizationProfileSelector.Opt, valueCount);
                states.Add($"{tensor.Name}:{values.Length}");
                if (states.Count >= 3)
                {
                    break;
                }
            }

            return states.Count == 0 ? "NoTensor" : string.Join(",", states);
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeTrt11EngineProfileTensorValueSnapshots(TensorRtEngine engine, IReadOnlyList<TensorRtTensorInfo> ioTensors)
    {
        try
        {
            List<string> states = new List<string>();
            foreach (TensorRtTensorInfo tensor in ioTensors)
            {
                int valueCount = Math.Max(1, tensor.Shape.Rank);
                TensorRtEngineProfileTensorValuesSnapshot snapshot =
                    engine.GetProfileTensorValuesSnapshot(tensor.Name, 0, TensorRtOptimizationProfileSelector.Opt, valueCount);
                bool trySnapshot = engine.TryGetProfileTensorValuesSnapshot(
                    tensor.Name,
                    0,
                    TensorRtOptimizationProfileSelector.Opt,
                    valueCount,
                    out TensorRtEngineProfileTensorValuesSnapshot tryGetSnapshot,
                    out string diagnostic);
                states.Add($"{tensor.Name}:V2={snapshot.ValuesV2.Count}:Legacy={snapshot.LegacyInt32Values.Count}:Try={trySnapshot}/{tryGetSnapshot.HasAnyValues}:Diag={snapshot.Diagnostics.Count}:{SanitizeSmokeValue(diagnostic)}");
                if (states.Count >= 3)
                {
                    break;
                }
            }

            return states.Count == 0 ? "NoTensor" : string.Join(",", states);
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeTrt8LegacyContextShapeBindingBoundary(TensorRtExecutionContext context)
    {
        try
        {
            int[] values = context.GetShapeBinding(0);
            return $"Values={values.Length}";
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string SanitizeSmokeValue(string value)
    {
        return string.IsNullOrEmpty(value)
            ? string.Empty
            : value.Replace(Environment.NewLine, " ").Replace('\r', ' ').Replace('\n', ' ').Replace(' ', '_');
    }

    static string ProbeTrt11Dims64(
        TensorRtNetworkDefinition network,
        TensorRtTensor input,
        TensorRtTensor output,
        TensorRtLayer identity,
        TensorRtOptimizationProfileShapeRange64 configuredProfileRange64,
        TensorRtEngine engine,
        TensorRtExecutionContext context,
        IReadOnlyList<TensorRtTensorInfo> ioTensors,
        int profileIndex)
    {
        try
        {
            List<string> states = new List<string>();
            TensorRtLayerTensorMetadata inputMetadata = identity.GetInputTensorMetadata(0);
            TensorRtLayerTensorMetadata outputMetadata = identity.GetOutputTensorMetadata(0);
            states.Add($"InputTensor={input.Shape64}/D0={input.GetDimensionExtent64(0)}");
            states.Add($"OutputTensor={output.Shape64}/D0={output.GetDimensionExtent64(0)}");
            states.Add($"NetworkI0={network.GetInputShape64(0)}/D0={network.GetInputDimensionExtent64(0, 0)}");
            states.Add($"NetworkI0D1={network.GetInputDimensionExtent64(0, 1)}");
            states.Add($"NetworkO0={network.GetOutputShape64(0)}/D0={network.GetOutputDimensionExtent64(0, 0)}");
            states.Add($"NetworkO0D1={network.GetOutputDimensionExtent64(0, 1)}");
            states.Add($"LayerI0={identity.GetInputTensorShape64(0)}/D0={identity.GetInputTensorDimensionExtent64(0, 0)}");
            states.Add($"LayerI0D1={identity.GetInputTensorDimensionExtent64(0, 1)}");
            states.Add($"LayerO0={identity.GetOutputTensorShape64(0)}/D0={identity.GetOutputTensorDimensionExtent64(0, 0)}");
            states.Add($"LayerO0D1={identity.GetOutputTensorDimensionExtent64(0, 1)}");
            states.Add($"Metadata64=I:{inputMetadata.Shape64}:{string.Join("/", inputMetadata.DimensionExtents64)}|O:{outputMetadata.Shape64}:{string.Join("/", outputMetadata.DimensionExtents64)}");
            states.Add($"ProfileConfigured64={configuredProfileRange64}");
            states.Add($"ProfileConfigured64D0={configuredProfileRange64.Min.Values[0]}/{configuredProfileRange64.Opt.Values[0]}/{configuredProfileRange64.Max.Values[0]}");

            foreach (TensorRtTensorInfo tensor in ioTensors.Take(2))
            {
                states.Add($"Engine64:{tensor.Name}={engine.GetTensorShape64(tensor.Name)}/D0={engine.GetTensorDimensionExtent64(tensor.Name, 0)}");
                states.Add($"Context64:{tensor.Name}={context.GetTensorShape64(tensor.Name)}/D0={context.GetTensorShapeDimensionExtent64(tensor.Name, 0)}");
                states.Add($"Context64D1:{tensor.Name}={context.GetTensorShapeDimensionExtent64(tensor.Name, 1)}");
                states.Add($"Strides64:{tensor.Name}={context.GetTensorStrides64(tensor.Name)}/D0={context.GetTensorStrideDimensionExtent64(tensor.Name, 0)}");
                states.Add($"Strides64D1:{tensor.Name}={context.GetTensorStrideDimensionExtent64(tensor.Name, 1)}");
                if (tensor.IOMode == TensorRtIOMode.Input)
                {
                    states.Add($"EngineProfile64:{tensor.Name}=Min{engine.GetProfileShape64(tensor.Name, profileIndex, TensorRtOptimizationProfileSelector.Min)}");
                    states.Add($"EngineProfile64Opt:{tensor.Name}={engine.GetProfileShape64(tensor.Name, profileIndex, TensorRtOptimizationProfileSelector.Opt)}");
                    states.Add($"EngineProfile64Max:{tensor.Name}={engine.GetProfileShape64(tensor.Name, profileIndex, TensorRtOptimizationProfileSelector.Max)}");
                    states.Add($"EngineProfile64D0:{tensor.Name}={engine.GetProfileShapeDimensionExtent64(tensor.Name, profileIndex, TensorRtOptimizationProfileSelector.Opt, 0)}");
                }
            }

            states.Add($"EvidenceCount={states.Count}");
            return string.Join(",", states);
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeOutputSizing(TensorRtExecutionContext context, IReadOnlyList<TensorRtTensorInfo> ioTensors)
    {
        IEnumerable<TensorRtTensorInfo> outputs = ioTensors.Where(static tensor => tensor.IOMode == TensorRtIOMode.Output);
        return string.Join(",", outputs.Select(tensor => $"{tensor.Name}:{context.GetMaxOutputSize(tensor.Name)}"));
    }

    static string ProbeTensorDebugState(TensorRtExecutionContext context, IReadOnlyList<TensorRtTensorInfo> ioTensors)
    {
        TensorRtTensorInfo? output = ioTensors.FirstOrDefault(static tensor => tensor.IOMode == TensorRtIOMode.Output);
        if (output == null)
        {
            return "NoOutputTensor";
        }

        if (!context.SupportsTensorDebugState)
        {
            return "Unsupported";
        }

        try
        {
            context.SetAllTensorsDebugState(false);
            bool before = context.GetTensorDebugState(output.Name);
            context.SetTensorDebugState(output.Name, false);
            bool after = context.GetTensorDebugState(output.Name);
            return $"{output.Name}:{before}->{after}";
        }
        catch (Exception exception)
        {
            return $"Skipped:{exception.Message}";
        }
    }

    static int EstimateTensorBytes(TensorRtTensorInfo tensor)
    {
        long elementCount = 1;
        foreach (int dimension in tensor.Shape.Values)
        {
            int normalizedDimension = dimension <= 0 ? 1 : dimension;
            checked
            {
                elementCount *= normalizedDimension;
            }
        }

        long byteCount = checked(elementCount * GetDataTypeSize(tensor.DataType));
        if (byteCount <= 0 || byteCount > int.MaxValue)
        {
            throw new InvalidOperationException($"Tensor {tensor.Name} has unsupported byte size {byteCount}.");
        }

        return (int)byteCount;
    }

    static int GetDataTypeSize(TensorRtDataType dataType)
    {
        switch (dataType)
        {
            case TensorRtDataType.Half:
            case TensorRtDataType.BFloat16:
                return 2;
            case TensorRtDataType.Float:
            case TensorRtDataType.Int32:
                return 4;
            case TensorRtDataType.Int64:
                return 8;
            case TensorRtDataType.Int8:
            case TensorRtDataType.Bool:
            case TensorRtDataType.UInt8:
            case TensorRtDataType.Float8:
                return 1;
            default:
                return 4;
        }
    }

    static void PrintDependencyProbe(TensorRtApiLine line)
    {
        TensorRtDependencyProbeReport dependencyProbe = TensorRtEnvironmentProbe.ProbeNativeDependencies(line);
        Console.WriteLine($"DependencyProbe Line={(int)line} BridgeInitialized={dependencyProbe.BridgeInitialized} Candidates={dependencyProbe.NativeBridgeCandidates.Count} Loaded={dependencyProbe.LoadedModuleCount} SearchPathCandidates={dependencyProbe.SearchPathCandidateCount} Diagnostics={dependencyProbe.Diagnostics.Count} Message={dependencyProbe.BridgeDiagnostic}");
    }

    static TensorRtApiLine ResolveProbeLine(string requestedLine)
    {
        if (string.Equals(requestedLine, "auto", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt11;
        }

        if (string.Equals(requestedLine, "8", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(requestedLine, "trt8", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt8;
        }

        if (string.Equals(requestedLine, "10", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(requestedLine, "trt10", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt10;
        }

        if (string.Equals(requestedLine, "11", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(requestedLine, "trt11", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt11;
        }

        throw new ArgumentException("TensorRT line must be auto, 8, 10, or 11.", nameof(requestedLine));
    }

    static string GetStringArgument(string[] args, string name, string defaultValue)
    {
        for (int index = 0; index < args.Length - 1; index++)
        {
            if (string.Equals(args[index], name, StringComparison.OrdinalIgnoreCase))
            {
                return args[index + 1];
            }
        }

        return defaultValue;
    }

    static bool HasSwitch(string[] args, string name)
    {
        return args.Any(argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
    }
}
