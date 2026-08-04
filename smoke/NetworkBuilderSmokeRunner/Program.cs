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
        try
        {
            Run(args);
        }
        catch (Exception exception) when (IsSkippableEnvironmentException(exception))
        {
            Console.WriteLine($"Skipped=True Reason={exception.GetType().Name}:{exception.Message}");
        }
    }

    private static void Run(string[] args)
    {
        TensorRtApiLine line = ResolveLine(JYPPX.SampleSupport.SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "10"));
        int batch = JYPPX.SampleSupport.SampleCommandLine.GetIntArgument(args, "--batch", 2);
        if (batch < 1 || batch > 4)
        {
            throw new ArgumentOutOfRangeException(nameof(batch), "Batch must be in the optimization profile range [1, 4].");
        }

        TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
        TensorRtAdapterInfo adapter = line switch
        {
            TensorRtApiLine.TensorRt8 => snapshot.TensorRt8,
            TensorRtApiLine.TensorRt10 => snapshot.TensorRt10,
            TensorRtApiLine.TensorRt11 => snapshot.TensorRt11,
            _ => snapshot.TensorRt10
        };
        Console.WriteLine($"NetworkBuilderSmokeRunner TensorRtLine={(int)line} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion} Batch={batch}");
        if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
        {
            Console.WriteLine($"Skipped=True Message={adapter.StatusMessage}");
            return;
        }

        using TensorRtLogger logger = new TensorRtLogger(line);
        using TensorRtRuntime runtime = new TensorRtRuntime(logger);
        using TensorRtBuilder builder = new TensorRtBuilder(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        using CudaStream stream = new CudaStream();
        const ulong workspaceMemoryPoolLimit = 64UL * 1024UL * 1024UL;
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, workspaceMemoryPoolLimit);
        ulong configuredWorkspaceMemoryPoolLimit = config.GetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace);
        config.SetOptimizationLevel(3);
        config.SetMaxAuxStreams(0);
        config.SetProfilingVerbosity(TensorRtProfilingVerbosity.Detailed);
        config.SetEngineCapability(TensorRtEngineCapability.Standard);
        config.SetHardwareCompatibilityLevel(TensorRtHardwareCompatibilityLevel.None);
        config.SetProfileStream(stream);
        bool profileStreamSet = config.IsProfileStreamSet;
        string builderScalarControlState = ProbeBuilderScalarControls(builder, config);
        string builderFlagMappingState = ProbeVersionedBuilderFlagMapping(config);
        string builderConfigDeploymentState = ProbeBuilderConfigDeploymentState(config, line);
        string pluginSerializationState = ProbeSerializedPluginPaths(config);
        int errorCodeUpperBound = TensorRtErrorCodeMetadata.GetExclusiveUpperBound(line);

        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        using TensorRtTensor inputTensor = network.AddInput("input", TensorRtDataType.Float, new TensorRtDims(new[] { -1, 4 }));
        using TensorRtLayer identity = network.AddIdentity(inputTensor);
        string layerDlaCapabilityState = ProbeLayerDlaCapability(config, identity);
        using TensorRtTensor outputTensor = identity.GetOutput(0);
        outputTensor.Name = "output";
        network.MarkOutput(outputTensor);

        Console.WriteLine($"Network Inputs={network.InputCount} Outputs={network.OutputCount} Input={inputTensor.Name}:{inputTensor.DataType}:{inputTensor.Shape} Output={outputTensor.Name}:{outputTensor.DataType}:{outputTensor.Shape}");

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

        using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
        int directEngineIoTensorCount;
        using (TensorRtEngine directEngine = builder.BuildEngineWithConfig(network, config))
        {
            directEngineIoTensorCount = directEngine.IOTensorCount;
        }
        using TensorRtEngine engine = runtime.Deserialize(hostMemory);
        string refitterLoggerState = ProbeRefitterLoggerPresence(engine, logger);
        string engineImplicitBatchState = ProbeEngineImplicitBatchCompatibility(engine);
        using TensorRtEngineInspector inspector = engine.CreateInspector();
        using TensorRtExecutionContext context = engine.CreateExecutionContext();
        context.ClearAuxStreams();
        TensorRtAuxiliaryStreamAssignmentSnapshot auxiliaryStreamSnapshot = context.GetAuxiliaryStreamAssignmentSnapshot();

        context.SetInputShape("input", new TensorRtDims(new[] { batch, 4 }));

        float[] inputValues = Enumerable.Range(0, batch * 4).Select(index => index + 0.5f).ToArray();
        using CudaMemory inputMemory = new CudaMemory(inputValues.Length * sizeof(float));
        using CudaMemory outputMemory = new CudaMemory(inputValues.Length * sizeof(float));
        inputMemory.CopyFrom(inputValues);
        outputMemory.Fill(0, outputMemory.SizeInBytes);

        context.SetInputTensorAddress("input", inputMemory);
        context.SetOutputTensorAddress("output", outputMemory);
        TensorRtExecutionContextReadiness readiness = context.GetReadiness(engine, runShapeInference: true);
        TensorRtEngineBindingReport bindingReport = engine.GetBindingReport(context, profileIndex, runShapeInference: false);
        context.EnqueueAsync(stream);
        stream.Synchronize();

        float[] outputValues = outputMemory.ToSingleArray(inputValues.Length);
        bool outputMatch = inputValues.SequenceEqual(outputValues);
        if (!outputMatch)
        {
            throw new InvalidOperationException($"Direct network output mismatch. Input=[{string.Join(", ", inputValues)}] Output=[{string.Join(", ", outputValues)}]");
        }

        IReadOnlyList<TensorRtTensorInfo> tensors = engine.GetIOTensors();
        string tensorSummary = string.Join("; ", tensors.Select(tensor => $"{tensor.Index}:{tensor.Name}:{tensor.IOMode}:{tensor.DataType}:{tensor.Shape}"));
        string bindingSummary = string.Join("; ", bindingReport.Tensors.Select(tensor => $"{tensor.Index}:{tensor.Name}:{tensor.IOMode}:{tensor.Format}:{tensor.VectorizedDimension}:{tensor.FormatDescription}"));
        string inspectorText = inspector.GetEngineInformation(TensorRtLayerInformationFormat.Oneline);
        ulong profileMemory = line == TensorRtApiLine.TensorRt10 ? engine.GetDeviceMemorySizeForProfileV2(profileIndex) : engine.DeviceMemorySizeInBytes;
        Console.WriteLine($"ProfileIndex={profileIndex} HostMemory={hostMemory.SizeInBytes} EngineIOTensors={engine.IOTensorCount}");
        Console.WriteLine($"BuilderConfig OptLevel={config.GetOptimizationLevel()} AuxStreams={config.GetMaxAuxStreams()} Profiling={config.GetProfilingVerbosity()} WorkspaceMemoryPoolLimit={configuredWorkspaceMemoryPoolLimit} ProfileStream={profileStreamSet} ProfileCount={configProfileCount} CalibrationProfile={calibrationProfileState} PluginSerialization={pluginSerializationState} {builderScalarControlState} {builderFlagMappingState} {layerDlaCapabilityState} {builderConfigDeploymentState}");
        Console.WriteLine($"EngineMemory Device={engine.DeviceMemorySizeInBytes} Profile={profileMemory} AuxStreams={engine.AuxiliaryStreamCount} ImplicitBatch={engineImplicitBatchState}");
        Console.WriteLine($"DirectEngineBuild=True IOTensors={directEngineIoTensorCount} RefitterHasLogger={refitterLoggerState} ErrorCodeUpperBound={errorCodeUpperBound}");
        Console.WriteLine($"ProfileConfigured Min={configuredProfileRange.Min} Opt={configuredProfileRange.Opt} Max={configuredProfileRange.Max} Valid={configuredProfileValid} ExtraMemoryTarget={profileExtraMemoryTarget} ShapeValueCount={inputShapeValueCount}");
        Console.WriteLine($"Readiness Ready={readiness.IsReadyForEnqueue} Bound={readiness.AllTensorAddressesBound} Missing={readiness.ShapeInferenceMissingTensorCount?.ToString() ?? "n/a"} ActiveProfile={readiness.ActiveOptimizationProfile} Tensors={readiness.Tensors.Count}");
        Console.WriteLine($"BindingReport Ready={bindingReport.IsReadyForEnqueue} Profile={bindingReport.ProfileIndex} Inputs={bindingReport.GetInputs().Count} Outputs={bindingReport.GetOutputs().Count} Tensors={bindingReport.Tensors.Count} Formats=[{bindingSummary}]");
        Console.WriteLine($"AuxiliaryStreams=Line:{(int)auxiliaryStreamSnapshot.Line};Assigned:{auxiliaryStreamSnapshot.AssignedStreamCount};Cleared:{auxiliaryStreamSnapshot.IsCleared};Lease:{auxiliaryStreamSnapshot.ManagedHandleLeaseActive}");
        Console.WriteLine($"IOTensors=[{tensorSummary}] InspectorBytes={inspectorText.Length} Enqueue=True OutputMatch=True");

    }

    static string ProbeRefitterLoggerPresence(TensorRtEngine engine, TensorRtLogger logger)
    {
        try
        {
            using TensorRtRefitter refitter = engine.CreateRefitter(logger);
            return refitter.HasLogger.ToString();
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            return $"Skipped:{exception.GetType().Name}";
        }
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


    static string ProbeBuilderScalarControls(TensorRtBuilder builder, TensorRtBuilderConfig config)
    {
        try
        {
            int originalMaxThreads = builder.MaxThreads;
            int targetMaxThreads = Math.Max(1, originalMaxThreads);
            bool maxThreadsAccepted = builder.SetMaxThreads(targetMaxThreads);
            int currentMaxThreads = builder.MaxThreads;
            int maxDlaBatchSize = builder.MaxDlaBatchSize;
            string maxBatchSizeCompatibility = ProbeBuilderMaxBatchSizeCompatibility(builder);
            config.SetDefaultDeviceType(TensorRtDeviceType.Gpu);
            TensorRtDeviceType defaultDeviceType = config.GetDefaultDeviceType();
            TensorRtBuilderFlags builderFlags = config.GetFlags();
            int dlaCore = config.GetDlaCore();
            string legacyConfig = ProbeBuilderConfigLegacyCompatibility(config);
            string quantization = ProbeQuantizationFlags(config);
            string tiling = ProbeTilingControls(config);
            string callbackPresence = ProbeBuilderConfigCallbackPresence(config);
            return $"ScalarControls=MaxThreads:{originalMaxThreads}->{currentMaxThreads}:Set={maxThreadsAccepted};MaxDlaBatch={maxDlaBatchSize};{maxBatchSizeCompatibility};DefaultDevice={defaultDeviceType};BuilderFlags={builderFlags};DlaCore={dlaCore};{legacyConfig};{quantization};{tiling};{callbackPresence}";
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            return $"ScalarControls=Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeBuilderMaxBatchSizeCompatibility(TensorRtBuilder builder)
    {
        if (builder.Line != TensorRtApiLine.TensorRt8)
        {
            return "MaxBatchCompatibility=Skipped:ModernExplicitBatch";
        }

        try
        {
            int before = builder.MaxBatchSizeCompatibility;
            builder.SetMaxBatchSizeCompatibility(before);
            int after = builder.MaxBatchSizeCompatibility;
            return $"MaxBatchCompatibility={before}->{after}:Set=True";
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            return $"MaxBatchCompatibility=Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeSerializedPluginPaths(TensorRtBuilderConfig config)
    {
        try
        {
            bool set = config.SetPluginsToSerialize(Array.Empty<string>());
            TensorRtBuilderConfigSerializedPluginSnapshot snapshot = config.GetSerializedPluginSnapshot();
            return $"Set={set}:Count={snapshot.Count}:Copied={snapshot.PluginLibraryPaths.Count}:Available={snapshot.HasPathInventory}";
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            return $"Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeEngineImplicitBatchCompatibility(TensorRtEngine engine)
    {
        if (engine.Line == TensorRtApiLine.TensorRt11)
        {
            return "Skipped:TensorRt11ExplicitBatch";
        }

        try
        {
            return engine.HasImplicitBatchDimensionCompatibility.ToString();
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            return $"Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeBuilderConfigLegacyCompatibility(TensorRtBuilderConfig config)
    {
        if (config.Line != TensorRtApiLine.TensorRt8)
        {
            return "LegacyConfigCompatibility=Skipped:ModernMemoryPools";
        }

        try
        {
            ulong workspaceBefore = config.MaxWorkspaceSizeCompatibilityInBytes;
            int minTimingBefore = config.MinTimingIterationsCompatibility;
            config.SetMaxWorkspaceSizeCompatibility(workspaceBefore);
            config.SetMinTimingIterationsCompatibility(minTimingBefore);
            ulong workspaceAfter = config.MaxWorkspaceSizeCompatibilityInBytes;
            int minTimingAfter = config.MinTimingIterationsCompatibility;
            config.SetFlags(config.GetFlags());
            return $"LegacyConfigCompatibility=Workspace:{workspaceBefore}->{workspaceAfter};MinTiming:{minTimingBefore}->{minTimingAfter};FlagsSet=True";
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            return $"LegacyConfigCompatibility=Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeQuantizationFlags(TensorRtBuilderConfig config)
    {
        if (config.Line == TensorRtApiLine.TensorRt11)
        {
            return "QuantizationFlags=Skipped:TensorRt11";
        }

        try
        {
            TensorRtQuantizationFlags originalFlags = config.GetQuantizationFlags();
            config.SetQuantizationFlag(TensorRtQuantizationFlag.CalibrateBeforeFusion);
            bool enabled = config.GetQuantizationFlag(TensorRtQuantizationFlag.CalibrateBeforeFusion);
            config.ClearQuantizationFlag(TensorRtQuantizationFlag.CalibrateBeforeFusion);
            TensorRtQuantizationFlags clearedFlags = config.GetQuantizationFlags();
            config.SetQuantizationFlags(originalFlags);
            return $"QuantizationFlags=Original:{originalFlags};Enabled={enabled};Cleared:{clearedFlags}";
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            return $"QuantizationFlags=Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeTilingControls(TensorRtBuilderConfig config)
    {
        if (config.Line == TensorRtApiLine.TensorRt8)
        {
            return "TilingControls=Skipped:TensorRt8";
        }

        try
        {
            int originalMaxTactics = config.GetMaxTactics();
            TensorRtTilingOptimizationLevel originalLevel = config.GetTilingOptimizationLevel();
            long originalL2Limit = config.GetL2LimitForTiling();
            config.SetMaxTactics(Math.Max(0, originalMaxTactics));
            bool tilingAccepted = config.SetTilingOptimizationLevel(TensorRtTilingOptimizationLevel.None);
            bool l2Accepted = config.SetL2LimitForTiling(0);
            TensorRtTilingOptimizationLevel currentLevel = config.GetTilingOptimizationLevel();
            long currentL2Limit = config.GetL2LimitForTiling();
            return $"TilingControls=MaxTactics:{originalMaxTactics};Tiling:{originalLevel}->{currentLevel}:Set={tilingAccepted};L2:{originalL2Limit}->{currentL2Limit}:Set={l2Accepted}";
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            return $"TilingControls=Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeBuilderConfigCallbackPresence(TensorRtBuilderConfig config)
    {
        try
        {
            bool hasAlgorithmSelector = config.HasAlgorithmSelectorCompatibility;
            bool hasInt8Calibrator = config.HasInt8CalibratorCompatibility;
            return $"CallbackPresence=AlgorithmSelector:{hasAlgorithmSelector};Int8Calibrator:{hasInt8Calibrator}";
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            return $"CallbackPresence=Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeLayerDlaCapability(TensorRtBuilderConfig config, TensorRtLayer layer)
    {
        try
        {
            return $"LayerDla={config.CanRunOnDla(layer)}";
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            return $"LayerDla=Skipped:{exception.GetType().Name}:{exception.Message}";
        }
    }

    static string ProbeVersionedBuilderFlagMapping(TensorRtBuilderConfig config)
    {
        if (config.Line != TensorRtApiLine.TensorRt8)
        {
            return "BuilderFlagMapping=NotRequired";
        }

        bool originalDirectIo = config.GetFlag(TensorRtBuilderFlag.DirectIO);
        bool originalPrefer = config.GetFlag(TensorRtBuilderFlag.PreferPrecisionConstraints);
        try
        {
            config.SetFlag(TensorRtBuilderFlag.DirectIO, true);
            bool directAfterSet = config.GetFlag(TensorRtBuilderFlag.DirectIO);
            bool preferAfterDirect = config.GetFlag(TensorRtBuilderFlag.PreferPrecisionConstraints);
            TensorRtBuilderFlags flagsAfterDirect = config.GetFlags();

            config.SetFlag(TensorRtBuilderFlag.PreferPrecisionConstraints, true);
            config.SetFlag(TensorRtBuilderFlag.DirectIO, false);
            bool directAfterClear = config.GetFlag(TensorRtBuilderFlag.DirectIO);
            bool preferAfterDirectClear = config.GetFlag(TensorRtBuilderFlag.PreferPrecisionConstraints);
            bool isolated = directAfterSet &&
                !preferAfterDirect &&
                (flagsAfterDirect & TensorRtBuilderFlags.DirectIO) != 0 &&
                (flagsAfterDirect & TensorRtBuilderFlags.PreferPrecisionConstraints) == 0 &&
                !directAfterClear &&
                preferAfterDirectClear;
            if (!isolated)
            {
                throw new InvalidOperationException("TensorRT 8 DirectIO and PreferPrecisionConstraints flag mapping is not isolated.");
            }

            return $"BuilderFlagMapping=TRT8DirectIORaw12PreferRaw11:Isolated:{isolated}";
        }
        finally
        {
            config.SetFlag(TensorRtBuilderFlag.DirectIO, originalDirectIo);
            config.SetFlag(TensorRtBuilderFlag.PreferPrecisionConstraints, originalPrefer);
        }
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

    private static bool IsSkippableEnvironmentException(Exception exception)
    {
        if (exception is BridgeProbeException bridgeProbe)
        {
            return bridgeProbe.StatusCode == BridgeStatusCode.DependencyMissing ||
                   bridgeProbe.StatusCode == BridgeStatusCode.NotSupported ||
                   bridgeProbe.StatusCode == BridgeStatusCode.InvalidState ||
                   IsKnownVendorSeh(bridgeProbe);
        }

        return false;
    }

    private static bool IsKnownVendorSeh(BridgeProbeException exception)
    {
        return exception.StatusCode == BridgeStatusCode.RuntimeError &&
               exception.Message.Contains("structured exception with code 3228369022", StringComparison.Ordinal);
    }
}
