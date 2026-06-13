using System;
using System.Collections.Generic;
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
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
        config.SetOptimizationLevel(3);
        config.SetMaxAuxStreams(0);
        config.SetProfilingVerbosity(TensorRtProfilingVerbosity.Detailed);
        config.SetEngineCapability(TensorRtEngineCapability.Standard);
        config.SetHardwareCompatibilityLevel(TensorRtHardwareCompatibilityLevel.None);
        config.SetProfileStream(stream);
        bool profileStreamSet = config.IsProfileStreamSet;
        string builderConfigDeploymentState = ProbeBuilderConfigDeploymentState(config, line);

        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        using TensorRtTensor inputTensor = network.AddInput("input", TensorRtDataType.Float, new TensorRtDims(new[] { -1, 4 }));
        using TensorRtLayer identity = network.AddIdentity(inputTensor);
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
        using TensorRtEngine engine = runtime.Deserialize(hostMemory);
        using TensorRtEngineInspector inspector = engine.CreateInspector();
        using TensorRtExecutionContext context = engine.CreateExecutionContext();

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
        Console.WriteLine($"BuilderConfig OptLevel={config.GetOptimizationLevel()} AuxStreams={config.GetMaxAuxStreams()} Profiling={config.GetProfilingVerbosity()} ProfileStream={profileStreamSet} ProfileCount={configProfileCount} CalibrationProfile={calibrationProfileState} {builderConfigDeploymentState}");
        Console.WriteLine($"EngineMemory Device={engine.DeviceMemorySizeInBytes} Profile={profileMemory} AuxStreams={engine.AuxiliaryStreamCount}");
        Console.WriteLine($"ProfileConfigured Min={configuredProfileRange.Min} Opt={configuredProfileRange.Opt} Max={configuredProfileRange.Max} Valid={configuredProfileValid} ExtraMemoryTarget={profileExtraMemoryTarget} ShapeValueCount={inputShapeValueCount}");
        Console.WriteLine($"Readiness Ready={readiness.IsReadyForEnqueue} Bound={readiness.AllTensorAddressesBound} Missing={readiness.ShapeInferenceMissingTensorCount?.ToString() ?? "n/a"} ActiveProfile={readiness.ActiveOptimizationProfile} Tensors={readiness.Tensors.Count}");
        Console.WriteLine($"BindingReport Ready={bindingReport.IsReadyForEnqueue} Profile={bindingReport.ProfileIndex} Inputs={bindingReport.GetInputs().Count} Outputs={bindingReport.GetOutputs().Count} Tensors={bindingReport.Tensors.Count} Formats=[{bindingSummary}]");
        Console.WriteLine($"IOTensors=[{tensorSummary}] InspectorBytes={inspectorText.Length} Enqueue=True OutputMatch=True");

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

        return $"Capability={capability} HardwareCompatibility={hardwareCompatibility} PreviewFeature={previewFeature}:{previewEnabled} RuntimePlatform={runtimePlatform}";
    }
}
