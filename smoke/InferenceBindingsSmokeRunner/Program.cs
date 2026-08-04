using System;
using System.Linq;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
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
        Console.WriteLine($"InferenceBindingsSmokeRunner TensorRtLine={(int)line} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion} Batch={batch}");
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
        config.SetProfileStream(stream);

        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        using TensorRtTensor inputTensor = network.AddInput("input", TensorRtDataType.Float, new TensorRtDims(new[] { -1, 4 }));
        using TensorRtLayer identity = network.AddIdentity(inputTensor);
        using TensorRtTensor outputTensor = identity.GetOutput(0);
        outputTensor.Name = "output";
        network.MarkOutput(outputTensor);

        using TensorRtOptimizationProfile profile = builder.CreateOptimizationProfile();
        profile.SetShape(
            "input",
            new TensorRtDims(new[] { 1, 4 }),
            new TensorRtDims(new[] { 2, 4 }),
            new TensorRtDims(new[] { 4, 4 }));
        int profileIndex = config.AddOptimizationProfile(profile);

        using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
        using TensorRtEngine engine = runtime.Deserialize(hostMemory);
        using TensorRtExecutionContext context = engine.CreateExecutionContext();
        TensorRtDims runtimeShape = new TensorRtDims(new[] { batch, 4 });
        float[] inputValues = Enumerable.Range(0, batch * 4).Select(index => index + 1.25f).ToArray();

        using TensorRtInferenceBindings bindings = new TensorRtInferenceBindings(engine, context, profileIndex);
        bindings.SetInputShape("input", runtimeShape)
                .CopyInputFromHost("input", inputValues, runtimeShape);
        bindings.AllocateDeviceBuffer("output", runtimeShape);
        bindings.BindAll();

        TensorRtExecutionContextReadiness readiness = bindings.GetReadiness(runShapeInference: true);
        if (!readiness.IsReadyForEnqueue)
        {
            throw new InvalidOperationException("Inference bindings are not ready: " + readiness);
        }

        TensorRtInferenceExecutionSummary executeV2Summary = bindings.ExecuteV2(runShapeInference: false);
        float[] executeV2Output = bindings.ReadOutputSingles("output", inputValues.Length);
        EnsureOutputMatches("executeV2", inputValues, executeV2Output);

        TensorRtInferenceExecutionSummary enqueueSummary = null!;
        float elapsedMilliseconds = stream.MeasureElapsedTime(cudaStream =>
        {
            enqueueSummary = bindings.EnqueueAsync(cudaStream, synchronize: false, runShapeInference: false);
        });

        float[] outputValues = bindings.ReadOutputSingles("output", inputValues.Length);
        EnsureOutputMatches("enqueueV3", inputValues, outputValues);

        TensorRtInferenceExecutionSummary? enqueueV2Summary = null;
        float enqueueV2ElapsedMilliseconds = 0;
        if (line == TensorRtApiLine.TensorRt8)
        {
            enqueueV2ElapsedMilliseconds = stream.MeasureElapsedTime(cudaStream =>
            {
                enqueueV2Summary = bindings.EnqueueV2AndSynchronize(cudaStream, runShapeInference: false);
            });
            float[] enqueueV2Output = bindings.ReadOutputSingles("output", inputValues.Length);
            EnsureOutputMatches("enqueueV2", inputValues, enqueueV2Output);
            Console.WriteLine($"LegacyExecute Skipped=True Reason=ExplicitBatchIdentityEngine");
        }

        ulong profileMemory = line == TensorRtApiLine.TensorRt10 ? engine.GetDeviceMemorySizeForProfileV2(profileIndex) : engine.DeviceMemorySizeInBytes;
        Console.WriteLine($"BindingReport {bindings.Report} Inputs={bindings.Report.GetInputs().Count} Outputs={bindings.Report.GetOutputs().Count}");
        Console.WriteLine($"EngineMemory Device={engine.DeviceMemorySizeInBytes} Profile={profileMemory} AuxStreams={engine.AuxiliaryStreamCount}");
        Console.WriteLine($"Readiness Ready={readiness.IsReadyForEnqueue} Bound={readiness.AllTensorAddressesBound} ActiveProfile={readiness.ActiveOptimizationProfile}");
        Console.WriteLine($"ExecuteV2 {executeV2Summary} OutputMatch=True");
        Console.WriteLine($"EnqueueV3 {enqueueSummary} ElapsedMs={elapsedMilliseconds:0.###} OutputMatch=True");
        if (enqueueV2Summary != null)
        {
            Console.WriteLine($"EnqueueV2 {enqueueV2Summary} ElapsedMs={enqueueV2ElapsedMilliseconds:0.###} Synchronized={enqueueV2Summary.Synchronized} OutputMatch=True");
        }
        Console.WriteLine(bindings.Describe());

    }

    private static void EnsureOutputMatches(string operation, float[] inputValues, float[] outputValues)
    {
        if (!inputValues.SequenceEqual(outputValues))
        {
            throw new InvalidOperationException(
                $"Inference binding {operation} output mismatch. Input=[{string.Join(", ", inputValues)}] Output=[{string.Join(", ", outputValues)}]");
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
}
