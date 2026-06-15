using System;
using System.Linq;
using JYPPX.CudaSharp;
using JYPPX.SampleSupport;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace InferenceBindingsSample;

internal static class Program
{
    public static int Main(string[] args)
    {
        try
        {
            return Run(args);
        }
        catch (Exception exception) when (TensorRtSampleSupport.IsDeploymentException(exception))
        {
            Console.WriteLine($"InferenceBindings=Skipped Reason={exception.Message}");
            return 0;
        }
        catch (ArgumentException exception)
        {
            Console.WriteLine($"InferenceBindings=InvalidArguments Reason={exception.Message}");
            PrintUsage();
            return 2;
        }
    }

    private static int Run(string[] args)
    {
        TensorRtApiLine line = TensorRtSampleSupport.ResolveLine(SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "10"));
        int batch = SampleCommandLine.GetIntArgument(args, "--batch", 2);
        if (batch < 1 || batch > 4)
        {
            throw new ArgumentOutOfRangeException(nameof(batch), "Batch must be in the optimization profile range [1, 4].");
        }

        TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
        TensorRtAdapterInfo adapter = TensorRtSampleSupport.SelectAdapter(snapshot, line);
        Console.WriteLine($"InferenceBindings TensorRtLine={(int)line} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion} Batch={batch}");
        if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
        {
            Console.WriteLine($"InferenceBindings=Skipped Reason={adapter.StatusMessage}");
            return 0;
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

        TensorRtInferenceExecutionSummary executionSummary = null!;
        float elapsedMilliseconds = stream.MeasureElapsedTime(cudaStream =>
        {
            executionSummary = bindings.EnqueueAsync(cudaStream, synchronize: false, runShapeInference: false);
        });

        float[] outputValues = bindings.ReadOutputSingles("output", inputValues.Length);
        if (!inputValues.SequenceEqual(outputValues))
        {
            throw new InvalidOperationException($"Inference binding output mismatch. Input=[{string.Join(", ", inputValues)}] Output=[{string.Join(", ", outputValues)}]");
        }

        ulong profileMemory = line == TensorRtApiLine.TensorRt10 ? engine.GetDeviceMemorySizeForProfileV2(profileIndex) : engine.DeviceMemorySizeInBytes;
        Console.WriteLine($"BindingReport Ready={bindings.Report.IsReadyForEnqueue} Inputs={bindings.Report.GetInputs().Count} Outputs={bindings.Report.GetOutputs().Count}");
        Console.WriteLine($"EngineMemory Device={engine.DeviceMemorySizeInBytes} Profile={profileMemory} AuxStreams={engine.AuxiliaryStreamCount}");
        Console.WriteLine($"Readiness Ready={readiness.IsReadyForEnqueue} Bound={readiness.AllTensorAddressesBound} ActiveProfile={readiness.ActiveOptimizationProfile}");
        Console.WriteLine($"Execution {executionSummary} ElapsedMs={elapsedMilliseconds:0.###} OutputMatch=True");
        Console.WriteLine(bindings.Describe());
        Console.WriteLine("InferenceBindings Passed=True");
        return 0;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("InferenceBindings sample");
        Console.WriteLine("Usage:");
        Console.WriteLine("  dotnet run --project samples/InferenceBindings -- --tensor-rt-line 10 --batch 2");
        Console.WriteLine("Options:");
        Console.WriteLine("  --tensor-rt-line <8|10|11>  TensorRT adapter line. Default: 10.");
        Console.WriteLine("  --batch <1..4>              Runtime batch inside the optimization profile. Default: 2.");
    }
}
