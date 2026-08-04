using System;
using System.Linq;
using JYPPX.CudaSharp;
using JYPPX.SampleSupport;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace DynamicShapeSample;

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
            Console.WriteLine($"DynamicShape=Skipped Reason={exception.Message}");
            return 0;
        }
        catch (ArgumentException exception)
        {
            Console.WriteLine($"DynamicShape=InvalidArguments Reason={exception.Message}");
            PrintUsage();
            return 2;
        }
    }

    private static int Run(string[] args)
    {
        TensorRtApiLine line = TensorRtSampleSupport.ResolveLine(SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "10"));
        int batch = SampleCommandLine.GetIntArgument(args, "--batch", 3);
        if (batch < 1 || batch > 4)
        {
            throw new ArgumentOutOfRangeException(nameof(batch), "Batch must be in the optimization profile range [1, 4].");
        }

        TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
        TensorRtAdapterInfo adapter = TensorRtSampleSupport.SelectAdapter(snapshot, line);
        Console.WriteLine($"DynamicShape TensorRtLine={(int)line} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion} Batch={batch}");
        if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
        {
            Console.WriteLine($"DynamicShape=Skipped Reason={adapter.StatusMessage}");
            return 0;
        }

        bool runtimeReady = TensorRtEnvironmentProbe.TryCreateRuntime(line, out string runtimeMessage);
        bool builderReady = TensorRtEnvironmentProbe.TryCreateBuilder(line, out string builderMessage);
        if (!runtimeReady || !builderReady)
        {
            Console.WriteLine($"DynamicShape=Skipped Reason=Runtime={runtimeMessage}; Builder={builderMessage}");
            return 0;
        }

        using TensorRtLogger logger = new TensorRtLogger(line);
        using TensorRtRuntime runtime = new TensorRtRuntime(logger);
        using TensorRtBuilder builder = new TensorRtBuilder(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        using CudaStream stream = new CudaStream(CudaStreamCreationFlags.NonBlocking);

        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
        config.SetOptimizationLevel(3);
        config.SetMaxAuxStreams(0);
        config.SetProfileStream(stream);

        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        using TensorRtTensor inputTensor = network.AddInput("input", TensorRtDataType.Float, new TensorRtDims(new[] { -1, 3, 4 }));
        using TensorRtLayer identity = network.AddIdentity(inputTensor);
        using TensorRtTensor outputTensor = identity.GetOutput(0);
        outputTensor.Name = "output";
        network.MarkOutput(outputTensor);

        using TensorRtOptimizationProfile profile = builder.CreateOptimizationProfile();
        profile.SetShape(
            "input",
            new TensorRtDims(new[] { 1, 3, 4 }),
            new TensorRtDims(new[] { 2, 3, 4 }),
            new TensorRtDims(new[] { 4, 3, 4 }));
        TensorRtOptimizationProfileShapeRange profileRange = profile.GetShapeRange("input");
        bool profileValid = profile.IsValid;
        int profileIndex = config.AddOptimizationProfile(profile);

        using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
        using TensorRtEngine engine = runtime.Deserialize(hostMemory);
        using TensorRtExecutionContext context = engine.CreateExecutionContext();

        TensorRtDims runtimeShape = new TensorRtDims(new[] { batch, 3, 4 });
        float[] inputValues = Enumerable.Range(0, batch * 3 * 4)
            .Select(index => (index + 1) / 10.0f)
            .ToArray();

        using TensorRtInferenceBindings bindings = new TensorRtInferenceBindings(engine, context, profileIndex);
        bindings.SetInputShape("input", runtimeShape)
                .CopyInputFromHost("input", inputValues, runtimeShape);
        bindings.AllocateDeviceBuffer("output", runtimeShape, checked(inputValues.Length * sizeof(float)));
        bindings.BindAll();

        TensorRtExecutionContextReadiness readiness = bindings.GetReadiness(runShapeInference: true);
        if (!readiness.IsReadyForEnqueue)
        {
            throw new InvalidOperationException("Dynamic shape bindings are not ready: " + readiness);
        }

        TensorRtInferenceExecutionSummary executionSummary = null!;
        float elapsedMilliseconds = stream.MeasureElapsedTime(cudaStream =>
        {
            executionSummary = bindings.EnqueueAsync(cudaStream, synchronize: false, runShapeInference: false);
        });

        float[] outputValues = bindings.ReadOutputSingles("output", inputValues.Length);
        bool outputMatch = inputValues.SequenceEqual(outputValues);
        if (!outputMatch)
        {
            throw new InvalidOperationException($"Dynamic shape output mismatch. Input=[{string.Join(", ", inputValues)}] Output=[{string.Join(", ", outputValues)}]");
        }

        TensorRtEngineBindingReport report = bindings.Report;
        Console.WriteLine($"Network Input={inputTensor.Name}:{inputTensor.Shape} Output={outputTensor.Name}:{outputTensor.Shape}");
        Console.WriteLine($"Profile Index={profileIndex} Min={profileRange.Min} Opt={profileRange.Opt} Max={profileRange.Max} Valid={profileValid}");
        Console.WriteLine($"RuntimeShape={runtimeShape} Values={inputValues.Length} HostMemory={hostMemory.SizeInBytes} EngineTensors={engine.IOTensorCount}");
        Console.WriteLine($"Readiness Ready={readiness.IsReadyForEnqueue} Bound={readiness.AllTensorAddressesBound} ActiveProfile={readiness.ActiveOptimizationProfile}");
        Console.WriteLine($"BindingReport Ready={report.IsReadyForEnqueue} Inputs={report.GetInputs().Count} Outputs={report.GetOutputs().Count}");
        Console.WriteLine($"Execution {executionSummary} ElapsedMs={elapsedMilliseconds:0.###} OutputMatch=True");
        Console.WriteLine("DynamicShape Passed=True");
        return 0;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("DynamicShape sample");
        Console.WriteLine("Usage:");
        Console.WriteLine("  dotnet run --project samples/DynamicShape -- --tensor-rt-line 10 --batch 3");
        Console.WriteLine("Options:");
        Console.WriteLine("  --tensor-rt-line <8|10|11>  TensorRT adapter line. Default: 10.");
        Console.WriteLine("  --batch <1..4>              Runtime batch inside the optimization profile. Default: 3.");
    }
}
