using System;
using System.Linq;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

internal static class Program
{
    private static void Main(string[] args)
    {
        TensorRtApiLine line = ResolveLine(JYPPX.SampleSupport.SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "10"));

        TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
        TensorRtAdapterInfo adapter = line switch
        {
            TensorRtApiLine.TensorRt8 => snapshot.TensorRt8,
            TensorRtApiLine.TensorRt10 => snapshot.TensorRt10,
            TensorRtApiLine.TensorRt11 => snapshot.TensorRt11,
            _ => snapshot.TensorRt10
        };
        Console.WriteLine($"NetworkLrnSmokeRunner TensorRtLine={(int)line} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");
        if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
        {
            Console.WriteLine($"Skipped=True Message={adapter.StatusMessage}");
            return;
        }

        float[] inputValues = { 1.0f, -2.0f, 3.5f, 8.0f };
        float[] expectedValues = inputValues.ToArray();

        using TensorRtLogger logger = new TensorRtLogger(line);
        using TensorRtRuntime runtime = new TensorRtRuntime(logger);
        using TensorRtBuilder builder = new TensorRtBuilder(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
        config.SetOptimizationLevel(3);
        config.SetMaxAuxStreams(0);
        config.SetAverageTimingIterations(1);

        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        network.Name = "lrn_identity_network";
        using TensorRtTensor inputTensor = network.AddInput("input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 1, 4 }));
        using TensorRtLayer lrnLayer = network.AddLrn(inputTensor, windowSize: 1, alpha: 0.0f, beta: 1.0f, k: 1.0f);
        lrnLayer.Name = "lrn_identity";

        if (lrnLayer.Type != TensorRtLayerType.Lrn)
        {
            throw new InvalidOperationException($"Unexpected LRN layer type: {lrnLayer.Type}");
        }

        if (lrnLayer.GetLrnWindowSize() != 1 ||
            Math.Abs(lrnLayer.GetLrnAlpha()) > 0.000001f ||
            Math.Abs(lrnLayer.GetLrnBeta() - 1.0f) > 0.000001f ||
            Math.Abs(lrnLayer.GetLrnK() - 1.0f) > 0.000001f)
        {
            throw new InvalidOperationException("LRN layer metadata does not match the configured identity parameters.");
        }

        using TensorRtTensor outputTensor = lrnLayer.GetOutput(0);
        outputTensor.Name = "output";
        network.MarkOutput(outputTensor);

        using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
        using TensorRtEngine engine = runtime.Deserialize(hostMemory);
        using TensorRtExecutionContext context = engine.CreateExecutionContext();
        using CudaStream stream = new CudaStream();
        using CudaMemory inputMemory = new CudaMemory(inputValues.Length * sizeof(float));
        using CudaMemory outputMemory = new CudaMemory(expectedValues.Length * sizeof(float));
        using CudaMemory? contextDeviceMemory = TryCreateAndSetContextDeploymentMemory(context, engine);

        inputMemory.CopyFrom(inputValues);
        outputMemory.Fill(0, outputMemory.SizeInBytes);
        context.SetTensorAddress("input", inputMemory);
        context.SetTensorAddress("output", outputMemory);
        context.EnqueueAsync(stream);
        stream.Synchronize();

        float[] outputValues = outputMemory.ToSingleArray(expectedValues.Length);
        AssertClose("LrnIdentity", expectedValues, outputValues, 0.0001f);
        Console.WriteLine($"LrnIdentityOutputMatch=True Layers={network.LayerCount} HostMemory={hostMemory.SizeInBytes} EngineMemory={engine.DeviceMemorySizeInBytes} ContextDeviceMemory={(contextDeviceMemory == null ? 0 : contextDeviceMemory.SizeInBytes)}");

    }

    static CudaMemory? TryCreateAndSetContextDeploymentMemory(TensorRtExecutionContext context, TensorRtEngine engine)
    {
        if (engine.DeviceMemorySizeInBytes == 0 || engine.DeviceMemorySizeInBytes > int.MaxValue)
        {
            Console.WriteLine($"ContextDeviceMemory Skipped=True Size={engine.DeviceMemorySizeInBytes}");
            return null;
        }

        CudaMemory deviceMemory = new CudaMemory(checked((int)engine.DeviceMemorySizeInBytes));
        context.SetDeviceMemory(deviceMemory);
        return deviceMemory;
    }

    static void AssertClose(string name, float[] expected, float[] actual, float tolerance)
    {
        bool match = expected.Length == actual.Length &&
            expected.Zip(actual, (expectedValue, actualValue) => Math.Abs(expectedValue - actualValue) <= tolerance).All(static value => value);
        if (!match)
        {
            throw new InvalidOperationException($"{name} output mismatch. Expected=[{string.Join(", ", expected)}] Actual=[{string.Join(", ", actual)}]");
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
