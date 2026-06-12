using System;
using System.Linq;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

TensorRtApiLine line = ResolveLine(JYPPX.SampleSupport.SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "10"));

TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
TensorRtAdapterInfo adapter = line switch
{
    TensorRtApiLine.TensorRt8 => snapshot.TensorRt8,
    TensorRtApiLine.TensorRt10 => snapshot.TensorRt10,
    TensorRtApiLine.TensorRt11 => snapshot.TensorRt11,
    _ => snapshot.TensorRt10
};
Console.WriteLine($"NetworkDeconvolutionSmokeRunner TensorRtLine={(int)line} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");
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
network.Name = "deconvolution_identity_network";
using TensorRtTensor inputTensor = network.AddInput("input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 2, 2 }));

TensorRtWeights deconvKernel = TensorRtWeights.FromSingleArray(new[] { 1.0f });
TensorRtWeights deconvBias = TensorRtWeights.FromSingleArray(new[] { 0.0f });
using TensorRtLayer deconvolutionLayer = network.AddDeconvolution(inputTensor, 1, new TensorRtDims(new[] { 1, 1 }), deconvKernel, deconvBias);
deconvolutionLayer.Name = "deconv_1x1_identity";
deconvolutionLayer.SetDeconvolutionOutputMaps(1);
deconvolutionLayer.SetDeconvolutionGroups(1);
deconvolutionLayer.SetDeconvolutionKernelSize(new TensorRtDims(new[] { 1, 1 }));
deconvolutionLayer.SetDeconvolutionStride(new TensorRtDims(new[] { 1, 1 }));
deconvolutionLayer.SetDeconvolutionDilation(new TensorRtDims(new[] { 1, 1 }));
deconvolutionLayer.SetDeconvolutionPadding(new TensorRtDims(new[] { 0, 0 }));
deconvolutionLayer.SetDeconvolutionPrePadding(new TensorRtDims(new[] { 0, 0 }));
deconvolutionLayer.SetDeconvolutionPostPadding(new TensorRtDims(new[] { 0, 0 }));
deconvolutionLayer.SetDeconvolutionPaddingMode(TensorRtPaddingMode.ExplicitRoundDown);

AssertEqual("Deconvolution output maps", 1, deconvolutionLayer.GetDeconvolutionOutputMaps());
AssertEqual("Deconvolution groups", 1, deconvolutionLayer.GetDeconvolutionGroups());
AssertDims("Deconvolution kernel", new[] { 1, 1 }, deconvolutionLayer.GetDeconvolutionKernelSize());
AssertDims("Deconvolution stride", new[] { 1, 1 }, deconvolutionLayer.GetDeconvolutionStride());
AssertDims("Deconvolution dilation", new[] { 1, 1 }, deconvolutionLayer.GetDeconvolutionDilation());
AssertDims("Deconvolution padding", new[] { 0, 0 }, deconvolutionLayer.GetDeconvolutionPadding());
AssertDims("Deconvolution pre padding", new[] { 0, 0 }, deconvolutionLayer.GetDeconvolutionPrePadding());
AssertDims("Deconvolution post padding", new[] { 0, 0 }, deconvolutionLayer.GetDeconvolutionPostPadding());
AssertEqual("Deconvolution padding mode", TensorRtPaddingMode.ExplicitRoundDown, deconvolutionLayer.GetDeconvolutionPaddingMode());
Console.WriteLine($"DeconvolutionMetadata OutputMaps={deconvolutionLayer.GetDeconvolutionOutputMaps()} Groups={deconvolutionLayer.GetDeconvolutionGroups()} Kernel={deconvolutionLayer.GetDeconvolutionKernelSize()} Stride={deconvolutionLayer.GetDeconvolutionStride()} Dilation={deconvolutionLayer.GetDeconvolutionDilation()} Padding={deconvolutionLayer.GetDeconvolutionPadding()} Pre={deconvolutionLayer.GetDeconvolutionPrePadding()} Post={deconvolutionLayer.GetDeconvolutionPostPadding()} PaddingMode={deconvolutionLayer.GetDeconvolutionPaddingMode()}");

TensorRtWeights complexKernel = TensorRtWeights.FromSingleArray(new[] { 1.0f, 0.0f, 0.0f, 1.0f });
TensorRtWeights complexBias = TensorRtWeights.FromSingleArray(new[] { 0.0f });
using TensorRtLayer complexDeconvolutionLayer = network.AddDeconvolution(inputTensor, 1, new TensorRtDims(new[] { 2, 2 }), complexKernel, complexBias);
complexDeconvolutionLayer.Name = "deconv_2x2_metadata_probe";
complexDeconvolutionLayer.SetDeconvolutionOutputMaps(1);
complexDeconvolutionLayer.SetDeconvolutionGroups(1);
complexDeconvolutionLayer.SetDeconvolutionKernelSize(new TensorRtDims(new[] { 2, 2 }));
complexDeconvolutionLayer.SetDeconvolutionStride(new TensorRtDims(new[] { 2, 2 }));
complexDeconvolutionLayer.SetDeconvolutionDilation(new TensorRtDims(new[] { 1, 1 }));
complexDeconvolutionLayer.SetDeconvolutionPadding(new TensorRtDims(new[] { 1, 1 }));
complexDeconvolutionLayer.SetDeconvolutionPrePadding(new TensorRtDims(new[] { 1, 1 }));
complexDeconvolutionLayer.SetDeconvolutionPostPadding(new TensorRtDims(new[] { 0, 0 }));
complexDeconvolutionLayer.SetDeconvolutionPaddingMode(TensorRtPaddingMode.ExplicitRoundDown);
AssertEqual("Complex deconvolution output maps", 1, complexDeconvolutionLayer.GetDeconvolutionOutputMaps());
AssertEqual("Complex deconvolution groups", 1, complexDeconvolutionLayer.GetDeconvolutionGroups());
AssertDims("Complex deconvolution kernel", new[] { 2, 2 }, complexDeconvolutionLayer.GetDeconvolutionKernelSize());
AssertDims("Complex deconvolution stride", new[] { 2, 2 }, complexDeconvolutionLayer.GetDeconvolutionStride());
AssertDims("Complex deconvolution dilation", new[] { 1, 1 }, complexDeconvolutionLayer.GetDeconvolutionDilation());
AssertDims("Complex deconvolution padding", new[] { 1, 1 }, complexDeconvolutionLayer.GetDeconvolutionPadding());
AssertDims("Complex deconvolution pre padding", new[] { 1, 1 }, complexDeconvolutionLayer.GetDeconvolutionPrePadding());
AssertDims("Complex deconvolution post padding", new[] { 0, 0 }, complexDeconvolutionLayer.GetDeconvolutionPostPadding());
AssertEqual("Complex deconvolution padding mode", TensorRtPaddingMode.ExplicitRoundDown, complexDeconvolutionLayer.GetDeconvolutionPaddingMode());
Console.WriteLine($"DeconvolutionComplexMetadata OutputMaps={complexDeconvolutionLayer.GetDeconvolutionOutputMaps()} Groups={complexDeconvolutionLayer.GetDeconvolutionGroups()} Kernel={complexDeconvolutionLayer.GetDeconvolutionKernelSize()} Stride={complexDeconvolutionLayer.GetDeconvolutionStride()} Dilation={complexDeconvolutionLayer.GetDeconvolutionDilation()} Padding={complexDeconvolutionLayer.GetDeconvolutionPadding()} Pre={complexDeconvolutionLayer.GetDeconvolutionPrePadding()} Post={complexDeconvolutionLayer.GetDeconvolutionPostPadding()} PaddingMode={complexDeconvolutionLayer.GetDeconvolutionPaddingMode()}");

using TensorRtTensor outputTensor = deconvolutionLayer.GetOutput(0);
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
AssertClose("DeconvolutionIdentity", expectedValues, outputValues, 0.0001f);
Console.WriteLine($"DeconvolutionIdentityOutputMatch=True Layers={network.LayerCount} HostMemory={hostMemory.SizeInBytes} EngineMemory={engine.DeviceMemorySizeInBytes} ContextDeviceMemory={(contextDeviceMemory == null ? 0 : contextDeviceMemory.SizeInBytes)}");

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

static void AssertDims(string name, int[] expected, TensorRtDims actual)
{
    bool match = expected.SequenceEqual(actual.Values);
    if (!match)
    {
        throw new InvalidOperationException($"{name} mismatch. Expected=[{string.Join(", ", expected)}] Actual={actual}");
    }
}

static void AssertEqual<T>(string name, T expected, T actual)
{
    if (!object.Equals(expected, actual))
    {
        throw new InvalidOperationException($"{name} mismatch. Expected={expected} Actual={actual}");
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
