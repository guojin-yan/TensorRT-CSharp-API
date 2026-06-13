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
Console.WriteLine($"NetworkQuantizeDequantizeSmokeRunner TensorRtLine={(int)line} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");
if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
{
    Console.WriteLine($"Skipped=True Message={adapter.StatusMessage}");
    return;
}

float[] inputValues = { -0.5f, 0.0f, 0.25f, 1.0f };
float[] expectedValues = inputValues.ToArray();

using TensorRtLogger logger = new TensorRtLogger(line);
using TensorRtRuntime runtime = new TensorRtRuntime(logger);
using TensorRtBuilder builder = new TensorRtBuilder(logger);
using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
config.SetFlag(TensorRtBuilderFlag.Int8, true);
config.SetOptimizationLevel(3);
config.SetMaxAuxStreams(0);
config.SetAverageTimingIterations(1);

string dequantizeBlockShapeMetadata = ProbeDequantizeBlockShapeIfSupported(line, builder, new TensorRtDims(new[] { 1, 1, 1, 1 }));

using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
network.Name = "qdq_identity_network";
using TensorRtTensor inputTensor = network.AddInput("input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 1, 4 }));

using TensorRtLayer scaleConstantLayer = network.AddConstant(new TensorRtDims(new[] { 1 }), TensorRtWeights.FromSingleArray(new[] { 0.25f }));
scaleConstantLayer.Name = "qdq_scale";
using TensorRtTensor scaleTensor = scaleConstantLayer.GetOutput(0);
scaleTensor.Name = "scale";

using TensorRtLayer quantizeLayer = network.AddQuantize(inputTensor, scaleTensor);
quantizeLayer.Name = "quantize_per_tensor";
quantizeLayer.SetQuantizeAxis(-1);
ValidateLayerType("Quantize", line, quantizeLayer.Type, TensorRtLayerType.QuantizeTrt8, TensorRtLayerType.QuantizeTrt10);
AssertEqual("Quantize axis", -1, quantizeLayer.GetQuantizeAxis());

using TensorRtTensor quantizedTensor = quantizeLayer.GetOutput(0);
quantizedTensor.Name = "quantized";

using TensorRtLayer dequantizeLayer = network.AddDequantize(quantizedTensor, scaleTensor);
dequantizeLayer.Name = "dequantize_per_tensor";
dequantizeLayer.SetDequantizeAxis(-1);
ValidateLayerType("Dequantize", line, dequantizeLayer.Type, TensorRtLayerType.DequantizeTrt8, TensorRtLayerType.DequantizeTrt10);
AssertEqual("Dequantize axis", -1, dequantizeLayer.GetDequantizeAxis());

Console.WriteLine($"QdqMetadata QuantizeType={quantizeLayer.Type} QuantizeAxis={quantizeLayer.GetQuantizeAxis()} DequantizeType={dequantizeLayer.Type} DequantizeAxis={dequantizeLayer.GetDequantizeAxis()} {dequantizeBlockShapeMetadata}");

using TensorRtTensor outputTensor = dequantizeLayer.GetOutput(0);
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
AssertClose("QdqIdentity", expectedValues, outputValues, 0.001f);
Console.WriteLine($"QdqOutputMatch=True Layers={network.LayerCount} HostMemory={hostMemory.SizeInBytes} EngineMemory={engine.DeviceMemorySizeInBytes} ContextDeviceMemory={(contextDeviceMemory == null ? 0 : contextDeviceMemory.SizeInBytes)}");

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

static void ValidateLayerType(string name, TensorRtApiLine line, TensorRtLayerType actual, TensorRtLayerType trt8Expected, TensorRtLayerType trt10Expected)
{
    TensorRtLayerType expected = line == TensorRtApiLine.TensorRt8 ? trt8Expected : trt10Expected;
    if (actual != expected)
    {
        throw new InvalidOperationException($"{name} layer type mismatch. Expected={expected} Actual={actual}");
    }
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

static void AssertEqual<T>(string name, T expected, T actual)
{
    if (!object.Equals(expected, actual))
    {
        throw new InvalidOperationException($"{name} mismatch. Expected={expected} Actual={actual}");
    }
}

static string ProbeDequantizeBlockShapeIfSupported(TensorRtApiLine line, TensorRtBuilder builder, TensorRtDims blockShape)
{
    if (line != TensorRtApiLine.TensorRt11)
    {
        return "BlockShape=Skipped:TRT11Only";
    }

    using TensorRtNetworkDefinition probeNetwork = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
    using TensorRtTensor probeInput = probeNetwork.AddInput("probe_input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 1, 4 }));
    using TensorRtLayer probeScaleLayer = probeNetwork.AddConstant(new TensorRtDims(new[] { 1 }), TensorRtWeights.FromSingleArray(new[] { 0.25f }));
    using TensorRtTensor probeScaleTensor = probeScaleLayer.GetOutput(0);
    using TensorRtLayer probeQuantizeLayer = probeNetwork.AddQuantize(probeInput, probeScaleTensor);
    using TensorRtTensor probeQuantizedTensor = probeQuantizeLayer.GetOutput(0);
    using TensorRtLayer dequantizeLayer = probeNetwork.AddDequantize(probeQuantizedTensor, probeScaleTensor);

    bool changed = dequantizeLayer.SetDequantizeBlockShape(blockShape);
    TensorRtDims probed = dequantizeLayer.GetDequantizeBlockShape();
    bool reset = dequantizeLayer.SetDequantizeBlockShape(new TensorRtDims(Array.Empty<int>()));
    TensorRtDims restored = dequantizeLayer.GetDequantizeBlockShape();
    AssertEqual("Dequantize block shape", blockShape.ToString(), probed.ToString());
    AssertEqual("Dequantize restored block shape", "[]", restored.ToString());
    return $"BlockShapeProbe={probed}:Changed={changed}:Reset={reset}:Restored={restored}";
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
