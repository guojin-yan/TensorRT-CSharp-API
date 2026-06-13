using System;
using System.Collections.Generic;
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
Console.WriteLine($"RefitWeightsSmokeRunner TensorRtLine={(int)line} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");
if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
{
    Console.WriteLine($"Skipped=True Message={adapter.StatusMessage}");
    return;
}

float[] inputValues = { 1.0f, 2.0f, -3.0f, 4.0f };
float[] expectedBefore = inputValues.ToArray();
float[] expectedAfter = inputValues.Select(static value => value * 2.0f).ToArray();

using TensorRtLogger logger = new TensorRtLogger(line);
using TensorRtRuntime runtime = new TensorRtRuntime(logger);
using TensorRtBuilder builder = new TensorRtBuilder(logger);
using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
config.SetFlag(TensorRtBuilderFlag.Refit);
config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
config.SetOptimizationLevel(3);
config.SetMaxAuxStreams(0);
config.SetAverageTimingIterations(1);

using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
network.Name = "refit_scale_network";
using TensorRtTensor inputTensor = network.AddInput("input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 1, 1, inputValues.Length }));

using TensorRtLayer scaleLayer = network.AddScale(
    inputTensor,
    TensorRtScaleMode.Uniform,
    TensorRtWeights.FromSingleArray(new[] { 0.0f }),
    TensorRtWeights.FromSingleArray(new[] { 1.0f }),
    TensorRtWeights.FromSingleArray(new[] { 1.0f }),
    channelAxis: 1);
scaleLayer.Name = "scale_refit";
using TensorRtTensor outputTensor = scaleLayer.GetOutput(0);
outputTensor.Name = "output";
network.MarkOutput(outputTensor);

using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
using TensorRtEngine engine = runtime.Deserialize(hostMemory);
Console.WriteLine($"Engine Refittable={engine.IsRefittable} HostMemory={hostMemory.SizeInBytes} DeviceMemory={engine.DeviceMemorySizeInBytes}");
if (!engine.IsRefittable)
{
    Console.WriteLine("RefitWeights=Skipped Reason=EngineNotRefittable");
    return;
}

float[] before = RunInference(engine, inputValues);
AssertClose("RefitBefore", expectedBefore, before, 0.0001f);

using TensorRtRefitter refitter = engine.CreateRefitter(logger);
IReadOnlyList<TensorRtRefitEntry> allEntries = refitter.GetAllEntries();
IReadOnlyList<TensorRtRefitEntry> missingEntries = refitter.GetMissingEntries();
string allSummary = string.Join(";", allEntries.Select(static entry => $"{entry.LayerName}:{entry.Role}"));
string missingSummary = string.Join(";", missingEntries.Select(static entry => $"{entry.LayerName}:{entry.Role}"));
Console.WriteLine($"RefitEntries All={allEntries.Count}/{refitter.AllRefittableWeightCount} Missing={missingEntries.Count}/{refitter.MissingWeightCount} All=[{allSummary}] Missing=[{missingSummary}]");

TensorRtRefitEntry scaleEntry = default;
bool scaleEntryFound = false;
foreach (TensorRtRefitEntry entry in allEntries)
{
    if (string.Equals(entry.LayerName, "scale_refit", StringComparison.Ordinal) &&
        entry.Role == TensorRtWeightsRole.Scale)
    {
        scaleEntry = entry;
        scaleEntryFound = true;
        break;
    }
}

if (!scaleEntryFound)
{
    Console.WriteLine("RefitWeights=Skipped Reason=ScaleRefitEntryMissing");
    return;
}

using TensorRtRefitWeightsBuffer scaleWeights = TensorRtRefitWeightsBuffer.FromSingleArray(new[] { 2.0f });
bool weightsSet = refitter.SetWeights(scaleEntry, scaleWeights);
if (!weightsSet)
{
    throw new InvalidOperationException("TensorRT rejected the scale refit weights.");
}

bool refitted = refitter.RefitCudaEngine();
if (!refitted)
{
    throw new InvalidOperationException("TensorRT refitCudaEngine returned false after weights were set.");
}

float[] after = RunInference(engine, inputValues);
AssertClose("RefitAfter", expectedAfter, after, 0.0001f);
Console.WriteLine($"RefitWeights Set=True Refit=True Before=[{string.Join(", ", before)}] After=[{string.Join(", ", after)}] OutputChanged=True");

static float[] RunInference(TensorRtEngine engine, float[] inputValues)
{
    using TensorRtExecutionContext context = engine.CreateExecutionContext();
    using CudaStream stream = new CudaStream();
    using CudaMemory inputMemory = new CudaMemory(inputValues.Length * sizeof(float));
    using CudaMemory outputMemory = new CudaMemory(inputValues.Length * sizeof(float));

    inputMemory.CopyFrom(inputValues);
    outputMemory.Fill(0, outputMemory.SizeInBytes);
    context.SetTensorAddress("input", inputMemory);
    context.SetTensorAddress("output", outputMemory);
    context.EnqueueAsync(stream);
    stream.Synchronize();
    return outputMemory.ToSingleArray(inputValues.Length);
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
