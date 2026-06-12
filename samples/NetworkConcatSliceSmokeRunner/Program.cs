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
Console.WriteLine($"NetworkConcatSliceSmokeRunner TensorRtLine={(int)line} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");
if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
{
    Console.WriteLine($"Skipped=True Message={adapter.StatusMessage}");
    return;
}

float[] inputValues = Enumerable.Range(1, 8).Select(static value => (float)value).ToArray();
float[] constantValues = { 10.0f, 20.0f, 30.0f, 40.0f };
float[] expectedValues = { 1.0f, 2.0f, 10.0f, 20.0f, 5.0f, 6.0f, 30.0f, 40.0f };

using TensorRtLogger logger = new TensorRtLogger(line);
using TensorRtRuntime runtime = new TensorRtRuntime(logger);
using TensorRtBuilder builder = new TensorRtBuilder(logger);
using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
config.SetOptimizationLevel(3);
config.SetMaxAuxStreams(0);
config.SetAverageTimingIterations(1);
config.SetProfilingVerbosity(TensorRtProfilingVerbosity.Detailed);

using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
using TensorRtTensor inputTensor = network.AddInput("input", TensorRtDataType.Float, new TensorRtDims(new[] { 2, 4 }));

using TensorRtLayer sliceLayer = network.AddSlice(
    inputTensor,
    new TensorRtDims(new[] { 0, 0 }),
    new TensorRtDims(new[] { 2, 2 }),
    new TensorRtDims(new[] { 1, 1 }));
sliceLayer.Name = "slice_first_two_columns";
sliceLayer.SetSliceStart(new TensorRtDims(new[] { 0, 0 }));
sliceLayer.SetSliceSize(new TensorRtDims(new[] { 2, 2 }));
sliceLayer.SetSliceStride(new TensorRtDims(new[] { 1, 1 }));
string sliceAxesMetadata = SetSliceAxesIfSupported(line, sliceLayer, new TensorRtDims(new[] { 0, 1 }));
using TensorRtTensor slicedTensor = sliceLayer.GetOutput(0);
slicedTensor.Name = "sliced";

using TensorRtLayer constantLayer = network.AddConstant(new TensorRtDims(new[] { 2, 2 }), TensorRtWeights.FromSingleArray(constantValues));
constantLayer.Name = "constant_tail";
using TensorRtTensor constantTensor = constantLayer.GetOutput(0);
constantTensor.Name = "constant_tail_tensor";

using TensorRtLayer concatLayer = network.AddConcatenation(slicedTensor, constantTensor);
concatLayer.Name = "concat_axis_1";
concatLayer.SetConcatenationAxis(1);
using TensorRtTensor outputTensor = concatLayer.GetOutput(0);
outputTensor.Name = "concat_output";
network.MarkOutput(outputTensor);

Console.WriteLine($"Config AvgTiming={config.GetAverageTimingIterations()} OptLevel={config.GetOptimizationLevel()} AuxStreams={config.GetMaxAuxStreams()}");
Console.WriteLine($"ConcatSliceMetadata Slice={sliceLayer.Name}:{sliceLayer.Type}:Start={sliceLayer.GetSliceStart()}:Size={sliceLayer.GetSliceSize()}:Stride={sliceLayer.GetSliceStride()}:{sliceAxesMetadata} Concat={concatLayer.Name}:{concatLayer.Type}:Axis={concatLayer.GetConcatenationAxis()}");
Console.WriteLine($"Network Inputs={network.InputCount} Outputs={network.OutputCount} Input={inputTensor.Name}:{inputTensor.Shape} Slice={slicedTensor.Name}:{slicedTensor.Shape} Output={outputTensor.Name}:{outputTensor.Shape}");

using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
using TensorRtEngine engine = runtime.Deserialize(hostMemory);
using TensorRtEngineInspector inspector = engine.CreateInspector();
using TensorRtExecutionContext context = engine.CreateExecutionContext();
using CudaStream stream = new CudaStream();
using CudaMemory inputMemory = new CudaMemory(inputValues.Length * sizeof(float));
using CudaMemory outputMemory = new CudaMemory(expectedValues.Length * sizeof(float));

inputMemory.CopyFrom(inputValues);
outputMemory.Fill(0, outputMemory.SizeInBytes);

context.SetTensorAddress("input", inputMemory);
context.SetTensorAddress("concat_output", outputMemory);
context.EnqueueAsync(stream);
stream.Synchronize();

float[] outputValues = outputMemory.ToSingleArray(expectedValues.Length);
bool outputMatch = expectedValues.Zip(outputValues, static (expected, actual) => Math.Abs(expected - actual) < 0.0001f).All(static value => value);
if (!outputMatch)
{
    throw new InvalidOperationException($"Concat/slice network output mismatch. Expected=[{string.Join(", ", expectedValues)}] Actual=[{string.Join(", ", outputValues)}]");
}

IReadOnlyList<TensorRtTensorInfo> tensors = engine.GetIOTensors();
string tensorSummary = string.Join("; ", tensors.Select(tensor => $"{tensor.Index}:{tensor.Name}:{tensor.IOMode}:{tensor.DataType}:{tensor.Shape}"));
string inspectorText = inspector.GetEngineInformation(TensorRtLayerInformationFormat.Oneline);
Console.WriteLine($"HostMemory={hostMemory.SizeInBytes} EngineIOTensors={engine.IOTensorCount} IOTensors=[{tensorSummary}] InspectorBytes={inspectorText.Length}");
Console.WriteLine("ConcatSliceOutputMatch=True");

static string SetSliceAxesIfSupported(TensorRtApiLine line, TensorRtLayer sliceLayer, TensorRtDims axes)
{
    if (line == TensorRtApiLine.TensorRt8)
    {
        return "Axes=Skipped:TRT10OrNewer";
    }

    sliceLayer.SetSliceAxes(axes);
    return $"Axes={sliceLayer.GetSliceAxes()}";
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
