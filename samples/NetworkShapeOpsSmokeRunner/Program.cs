using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

TensorRtApiLine line = ResolveLine(GetStringArgument(args, "--tensor-rt-line", "10"));

TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
TensorRtAdapterInfo adapter = line switch
{
    TensorRtApiLine.TensorRt8 => snapshot.TensorRt8,
    TensorRtApiLine.TensorRt10 => snapshot.TensorRt10,
    TensorRtApiLine.TensorRt11 => snapshot.TensorRt11,
    _ => snapshot.TensorRt10
};
Console.WriteLine($"NetworkShapeOpsSmokeRunner TensorRtLine={(int)line} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");
if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
{
    Console.WriteLine($"Skipped=True Message={adapter.StatusMessage}");
    return;
}

float[] inputValues = Enumerable.Range(1, 8).Select(static value => (float)value).ToArray();
float[] expectedValues = { 3.0f, 7.0f, 11.0f, 15.0f };

using TensorRtLogger logger = new TensorRtLogger(line);
using TensorRtRuntime runtime = new TensorRtRuntime(logger);
using TensorRtBuilder builder = new TensorRtBuilder(logger);
using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
config.SetOptimizationLevel(3);
config.SetMaxAuxStreams(0);
config.SetProfilingVerbosity(TensorRtProfilingVerbosity.Detailed);

using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
using TensorRtTensor inputTensor = network.AddInput("input", TensorRtDataType.Float, new TensorRtDims(new[] { 2, 4 }));

using TensorRtLayer shapeLayer = network.AddShape(inputTensor);
shapeLayer.Name = "shape_of_input";
using TensorRtTensor shapeTensor = shapeLayer.GetOutput(0);
shapeTensor.Name = "input_shape";

using TensorRtLayer shuffleLayer = network.AddShuffle(inputTensor, new TensorRtDims(new[] { 2, 2, 2 }));
shuffleLayer.Name = "reshape_2x2x2";
shuffleLayer.SetShuffleFirstTranspose(new TensorRtDims(new[] { 0, 1 }));
shuffleLayer.SetShuffleSecondTranspose(new TensorRtDims(new[] { 0, 1, 2 }));
shuffleLayer.SetShuffleZeroIsPlaceholder(false);
using TensorRtTensor shuffledTensor = shuffleLayer.GetOutput(0);
shuffledTensor.Name = "reshaped";

using TensorRtLayer reduceLayer = network.AddReduce(shuffledTensor, TensorRtReduceOperation.Sum, 1u << 2, keepDimensions: true);
reduceLayer.Name = "sum_last_dim";
using TensorRtTensor outputTensor = reduceLayer.GetOutput(0);
outputTensor.Name = "reduced_output";
network.MarkOutput(outputTensor);

Console.WriteLine($"ShapeOpsMetadata Shape={shapeLayer.Name}:{shapeLayer.Type}:Out={shapeTensor.Name}:{shapeTensor.Shape} Shuffle={shuffleLayer.Name}:{shuffleLayer.Type}:Reshape={shuffleLayer.GetShuffleReshapeDimensions()}:FirstTranspose={shuffleLayer.GetShuffleFirstTranspose()}:SecondTranspose={shuffleLayer.GetShuffleSecondTranspose()}:ZeroPlaceholder={shuffleLayer.GetShuffleZeroIsPlaceholder()} Reduce={reduceLayer.Name}:{reduceLayer.Type}:Op={reduceLayer.GetReduceOperation()}:Axes={reduceLayer.GetReduceAxes()}:Keep={reduceLayer.GetReduceKeepDimensions()}");
Console.WriteLine($"Network Inputs={network.InputCount} Outputs={network.OutputCount} ImplicitBatch={network.HasImplicitBatchDimension} Input={inputTensor.Name}:{inputTensor.Shape} Shuffled={shuffledTensor.Name}:{shuffledTensor.Shape} Output={outputTensor.Name}:{outputTensor.Shape}");

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
context.SetTensorAddress("reduced_output", outputMemory);
context.EnqueueAsync(stream);
stream.Synchronize();

float[] outputValues = outputMemory.ToSingleArray(expectedValues.Length);
bool outputMatch = expectedValues.Zip(outputValues, static (expected, actual) => Math.Abs(expected - actual) < 0.0001f).All(static value => value);
if (!outputMatch)
{
    throw new InvalidOperationException($"Shape ops network output mismatch. Expected=[{string.Join(", ", expectedValues)}] Actual=[{string.Join(", ", outputValues)}]");
}

IReadOnlyList<TensorRtTensorInfo> tensors = engine.GetIOTensors();
string tensorSummary = string.Join("; ", tensors.Select(tensor => $"{tensor.Index}:{tensor.Name}:{tensor.IOMode}:{tensor.DataType}:{tensor.Shape}"));
string inspectorText = inspector.GetEngineInformation(TensorRtLayerInformationFormat.Oneline);
Console.WriteLine($"HostMemory={hostMemory.SizeInBytes} EngineIOTensors={engine.IOTensorCount} IOTensors=[{tensorSummary}] InspectorBytes={inspectorText.Length}");
Console.WriteLine("ShapeOpsOutputMatch=True");

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

static string GetStringArgument(string[] args, string name, string defaultValue)
{
    for (int index = 0; index < args.Length - 1; index++)
    {
        if (string.Equals(args[index], name, StringComparison.OrdinalIgnoreCase))
        {
            return args[index + 1];
        }
    }

    return defaultValue;
}
