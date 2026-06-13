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
Console.WriteLine($"NetworkSoftmaxTopKSmokeRunner TensorRtLine={(int)line} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");
if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
{
    Console.WriteLine($"Skipped=True Message={adapter.StatusMessage}");
    return;
}

const int batch = 2;
const int width = 4;
uint featureAxis = 1u << 1;
float[] inputValues = { -1.0f, 2.0f, -3.0f, 4.0f, 4.0f, -3.0f, 2.0f, -1.0f };
int[] gatherIndices = { 1, 3 };
float[] expectedSoftmax = ComputeRowSoftmax(inputValues, batch, width);
float[] expectedTopK = ComputeRowMax(expectedSoftmax, batch, width);
float[] expectedAbs = inputValues.Select(static value => Math.Abs(value)).ToArray();
float[] expectedGather = ComputeGatherColumns(inputValues, batch, width, gatherIndices);

using TensorRtLogger logger = new TensorRtLogger(line);
using TensorRtRuntime runtime = new TensorRtRuntime(logger);
using TensorRtBuilder builder = new TensorRtBuilder(logger);
using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
config.SetOptimizationLevel(3);
config.SetMaxAuxStreams(0);
config.SetAverageTimingIterations(1);
config.SetProfilingVerbosity(TensorRtProfilingVerbosity.Detailed);
TensorRtTacticSources defaultTacticSources = config.GetTacticSources();
if (defaultTacticSources != TensorRtTacticSources.None)
{
    config.SetTacticSources(defaultTacticSources);
}

using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
using TensorRtTensor inputTensor = network.AddInput("input", TensorRtDataType.Float, new TensorRtDims(new[] { batch, width }));

using TensorRtLayer softmaxLayer = network.AddSoftMax(inputTensor, featureAxis);
softmaxLayer.Name = "softmax_axis_1";
using TensorRtTensor softmaxTensor = softmaxLayer.GetOutput(0);
softmaxTensor.Name = "softmax_output";
network.MarkOutput(softmaxTensor);

using TensorRtLayer topKLayer = network.AddTopK(softmaxTensor, TensorRtTopKOperation.Max, 1, featureAxis);
topKLayer.Name = "topk_max_axis_1";
string topKIndicesMetadata = SetTopKIndicesTypeIfSupported(line, topKLayer, TensorRtDataType.Int64);
using TensorRtTensor topKValueTensor = topKLayer.GetOutput(0);
topKValueTensor.Name = "topk_value_output";
network.MarkOutput(topKValueTensor);

using TensorRtLayer unaryLayer = network.AddUnary(inputTensor, TensorRtUnaryOperation.Abs);
unaryLayer.Name = "unary_abs";
using TensorRtTensor absTensor = unaryLayer.GetOutput(0);
absTensor.Name = "abs_output";
network.MarkOutput(absTensor);

using TensorRtLayer indicesLayer = network.AddConstant(new TensorRtDims(new[] { gatherIndices.Length }), TensorRtWeights.FromInt32Array(gatherIndices));
indicesLayer.Name = "gather_indices";
using TensorRtTensor indicesTensor = indicesLayer.GetOutput(0);
indicesTensor.Name = "gather_indices_tensor";

using TensorRtLayer gatherLayer = network.AddGather(inputTensor, indicesTensor, 1);
gatherLayer.Name = "gather_columns";
string gatherMetadataText;
if (line == TensorRtApiLine.TensorRt11)
{
    gatherLayer.SetGatherMode(TensorRtGatherMode.Default);
    gatherLayer.SetGatherElementWiseDimensions(0);
    gatherMetadataText = $"Mode={gatherLayer.GetGatherMode()}:ElementWiseDims={gatherLayer.GetGatherElementWiseDimensions()}";
}
else
{
    gatherMetadataText = "Mode=Skipped:TRT11Only";
}
using TensorRtTensor gatherTensor = gatherLayer.GetOutput(0);
gatherTensor.Name = "gather_output";
network.MarkOutput(gatherTensor);

Console.WriteLine($"Config AvgTiming={config.GetAverageTimingIterations()} OptLevel={config.GetOptimizationLevel()} AuxStreams={config.GetMaxAuxStreams()} TacticSources={config.GetTacticSources()}");
Console.WriteLine($"LayerMetadata SoftMax={softmaxLayer.Name}:{softmaxLayer.Type}:Axes={softmaxLayer.GetSoftMaxAxes()} TopK={topKLayer.Name}:{topKLayer.Type}:Op={topKLayer.GetTopKOperation()}:K={topKLayer.GetTopKValue()}:Axes={topKLayer.GetTopKAxes()}:{topKIndicesMetadata} Unary={unaryLayer.Name}:{unaryLayer.Type}:Op={unaryLayer.GetUnaryOperation()} Gather={gatherLayer.Name}:{gatherLayer.Type}:Axis={gatherLayer.GetGatherAxis()}:{gatherMetadataText}");
Console.WriteLine($"Network Inputs={network.InputCount} Outputs={network.OutputCount} Input={inputTensor.Name}:{inputTensor.Shape} SoftMax={softmaxTensor.Name}:{softmaxTensor.Shape} TopK={topKValueTensor.Name}:{topKValueTensor.Shape} Abs={absTensor.Name}:{absTensor.Shape} Gather={gatherTensor.Name}:{gatherTensor.Shape}");

using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
using TensorRtEngine engine = runtime.Deserialize(hostMemory);
using TensorRtEngineInspector inspector = engine.CreateInspector();
using TensorRtExecutionContext context = engine.CreateExecutionContext();
using CudaStream stream = new CudaStream();
using CudaMemory inputMemory = new CudaMemory(inputValues.Length * sizeof(float));
using CudaMemory softmaxMemory = new CudaMemory(expectedSoftmax.Length * sizeof(float));
using CudaMemory topKMemory = new CudaMemory(expectedTopK.Length * sizeof(float));
using CudaMemory absMemory = new CudaMemory(expectedAbs.Length * sizeof(float));
using CudaMemory gatherMemory = new CudaMemory(expectedGather.Length * sizeof(float));

inputMemory.CopyFrom(inputValues);
softmaxMemory.Fill(0, softmaxMemory.SizeInBytes);
topKMemory.Fill(0, topKMemory.SizeInBytes);
absMemory.Fill(0, absMemory.SizeInBytes);
gatherMemory.Fill(0, gatherMemory.SizeInBytes);

context.SetTensorAddress("input", inputMemory);
context.SetTensorAddress("softmax_output", softmaxMemory);
context.SetTensorAddress("topk_value_output", topKMemory);
context.SetTensorAddress("abs_output", absMemory);
context.SetTensorAddress("gather_output", gatherMemory);
context.EnqueueAsync(stream);
stream.Synchronize();

float[] softmaxValues = softmaxMemory.ToSingleArray(expectedSoftmax.Length);
float[] topKValues = topKMemory.ToSingleArray(expectedTopK.Length);
float[] absValues = absMemory.ToSingleArray(expectedAbs.Length);
float[] gatherValues = gatherMemory.ToSingleArray(expectedGather.Length);

AssertClose("SoftMax", expectedSoftmax, softmaxValues, 0.0002f);
AssertClose("TopK", expectedTopK, topKValues, 0.0002f);
AssertClose("UnaryAbs", expectedAbs, absValues, 0.0001f);
AssertClose("Gather", expectedGather, gatherValues, 0.0001f);

IReadOnlyList<TensorRtTensorInfo> tensors = engine.GetIOTensors();
string tensorSummary = string.Join("; ", tensors.Select(tensor => $"{tensor.Index}:{tensor.Name}:{tensor.IOMode}:{tensor.DataType}:{tensor.Shape}"));
string inspectorText = inspector.GetEngineInformation(TensorRtLayerInformationFormat.Oneline);
Console.WriteLine($"HostMemory={hostMemory.SizeInBytes} EngineIOTensors={engine.IOTensorCount} IOTensors=[{tensorSummary}] InspectorBytes={inspectorText.Length}");
Console.WriteLine("SoftMaxTopKOutputMatch=True UnaryOutputMatch=True GatherOutputMatch=True");

static float[] ComputeRowSoftmax(float[] values, int batch, int width)
{
    float[] output = new float[values.Length];
    for (int row = 0; row < batch; row++)
    {
        int offset = row * width;
        float max = values.Skip(offset).Take(width).Max();
        double sum = 0;
        for (int column = 0; column < width; column++)
        {
            sum += Math.Exp(values[offset + column] - max);
        }

        for (int column = 0; column < width; column++)
        {
            output[offset + column] = (float)(Math.Exp(values[offset + column] - max) / sum);
        }
    }

    return output;
}

static float[] ComputeRowMax(float[] values, int batch, int width)
{
    float[] output = new float[batch];
    for (int row = 0; row < batch; row++)
    {
        output[row] = values.Skip(row * width).Take(width).Max();
    }

    return output;
}

static float[] ComputeGatherColumns(float[] values, int batch, int width, int[] columns)
{
    float[] output = new float[batch * columns.Length];
    for (int row = 0; row < batch; row++)
    {
        for (int index = 0; index < columns.Length; index++)
        {
            output[(row * columns.Length) + index] = values[(row * width) + columns[index]];
        }
    }

    return output;
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

static string SetTopKIndicesTypeIfSupported(TensorRtApiLine line, TensorRtLayer topKLayer, TensorRtDataType dataType)
{
    if (line != TensorRtApiLine.TensorRt11)
    {
        return "IndicesType=Skipped:TRT11Only";
    }

    bool changed = topKLayer.SetTopKIndicesType(dataType);
    return $"IndicesType={topKLayer.GetTopKIndicesType()}:Changed={changed}";
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
