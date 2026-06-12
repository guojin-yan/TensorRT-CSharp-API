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
Console.WriteLine($"NetworkMatrixFillSelectSmokeRunner TensorRtLine={(int)line} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");
if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
{
    Console.WriteLine($"Skipped=True Message={adapter.StatusMessage}");
    return;
}

float[] inputValues = { 1.0f, 2.0f, 3.0f, 4.0f };
float[] identityValues = { 1.0f, 0.0f, 0.0f, 1.0f };
bool[] conditionValues = { true, false, true, false };
float[] expectedValues = { 1.0f, 10.0f, 3.0f, 10.0f };

using TensorRtLogger logger = new TensorRtLogger(line);
using TensorRtRuntime runtime = new TensorRtRuntime(logger);
using TensorRtBuilder builder = new TensorRtBuilder(logger);
using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 64UL * 1024UL * 1024UL);
config.SetOptimizationLevel(3);
config.SetMaxAuxStreams(0);
config.SetProfilingVerbosity(TensorRtProfilingVerbosity.Detailed);

using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
using TensorRtTensor inputTensor = network.AddInput("input", TensorRtDataType.Float, new TensorRtDims(new[] { 2, 2 }));

using TensorRtLayer identityConstantLayer = network.AddConstant(new TensorRtDims(new[] { 2, 2 }), TensorRtWeights.FromSingleArray(identityValues));
identityConstantLayer.Name = "identity_constant";
using TensorRtTensor identityTensor = identityConstantLayer.GetOutput(0);
identityTensor.Name = "identity";

using TensorRtLayer matrixLayer = network.AddMatrixMultiply(inputTensor, TensorRtMatrixOperation.None, identityTensor, TensorRtMatrixOperation.None);
matrixLayer.Name = "matrix_identity";
using TensorRtTensor matrixOutput = matrixLayer.GetOutput(0);
matrixOutput.Name = "matrix_output";

using TensorRtLayer fillLayer = network.AddFill(new TensorRtDims(new[] { 2, 2 }), TensorRtFillOperation.Linspace);
fillLayer.Name = "fill_ten";
fillLayer.SetFillAlpha(10.0);
fillLayer.SetFillBeta(0.0);
using TensorRtTensor fillOutput = fillLayer.GetOutput(0);
fillOutput.Name = "fill_output";

using TensorRtLayer conditionConstantLayer = network.AddConstant(new TensorRtDims(new[] { 2, 2 }), TensorRtWeights.FromBooleanArray(conditionValues));
conditionConstantLayer.Name = "condition_constant";
using TensorRtTensor conditionTensor = conditionConstantLayer.GetOutput(0);
conditionTensor.Name = "condition";

using TensorRtLayer selectLayer = network.AddSelect(conditionTensor, matrixOutput, fillOutput);
selectLayer.Name = "select_matrix_or_fill";
using TensorRtTensor outputTensor = selectLayer.GetOutput(0);
outputTensor.Name = "selected_output";
network.MarkOutput(outputTensor);

Console.WriteLine(
    $"LayerMetadata Matrix={matrixLayer.Type}:Op0={matrixLayer.GetMatrixMultiplyOperation(0)}:Op1={matrixLayer.GetMatrixMultiplyOperation(1)} " +
    $"Fill={fillLayer.Type}:Dims={fillLayer.GetFillDimensions()}:Op={fillLayer.GetFillOperation()}:Alpha={fillLayer.GetFillAlpha()}:Beta={fillLayer.GetFillBeta()} " +
    $"Select={selectLayer.Type}:Inputs={selectLayer.InputCount}:Outputs={selectLayer.OutputCount}");

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
context.SetTensorAddress("selected_output", outputMemory);
context.EnqueueAsync(stream);
stream.Synchronize();

float[] outputValues = outputMemory.ToSingleArray(expectedValues.Length);
bool outputMatch = expectedValues.Zip(outputValues, static (expected, actual) => Math.Abs(expected - actual) < 0.0001f).All(static value => value);
if (!outputMatch)
{
    throw new InvalidOperationException($"Matrix/fill/select network output mismatch. Expected=[{string.Join(", ", expectedValues)}] Actual=[{string.Join(", ", outputValues)}]");
}

IReadOnlyList<TensorRtTensorInfo> tensors = engine.GetIOTensors();
string tensorSummary = string.Join("; ", tensors.Select(tensor => $"{tensor.Index}:{tensor.Name}:{tensor.IOMode}:{tensor.DataType}:{tensor.Shape}"));
string engineName0 = engine.GetIOTensorName(0);
int inputIndex = engine.GetTensorIndex("input");
int outputIndex = engine.GetTensorIndex("selected_output");
TensorRtDataType outputDataType = engine.GetTensorDataType("selected_output");
TensorRtDims outputShape = engine.GetTensorShape("selected_output");
TensorRtIOMode outputMode = engine.GetTensorIOMode("selected_output");
string inspectorText = inspector.GetEngineInformation(TensorRtLayerInformationFormat.Oneline);

Console.WriteLine(
    $"EngineMetadata DeviceMemory={engine.DeviceMemorySizeInBytes} Profiles={engine.OptimizationProfileCount} " +
    $"Name0={engineName0} InputIndex={inputIndex} OutputIndex={outputIndex} OutputType={outputDataType} OutputShape={outputShape} OutputMode={outputMode}");
Console.WriteLine($"HostMemory={hostMemory.SizeInBytes} EngineIOTensors={engine.IOTensorCount} IOTensors=[{tensorSummary}] InspectorBytes={inspectorText.Length}");
Console.WriteLine("MatrixFillSelectOutputMatch=True");

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
