using System;
using System.Collections.Generic;
using System.IO;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.SampleSupport;

internal sealed class OnnxSampleOptions
{
    private OnnxSampleOptions(
        TensorRtApiLine line,
        string modelPath,
        string inputName,
        string outputName,
        TensorRtDims inputShape,
        TensorRtDims minShape,
        TensorRtDims optShape,
        TensorRtDims maxShape,
        bool hasProfileOverride,
        string inputPattern)
    {
        Line = line;
        ModelPath = modelPath;
        InputName = inputName;
        OutputName = outputName;
        InputShape = inputShape;
        MinShape = minShape;
        OptShape = optShape;
        MaxShape = maxShape;
        HasProfileOverride = hasProfileOverride;
        InputPattern = inputPattern;
    }

    public TensorRtApiLine Line { get; }

    public string ModelPath { get; }

    public string InputName { get; }

    public string OutputName { get; }

    public TensorRtDims InputShape { get; }

    public TensorRtDims MinShape { get; }

    public TensorRtDims OptShape { get; }

    public TensorRtDims MaxShape { get; }

    public bool HasProfileOverride { get; }

    public string InputPattern { get; }

    public static OnnxSampleOptions FromArgs(string[] args, string defaultInputShape)
    {
        string modelPath = SampleCommandLine.GetStringArgument(args, "--model", string.Empty);
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            throw new ArgumentException("Missing --model <path-to-model.onnx>.");
        }

        string fullModelPath = Path.GetFullPath(modelPath);
        if (!File.Exists(fullModelPath))
        {
            throw new FileNotFoundException("ONNX model file was not found.", fullModelPath);
        }

        TensorRtApiLine line = TensorRtOnnxSample.ResolveLine(SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "10"));
        TensorRtDims inputShape = TensorRtOnnxSample.ParseShape(SampleCommandLine.GetStringArgument(args, "--input-shape", defaultInputShape), "--input-shape");
        string minShapeText = SampleCommandLine.GetStringArgument(args, "--min-shape", string.Empty);
        string optShapeText = SampleCommandLine.GetStringArgument(args, "--opt-shape", string.Empty);
        string maxShapeText = SampleCommandLine.GetStringArgument(args, "--max-shape", string.Empty);
        bool hasProfileOverride = !string.IsNullOrWhiteSpace(minShapeText) ||
                                  !string.IsNullOrWhiteSpace(optShapeText) ||
                                  !string.IsNullOrWhiteSpace(maxShapeText);

        TensorRtDims minShape = string.IsNullOrWhiteSpace(minShapeText) ? inputShape : TensorRtOnnxSample.ParseShape(minShapeText, "--min-shape");
        TensorRtDims optShape = string.IsNullOrWhiteSpace(optShapeText) ? inputShape : TensorRtOnnxSample.ParseShape(optShapeText, "--opt-shape");
        TensorRtDims maxShape = string.IsNullOrWhiteSpace(maxShapeText) ? inputShape : TensorRtOnnxSample.ParseShape(maxShapeText, "--max-shape");

        return new OnnxSampleOptions(
            line,
            fullModelPath,
            SampleCommandLine.GetStringArgument(args, "--input-name", string.Empty),
            SampleCommandLine.GetStringArgument(args, "--output-name", string.Empty),
            inputShape,
            minShape,
            optShape,
            maxShape,
            hasProfileOverride,
            SampleCommandLine.GetStringArgument(args, "--input-pattern", "ramp"));
    }
}

internal sealed class OnnxSampleResult
{
    public OnnxSampleResult(
        TensorRtApiLine line,
        string inputName,
        string outputName,
        TensorRtDims inputShape,
        TensorRtDims outputShape,
        float[] outputValues,
        TensorRtEngineBindingReport report,
        TensorRtInferenceExecutionSummary executionSummary,
        float elapsedMilliseconds,
        int profileIndex,
        ulong engineDeviceMemoryBytes)
    {
        Line = line;
        InputName = inputName;
        OutputName = outputName;
        InputShape = inputShape;
        OutputShape = outputShape;
        OutputValues = outputValues;
        Report = report;
        ExecutionSummary = executionSummary;
        ElapsedMilliseconds = elapsedMilliseconds;
        ProfileIndex = profileIndex;
        EngineDeviceMemoryBytes = engineDeviceMemoryBytes;
    }

    public TensorRtApiLine Line { get; }

    public string InputName { get; }

    public string OutputName { get; }

    public TensorRtDims InputShape { get; }

    public TensorRtDims OutputShape { get; }

    public float[] OutputValues { get; }

    public TensorRtEngineBindingReport Report { get; }

    public TensorRtInferenceExecutionSummary ExecutionSummary { get; }

    public float ElapsedMilliseconds { get; }

    public int ProfileIndex { get; }

    public ulong EngineDeviceMemoryBytes { get; }
}

internal sealed class SampleSkippedException : Exception
{
    public SampleSkippedException(string message)
        : base(message)
    {
    }
}

internal static class TensorRtOnnxSample
{
    public static TensorRtApiLine ResolveLine(string value)
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

    public static TensorRtDims ParseShape(string value, string argumentName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new ArgumentException($"{argumentName} must not be empty.");
        }

        string[] tokens = value.Split(new[] { 'x', 'X', ',', ';' }, StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length == 0)
        {
            throw new ArgumentException($"{argumentName} must contain at least one dimension.");
        }

        int[] values = new int[tokens.Length];
        for (int index = 0; index < tokens.Length; index++)
        {
            if (!int.TryParse(tokens[index].Trim(), out int dimension) || dimension <= 0)
            {
                throw new ArgumentException($"{argumentName} must contain positive integer dimensions.");
            }

            values[index] = dimension;
        }

        return new TensorRtDims(values);
    }

    public static OnnxSampleResult RunSingleFloatInputOutput(OnnxSampleOptions options)
    {
        TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
        TensorRtAdapterInfo adapter = SelectAdapter(snapshot, options.Line);
        if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
        {
            throw new SampleSkippedException(adapter.StatusMessage);
        }

        using TensorRtLogger logger = new TensorRtLogger(options.Line);
        using TensorRtRuntime runtime = new TensorRtRuntime(logger);
        using TensorRtBuilder builder = new TensorRtBuilder(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        using CudaStream stream = new CudaStream(CudaStreamCreationFlags.NonBlocking);
        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        using TensorRtOnnxParser parser = new TensorRtOnnxParser(logger, network);

        if (!parser.ParseFromFile(options.ModelPath))
        {
            throw new InvalidOperationException(parser.GetErrorSummary());
        }

        if (network.InputCount != 1)
        {
            throw new NotSupportedException($"This sample supports one float input tensor. Model input count: {network.InputCount}.");
        }

        using TensorRtTensor inputTensor = ResolveNetworkInput(network, options.InputName);
        if (inputTensor.DataType != TensorRtDataType.Float)
        {
            throw new NotSupportedException($"This sample supports float input tensors only. Input '{inputTensor.Name}' is {inputTensor.DataType}.");
        }

        bool dynamicInput = HasDynamicDimension(inputTensor.Shape);
        int profileIndex = 0;
        if (dynamicInput || options.HasProfileOverride)
        {
            using TensorRtOptimizationProfile profile = builder.CreateOptimizationProfile();
            profile.SetShape(inputTensor.Name, options.MinShape, options.OptShape, options.MaxShape);
            profileIndex = config.AddOptimizationProfile(profile);
        }

        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 256UL * 1024UL * 1024UL);
        config.SetOptimizationLevel(3);
        config.SetMaxAuxStreams(0);
        config.SetProfileStream(stream);

        using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
        using TensorRtEngine engine = runtime.Deserialize(hostMemory);
        using TensorRtExecutionContext context = engine.CreateExecutionContext();
        using TensorRtInferenceBindings bindings = new TensorRtInferenceBindings(engine, context, profileIndex);

        string outputName = ResolveOutputName(bindings.Report, options.OutputName);
        TensorRtEngineTensorBinding outputBinding = bindings.Report.GetTensor(outputName);
        if (outputBinding.DataType != TensorRtDataType.Float)
        {
            throw new NotSupportedException($"This sample supports float output tensors only. Output '{outputName}' is {outputBinding.DataType}.");
        }

        if (dynamicInput || options.HasProfileOverride)
        {
            bindings.SetInputShape(inputTensor.Name, options.InputShape);
        }

        float[] inputValues = CreateInputValues(CountElements(options.InputShape), options.InputPattern);
        bindings.CopyInputFromHost(inputTensor.Name, inputValues, options.InputShape);
        _ = bindings.GetReadiness(runShapeInference: true);
        TensorRtInferenceBuffer outputBuffer = bindings.AllocateDeviceBuffer(outputName);

        TensorRtInferenceExecutionSummary executionSummary = null!;
        float elapsedMilliseconds = stream.MeasureElapsedTime(cudaStream =>
        {
            executionSummary = bindings.EnqueueAsync(cudaStream, synchronize: false, runShapeInference: true);
        });

        int outputElementCount = CountElements(outputBuffer.RuntimeShape!);
        float[] outputValues = bindings.ReadOutputSingles(outputName, outputElementCount);

        return new OnnxSampleResult(
            options.Line,
            inputTensor.Name,
            outputName,
            options.InputShape,
            outputBuffer.RuntimeShape!,
            outputValues,
            bindings.Report,
            executionSummary,
            elapsedMilliseconds,
            profileIndex,
            engine.DeviceMemorySizeInBytes);
    }

    public static bool IsDeploymentException(Exception exception)
    {
        return exception is TensorRtException ||
               exception is CudaException ||
               exception is DllNotFoundException ||
               exception is BadImageFormatException;
    }

    public static int CountElements(TensorRtDims shape)
    {
        if (shape == null)
        {
            throw new ArgumentNullException(nameof(shape));
        }

        int result = 1;
        foreach (int value in shape.Values)
        {
            if (value <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(shape), "Shape must be concrete and positive.");
            }

            result = checked(result * value);
        }

        return result;
    }

    public static IReadOnlyList<string> ReadLabels(string labelsPath)
    {
        if (string.IsNullOrWhiteSpace(labelsPath))
        {
            return Array.Empty<string>();
        }

        string fullPath = Path.GetFullPath(labelsPath);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException("Label file was not found.", fullPath);
        }

        return File.ReadAllLines(fullPath);
    }

    public static string LabelOrIndex(IReadOnlyList<string> labels, int index)
    {
        return index >= 0 && index < labels.Count && !string.IsNullOrWhiteSpace(labels[index])
            ? labels[index]
            : index.ToString();
    }

    private static TensorRtAdapterInfo SelectAdapter(TensorRtEnvironmentSnapshot snapshot, TensorRtApiLine line)
    {
        return line switch
        {
            TensorRtApiLine.TensorRt8 => snapshot.TensorRt8,
            TensorRtApiLine.TensorRt10 => snapshot.TensorRt10,
            TensorRtApiLine.TensorRt11 => snapshot.TensorRt11,
            _ => snapshot.TensorRt10
        };
    }

    private static TensorRtTensor ResolveNetworkInput(TensorRtNetworkDefinition network, string requestedName)
    {
        using TensorRtTensor firstInput = network.GetInput(0);
        if (string.IsNullOrWhiteSpace(requestedName) || string.Equals(firstInput.Name, requestedName, StringComparison.Ordinal))
        {
            return network.GetInput(0);
        }

        throw new ArgumentException($"Input tensor '{requestedName}' was not found. This sample model exposes '{firstInput.Name}'.");
    }

    private static string ResolveOutputName(TensorRtEngineBindingReport report, string requestedName)
    {
        IReadOnlyList<TensorRtEngineTensorBinding> outputs = report.GetOutputs();
        if (outputs.Count == 0)
        {
            throw new InvalidOperationException("The model did not expose any TensorRT output tensors.");
        }

        if (string.IsNullOrWhiteSpace(requestedName))
        {
            return outputs[0].Name;
        }

        foreach (TensorRtEngineTensorBinding output in outputs)
        {
            if (string.Equals(output.Name, requestedName, StringComparison.Ordinal))
            {
                return output.Name;
            }
        }

        throw new ArgumentException($"Output tensor '{requestedName}' was not found.");
    }

    private static bool HasDynamicDimension(TensorRtDims shape)
    {
        foreach (int value in shape.Values)
        {
            if (value <= 0)
            {
                return true;
            }
        }

        return false;
    }

    private static float[] CreateInputValues(int count, string pattern)
    {
        float[] values = new float[count];
        if (string.Equals(pattern, "ones", StringComparison.OrdinalIgnoreCase))
        {
            Array.Fill(values, 1.0f);
            return values;
        }

        if (string.Equals(pattern, "zeros", StringComparison.OrdinalIgnoreCase))
        {
            return values;
        }

        for (int index = 0; index < values.Length; index++)
        {
            values[index] = (index % 255) / 255.0f;
        }

        return values;
    }
}
