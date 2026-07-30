using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
        string inputPattern,
        string inputPath,
        string inputDataPath,
        IReadOnlyList<OnnxSampleInputOptions> inputs,
        OnnxSampleReferenceOptions referenceOutputs)
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
        InputPath = inputPath;
        InputDataPath = inputDataPath;
        Inputs = inputs ?? throw new ArgumentNullException(nameof(inputs));
        ReferenceOutputs = referenceOutputs ?? throw new ArgumentNullException(nameof(referenceOutputs));
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

    public string InputPath { get; }

    public string InputDataPath { get; }

    public IReadOnlyList<OnnxSampleInputOptions> Inputs { get; }

    public OnnxSampleReferenceOptions ReferenceOutputs { get; }

    public bool UsesExternalInput => Inputs.Any(static input => input.UsesExternalInput);

    public bool UsesNamedInputContract => Inputs.Count > 1 || (Inputs.Count == 1 && !string.IsNullOrWhiteSpace(Inputs[0].Name) && !string.Equals(Inputs[0].Name, InputName, StringComparison.Ordinal));

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

        TensorRtApiLine line = TensorRtSampleSupport.ResolveLine(SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "10"));
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

        string inputName = SampleCommandLine.GetStringArgument(args, "--input-name", string.Empty);
        string inputPattern = SampleCommandLine.GetStringArgument(args, "--input-pattern", "ramp");
        string inputPath = ResolveOptionalInputFile(args, "--input");
        string inputDataPath = ResolveOptionalInputFile(args, "--input-data");
        IReadOnlyList<OnnxSampleInputOptions> inputs = OnnxSampleInputOptions.FromArgs(
            args,
            inputName,
            inputShape,
            minShape,
            optShape,
            maxShape,
            hasProfileOverride,
            inputPattern,
            inputPath,
            inputDataPath);

        return new OnnxSampleOptions(
            line,
            fullModelPath,
            inputName,
            SampleCommandLine.GetStringArgument(args, "--output-name", string.Empty),
            inputShape,
            minShape,
            optShape,
            maxShape,
            hasProfileOverride,
            inputPattern,
            inputPath,
            inputDataPath,
            inputs,
            OnnxSampleReferenceOptions.FromArgs(args));
    }

    private static string ResolveOptionalInputFile(string[] args, string name)
    {
        string value = SampleCommandLine.GetStringArgument(args, name, string.Empty);
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        string fullPath = Path.GetFullPath(value);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException($"Input file for {name} was not found.", fullPath);
        }

        return fullPath;
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
        ulong engineDeviceMemoryBytes,
        IReadOnlyList<OnnxSampleInputTensor> inputs,
        OnnxSampleReferenceValidationResult referenceValidation)
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
        Inputs = inputs ?? Array.Empty<OnnxSampleInputTensor>();
        ReferenceValidation = referenceValidation ?? throw new ArgumentNullException(nameof(referenceValidation));
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

    public IReadOnlyList<OnnxSampleInputTensor> Inputs { get; }

    public OnnxSampleReferenceValidationResult ReferenceValidation { get; }
}

internal sealed class OnnxSampleOutputTensor
{
    public OnnxSampleOutputTensor(string name, TensorRtDims shape, float[] values)
    {
        Name = name ?? string.Empty;
        Shape = shape ?? throw new ArgumentNullException(nameof(shape));
        Values = values ?? throw new ArgumentNullException(nameof(values));
    }

    public string Name { get; }

    public TensorRtDims Shape { get; }

    public float[] Values { get; }

    public int ElementCount => Values.Length;

    public override string ToString()
    {
        return $"{Name}:{Shape} Values={ElementCount}";
    }
}

internal sealed class OnnxSampleMultiOutputResult
{
    public OnnxSampleMultiOutputResult(
        TensorRtApiLine line,
        string inputName,
        string primaryOutputName,
        TensorRtDims inputShape,
        IReadOnlyList<OnnxSampleOutputTensor> outputs,
        TensorRtEngineBindingReport report,
        TensorRtInferenceExecutionSummary executionSummary,
        float elapsedMilliseconds,
        int profileIndex,
        ulong engineDeviceMemoryBytes,
        IReadOnlyList<OnnxSampleInputTensor> inputs,
        OnnxSampleReferenceValidationResult referenceValidation)
    {
        Line = line;
        InputName = inputName ?? string.Empty;
        PrimaryOutputName = primaryOutputName ?? string.Empty;
        InputShape = inputShape ?? throw new ArgumentNullException(nameof(inputShape));
        Outputs = outputs ?? Array.Empty<OnnxSampleOutputTensor>();
        Report = report;
        ExecutionSummary = executionSummary;
        ElapsedMilliseconds = elapsedMilliseconds;
        ProfileIndex = profileIndex;
        EngineDeviceMemoryBytes = engineDeviceMemoryBytes;
        Inputs = inputs ?? Array.Empty<OnnxSampleInputTensor>();
        ReferenceValidation = referenceValidation ?? throw new ArgumentNullException(nameof(referenceValidation));
    }

    public TensorRtApiLine Line { get; }

    public string InputName { get; }

    public string PrimaryOutputName { get; }

    public TensorRtDims InputShape { get; }

    public IReadOnlyList<OnnxSampleOutputTensor> Outputs { get; }

    public TensorRtEngineBindingReport Report { get; }

    public TensorRtInferenceExecutionSummary ExecutionSummary { get; }

    public float ElapsedMilliseconds { get; }

    public int ProfileIndex { get; }

    public ulong EngineDeviceMemoryBytes { get; }

    public IReadOnlyList<OnnxSampleInputTensor> Inputs { get; }

    public OnnxSampleReferenceValidationResult ReferenceValidation { get; }

    public OnnxSampleOutputTensor PrimaryOutput => GetOutput(PrimaryOutputName);

    public OnnxSampleOutputTensor GetOutput(string name)
    {
        foreach (OnnxSampleOutputTensor output in Outputs)
        {
            if (string.Equals(output.Name, name, StringComparison.Ordinal))
            {
                return output;
            }
        }

        throw new ArgumentException($"Output tensor '{name}' was not captured.", nameof(name));
    }
}

internal sealed class SampleSkippedException : Exception
{
    public SampleSkippedException(string message)
        : base(message)
    {
    }
}

internal static partial class TensorRtOnnxSample
{
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
        OnnxSampleMultiOutputResult result = RunSingleFloatInputOutputsCore(options, captureAllOutputs: false);
        OnnxSampleOutputTensor output = result.PrimaryOutput;
        return new OnnxSampleResult(
            result.Line,
            result.InputName,
            output.Name,
            result.InputShape,
            output.Shape,
            output.Values,
            result.Report,
            result.ExecutionSummary,
            result.ElapsedMilliseconds,
            result.ProfileIndex,
            result.EngineDeviceMemoryBytes,
            result.Inputs,
            result.ReferenceValidation);
    }

    public static OnnxSampleMultiOutputResult RunSingleFloatInputOutputs(OnnxSampleOptions options)
    {
        return RunSingleFloatInputOutputsCore(options, captureAllOutputs: true);
    }

    private static OnnxSampleMultiOutputResult RunSingleFloatInputOutputsCore(OnnxSampleOptions options, bool captureAllOutputs)
    {
        TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
        TensorRtAdapterInfo adapter = TensorRtSampleSupport.SelectAdapter(snapshot, options.Line);
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

        List<OnnxSampleNetworkInput> networkInputs = ResolveNetworkInputs(network, options);
        bool dynamicInput = networkInputs.Any(static input => HasDynamicDimension(input.NetworkShape));
        int profileIndex = 0;
        if (dynamicInput || networkInputs.Any(static input => input.Options.HasProfileOverride))
        {
            using TensorRtOptimizationProfile profile = builder.CreateOptimizationProfile();
            foreach (OnnxSampleNetworkInput input in networkInputs)
            {
                if (HasDynamicDimension(input.NetworkShape) || input.Options.HasProfileOverride)
                {
                    profile.SetShape(input.Name, input.Options.MinShape, input.Options.OptShape, input.Options.MaxShape);
                }
            }
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

        string primaryOutputName = ResolveOutputName(bindings.Report, options.OutputName);
        List<string> outputNames = ResolveOutputNames(
            bindings.Report,
            primaryOutputName,
            captureAllOutputs || options.ReferenceOutputs.Requested);
        foreach (string outputName in outputNames)
        {
            TensorRtEngineTensorBinding outputBinding = bindings.Report.GetTensor(outputName);
            if (outputBinding.DataType != TensorRtDataType.Float)
            {
                throw new NotSupportedException($"This sample supports float output tensors only. Output '{outputName}' is {outputBinding.DataType}.");
            }
        }

        List<OnnxSampleInputTensor> inputArtifacts = new List<OnnxSampleInputTensor>(networkInputs.Count);
        foreach (OnnxSampleNetworkInput input in networkInputs)
        {
            if (HasDynamicDimension(input.NetworkShape) || input.Options.HasProfileOverride || !ShapesEqual(input.NetworkShape, input.Options.Shape))
            {
                bindings.SetInputShape(input.Name, input.Options.Shape);
            }

            float[] inputValues = CreateInputValues(CountElements(input.Options.Shape), input.Options);
            bindings.CopyInputFromHost(input.Name, inputValues, input.Options.Shape);
            inputArtifacts.Add(new OnnxSampleInputTensor(
                input.Name,
                input.Options.Shape,
                inputValues,
                input.Options.SourceClassification,
                input.Options.SourcePath));
        }
        _ = bindings.GetReadiness(runShapeInference: true);

        Dictionary<string, TensorRtInferenceBuffer> outputBuffers = new Dictionary<string, TensorRtInferenceBuffer>(StringComparer.Ordinal);
        foreach (string outputName in outputNames)
        {
            outputBuffers[outputName] = bindings.AllocateDeviceBuffer(outputName);
        }

        TensorRtInferenceExecutionSummary executionSummary = null!;
        float elapsedMilliseconds = stream.MeasureElapsedTime(cudaStream =>
        {
            executionSummary = bindings.EnqueueAsync(cudaStream, synchronize: false, runShapeInference: true);
        });

        List<OnnxSampleOutputTensor> outputs = new List<OnnxSampleOutputTensor>();
        foreach (string outputName in outputNames)
        {
            TensorRtInferenceBuffer outputBuffer = outputBuffers[outputName];
            int outputElementCount = CountElements(outputBuffer.RuntimeShape!);
            float[] outputValues = bindings.ReadOutputSingles(outputName, outputElementCount);
            outputs.Add(new OnnxSampleOutputTensor(outputName, outputBuffer.RuntimeShape!, outputValues));
        }

        OnnxSampleReferenceValidationResult referenceValidation = ValidateReferenceOutputs(outputs, options.ReferenceOutputs);
        OnnxSampleInputTensor primaryInput = inputArtifacts[0];

        return new OnnxSampleMultiOutputResult(
            options.Line,
            primaryInput.Name,
            primaryOutputName,
            primaryInput.Shape,
            outputs,
            bindings.Report,
            executionSummary,
            elapsedMilliseconds,
            profileIndex,
            engine.DeviceMemorySizeInBytes,
            inputArtifacts,
            referenceValidation);
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

    private static List<OnnxSampleNetworkInput> ResolveNetworkInputs(TensorRtNetworkDefinition network, OnnxSampleOptions options)
    {
        if (network.InputCount == 0)
        {
            throw new NotSupportedException("This sample requires at least one model input tensor.");
        }

        List<string> names = new List<string>(network.InputCount);
        List<TensorRtDataType> dataTypes = new List<TensorRtDataType>(network.InputCount);
        List<TensorRtDims> networkShapes = new List<TensorRtDims>(network.InputCount);
        for (int index = 0; index < network.InputCount; index++)
        {
            using TensorRtTensor tensor = network.GetInput(index);
            names.Add(tensor.Name);
            dataTypes.Add(tensor.DataType);
            networkShapes.Add(tensor.Shape);
        }

        Dictionary<string, OnnxSampleInputOptions> configured = new Dictionary<string, OnnxSampleInputOptions>(StringComparer.Ordinal);
        if (options.UsesNamedInputContract)
        {
            foreach (OnnxSampleInputOptions input in options.Inputs)
            {
                if (!configured.TryAdd(input.Name, input))
                {
                    throw new ArgumentException($"Input contract contains a duplicate tensor '{input.Name}'.");
                }
            }

            HashSet<string> engineNames = new HashSet<string>(names, StringComparer.Ordinal);
            string[] missing = engineNames.Except(configured.Keys).OrderBy(static name => name, StringComparer.Ordinal).ToArray();
            string[] extra = configured.Keys.Except(engineNames).OrderBy(static name => name, StringComparer.Ordinal).ToArray();
            if (missing.Length != 0 || extra.Length != 0)
            {
                throw new ArgumentException($"Named input contract must match model inputs exactly. Missing=[{string.Join(",", missing)}] Extra=[{string.Join(",", extra)}].");
            }
        }
        else
        {
            if (network.InputCount != 1)
            {
                throw new NotSupportedException($"Models with {network.InputCount} inputs require named --input-shapes and per-input source mappings.");
            }

            OnnxSampleInputOptions input = options.Inputs[0];
            if (!string.IsNullOrWhiteSpace(input.Name) && !string.Equals(input.Name, names[0], StringComparison.Ordinal))
            {
                throw new ArgumentException($"Input tensor '{input.Name}' was not found. This sample model exposes '{names[0]}'.");
            }

            configured[names[0]] = input;
        }

        List<OnnxSampleNetworkInput> result = new List<OnnxSampleNetworkInput>(network.InputCount);
        for (int index = 0; index < names.Count; index++)
        {
            if (dataTypes[index] != TensorRtDataType.Float)
            {
                throw new NotSupportedException($"This sample supports float input tensors only. Input '{names[index]}' is {dataTypes[index]}.");
            }

            OnnxSampleInputOptions input = configured[names[index]];
            ValidateInputShapes(names[index], networkShapes[index], input);
            result.Add(new OnnxSampleNetworkInput(names[index], dataTypes[index], networkShapes[index], input));
        }

        return result;
    }

    private static void ValidateInputShapes(string name, TensorRtDims networkShape, OnnxSampleInputOptions input)
    {
        int[] network = networkShape.Values;
        int[] runtime = input.Shape.Values;
        int[] minimum = input.MinShape.Values;
        int[] optimum = input.OptShape.Values;
        int[] maximum = input.MaxShape.Values;
        if (runtime.Length != network.Length || minimum.Length != network.Length || optimum.Length != network.Length || maximum.Length != network.Length)
        {
            throw new ArgumentException($"Input '{name}' runtime/profile shapes must all have rank {network.Length}.");
        }

        for (int index = 0; index < network.Length; index++)
        {
            if (minimum[index] > optimum[index] || optimum[index] > maximum[index])
            {
                throw new ArgumentException($"Input '{name}' profile dimension {index} must satisfy min <= opt <= max.");
            }
            if (runtime[index] < minimum[index] || runtime[index] > maximum[index])
            {
                throw new ArgumentException($"Input '{name}' runtime dimension {index} must be inside the profile range.");
            }
            if (network[index] > 0 &&
                (runtime[index] != network[index] || minimum[index] != network[index] || optimum[index] != network[index] || maximum[index] != network[index]))
            {
                throw new ArgumentException($"Input '{name}' dimension {index} is static ({network[index]}) and cannot be overridden.");
            }
        }
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

    private static List<string> ResolveOutputNames(TensorRtEngineBindingReport report, string primaryOutputName, bool captureAllOutputs)
    {
        List<string> names = new List<string>();
        if (!captureAllOutputs)
        {
            names.Add(primaryOutputName);
            return names;
        }

        foreach (TensorRtEngineTensorBinding output in report.GetOutputs())
        {
            names.Add(output.Name);
        }

        return names;
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

    private static bool ShapesEqual(TensorRtDims left, TensorRtDims right)
    {
        return left.Values.SequenceEqual(right.Values);
    }

    public static float[] CreateInputValuesForTesting(int count, string inputPattern, string inputPath = "", string inputDataPath = "")
    {
        return CreateInputValues(
            count,
            new ExternalInputRequest(inputPattern ?? string.Empty, inputPath ?? string.Empty, inputDataPath ?? string.Empty));
    }

    private static float[] CreateInputValues(int count, OnnxSampleOptions options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        return CreateInputValues(count, new ExternalInputRequest(options.InputPattern, options.InputPath, options.InputDataPath));
    }

    private static float[] CreateInputValues(int count, OnnxSampleInputOptions options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        return CreateInputValues(count, new ExternalInputRequest(options.InputPattern, options.InputPath, options.InputDataPath));
    }

    private static float[] CreateInputValues(int count, ExternalInputRequest request)
    {
        if (count <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(count), "Input element count must be positive.");
        }

        if (!string.IsNullOrWhiteSpace(request.InputDataPath))
        {
            return ReadFloatInputData(count, request.InputDataPath);
        }

        if (!string.IsNullOrWhiteSpace(request.InputPath))
        {
            return ReadByteInputData(count, request.InputPath);
        }

        float[] values = new float[count];
        if (string.Equals(request.Pattern, "ones", StringComparison.OrdinalIgnoreCase))
        {
            Array.Fill(values, 1.0f);
            return values;
        }

        if (string.Equals(request.Pattern, "zeros", StringComparison.OrdinalIgnoreCase))
        {
            return values;
        }

        if (!string.Equals(request.Pattern, "ramp", StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException("Input pattern must be zeros, ones, or ramp when no --input/--input-data file is supplied.", nameof(request));
        }

        for (int index = 0; index < values.Length; index++)
        {
            values[index] = (index % 255) / 255.0f;
        }

        return values;
    }

    private static float[] ReadFloatInputData(int expectedCount, string path)
    {
        string extension = Path.GetExtension(path);
        if (string.Equals(extension, ".bin", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(extension, ".raw", StringComparison.OrdinalIgnoreCase))
        {
            byte[] bytes = File.ReadAllBytes(path);
            if (bytes.Length % sizeof(float) != 0)
            {
                throw new ArgumentException($"Float input file byte length must be divisible by {sizeof(float)}.");
            }

            int actualCount = bytes.Length / sizeof(float);
            if (actualCount != expectedCount)
            {
                throw new ArgumentException($"Float input file has {actualCount} elements, expected {expectedCount}.");
            }

            float[] values = new float[actualCount];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }

        string text = File.ReadAllText(path);
        float[] parsed = text
            .Split(new[] { ',', ';', ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(static value => float.Parse(value, System.Globalization.CultureInfo.InvariantCulture))
            .ToArray();
        if (parsed.Length != expectedCount)
        {
            throw new ArgumentException($"Text input file has {parsed.Length} float values, expected {expectedCount}.");
        }

        return parsed;
    }

    private static float[] ReadByteInputData(int expectedCount, string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length != expectedCount)
        {
            throw new ArgumentException("--input currently accepts raw byte tensors only. Use --input-data for text/.bin float tensors, or provide a raw byte file whose length matches input element count.");
        }

        float[] values = new float[bytes.Length];
        for (int index = 0; index < bytes.Length; index++)
        {
            values[index] = bytes[index] / 255.0f;
        }

        return values;
    }

    private readonly struct ExternalInputRequest
    {
        public ExternalInputRequest(string pattern, string inputPath, string inputDataPath)
        {
            Pattern = string.IsNullOrWhiteSpace(pattern) ? "ramp" : pattern;
            InputPath = inputPath ?? string.Empty;
            InputDataPath = inputDataPath ?? string.Empty;
        }

        public string Pattern { get; }

        public string InputPath { get; }

        public string InputDataPath { get; }
    }
}
