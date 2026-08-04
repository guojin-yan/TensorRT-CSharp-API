using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

internal static class Program
{
    public static int Main(string[] args)
    {
        try
        {
            if (args.Length != 9)
            {
                Console.Error.WriteLine(
                    "Usage: RefittedPlan.PackageConsumer <plan> <float-input> <raw-output> <expected-output-sha256> <reference-json> <abs-tolerance> <rel-tolerance> <nan-policy> <infinity-policy>");
                return 2;
            }

            string planPath = Path.GetFullPath(args[0]);
            string inputPath = Path.GetFullPath(args[1]);
            string outputPath = Path.GetFullPath(args[2]);
            string expectedOutputSha256 = args[3].Trim().ToLowerInvariant();
            string referencePath = Path.GetFullPath(args[4]);
            float absoluteTolerance = ParseTolerance(args[5], "absolute");
            float relativeTolerance = ParseTolerance(args[6], "relative");
            string nanPolicy = ParseNaNPolicy(args[7]);
            string infinityPolicy = ParseInfinityPolicy(args[8]);
            if (!File.Exists(planPath))
            {
                throw new FileNotFoundException("Persisted refitted plan was not found.", planPath);
            }

            if (!File.Exists(inputPath))
            {
                throw new FileNotFoundException("Float input artifact was not found.", inputPath);
            }

            ReferenceTensor reference = ReadReference(referencePath);

            byte[] inputBytes = File.ReadAllBytes(inputPath);
            if (inputBytes.Length == 0 || inputBytes.Length % sizeof(float) != 0)
            {
                throw new InvalidDataException("Float input byte length must be non-zero and divisible by four.");
            }

            float[] inputValues = new float[inputBytes.Length / sizeof(float)];
            Buffer.BlockCopy(inputBytes, 0, inputValues, 0, inputBytes.Length);

            Console.WriteLine("PackageReferenceOnly=True");
            Console.WriteLine("ProjectReference=False");
            Console.WriteLine("ManualManagedAssemblyLoad=False");
            Console.WriteLine("CoreAssemblyLocation=" + typeof(TensorRtRuntime).Assembly.Location);
            Console.WriteLine("BridgeFileName=" + NativeBridgePathResolver.GetBridgeFileName());
            Console.WriteLine("PlanSha256=" + ComputeFileSha256(planPath));
            Console.WriteLine("PlanLengthBytes=" + new FileInfo(planPath).Length);
            Console.WriteLine("InputSha256=" + ComputeSha256(inputBytes));
            Console.WriteLine("InputLengthBytes=" + inputBytes.LongLength);
            Console.WriteLine("ReferenceSha256=" + ComputeFileSha256(referencePath));
            Console.WriteLine("ReferenceSourceClassification=" + reference.SourceClassification);
            Console.WriteLine("ReferenceTensor=" + reference.TensorName);
            Console.WriteLine("ReferenceShape=" + FormatShape(reference.Shape));
            Console.WriteLine("ReferenceElementCount=" + reference.Values.Length);
            Console.WriteLine("ReferenceAbsTolerance=" + absoluteTolerance.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
            Console.WriteLine("ReferenceRelTolerance=" + relativeTolerance.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
            Console.WriteLine("ReferenceNaNPolicy=" + nanPolicy);
            Console.WriteLine("ReferenceInfinityPolicy=" + infinityPolicy);

            TensorRtEnvironmentSnapshot environment = TensorRtEnvironmentProbe.GetCurrent();
            Console.WriteLine(
                "RuntimeEnvironment TRT=" + environment.BuildInfo.TensorRtVersion +
                " CUDA=" + environment.BuildInfo.CudaToolkitVersion +
                " TensorRtAvailable=" + environment.RuntimeInfo.TensorRtAvailable +
                " CudaAvailable=" + environment.RuntimeInfo.CudaToolkitAvailable);

            bool outputValidated = RunPersistedPlan(
                planPath,
                outputPath,
                inputValues,
                expectedOutputSha256,
                reference,
                absoluteTolerance,
                relativeTolerance,
                nanPolicy,
                infinityPolicy);

            Console.WriteLine("OwnerScopeExited=True");
            if (!outputValidated)
            {
                Console.Error.WriteLine("PackageConsumerRuntime=Failed");
                return 1;
            }

            Console.WriteLine("PackageConsumerRuntime=Passed");
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine("PackageConsumerRuntime=Failed");
            Console.Error.WriteLine(exception.ToString());
            return 1;
        }
    }

    private static bool RunPersistedPlan(
        string planPath,
        string outputPath,
        float[] inputValues,
        string expectedOutputSha256,
        ReferenceTensor reference,
        float absoluteTolerance,
        float relativeTolerance,
        string nanPolicy,
        string infinityPolicy)
    {
        using TensorRtLogger logger = new TensorRtLogger(TensorRtApiLine.TensorRt10);
        using TensorRtRuntime runtime = new TensorRtRuntime(logger);
        using TensorRtEngine engine = runtime.DeserializeFromFile(planPath);
        using TensorRtExecutionContext context = engine.CreateExecutionContext();
        using TensorRtInferenceBindings bindings = new TensorRtInferenceBindings(engine, context, profileIndex: 0);
        using CudaStream stream = new CudaStream(CudaStreamCreationFlags.NonBlocking);

        TensorRtEngineTensorBinding input = Single(bindings.Report.GetInputs(), "input");
        TensorRtEngineTensorBinding output = Single(bindings.Report.GetOutputs(), "output");
        EnsureFloatDeviceTensor(input, "input");
        EnsureFloatDeviceTensor(output, "output");

        TensorRtDims inputShape = ResolveConcreteShape(context, input);
        TensorRtDims outputShape = ResolveConcreteShape(context, output);
        int inputElementCount = CountElements(inputShape);
        int outputElementCount = CountElements(outputShape);
        if (inputElementCount != inputValues.Length)
        {
            throw new InvalidDataException(
                $"Input element count mismatch. Tensor={inputElementCount} File={inputValues.Length}.");
        }

        if (outputElementCount <= 0)
        {
            throw new InvalidDataException("Output element count must be positive.");
        }

        bindings.CopyInputFromHost(input.Name, inputValues, inputShape);
        bindings.AllocateDeviceBuffer(output.Name, outputShape, output.EstimateByteSize(outputShape));
        bindings.BindAll();
        TensorRtExecutionContextReadiness readiness = bindings.GetReadiness(runShapeInference: true);
        if (!readiness.IsReadyForEnqueue)
        {
            throw new InvalidOperationException("Bindings are not ready for enqueue: " + readiness);
        }

        TensorRtInferenceExecutionSummary execution = bindings.EnqueueAsync(
            stream,
            synchronize: true,
            runShapeInference: false);
        float[] outputValues = bindings.ReadOutputSingles(output.Name, outputElementCount);
        byte[] outputBytes = new byte[checked(outputValues.Length * sizeof(float))];
        Buffer.BlockCopy(outputValues, 0, outputBytes, 0, outputBytes.Length);

        string? outputDirectory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
        }

        File.WriteAllBytes(outputPath, outputBytes);
        string outputSha256 = ComputeSha256(outputBytes);
        int predictedIndex = ArgMax(outputValues);
        bool exactMatch = string.Equals(outputSha256, expectedOutputSha256, StringComparison.OrdinalIgnoreCase);
        ReferenceValidationResult referenceValidation = ValidateReference(
            output.Name,
            outputShape,
            outputValues,
            reference,
            absoluteTolerance,
            relativeTolerance,
            nanPolicy,
            infinityPolicy);

        Console.WriteLine("EngineRefittable=" + engine.IsRefittable);
        Console.WriteLine("EngineIOTensorCount=" + engine.IOTensorCount);
        Console.WriteLine("EngineLayerCount=" + engine.LayerCount);
        Console.WriteLine("EngineOptimizationProfileCount=" + engine.OptimizationProfileCount);
        Console.WriteLine("InputTensor=" + input.Name);
        Console.WriteLine("InputShape=" + FormatShape(inputShape));
        Console.WriteLine("InputElementCount=" + inputElementCount);
        Console.WriteLine("OutputTensor=" + output.Name);
        Console.WriteLine("OutputShape=" + FormatShape(outputShape));
        Console.WriteLine("OutputElementCount=" + outputElementCount);
        Console.WriteLine("BindingsReadyForEnqueue=" + readiness.IsReadyForEnqueue);
        Console.WriteLine("EnqueueCompleted=" + execution.Synchronized);
        Console.WriteLine("ExecutionSummary=" + execution);
        Console.WriteLine("OutputLengthBytes=" + outputBytes.LongLength);
        Console.WriteLine("OutputSha256=" + outputSha256);
        Console.WriteLine("PredictedIndex=" + predictedIndex);
        Console.WriteLine("OutputExactMatch=" + exactMatch);
        Console.WriteLine("ReferenceValidationCompleted=" + referenceValidation.Completed);
        Console.WriteLine("ReferenceValidationPassed=" + referenceValidation.Passed);
        Console.WriteLine("OutputValidated=" + (referenceValidation.Completed && referenceValidation.Passed));
        Console.WriteLine("ReferenceComparedElementCount=" + referenceValidation.ComparedElementCount);
        Console.WriteLine("ReferenceMismatchCount=" + referenceValidation.MismatchCount);
        Console.WriteLine("ReferenceFirstMismatchIndex=" + referenceValidation.FirstMismatchIndex);
        Console.WriteLine("ReferenceMaximumAbsoluteError=" + referenceValidation.MaximumAbsoluteError.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
        Console.WriteLine("ReferenceMaximumRelativeError=" + referenceValidation.MaximumRelativeError.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
        Console.WriteLine("ReferenceDiagnostic=" + referenceValidation.Diagnostic);

        if (!exactMatch)
        {
            throw new InvalidDataException(
                $"Output SHA256 mismatch. Expected={expectedOutputSha256} Actual={outputSha256}.");
        }

        if (!referenceValidation.Passed)
        {
            return false;
        }

        return true;
    }

    private static ReferenceTensor ReadReference(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("Structured reference JSON was not found.", path);
        }

        ReferenceTensor? reference = JsonSerializer.Deserialize<ReferenceTensor>(
            File.ReadAllBytes(path),
            new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
            });
        if (reference == null || reference.SchemaVersion != 1 || string.IsNullOrWhiteSpace(reference.TensorName) ||
            reference.Shape == null || reference.Shape.Length == 0 || reference.Shape.Any(static value => value <= 0) ||
            reference.Values == null || string.IsNullOrWhiteSpace(reference.SourceClassification))
        {
            throw new InvalidDataException("Structured reference JSON must contain schemaVersion=1, tensorName, positive shape, values, and sourceClassification.");
        }

        return reference;
    }

    private static ReferenceValidationResult ValidateReference(
        string outputTensorName,
        TensorRtDims outputShape,
        float[] outputValues,
        ReferenceTensor reference,
        float absoluteTolerance,
        float relativeTolerance,
        string nanPolicy,
        string infinityPolicy)
    {
        if (!string.Equals(outputTensorName, reference.TensorName, StringComparison.Ordinal))
        {
            return ReferenceValidationResult.MetadataMismatch(
                $"reference tensorName does not match engine output name; actual={outputTensorName}; expected={reference.TensorName}");
        }

        if (!outputShape.Values.SequenceEqual(reference.Shape))
        {
            return ReferenceValidationResult.MetadataMismatch(
                $"reference shape does not match runtime output shape; actual={FormatShape(outputShape)}; expected={FormatShape(reference.Shape)}");
        }

        if (outputValues.Length != reference.Values.Length)
        {
            return ReferenceValidationResult.MetadataMismatch(
                $"reference value count does not match runtime output element count; actual={outputValues.Length}; expected={reference.Values.Length}");
        }

        int mismatchCount = 0;
        int firstMismatchIndex = -1;
        float maximumAbsoluteError = 0.0f;
        float maximumRelativeError = 0.0f;
        for (int index = 0; index < outputValues.Length; index++)
        {
            bool matches = ReferenceValuesMatch(
                outputValues[index],
                reference.Values[index],
                absoluteTolerance,
                relativeTolerance,
                nanPolicy,
                infinityPolicy,
                out float absoluteError,
                out float relativeError);
            maximumAbsoluteError = Math.Max(maximumAbsoluteError, absoluteError);
            maximumRelativeError = Math.Max(maximumRelativeError, relativeError);
            if (!matches)
            {
                mismatchCount++;
                if (firstMismatchIndex < 0)
                {
                    firstMismatchIndex = index;
                }
            }
        }

        return new ReferenceValidationResult(
            completed: true,
            passed: mismatchCount == 0,
            comparedElementCount: outputValues.Length,
            mismatchCount,
            firstMismatchIndex,
            maximumAbsoluteError,
            maximumRelativeError,
            diagnostic: mismatchCount == 0
                ? "all reference values matched"
                : $"{mismatchCount} value(s) exceeded tolerance or special-value policy; first mismatch index {firstMismatchIndex}");
    }

    private static bool ReferenceValuesMatch(
        float actual,
        float expected,
        float absoluteTolerance,
        float relativeTolerance,
        string nanPolicy,
        string infinityPolicy,
        out float absoluteError,
        out float relativeError)
    {
        if (float.IsNaN(actual) || float.IsNaN(expected))
        {
            bool bothNaN = float.IsNaN(actual) && float.IsNaN(expected);
            absoluteError = bothNaN ? 0.0f : float.MaxValue;
            relativeError = absoluteError;
            return bothNaN && string.Equals(nanPolicy, "equal", StringComparison.Ordinal);
        }

        if (float.IsInfinity(actual) || float.IsInfinity(expected))
        {
            bool exact = actual.Equals(expected);
            absoluteError = exact ? 0.0f : float.MaxValue;
            relativeError = absoluteError;
            return exact && string.Equals(infinityPolicy, "exact", StringComparison.Ordinal);
        }

        absoluteError = Math.Abs(actual - expected);
        float scale = Math.Max(Math.Abs(actual), Math.Abs(expected));
        relativeError = scale == 0.0f ? absoluteError : absoluteError / scale;
        return absoluteError <= absoluteTolerance || absoluteError <= relativeTolerance * scale;
    }

    private static float ParseTolerance(string value, string name)
    {
        if (!float.TryParse(value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float parsed) ||
            !float.IsFinite(parsed) || parsed < 0.0f)
        {
            throw new ArgumentException($"Reference {name} tolerance must be finite and non-negative.", name);
        }

        return parsed;
    }

    private static string ParseNaNPolicy(string value)
    {
        if (string.Equals(value, "reject", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, "equal", StringComparison.OrdinalIgnoreCase))
        {
            return value.ToLowerInvariant();
        }

        throw new ArgumentException("Reference NaN policy must be reject or equal.", nameof(value));
    }

    private static string ParseInfinityPolicy(string value)
    {
        if (string.Equals(value, "exact", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value, "reject", StringComparison.OrdinalIgnoreCase))
        {
            return value.ToLowerInvariant();
        }

        throw new ArgumentException("Reference infinity policy must be exact or reject.", nameof(value));
    }

    private static TensorRtEngineTensorBinding Single(
        System.Collections.Generic.IReadOnlyList<TensorRtEngineTensorBinding> tensors,
        string role)
    {
        if (tensors.Count != 1)
        {
            throw new NotSupportedException($"The package consumer requires exactly one {role} tensor. Count={tensors.Count}.");
        }

        return tensors[0];
    }

    private static void EnsureFloatDeviceTensor(TensorRtEngineTensorBinding tensor, string role)
    {
        if (tensor.DataType != TensorRtDataType.Float || tensor.Location != TensorRtTensorLocation.Device)
        {
            throw new NotSupportedException(
                $"The {role} tensor must be a device float tensor. Name={tensor.Name} Type={tensor.DataType} Location={tensor.Location}.");
        }
    }

    private static TensorRtDims ResolveConcreteShape(
        TensorRtExecutionContext context,
        TensorRtEngineTensorBinding tensor)
    {
        TensorRtDims shape = tensor.EngineShape;
        if (shape.Values.Length == 0 || shape.Values.Any(static value => value <= 0))
        {
            shape = context.GetTensorShape(tensor.Name);
        }

        if (shape.Values.Length == 0 || shape.Values.Any(static value => value <= 0))
        {
            throw new InvalidDataException($"Tensor '{tensor.Name}' does not have a concrete runtime shape: {shape}.");
        }

        return shape;
    }

    private static int CountElements(TensorRtDims shape)
    {
        int count = 1;
        foreach (int value in shape.Values)
        {
            count = checked(count * value);
        }

        return count;
    }

    private static int ArgMax(float[] values)
    {
        if (values.Length == 0)
        {
            throw new ArgumentException("Output values must not be empty.", nameof(values));
        }

        int index = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[index])
            {
                index = i;
            }
        }

        return index;
    }

    private static string FormatShape(TensorRtDims shape)
    {
        return "[" + string.Join(",", shape.Values) + "]";
    }

    private static string FormatShape(int[] shape)
    {
        return "[" + string.Join(",", shape) + "]";
    }

    private static string ComputeFileSha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static string ComputeSha256(byte[] bytes)
    {
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }

    private sealed class ReferenceTensor
    {
        public int SchemaVersion { get; set; }

        public string TensorName { get; set; } = string.Empty;

        public int[] Shape { get; set; } = Array.Empty<int>();

        public float[] Values { get; set; } = Array.Empty<float>();

        public string SourceClassification { get; set; } = string.Empty;
    }

    private sealed class ReferenceValidationResult
    {
        public ReferenceValidationResult(
            bool completed,
            bool passed,
            int comparedElementCount,
            int mismatchCount,
            int firstMismatchIndex,
            float maximumAbsoluteError,
            float maximumRelativeError,
            string diagnostic)
        {
            Completed = completed;
            Passed = passed;
            ComparedElementCount = comparedElementCount;
            MismatchCount = mismatchCount;
            FirstMismatchIndex = firstMismatchIndex;
            MaximumAbsoluteError = maximumAbsoluteError;
            MaximumRelativeError = maximumRelativeError;
            Diagnostic = diagnostic ?? string.Empty;
        }

        public static ReferenceValidationResult MetadataMismatch(string diagnostic)
        {
            return new ReferenceValidationResult(
                completed: false,
                passed: false,
                comparedElementCount: 0,
                mismatchCount: 0,
                firstMismatchIndex: -1,
                maximumAbsoluteError: 0.0f,
                maximumRelativeError: 0.0f,
                diagnostic);
        }

        public bool Completed { get; }

        public bool Passed { get; }

        public int ComparedElementCount { get; }

        public int MismatchCount { get; }

        public int FirstMismatchIndex { get; }

        public float MaximumAbsoluteError { get; }

        public float MaximumRelativeError { get; }

        public string Diagnostic { get; }
    }
}
