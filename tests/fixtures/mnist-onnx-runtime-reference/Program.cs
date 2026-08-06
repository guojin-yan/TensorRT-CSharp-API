using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;

internal static class Program
{
    private const string ExpectedInputName = "Input3";
    private const string ExpectedOutputName = "Plus214_Output_0";
    private const string SourceClassification = "onnxruntime-cpu-1.23.2-derived-unreviewed";
    private static readonly int[] ExpectedInputShape = { 1, 1, 28, 28 };
    private static readonly JsonSerializerOptions JsonOptions = new JsonSerializerOptions
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true
    };

    public static int Main(string[] args)
    {
        try
        {
            if (args.Length != 6)
            {
                Console.Error.WriteLine(
                    "Usage: Mnist.OnnxRuntimeReference <model> <input-f32> <tensorrt-reference> <onnxruntime-reference> <raw-output> <run-report>");
                return 2;
            }

            string modelPath = RequireFile(args[0], "MNIST ONNX model");
            string inputPath = RequireFile(args[1], "MNIST float input");
            string tensorRtReferencePath = RequireFile(args[2], "TensorRT structured reference");
            string referenceOutputPath = Path.GetFullPath(args[3]);
            string rawOutputPath = Path.GetFullPath(args[4]);
            string reportPath = Path.GetFullPath(args[5]);
            string outputDirectory = Path.GetDirectoryName(reportPath)
                ?? throw new InvalidOperationException("Run report must have a parent directory.");
            Directory.CreateDirectory(outputDirectory);
            Directory.CreateDirectory(Path.GetDirectoryName(referenceOutputPath)!);
            Directory.CreateDirectory(Path.GetDirectoryName(rawOutputPath)!);
            Environment.CurrentDirectory = outputDirectory;

            byte[] inputBytes = File.ReadAllBytes(inputPath);
            if (inputBytes.Length != 784 * sizeof(float))
            {
                throw new InvalidDataException($"MNIST input must contain 784 float32 values. Bytes={inputBytes.Length}.");
            }

            float[] inputValues = new float[784];
            Buffer.BlockCopy(inputBytes, 0, inputValues, 0, inputBytes.Length);

            OrtEnv environment = OrtEnv.Instance();
            string runtimeVersion = environment.GetVersionString();
            string[] availableProviders = environment.GetAvailableProviders();
            if (!availableProviders.Contains("CPUExecutionProvider", StringComparer.Ordinal))
            {
                throw new NotSupportedException("ONNX Runtime CPUExecutionProvider is unavailable.");
            }

            string profilePrefix = Path.Combine(outputDirectory, "mnist-onnxruntime-cpu-profile-");
            string profilePath;
            InferenceOutput firstRun;
            InferenceOutput secondRun;
            using (SessionOptions sessionOptions = new SessionOptions())
            {
                sessionOptions.AppendExecutionProvider_CPU(1);
                sessionOptions.EnableProfiling = true;
                sessionOptions.ProfileOutputPathPrefix = profilePrefix;
                sessionOptions.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
                sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                sessionOptions.IntraOpNumThreads = 1;
                sessionOptions.InterOpNumThreads = 1;

                using InferenceSession session = new InferenceSession(modelPath, sessionOptions);
                ValidateModelContract(session);
                firstRun = Execute(session, inputValues);
                secondRun = Execute(session, inputValues);
                profilePath = Path.GetFullPath(session.EndProfiling());
            }

            byte[] firstRaw = ToBytes(firstRun.Values);
            byte[] secondRaw = ToBytes(secondRun.Values);
            bool deterministic = firstRaw.AsSpan().SequenceEqual(secondRaw);
            if (!deterministic)
            {
                throw new InvalidDataException("Two ONNX Runtime CPU runs produced different float32 bytes.");
            }

            File.WriteAllBytes(rawOutputPath, firstRaw);
            string[] profileProviders = ReadProfileProviders(profilePath);
            bool providerValidated = profileProviders.Contains("CPUExecutionProvider", StringComparer.Ordinal);
            if (!providerValidated || profileProviders.Any(provider => !string.Equals(provider, "CPUExecutionProvider", StringComparison.Ordinal)))
            {
                throw new InvalidDataException(
                    $"ONNX Runtime profiling did not prove a CPU-only run. Providers=[{string.Join(",", profileProviders)}].");
            }

            ReferenceTensor tensorRtReference = ReadReference(tensorRtReferencePath);
            ReferenceComparison comparison = Compare(firstRun, tensorRtReference, 0.0001f, 0.0001f);
            if (!comparison.Passed)
            {
                throw new InvalidDataException(
                    $"ONNX Runtime output does not match the retained TensorRT reference. Mismatches={comparison.MismatchCount}.");
            }

            var reference = new
            {
                schemaVersion = 1,
                tensorName = firstRun.TensorName,
                shape = firstRun.Shape,
                values = firstRun.Values,
                sourceClassification = SourceClassification
            };
            WriteJson(referenceOutputPath, reference);

            int predictedIndex = Enumerable.Range(0, firstRun.Values.Length)
                .OrderByDescending(index => firstRun.Values[index])
                .First();
            string managedAssemblyPath = typeof(InferenceSession).Assembly.Location;
            string nativeLibraryPath = Process.GetCurrentProcess().Modules
                .Cast<ProcessModule>()
                .Where(module => string.Equals(module.ModuleName, "onnxruntime.dll", StringComparison.OrdinalIgnoreCase))
                .Select(module => module.FileName)
                .Single();

            var report = new
            {
                schemaVersion = "tensorrtexec-mnist-onnxruntime-reference-run.v1",
                success = true,
                providerValidated,
                deterministicOutput = deterministic,
                runtime = new
                {
                    version = runtimeVersion,
                    requestedProvider = "CPUExecutionProvider",
                    availableProviders,
                    profileProviders,
                    managedAssemblyPath,
                    managedAssemblySha256 = ComputeSha256(managedAssemblyPath),
                    nativeLibraryPath,
                    nativeLibrarySha256 = ComputeSha256(nativeLibraryPath)
                },
                model = new
                {
                    path = modelPath,
                    sha256 = ComputeSha256(modelPath),
                    inputName = ExpectedInputName,
                    inputShape = ExpectedInputShape,
                    outputName = ExpectedOutputName
                },
                input = new
                {
                    path = inputPath,
                    sha256 = ComputeSha256(inputPath),
                    elementCount = inputValues.Length
                },
                output = new
                {
                    tensorName = firstRun.TensorName,
                    shape = firstRun.Shape,
                    elementCount = firstRun.Values.Length,
                    rawPath = rawOutputPath,
                    rawSha256 = ComputeSha256(rawOutputPath),
                    secondRunRawSha256 = ComputeSha256(secondRaw),
                    predictedIndex
                },
                reference = new
                {
                    path = referenceOutputPath,
                    sha256 = ComputeSha256(referenceOutputPath),
                    sourceClassification = SourceClassification
                },
                tensorRtComparison = new
                {
                    referencePath = tensorRtReferencePath,
                    referenceSha256 = ComputeSha256(tensorRtReferencePath),
                    comparedElementCount = comparison.ComparedElementCount,
                    mismatchCount = comparison.MismatchCount,
                    firstMismatchIndex = comparison.FirstMismatchIndex,
                    maximumAbsoluteError = comparison.MaximumAbsoluteError,
                    maximumRelativeError = comparison.MaximumRelativeError,
                    absoluteTolerance = 0.0001f,
                    relativeTolerance = 0.0001f,
                    passed = comparison.Passed
                },
                profile = new
                {
                    path = profilePath,
                    sha256 = ComputeSha256(profilePath),
                    providers = profileProviders
                },
                proofBoundary = new
                {
                    independentFromTensorRtExecution = true,
                    ownerReviewedGolden = false,
                    repositoryRedistributionApproved = false,
                    publicPackageProof = false,
                    postPublishProof = false,
                    statement = "ONNX Runtime CPU execution is independent of TensorRT execution, but the model, input, and generated reference remain unreviewed Owner inputs."
                }
            };
            WriteJson(reportPath, report);

            Console.WriteLine("OnnxRuntimeReference=Passed");
            Console.WriteLine("OnnxRuntimeVersion=" + runtimeVersion);
            Console.WriteLine("ProviderValidated=" + providerValidated);
            Console.WriteLine("ProfileProviders=" + string.Join(",", profileProviders));
            Console.WriteLine("DeterministicOutput=" + deterministic);
            Console.WriteLine("TensorRtReferenceComparisonPassed=" + comparison.Passed);
            Console.WriteLine("MaximumAbsoluteError=" + comparison.MaximumAbsoluteError.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
            Console.WriteLine("MaximumRelativeError=" + comparison.MaximumRelativeError.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
            Console.WriteLine("PredictedIndex=" + predictedIndex);
            Console.WriteLine("ReferenceSha256=" + ComputeSha256(referenceOutputPath));
            Console.WriteLine("RawOutputSha256=" + ComputeSha256(rawOutputPath));
            Console.WriteLine("ProfileSha256=" + ComputeSha256(profilePath));
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine("OnnxRuntimeReference=Failed");
            Console.Error.WriteLine(exception);
            return 1;
        }
    }

    private static void ValidateModelContract(InferenceSession session)
    {
        if (session.InputNames.Count != 1 || session.OutputNames.Count != 1 ||
            !string.Equals(session.InputNames[0], ExpectedInputName, StringComparison.Ordinal) ||
            !string.Equals(session.OutputNames[0], ExpectedOutputName, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Unexpected MNIST model I/O. Inputs=[{string.Join(",", session.InputNames)}] Outputs=[{string.Join(",", session.OutputNames)}].");
        }

        NodeMetadata inputMetadata = session.InputMetadata[ExpectedInputName];
        if (!inputMetadata.IsTensor || inputMetadata.ElementType != typeof(float) ||
            !inputMetadata.Dimensions.SequenceEqual(ExpectedInputShape))
        {
            throw new InvalidDataException("MNIST Input3 must be a float tensor with shape [1,1,28,28].");
        }
    }

    private static InferenceOutput Execute(InferenceSession session, float[] inputValues)
    {
        using OrtValue inputValue = OrtValue.CreateTensorValueFromMemory(
            inputValues,
            ExpectedInputShape.Select(static value => (long)value).ToArray());
        using RunOptions runOptions = new RunOptions();
        var inputs = new Dictionary<string, OrtValue>(StringComparer.Ordinal)
        {
            [ExpectedInputName] = inputValue
        };
        using IDisposableReadOnlyCollection<OrtValue> outputs = session.Run(
            runOptions,
            inputs,
            new[] { ExpectedOutputName });
        if (outputs.Count != 1)
        {
            throw new InvalidDataException($"MNIST ONNX Runtime output count must be one. Count={outputs.Count}.");
        }

        OrtValue output = outputs.Single();
        OrtTensorTypeAndShapeInfo typeAndShape = output.GetTensorTypeAndShape();
        if (typeAndShape.ElementDataType != Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float)
        {
            throw new InvalidDataException($"MNIST output must be float32. Type={typeAndShape.ElementDataType}.");
        }

        return new InferenceOutput(
            ExpectedOutputName,
            typeAndShape.Shape.Select(checkedValue => checked((int)checkedValue)).ToArray(),
            output.GetTensorDataAsSpan<float>().ToArray());
    }

    private static ReferenceTensor ReadReference(string path)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement root = document.RootElement;
        if (root.GetProperty("schemaVersion").GetInt32() != 1)
        {
            throw new InvalidDataException("TensorRT reference schemaVersion must be 1.");
        }

        return new ReferenceTensor(
            root.GetProperty("tensorName").GetString() ?? string.Empty,
            root.GetProperty("shape").EnumerateArray().Select(static value => value.GetInt32()).ToArray(),
            root.GetProperty("values").EnumerateArray().Select(static value => value.GetSingle()).ToArray());
    }

    private static ReferenceComparison Compare(
        InferenceOutput actual,
        ReferenceTensor expected,
        float absoluteTolerance,
        float relativeTolerance)
    {
        if (!string.Equals(actual.TensorName, expected.TensorName, StringComparison.Ordinal) ||
            !actual.Shape.SequenceEqual(expected.Shape) ||
            actual.Values.Length != expected.Values.Length)
        {
            throw new InvalidDataException("TensorRT and ONNX Runtime reference metadata do not match.");
        }

        int mismatchCount = 0;
        int firstMismatchIndex = -1;
        float maximumAbsoluteError = 0.0f;
        float maximumRelativeError = 0.0f;
        for (int index = 0; index < actual.Values.Length; index++)
        {
            float actualValue = actual.Values[index];
            float expectedValue = expected.Values[index];
            if (!float.IsFinite(actualValue) || !float.IsFinite(expectedValue))
            {
                throw new InvalidDataException("MNIST reference comparison requires finite logits.");
            }

            float absoluteError = Math.Abs(actualValue - expectedValue);
            float scale = Math.Max(Math.Abs(actualValue), Math.Abs(expectedValue));
            float relativeError = scale == 0.0f ? absoluteError : absoluteError / scale;
            maximumAbsoluteError = Math.Max(maximumAbsoluteError, absoluteError);
            maximumRelativeError = Math.Max(maximumRelativeError, relativeError);
            if (absoluteError <= absoluteTolerance || absoluteError <= relativeTolerance * scale)
            {
                continue;
            }

            mismatchCount++;
            if (firstMismatchIndex < 0)
            {
                firstMismatchIndex = index;
            }
        }

        return new ReferenceComparison(
            mismatchCount == 0,
            actual.Values.Length,
            mismatchCount,
            firstMismatchIndex,
            maximumAbsoluteError,
            maximumRelativeError);
    }

    private static string[] ReadProfileProviders(string profilePath)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(profilePath));
        var providers = new HashSet<string>(StringComparer.Ordinal);
        foreach (JsonElement item in document.RootElement.EnumerateArray())
        {
            if (item.TryGetProperty("args", out JsonElement eventArgs) &&
                eventArgs.TryGetProperty("provider", out JsonElement provider) &&
                provider.ValueKind == JsonValueKind.String &&
                !string.IsNullOrWhiteSpace(provider.GetString()))
            {
                providers.Add(provider.GetString()!);
            }
        }

        return providers.OrderBy(static value => value, StringComparer.Ordinal).ToArray();
    }

    private static string RequireFile(string value, string description)
    {
        string path = Path.GetFullPath(value);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(description + " was not found.", path);
        }

        return path;
    }

    private static void WriteJson(string path, object value)
    {
        File.WriteAllText(path, JsonSerializer.Serialize(value, JsonOptions) + "\n", new UTF8Encoding(false));
    }

    private static byte[] ToBytes(float[] values)
    {
        byte[] bytes = new byte[checked(values.Length * sizeof(float))];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static string ComputeSha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static string ComputeSha256(byte[] bytes)
    {
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }

    private sealed record InferenceOutput(string TensorName, int[] Shape, float[] Values);
    private sealed record ReferenceTensor(string TensorName, int[] Shape, float[] Values);
    private sealed record ReferenceComparison(
        bool Passed,
        int ComparedElementCount,
        int MismatchCount,
        int FirstMismatchIndex,
        float MaximumAbsoluteError,
        float MaximumRelativeError);
}
