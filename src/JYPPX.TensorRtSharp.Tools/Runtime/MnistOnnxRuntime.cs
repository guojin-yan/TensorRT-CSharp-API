using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

/// <summary>
/// Options for the model-specific MNIST ONNX runtime path.
/// MNIST ONNX 模型专用运行路径的选项。
/// </summary>
public sealed class MnistOnnxRuntimeOptions
{
    public MnistOnnxRuntimeOptions(
        TensorRtApiLine tensorRtLine,
        string onnxPath,
        string inputPgmPath,
        int expectedDigit,
        string saveEnginePath,
        string exportReportPath,
        string exportOutputPath,
        string exportPreprocessedInputPath,
        ulong workspaceBytes,
        float minimumConfidence)
    {
        TensorRtLine = tensorRtLine;
        OnnxPath = onnxPath ?? string.Empty;
        InputPgmPath = inputPgmPath ?? string.Empty;
        ExpectedDigit = expectedDigit;
        SaveEnginePath = saveEnginePath ?? string.Empty;
        ExportReportPath = exportReportPath ?? string.Empty;
        ExportOutputPath = exportOutputPath ?? string.Empty;
        ExportPreprocessedInputPath = exportPreprocessedInputPath ?? string.Empty;
        WorkspaceBytes = workspaceBytes;
        MinimumConfidence = minimumConfidence;
    }

    public TensorRtApiLine TensorRtLine { get; }

    public string OnnxPath { get; }

    public string InputPgmPath { get; }

    public int ExpectedDigit { get; }

    public string SaveEnginePath { get; }

    public string ExportReportPath { get; }

    public string ExportOutputPath { get; }

    public string ExportPreprocessedInputPath { get; }

    public ulong WorkspaceBytes { get; }

    public float MinimumConfidence { get; }

    public string ToCommandLine()
    {
        return string.Join(" ", new[]
        {
            "--mnist",
            "--tensor-rt-line " + (int)TensorRtLine,
            "--onnx " + Quote(OnnxPath),
            "--mnistInput " + Quote(InputPgmPath),
            "--expectedDigit " + ExpectedDigit.ToString(CultureInfo.InvariantCulture),
            "--saveEngine " + Quote(SaveEnginePath),
            "--exportReport " + Quote(ExportReportPath),
            "--exportOutput " + Quote(ExportOutputPath),
            "--exportPreprocessedInput " + Quote(ExportPreprocessedInputPath),
            "--workspace " + (WorkspaceBytes / (1024UL * 1024UL)).ToString(CultureInfo.InvariantCulture),
            "--minimumConfidence " + MinimumConfidence.ToString("0.####", CultureInfo.InvariantCulture)
        }.Where(static value => !value.EndsWith("\"\"", StringComparison.Ordinal)));
    }

    private static string Quote(string value)
    {
        return string.IsNullOrWhiteSpace(value) ? string.Empty : "\"" + value.Replace("\"", "\\\"", StringComparison.Ordinal) + "\"";
    }
}

/// <summary>
/// One binary PGM image used by the TensorRT MNIST sample.
/// TensorRT MNIST 样例使用的一张二进制 PGM 图像。
/// </summary>
public sealed class MnistPgmImage
{
    public MnistPgmImage(int width, int height, int maxValue, byte[] pixels)
    {
        Width = width;
        Height = height;
        MaxValue = maxValue;
        Pixels = pixels ?? throw new ArgumentNullException(nameof(pixels));
    }

    public int Width { get; }

    public int Height { get; }

    public int MaxValue { get; }

    public byte[] Pixels { get; }
}

/// <summary>
/// Reads and preprocesses the P5 PGM assets shipped with TensorRT MNIST samples.
/// 读取并预处理 TensorRT MNIST 样例附带的 P5 PGM 资产。
/// </summary>
public static class MnistPgmReader
{
    public static MnistPgmImage Read(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("PGM path must not be empty.", nameof(path));
        }

        byte[] bytes = File.ReadAllBytes(path);
        int offset = 0;
        string magic = ReadToken(bytes, ref offset);
        if (!string.Equals(magic, "P5", StringComparison.Ordinal))
        {
            throw new InvalidDataException("MNIST input must be a binary P5 PGM file.");
        }

        int width = ParsePositiveInt(ReadToken(bytes, ref offset), "width");
        int height = ParsePositiveInt(ReadToken(bytes, ref offset), "height");
        int maxValue = ParsePositiveInt(ReadToken(bytes, ref offset), "max value");
        if (maxValue != 255)
        {
            throw new InvalidDataException("MNIST PGM max value must be 255.");
        }

        ConsumePixelSeparator(bytes, ref offset);
        int pixelCount = checked(width * height);
        if (bytes.Length - offset != pixelCount)
        {
            throw new InvalidDataException(
                $"PGM pixel payload length mismatch. Expected={pixelCount} Actual={bytes.Length - offset}.");
        }

        byte[] pixels = new byte[pixelCount];
        Buffer.BlockCopy(bytes, offset, pixels, 0, pixelCount);
        return new MnistPgmImage(width, height, maxValue, pixels);
    }

    public static float[] ToTensorInput(MnistPgmImage image)
    {
        if (image == null)
        {
            throw new ArgumentNullException(nameof(image));
        }

        float[] values = new float[image.Pixels.Length];
        for (int index = 0; index < image.Pixels.Length; index++)
        {
            values[index] = 1.0f - (image.Pixels[index] / 255.0f);
        }

        return values;
    }

    private static string ReadToken(byte[] bytes, ref int offset)
    {
        SkipWhitespaceAndComments(bytes, ref offset);
        if (offset >= bytes.Length)
        {
            throw new InvalidDataException("Unexpected end of PGM header.");
        }

        int start = offset;
        while (offset < bytes.Length && !IsWhitespace(bytes[offset]) && bytes[offset] != (byte)'#')
        {
            offset++;
        }

        if (offset == start)
        {
            throw new InvalidDataException("PGM header token is empty.");
        }

        return Encoding.ASCII.GetString(bytes, start, offset - start);
    }

    private static void SkipWhitespaceAndComments(byte[] bytes, ref int offset)
    {
        while (offset < bytes.Length)
        {
            while (offset < bytes.Length && IsWhitespace(bytes[offset]))
            {
                offset++;
            }

            if (offset >= bytes.Length || bytes[offset] != (byte)'#')
            {
                return;
            }

            while (offset < bytes.Length && bytes[offset] != (byte)'\n')
            {
                offset++;
            }
        }
    }

    private static void ConsumePixelSeparator(byte[] bytes, ref int offset)
    {
        if (offset >= bytes.Length || !IsWhitespace(bytes[offset]))
        {
            throw new InvalidDataException("PGM header must be followed by a whitespace separator.");
        }

        byte first = bytes[offset++];
        if (first == (byte)'\r' && offset < bytes.Length && bytes[offset] == (byte)'\n')
        {
            offset++;
        }
    }

    private static bool IsWhitespace(byte value)
    {
        return value == (byte)' ' ||
            value == (byte)'\t' ||
            value == (byte)'\r' ||
            value == (byte)'\n' ||
            value == (byte)'\f';
    }

    private static int ParsePositiveInt(string value, string fieldName)
    {
        if (!int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int parsed) || parsed <= 0)
        {
            throw new InvalidDataException($"PGM {fieldName} is invalid.");
        }

        return parsed;
    }
}

/// <summary>
/// Classification output for the ten MNIST logits.
/// 十个 MNIST logits 的分类结果。
/// </summary>
public sealed class MnistClassification
{
    public MnistClassification(float[] probabilities, int predictedDigit, float confidence)
    {
        Probabilities = probabilities ?? throw new ArgumentNullException(nameof(probabilities));
        PredictedDigit = predictedDigit;
        Confidence = confidence;
    }

    public float[] Probabilities { get; }

    public int PredictedDigit { get; }

    public float Confidence { get; }
}

/// <summary>
/// Stable softmax and argmax helper for MNIST output validation.
/// 用于 MNIST 输出验证的稳定 softmax 与 argmax 帮助器。
/// </summary>
public static class MnistOutputClassifier
{
    public static MnistClassification Classify(float[] logits)
    {
        if (logits == null)
        {
            throw new ArgumentNullException(nameof(logits));
        }

        if (logits.Length != 10)
        {
            throw new ArgumentException("MNIST output must contain exactly ten logits.", nameof(logits));
        }

        float max = logits.Max();
        double[] exponentials = new double[logits.Length];
        double sum = 0.0;
        for (int index = 0; index < logits.Length; index++)
        {
            exponentials[index] = Math.Exp(logits[index] - max);
            sum += exponentials[index];
        }

        float[] probabilities = new float[logits.Length];
        int predictedDigit = 0;
        float confidence = float.MinValue;
        for (int index = 0; index < exponentials.Length; index++)
        {
            probabilities[index] = (float)(exponentials[index] / sum);
            if (probabilities[index] > confidence)
            {
                confidence = probabilities[index];
                predictedDigit = index;
            }
        }

        return new MnistClassification(probabilities, predictedDigit, confidence);
    }
}

/// <summary>
/// Captured runtime environment for one MNIST proof attempt.
/// 一次 MNIST proof 尝试所采集的运行环境。
/// </summary>
public sealed class MnistRuntimeEnvironment
{
    public MnistRuntimeEnvironment(
        string hostOs,
        string processArchitecture,
        string machineName,
        string gpuName,
        string computeCapability,
        ulong gpuMemoryBytes,
        int cudaDriverVersion,
        int cudaRuntimeVersion,
        string cudaToolkitVersion,
        string tensorRtVersion,
        string bridgeVersion)
    {
        HostOs = hostOs;
        ProcessArchitecture = processArchitecture;
        MachineName = machineName;
        GpuName = gpuName;
        ComputeCapability = computeCapability;
        GpuMemoryBytes = gpuMemoryBytes;
        CudaDriverVersion = cudaDriverVersion;
        CudaRuntimeVersion = cudaRuntimeVersion;
        CudaToolkitVersion = cudaToolkitVersion;
        TensorRtVersion = tensorRtVersion;
        BridgeVersion = bridgeVersion;
    }

    public string HostOs { get; }

    public string ProcessArchitecture { get; }

    public string MachineName { get; }

    public string GpuName { get; }

    public string ComputeCapability { get; }

    public ulong GpuMemoryBytes { get; }

    public int CudaDriverVersion { get; }

    public int CudaRuntimeVersion { get; }

    public string CudaToolkitVersion { get; }

    public string TensorRtVersion { get; }

    public string BridgeVersion { get; }

    public static MnistRuntimeEnvironment Capture(TensorRtEnvironmentSnapshot snapshot)
    {
        CudaDeviceProperties device = CudaDevice.CurrentProperties;
        BridgeBuildInfo build = snapshot.BuildInfo;
        return new MnistRuntimeEnvironment(
            RuntimeInformation.OSDescription,
            RuntimeInformation.ProcessArchitecture.ToString(),
            Environment.MachineName,
            device.Info.Name,
            device.ComputeCapabilityLabel,
            device.Info.TotalGlobalMemory,
            CudaDevice.DriverVersion,
            CudaDevice.RuntimeVersion,
            build.CudaToolkitVersion,
            build.TensorRtVersion,
            $"{build.BridgeVersionMajor}.{build.BridgeVersionMinor}.{build.BridgeVersionPatch}");
    }
}

/// <summary>
/// Result of one model-specific MNIST ONNX execution.
/// 一次 MNIST ONNX 模型专用执行的结果。
/// </summary>
public sealed class MnistOnnxRuntimeResult
{
    public MnistOnnxRuntimeResult(
        bool success,
        bool skipped,
        string state,
        TensorRtApiLine tensorRtLine,
        string modelPath,
        string inputPath,
        string enginePath,
        string modelSha256,
        string inputSha256,
        string preprocessedInputSha256,
        string engineSha256,
        bool parsed,
        bool engineSaved,
        bool engineFileRoundTrip,
        bool inferenceRan,
        bool outputMatch,
        int expectedDigit,
        int predictedDigit,
        float confidence,
        float minimumConfidence,
        string inputTensorName,
        int[] inputShape,
        string inputDataType,
        string outputTensorName,
        int[] outputShape,
        string outputDataType,
        float[] logits,
        float[] probabilities,
        float? elapsedMilliseconds,
        string skipReason,
        string normalizedCommandLine,
        MnistRuntimeEnvironment? environment,
        IReadOnlyList<string> logLines)
    {
        Success = success;
        Skipped = skipped;
        State = state ?? string.Empty;
        TensorRtLine = tensorRtLine;
        ModelPath = modelPath ?? string.Empty;
        InputPath = inputPath ?? string.Empty;
        EnginePath = enginePath ?? string.Empty;
        ModelSha256 = modelSha256 ?? string.Empty;
        InputSha256 = inputSha256 ?? string.Empty;
        PreprocessedInputSha256 = preprocessedInputSha256 ?? string.Empty;
        EngineSha256 = engineSha256 ?? string.Empty;
        Parsed = parsed;
        EngineSaved = engineSaved;
        EngineFileRoundTrip = engineFileRoundTrip;
        InferenceRan = inferenceRan;
        OutputMatch = outputMatch;
        ExpectedDigit = expectedDigit;
        PredictedDigit = predictedDigit;
        Confidence = confidence;
        MinimumConfidence = minimumConfidence;
        InputTensorName = inputTensorName ?? string.Empty;
        InputShape = inputShape ?? Array.Empty<int>();
        InputDataType = inputDataType ?? string.Empty;
        OutputTensorName = outputTensorName ?? string.Empty;
        OutputShape = outputShape ?? Array.Empty<int>();
        OutputDataType = outputDataType ?? string.Empty;
        Logits = logits ?? Array.Empty<float>();
        Probabilities = probabilities ?? Array.Empty<float>();
        ElapsedMilliseconds = elapsedMilliseconds;
        SkipReason = skipReason ?? string.Empty;
        NormalizedCommandLine = normalizedCommandLine ?? string.Empty;
        Environment = environment;
        LogLines = logLines ?? Array.Empty<string>();
    }

    public bool Success { get; }

    public bool Skipped { get; }

    public string State { get; }

    public TensorRtApiLine TensorRtLine { get; }

    public string ModelPath { get; }

    public string InputPath { get; }

    public string EnginePath { get; }

    public string ModelSha256 { get; }

    public string InputSha256 { get; }

    public string PreprocessedInputSha256 { get; }

    public string EngineSha256 { get; }

    public bool Parsed { get; }

    public bool EngineSaved { get; }

    public bool EngineFileRoundTrip { get; }

    public bool InferenceRan { get; }

    public bool OutputMatch { get; }

    public int ExpectedDigit { get; }

    public int PredictedDigit { get; }

    public float Confidence { get; }

    public float MinimumConfidence { get; }

    public string InputTensorName { get; }

    public int[] InputShape { get; }

    public string InputDataType { get; }

    public string OutputTensorName { get; }

    public int[] OutputShape { get; }

    public string OutputDataType { get; }

    public float[] Logits { get; }

    public float[] Probabilities { get; }

    public float? ElapsedMilliseconds { get; }

    public string SkipReason { get; }

    public string NormalizedCommandLine { get; }

    public string NormalizedCommandSha256 => HashText(NormalizedCommandLine);

    public MnistRuntimeEnvironment? Environment { get; }

    public IReadOnlyList<string> LogLines { get; }

    public string ProofClassification => Skipped
        ? "dependency-probe-only"
        : InferenceRan && OutputMatch && Confidence >= MinimumConfidence
            ? "real-model-runtime"
            : InferenceRan
                ? "runtime-output-mismatch"
                : "build-only";

    public bool IsRealModelRuntimeProof => string.Equals(ProofClassification, "real-model-runtime", StringComparison.Ordinal);

    public bool IsPackageConsumerRuntimeProof => false;

    public bool CanPublishPublicly => false;

    public bool CanCloseReleaseIssue => false;

    public string ProofBoundary =>
        "real-model-runtime requires external ONNX inference, validated MNIST digit output, and minimum confidence; " +
        "this source-tree execution is not package-consumer-runtime, post-publish proof, or release authorization.";

    private static string HashText(string value)
    {
        return MnistOnnxRuntimeService.ComputeSha256(Encoding.UTF8.GetBytes(value ?? string.Empty));
    }
}

/// <summary>
/// Executes the ONNX Model Zoo MNIST model with explicit model semantics.
/// 使用明确模型语义执行 ONNX Model Zoo MNIST 模型。
/// </summary>
public sealed class MnistOnnxRuntimeService
{
    public MnistOnnxRuntimeResult Execute(MnistOnnxRuntimeOptions options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        ValidateOptions(options);
        string modelPath = Path.GetFullPath(options.OnnxPath);
        string inputPath = Path.GetFullPath(options.InputPgmPath);
        string enginePath = string.IsNullOrWhiteSpace(options.SaveEnginePath)
            ? Path.Combine(Path.GetTempPath(), $"jyppx-mnist-{Guid.NewGuid():N}.plan")
            : Path.GetFullPath(options.SaveEnginePath);
        bool deleteEngine = string.IsNullOrWhiteSpace(options.SaveEnginePath);
        byte[] model = File.ReadAllBytes(modelPath);
        byte[] inputFile = File.ReadAllBytes(inputPath);
        MnistPgmImage image = MnistPgmReader.Read(inputPath);
        float[] inputValues = MnistPgmReader.ToTensorInput(image);
        byte[] preprocessedInput = ToBytes(inputValues);
        List<string> log = new List<string>
        {
            $"MnistOnnxRuntime TensorRtLine={(int)options.TensorRtLine} Model={modelPath} Input={inputPath} ExpectedDigit={options.ExpectedDigit}",
            $"MnistPreprocess Width={image.Width} Height={image.Height} Formula=1-pixel/255 ElementCount={inputValues.Length}",
            $"MnistHashes Model={ComputeSha256(model)} Input={ComputeSha256(inputFile)} PreprocessedInput={ComputeSha256(preprocessedInput)}"
        };

        TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
        TensorRtAdapterInfo adapter = TensorRtToolSupport.SelectAdapter(snapshot, options.TensorRtLine);
        log.Add($"MnistPreflight TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion} Runtime={adapter.RuntimeCreationSupported} Builder={adapter.BuilderCreationSupported}");
        if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
        {
            MnistOnnxRuntimeResult skipped = new MnistOnnxRuntimeResult(
                success: true,
                skipped: true,
                state: "dependency-unavailable",
                options.TensorRtLine,
                modelPath,
                inputPath,
                enginePath,
                ComputeSha256(model),
                ComputeSha256(inputFile),
                ComputeSha256(preprocessedInput),
                engineSha256: string.Empty,
                parsed: false,
                engineSaved: false,
                engineFileRoundTrip: false,
                inferenceRan: false,
                outputMatch: false,
                options.ExpectedDigit,
                predictedDigit: -1,
                confidence: 0.0f,
                options.MinimumConfidence,
                inputTensorName: string.Empty,
                inputShape: Array.Empty<int>(),
                inputDataType: string.Empty,
                outputTensorName: string.Empty,
                outputShape: Array.Empty<int>(),
                outputDataType: string.Empty,
                logits: Array.Empty<float>(),
                probabilities: Array.Empty<float>(),
                elapsedMilliseconds: null,
                skipReason: adapter.StatusMessage,
                normalizedCommandLine: options.ToCommandLine(),
                environment: null,
                log);
            MnistOnnxRuntimeDiagnostics.WriteArtifacts(skipped, options);
            return skipped;
        }

        string? engineDirectory = Path.GetDirectoryName(enginePath);
        if (!string.IsNullOrWhiteSpace(engineDirectory))
        {
            Directory.CreateDirectory(engineDirectory);
        }

        try
        {
            using TensorRtLogger logger = new TensorRtLogger(options.TensorRtLine);
            using TensorRtRuntime runtime = new TensorRtRuntime(logger);
            using TensorRtBuilder builder = new TensorRtBuilder(logger);
            using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
            using CudaStream stream = new CudaStream();
            config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, options.WorkspaceBytes);
            config.SetProfileStream(stream);
            config.SetEngineCapability(TensorRtEngineCapability.Standard);

            using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
            using TensorRtOnnxParser parser = new TensorRtOnnxParser(logger, network);
            if (!parser.Parse(model, Path.GetFileName(modelPath)))
            {
                throw new InvalidOperationException(parser.GetErrorSummary());
            }

            using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
            hostMemory.SaveToFile(enginePath);
            using TensorRtEngine engine = runtime.DeserializeFromFile(enginePath);
            using TensorRtExecutionContext context = engine.CreateExecutionContext();
            using TensorRtInferenceBindings bindings = new TensorRtInferenceBindings(engine, context, profileIndex: 0);

            TensorRtEngineTensorBinding input = SingleTensor(bindings.Report.GetInputs(), "input");
            TensorRtEngineTensorBinding output = SingleTensor(bindings.Report.GetOutputs(), "output");
            EnsureFloatDeviceTensor(input, "input");
            EnsureFloatDeviceTensor(output, "output");

            TensorRtDims inputShape = ResolveShape(context, input);
            TensorRtDims outputShape = ResolveShape(context, output);
            int inputElementCount = ElementCount(inputShape);
            int outputElementCount = ElementCount(outputShape);
            if (inputElementCount != inputValues.Length)
            {
                throw new InvalidOperationException(
                    $"MNIST input element count mismatch. Tensor={inputElementCount} Image={inputValues.Length}.");
            }

            if (outputElementCount != 10)
            {
                throw new InvalidOperationException($"MNIST output must contain ten logits. Actual={outputElementCount}.");
            }

            bindings.CopyInputFromHost(input.Name, inputValues, inputShape);
            bindings.AllocateDeviceBuffer(output.Name, outputShape, checked(outputElementCount * sizeof(float)));
            bindings.BindAll();

            TensorRtInferenceExecutionSummary executionSummary = null!;
            float elapsedMilliseconds = stream.MeasureElapsedTime(cudaStream =>
            {
                executionSummary = bindings.EnqueueAsync(cudaStream, synchronize: false, runShapeInference: false);
            });
            stream.Synchronize();

            float[] logits = bindings.ReadOutputSingles(output.Name, outputElementCount);
            MnistClassification classification = MnistOutputClassifier.Classify(logits);
            bool outputMatch = classification.PredictedDigit == options.ExpectedDigit &&
                classification.Confidence >= options.MinimumConfidence;
            string engineSha256 = ComputeFileSha256(enginePath);
            MnistRuntimeEnvironment environment = MnistRuntimeEnvironment.Capture(snapshot);

            log.Add($"MnistBindings Input={input.Name}:{input.DataType}:{inputShape} Output={output.Name}:{output.DataType}:{outputShape}");
            log.Add($"MnistExecution {executionSummary} ElapsedMs={elapsedMilliseconds:0.###}");
            log.Add($"MnistClassification Expected={options.ExpectedDigit} Predicted={classification.PredictedDigit} Confidence={classification.Confidence:0.000000} Minimum={options.MinimumConfidence:0.000000} OutputMatch={outputMatch}");
            log.Add($"MnistEngine Path={enginePath} Sha256={engineSha256} Bytes={new FileInfo(enginePath).Length}");

            MnistOnnxRuntimeResult result = new MnistOnnxRuntimeResult(
                success: outputMatch,
                skipped: false,
                state: outputMatch ? "mnist-real-model-runtime" : "mnist-output-mismatch",
                options.TensorRtLine,
                modelPath,
                inputPath,
                enginePath,
                ComputeSha256(model),
                ComputeSha256(inputFile),
                ComputeSha256(preprocessedInput),
                engineSha256,
                parsed: true,
                engineSaved: true,
                engineFileRoundTrip: true,
                inferenceRan: true,
                outputMatch,
                options.ExpectedDigit,
                classification.PredictedDigit,
                classification.Confidence,
                options.MinimumConfidence,
                input.Name,
                inputShape.Values,
                input.DataType.ToString(),
                output.Name,
                outputShape.Values,
                output.DataType.ToString(),
                logits,
                classification.Probabilities,
                elapsedMilliseconds,
                skipReason: string.Empty,
                normalizedCommandLine: options.ToCommandLine(),
                environment,
                log);
            MnistOnnxRuntimeDiagnostics.WriteArtifacts(result, options, preprocessedInput);
            return result;
        }
        finally
        {
            if (deleteEngine && File.Exists(enginePath))
            {
                File.Delete(enginePath);
            }
        }
    }

    internal static string ComputeSha256(byte[] bytes)
    {
        using SHA256 sha256 = SHA256.Create();
        byte[] hash = sha256.ComputeHash(bytes);
        StringBuilder builder = new StringBuilder(hash.Length * 2);
        foreach (byte item in hash)
        {
            builder.Append(item.ToString("x2", CultureInfo.InvariantCulture));
        }

        return builder.ToString();
    }

    private static string ComputeFileSha256(string path)
    {
        return ComputeSha256(File.ReadAllBytes(path));
    }

    private static byte[] ToBytes(float[] values)
    {
        byte[] bytes = new byte[checked(values.Length * sizeof(float))];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static TensorRtEngineTensorBinding SingleTensor(
        IReadOnlyList<TensorRtEngineTensorBinding> tensors,
        string role)
    {
        if (tensors.Count != 1)
        {
            throw new InvalidOperationException($"MNIST runtime expects exactly one {role} tensor. Actual={tensors.Count}.");
        }

        return tensors[0];
    }

    private static void EnsureFloatDeviceTensor(TensorRtEngineTensorBinding tensor, string role)
    {
        if (tensor.DataType != TensorRtDataType.Float)
        {
            throw new NotSupportedException($"MNIST {role} tensor must use Float. Actual={tensor.DataType}.");
        }

        if (tensor.Location != TensorRtTensorLocation.Device)
        {
            throw new NotSupportedException($"MNIST {role} tensor must use device memory. Actual={tensor.Location}.");
        }
    }

    private static TensorRtDims ResolveShape(
        TensorRtExecutionContext context,
        TensorRtEngineTensorBinding tensor)
    {
        TensorRtDims contextShape = context.GetTensorShape(tensor.Name);
        if (IsConcrete(contextShape))
        {
            return contextShape;
        }

        if (IsConcrete(tensor.EngineShape))
        {
            return tensor.EngineShape;
        }

        throw new InvalidOperationException($"MNIST tensor '{tensor.Name}' does not have a concrete shape.");
    }

    private static bool IsConcrete(TensorRtDims shape)
    {
        return shape.Values.Length > 0 && shape.Values.All(static value => value > 0);
    }

    private static int ElementCount(TensorRtDims shape)
    {
        int count = 1;
        foreach (int value in shape.Values)
        {
            count = checked(count * value);
        }

        return count;
    }

    private static void ValidateOptions(MnistOnnxRuntimeOptions options)
    {
        if (string.IsNullOrWhiteSpace(options.OnnxPath) || !File.Exists(options.OnnxPath))
        {
            throw new FileNotFoundException("MNIST ONNX model was not found.", options.OnnxPath);
        }

        if (string.IsNullOrWhiteSpace(options.InputPgmPath) || !File.Exists(options.InputPgmPath))
        {
            throw new FileNotFoundException("MNIST PGM input was not found.", options.InputPgmPath);
        }

        if (options.ExpectedDigit < 0 || options.ExpectedDigit > 9)
        {
            throw new ArgumentOutOfRangeException(nameof(options.ExpectedDigit), "Expected digit must be in [0, 9].");
        }

        if (options.WorkspaceBytes == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.WorkspaceBytes), "Workspace size must be greater than zero.");
        }

        if (options.MinimumConfidence <= 0.0f || options.MinimumConfidence > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options.MinimumConfidence), "Minimum confidence must be in (0, 1].");
        }
    }
}

/// <summary>
/// Writes auditable MNIST runtime report, output, and preprocessed input artifacts.
/// 写入可审计的 MNIST runtime 报告、输出和预处理输入产物。
/// </summary>
public static class MnistOnnxRuntimeDiagnostics
{
    private static readonly JsonSerializerOptions JsonOptions = new JsonSerializerOptions
    {
        WriteIndented = true
    };

    public static void WriteArtifacts(
        MnistOnnxRuntimeResult result,
        MnistOnnxRuntimeOptions options,
        byte[]? preprocessedInput = null)
    {
        if (!string.IsNullOrWhiteSpace(options.ExportReportPath))
        {
            WriteJson(options.ExportReportPath, result);
        }

        if (!string.IsNullOrWhiteSpace(options.ExportOutputPath))
        {
            WriteJson(options.ExportOutputPath, new
            {
                ArtifactKind = "mnist-real-model-output",
                result.State,
                result.ProofClassification,
                result.IsRealModelRuntimeProof,
                result.IsPackageConsumerRuntimeProof,
                result.ExpectedDigit,
                result.PredictedDigit,
                result.Confidence,
                result.MinimumConfidence,
                result.OutputMatch,
                result.OutputTensorName,
                result.OutputShape,
                result.OutputDataType,
                result.Logits,
                result.Probabilities,
                result.ElapsedMilliseconds,
                result.ProofBoundary
            });
        }

        if (!string.IsNullOrWhiteSpace(options.ExportPreprocessedInputPath) && preprocessedInput != null)
        {
            WriteBytes(options.ExportPreprocessedInputPath, preprocessedInput);
        }
    }

    private static void WriteJson(string path, object value)
    {
        WriteText(path, JsonSerializer.Serialize(value, JsonOptions));
    }

    private static void WriteText(string path, string content)
    {
        string fullPath = Path.GetFullPath(path);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(fullPath, content, Encoding.UTF8);
    }

    private static void WriteBytes(string path, byte[] bytes)
    {
        string fullPath = Path.GetFullPath(path);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllBytes(fullPath, bytes);
    }
}
