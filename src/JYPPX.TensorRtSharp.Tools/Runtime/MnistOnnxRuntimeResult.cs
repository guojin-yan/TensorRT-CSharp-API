using System;
using System.Collections.Generic;
using System.Text;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Tools;

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
