using System.IO;
using System.Text;
using System.Text.Json;

namespace JYPPX.TensorRtSharp.Tools;

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
