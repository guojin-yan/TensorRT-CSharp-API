using System;
using System.Globalization;
using System.Linq;
using JYPPX.TensorRtSharp.Shared.Interop;

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
