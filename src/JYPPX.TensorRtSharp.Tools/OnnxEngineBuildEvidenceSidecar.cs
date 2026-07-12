using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class OnnxEngineBuildEvidenceSidecar
{
    public OnnxEngineBuildEvidenceSidecar(
        string path,
        string modelSource,
        string modelSha256,
        string modelLicense,
        string inputAssetName,
        string inputAssetSha256,
        string stdoutSummary,
        string stderrSummary,
        string proofClassification,
        IReadOnlyList<string> diagnostics)
    {
        Path = path ?? string.Empty;
        ModelSource = modelSource ?? string.Empty;
        ModelSha256 = modelSha256 ?? string.Empty;
        ModelLicense = modelLicense ?? string.Empty;
        InputAssetName = inputAssetName ?? string.Empty;
        InputAssetSha256 = inputAssetSha256 ?? string.Empty;
        StdoutSummary = stdoutSummary ?? string.Empty;
        StderrSummary = stderrSummary ?? string.Empty;
        ProofClassification = proofClassification ?? string.Empty;
        Diagnostics = diagnostics ?? Array.Empty<string>();
    }

    public string Path { get; }

    public string ModelSource { get; }

    public string ModelSha256 { get; }

    public string ModelLicense { get; }

    public string InputAssetName { get; }

    public string InputAssetSha256 { get; }

    public string StdoutSummary { get; }

    public string StderrSummary { get; }

    public string ProofClassification { get; }

    public IReadOnlyList<string> Diagnostics { get; }

    public bool HasModelEvidence =>
        !string.IsNullOrWhiteSpace(ModelSha256) ||
        !string.IsNullOrWhiteSpace(ModelLicense) ||
        !string.IsNullOrWhiteSpace(InputAssetName) ||
        !string.IsNullOrWhiteSpace(InputAssetSha256);

    public OnnxEngineBuildModelEvidence ToModelEvidence(string fallbackModelSource)
    {
        return new OnnxEngineBuildModelEvidence(
            string.IsNullOrWhiteSpace(ModelSource) ? fallbackModelSource : ModelSource,
            ModelSha256,
            ModelLicense,
            InputAssetName,
            InputAssetSha256);
    }
}

public static class OnnxEngineBuildEvidenceSidecarReader
{
    public static OnnxEngineBuildEvidenceSidecar Empty { get; } = new OnnxEngineBuildEvidenceSidecar(
        string.Empty,
        string.Empty,
        string.Empty,
        string.Empty,
        string.Empty,
        string.Empty,
        string.Empty,
        string.Empty,
        string.Empty,
        Array.Empty<string>());

    public static OnnxEngineBuildEvidenceSidecar Read(string sidecarPath)
    {
        if (string.IsNullOrWhiteSpace(sidecarPath))
        {
            return Empty;
        }

        string fullPath = System.IO.Path.GetFullPath(sidecarPath);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException("Evidence sidecar file was not found.", fullPath);
        }

        using FileStream stream = File.OpenRead(fullPath);
        using JsonDocument document = JsonDocument.Parse(stream, new JsonDocumentOptions { AllowTrailingCommas = true });
        JsonElement root = document.RootElement;

        string modelSource = GetString(root, "modelSource");
        string modelSha256 = GetString(root, "modelSha256");
        string modelLicense = GetString(root, "modelLicense");
        string inputAssetName = GetString(root, "inputAssetName");
        string inputAssetSha256 = GetString(root, "inputAssetSha256");
        string stdoutSummary = GetString(root, "stdoutSummary");
        string stderrSummary = GetString(root, "stderrSummary");
        string proofClassification = GetString(root, "proofClassification");

        if (root.TryGetProperty("modelEvidence", out JsonElement evidence) && evidence.ValueKind == JsonValueKind.Object)
        {
            modelSource = FirstNonEmpty(GetString(evidence, "modelSource"), GetString(evidence, "ModelSource"), modelSource);
            modelSha256 = FirstNonEmpty(GetString(evidence, "modelSha256"), GetString(evidence, "ModelSha256"), modelSha256);
            modelLicense = FirstNonEmpty(GetString(evidence, "modelLicense"), GetString(evidence, "ModelLicense"), modelLicense);
            inputAssetName = FirstNonEmpty(GetString(evidence, "inputAssetName"), GetString(evidence, "InputAssetName"), inputAssetName);
            inputAssetSha256 = FirstNonEmpty(GetString(evidence, "inputAssetSha256"), GetString(evidence, "InputAssetSha256"), inputAssetSha256);
        }

        List<string> diagnostics = new List<string>();
        diagnostics.Add("EvidenceSidecar=" + fullPath);
        if (!string.IsNullOrWhiteSpace(proofClassification))
        {
            diagnostics.Add("EvidenceSidecarProofClassification=" + proofClassification);
        }

        if (string.Equals(proofClassification, "package-consumer-runtime", StringComparison.Ordinal))
        {
            diagnostics.Add("Evidence sidecar package-consumer-runtime is ignored by build reports; release proof records own that classification.");
        }

        if (string.Equals(proofClassification, "real-model-runtime", StringComparison.Ordinal) &&
            (!IsSha256(modelSha256) ||
             string.IsNullOrWhiteSpace(modelLicense) ||
             string.IsNullOrWhiteSpace(inputAssetName) ||
             !IsSha256(inputAssetSha256) ||
             (string.IsNullOrWhiteSpace(stdoutSummary) && string.IsNullOrWhiteSpace(stderrSummary))))
        {
            diagnostics.Add("Evidence sidecar real-model-runtime is incomplete and cannot promote sample manifest state.");
        }

        if (string.Equals(proofClassification, "real-model-runtime", StringComparison.Ordinal))
        {
            diagnostics.Add("Evidence sidecar real-model-runtime is recorded for sample-run evidence; TensorRtExec/OnnxToEngine build reports remain build/sample evidence.");
        }

        return new OnnxEngineBuildEvidenceSidecar(
            fullPath,
            modelSource,
            modelSha256,
            modelLicense,
            inputAssetName,
            inputAssetSha256,
            stdoutSummary,
            stderrSummary,
            proofClassification,
            diagnostics);
    }

    private static string GetString(JsonElement root, string propertyName)
    {
        return root.TryGetProperty(propertyName, out JsonElement value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? string.Empty
            : string.Empty;
    }

    private static string FirstNonEmpty(params string[] values)
    {
        foreach (string value in values)
        {
            if (!string.IsNullOrWhiteSpace(value))
            {
                return value;
            }
        }

        return string.Empty;
    }

    private static bool IsSha256(string value)
    {
        if (value.Length != 64)
        {
            return false;
        }

        foreach (char item in value)
        {
            bool digit = item >= '0' && item <= '9';
            bool lower = item >= 'a' && item <= 'f';
            bool upper = item >= 'A' && item <= 'F';
            if (!digit && !lower && !upper)
            {
                return false;
            }
        }

        return true;
    }
}
