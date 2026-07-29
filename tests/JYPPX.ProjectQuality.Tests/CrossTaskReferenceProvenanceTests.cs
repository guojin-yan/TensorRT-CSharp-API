using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CrossTaskReferenceProvenanceTests
{
    [Fact]
    public void ExporterAndStrictValidatorProduceSevenOwnerActionRows()
    {
        string exporterOutput = RunPowerShell("eng", "Export-CrossTaskReferenceProvenanceMatrix.ps1");
        string validatorOutput = RunPowerShell("eng", "Test-CrossTaskReferenceProvenanceMatrix.ps1", "-Strict");

        Assert.Contains("Rows=7 Ready=0 OwnerActionRequired=7", exporterOutput, StringComparison.Ordinal);
        Assert.Contains("MnistReferenceReusableForMatrixTasks=False", exporterOutput, StringComparison.Ordinal);
        Assert.Contains("CrossTaskReferenceProvenanceValidation=93/93", validatorOutput, StringComparison.Ordinal);

        using JsonDocument validation = ReadJson(
            "artifacts",
            "interface-coverage",
            "cross-task-reference-provenance-validation.json");
        Assert.Equal("cross-task-reference-provenance-validation.v1", validation.RootElement.GetProperty("schemaVersion").GetString());
        Assert.True(validation.RootElement.GetProperty("strict").GetBoolean());
        Assert.Equal(93, validation.RootElement.GetProperty("checkCount").GetInt32());
        Assert.Equal(93, validation.RootElement.GetProperty("passedCount").GetInt32());
        Assert.Equal(0, validation.RootElement.GetProperty("failureCount").GetInt32());
    }

    [Fact]
    public void ContractSeparatesGenericClassificationFromEveryYoloTaskSemanticProfile()
    {
        using JsonDocument contract = ReadJson(
            "samples",
            "assets",
            "cross-task-reference-provenance-contract.json");
        JsonElement root = contract.RootElement;
        JsonElement[] profiles = root.GetProperty("taskProfiles").EnumerateArray().ToArray();

        Assert.Equal("cross-task-reference-provenance-contract.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal(
            new[] { "classification", "yolo-det", "yolo-cls", "yolo-seg", "yolo-obb", "yolo-pose", "yolo-sem" },
            profiles.Select(static item => item.GetProperty("id").GetString()).ToArray());
        Assert.Equal(6, root.GetProperty("reuseFingerprintFields").GetArrayLength());
        Assert.Contains(
            root.GetProperty("reuseRules").EnumerateArray().Select(static item => item.GetString()!),
            static rule => rule.Contains("same-runtime", StringComparison.Ordinal));

        JsonElement classification = Assert.Single(profiles, static item => item.GetProperty("id").GetString() == "classification");
        JsonElement yoloClassification = Assert.Single(profiles, static item => item.GetProperty("id").GetString() == "yolo-cls");
        string[] classificationFields = classification.GetProperty("requiredSemanticFields").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        string[] yoloFields = yoloClassification.GetProperty("requiredSemanticFields").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("imageResizePolicy", classificationFields);
        Assert.Contains("mean", classificationFields);
        Assert.Contains("outputValueKind", classificationFields);
        Assert.Contains("classScoreField", yoloFields);
        Assert.Contains("softmaxApplied", yoloFields);
        Assert.DoesNotContain("imageResizePolicy", yoloFields);
    }

    [Fact]
    public void MatrixRejectsMnistReuseAndKeepsCurrentSemanticGapsExplicit()
    {
        using JsonDocument matrix = ReadJson(
            "artifacts",
            "interface-coverage",
            "cross-task-reference-provenance-matrix.json");
        JsonElement root = matrix.RootElement;
        JsonElement[] rows = root.GetProperty("rows").EnumerateArray().ToArray();
        JsonElement candidate = Assert.Single(root.GetProperty("independentReferenceCandidates").EnumerateArray());

        Assert.Equal(7, root.GetProperty("rowCount").GetInt32());
        Assert.Equal(0, root.GetProperty("readyRowCount").GetInt32());
        Assert.Equal(7, root.GetProperty("ownerActionRequiredRowCount").GetInt32());
        Assert.All(rows, static row =>
        {
            Assert.True(row.GetProperty("missingFieldCount").GetInt32() > 0);
            Assert.False(row.GetProperty("referenceReuseEligible").GetBoolean());
            Assert.False(row.GetProperty("ownerReviewedGolden").GetBoolean());
            Assert.False(row.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.False(row.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        });

        JsonElement classification = Assert.Single(rows, static item => item.GetProperty("id").GetString() == "classification");
        string[] missingClassification = classification.GetProperty("taskSemanticChecks").EnumerateArray()
            .Where(static item => !item.GetProperty("ready").GetBoolean())
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        Assert.Contains("scoreTransform", missingClassification);
        Assert.Contains("argmaxRule", missingClassification);
        Assert.Contains("labelMappingSha256", missingClassification);

        JsonElement detection = Assert.Single(rows, static item => item.GetProperty("id").GetString() == "yolo-det");
        string[] missingDetection = detection.GetProperty("taskSemanticChecks").EnumerateArray()
            .Where(static item => !item.GetProperty("ready").GetBoolean())
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        Assert.Contains("boxFormat", missingDetection);
        Assert.Contains("scoreRule", missingDetection);
        Assert.Contains("sourceImageInversePolicy", missingDetection);

        Assert.Equal("mnist-onnxruntime-cpu-1.23.2", candidate.GetProperty("id").GetString());
        Assert.Equal(new[] { "mnist" }, candidate.GetProperty("eligibleTaskIds").EnumerateArray().Select(static item => item.GetString()).ToArray());
        Assert.Equal(rows.Select(static row => row.GetProperty("id").GetString()).ToArray(), candidate.GetProperty("ineligibleMatrixTaskIds").EnumerateArray().Select(static item => item.GetString()).ToArray());
        Assert.False(candidate.GetProperty("allReuseFingerprintsRecorded").GetBoolean());
        Assert.False(candidate.GetProperty("ownerReviewedGolden").GetBoolean());
        Assert.False(candidate.GetProperty("canReuseForAnyMatrixTask").GetBoolean());

        JsonElement boundary = root.GetProperty("proofBoundary");
        Assert.False(boundary.GetProperty("crossTaskReferenceReuseProved").GetBoolean());
        Assert.False(boundary.GetProperty("publicPackageProof").GetBoolean());
        Assert.False(boundary.GetProperty("postPublishProof").GetBoolean());
        Assert.False(boundary.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(boundary.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    [Fact]
    public void SampleDocumentationNamesFingerprintAndTaskSemanticBoundaries()
    {
        string classification = File.ReadAllText(RepositoryPath("samples", "Classification", "README.md"));
        string yolo = File.ReadAllText(RepositoryPath("samples", "YoloVision", "README.md"));
        string assets = File.ReadAllText(RepositoryPath("samples", "assets", "README.md"));

        Assert.Contains("cross-task-reference-provenance-contract.json", classification, StringComparison.Ordinal);
        Assert.Contains("raw logits or probabilities", classification, StringComparison.Ordinal);
        Assert.Contains("model/input/preprocess/output/labels/task-semantics fingerprints", classification, StringComparison.Ordinal);
        Assert.Contains("cross-task-reference-provenance-contract.json", yolo, StringComparison.Ordinal);
        Assert.Contains("`det`, `cls`, `seg`, `obb`, `pose`, and `sem`", yolo, StringComparison.Ordinal);
        Assert.Contains("eligible only for its MNIST task", yolo, StringComparison.Ordinal);
        Assert.Contains("all six reuse fingerprints match", assets, StringComparison.Ordinal);
        Assert.Contains("Test-CrossTaskReferenceProvenanceMatrix.ps1 -Strict", assets, StringComparison.Ordinal);
    }

    private static JsonDocument ReadJson(params string[] parts)
    {
        string path = RepositoryPath(parts);
        Assert.True(File.Exists(path), "Required JSON file is missing: " + path);
        return JsonDocument.Parse(File.ReadAllText(path));
    }

    private static string RepositoryPath(params string[] parts)
    {
        return Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray());
    }

    private static string RunPowerShell(params string[] arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = "pwsh",
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        startInfo.ArgumentList.Add("-NoProfile");
        startInfo.ArgumentList.Add("-ExecutionPolicy");
        startInfo.ArgumentList.Add("Bypass");
        startInfo.ArgumentList.Add("-File");
        startInfo.ArgumentList.Add(RepositoryPath(arguments[0], arguments[1]));
        foreach (string argument in arguments.Skip(2))
        {
            startInfo.ArgumentList.Add(argument);
        }

        using Process process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Failed to start PowerShell.");
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        Assert.True(process.WaitForExit(120_000), $"PowerShell timed out.{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        Assert.Equal(0, process.ExitCode);
        Assert.True(string.IsNullOrWhiteSpace(stderr), stderr);
        return stdout;
    }
}
