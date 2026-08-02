using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ClassificationResNet18RealModelTests
{
    [Fact]
    public void OfficialManifestPinsReferenceContractsWithoutPublishingAssets()
    {
        using JsonDocument document = ReadJson("samples", "assets", "classification-resnet18-official-assets.json");
        JsonElement root = document.RootElement;
        JsonElement reference = root.GetProperty("referenceContract");

        Assert.Equal("v0.25.0", root.GetProperty("upstreamSourceTag").GetString());
        Assert.Equal(
            "ead3558569edd88aa73a4eb46acbe6c38dee113933234547f04a0f6e48169903",
            root.GetProperty("exportContract").GetProperty("onnxSha256").GetString());
        Assert.Equal("eng/Invoke-ClassificationResNet18Reference.py", reference.GetProperty("script").GetString());
        Assert.Equal("logits", reference.GetProperty("rawTensorName").GetString());
        Assert.Equal(new[] { 1, 1000 }, reference.GetProperty("rawTensorShape").EnumerateArray().Select(item => item.GetInt32()));
        Assert.Equal("probabilities", reference.GetProperty("taskValueKind").GetString());
        Assert.Equal(64, reference.GetProperty("preprocessContractSha256").GetString()!.Length);
        Assert.False(root.GetProperty("proofBoundary").GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("uploadsAssets").GetBoolean());
    }

    [Fact]
    public void RuntimeEvidenceClosesIndependentRawTaskTopKAndNegativeValidation()
    {
        using JsonDocument document = ReadJson(
            "samples",
            "assets",
            "classification-resnet18-real-model-runtime-evidence.json");
        JsonElement root = document.RootElement;
        JsonElement independent = root.GetProperty("independentReferenceValidation");
        JsonElement runtime = root.GetProperty("runtimeValidation");
        JsonElement raw = runtime.GetProperty("rawLogits");
        JsonElement task = runtime.GetProperty("taskProbabilities");
        JsonElement negative = root.GetProperty("controlledNegativeValidation");

        Assert.Equal("Classification", root.GetProperty("sampleName").GetString());
        Assert.Equal("real-model-runtime", root.GetProperty("proofClassification").GetString());
        Assert.Equal("real-model-runtime", root.GetProperty("validatorState").GetString());
        Assert.True(root.GetProperty("isSmokePassed").GetBoolean());
        Assert.True(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.Equal(64, root.GetProperty("modelSha256").GetString()!.Length);
        Assert.Equal(64, root.GetProperty("labelsSha256").GetString()!.Length);
        Assert.Equal(64, root.GetProperty("inputAssetSha256").GetString()!.Length);
        Assert.Equal(64, root.GetProperty("preprocessedInputTensorSha256").GetString()!.Length);
        Assert.Equal(150528, root.GetProperty("preprocessedInputTensorElementCount").GetInt32());
        Assert.False(string.IsNullOrWhiteSpace(root.GetProperty("evidenceSidecarPath").GetString()));
        Assert.False(string.IsNullOrWhiteSpace(root.GetProperty("buildReportPath").GetString()));
        Assert.False(string.IsNullOrWhiteSpace(root.GetProperty("sampleRunCommand").GetString()));
        Assert.Equal(64, root.GetProperty("sampleRunLogSha256").GetString()!.Length);
        Assert.Contains(
            root.GetProperty("expectedEvidenceLines").EnumerateArray(),
            item => item.GetString() == "Classification Passed=True");
        Assert.True(independent.GetProperty("passed").GetBoolean());
        Assert.Equal(1000, independent.GetProperty("comparedLogitCount").GetInt32());
        Assert.True(independent.GetProperty("sameArgmax").GetBoolean());
        Assert.True(runtime.GetProperty("tf32Disabled").GetBoolean());
        Assert.True(runtime.GetProperty("outputValidated").GetBoolean());
        Assert.True(raw.GetProperty("passed").GetBoolean());
        Assert.Equal(1000, raw.GetProperty("comparedValueCount").GetInt32());
        Assert.Equal(0, raw.GetProperty("mismatchCount").GetInt32());
        Assert.True(task.GetProperty("passed").GetBoolean());
        Assert.Equal(1000, task.GetProperty("comparedValueCount").GetInt32());
        Assert.Equal(0, task.GetProperty("mismatchCount").GetInt32());
        Assert.Equal("Samoyed", root.GetProperty("top5Validation").GetProperty("predictions")[0].GetProperty("className").GetString());
        Assert.Equal(1, negative.GetProperty("exitCode").GetInt32());
        Assert.Equal(1, negative.GetProperty("taskMismatchCount").GetInt32());
        Assert.Equal(0, negative.GetProperty("taskFirstMismatchIndex").GetInt32());
        Assert.True(negative.GetProperty("rawReferenceStillPassed").GetBoolean());
        Assert.True(negative.GetProperty("failClosed").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("publicWeightRedistributionApproved").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("packageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("uploadsAssets").GetBoolean());
    }

    [Fact]
    public void ReferenceGeneratorUsesExactCSharpTensorAndWritesPositiveAndTamperedReferences()
    {
        string script = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Invoke-ClassificationResNet18Reference.py"));

        Assert.Contains("EXPECTED_INPUT_SHA256", script, StringComparison.Ordinal);
        Assert.Contains("CPUExecutionProvider", script, StringComparison.Ordinal);
        Assert.Contains("pytorchOnnxRuntimeComparison", script, StringComparison.Ordinal);
        Assert.Contains("classification.onnxruntime.reference.json", script, StringComparison.Ordinal);
        Assert.Contains("classification.onnxruntime.tampered.reference.json", script, StringComparison.Ordinal);
        Assert.Contains("[0] = float", script, StringComparison.Ordinal);
        Assert.Contains("+ 0.125", script, StringComparison.Ordinal);
        Assert.DoesNotContain("github upload", script, StringComparison.OrdinalIgnoreCase);
    }

    private static JsonDocument ReadJson(params string[] parts)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            new[] { RepositoryPaths.Root }.Concat(parts).ToArray())));
    }
}
