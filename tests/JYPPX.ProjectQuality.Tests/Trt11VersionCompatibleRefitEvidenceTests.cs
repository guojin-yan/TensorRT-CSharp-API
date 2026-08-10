using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class Trt11VersionCompatibleRefitEvidenceTests
{
    [Fact]
    public void CompactEvidenceRecordsTwoProcessRuntimeWithoutPromotingReleaseProof()
    {
        using JsonDocument document = JsonDocument.Parse(ReadSource(
            "artifacts", "interface-coverage", "trt11-version-compatible-refit-runtime-evidence.json"));
        JsonElement root = document.RootElement;
        JsonElement sameProcess = root.GetProperty("processes").GetProperty("sameProcess");
        JsonElement secondProcess = root.GetProperty("processes").GetProperty("secondProcess");
        JsonElement boundary = root.GetProperty("proofBoundary");

        Assert.Equal("external-onnx-refit-reload-reference-validated-runtime", root.GetProperty("state").GetString());
        Assert.Equal("synthetic-input-runtime", root.GetProperty("proofClassification").GetString());
        Assert.Equal("11.0.0", root.GetProperty("tensorRtVersion").GetString());
        Assert.False(root.GetProperty("sourceTreeDirtyAtExecution").GetBoolean());
        Assert.Equal(0, sameProcess.GetProperty("exitCode").GetProperty("value").GetInt32());
        Assert.Equal(69, sameProcess.GetProperty("strictValidation").GetProperty("checkCount").GetInt32());
        Assert.Equal(69, sameProcess.GetProperty("strictValidation").GetProperty("passedCount").GetInt32());
        Assert.Equal(0, secondProcess.GetProperty("exitCode").GetProperty("value").GetInt32());
        Assert.Equal(69, secondProcess.GetProperty("strictValidation").GetProperty("checkCount").GetInt32());
        Assert.Equal(69, secondProcess.GetProperty("strictValidation").GetProperty("passedCount").GetInt32());
        Assert.Matches("^[0-9a-f]{64}$", root.GetProperty("outputSha256").GetString());
        Assert.True(boundary.GetProperty("isTrt11VersionCompatibleRefitRuntimeEvidence").GetBoolean());
        Assert.True(boundary.GetProperty("isBuilderAndEnginePolicyEvidence").GetBoolean());
        Assert.True(boundary.GetProperty("isReferenceValidatedSyntheticRuntime").GetBoolean());
        Assert.False(boundary.GetProperty("isCrossVersionLeanRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("isModelAccuracyProof").GetBoolean());
        Assert.False(boundary.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(boundary.GetProperty("canPublishPublicly").GetBoolean());
    }

    [Fact]
    public void StrictEvidenceValidationPassesEveryCheck()
    {
        using JsonDocument document = JsonDocument.Parse(ReadSource(
            "artifacts", "interface-coverage", "trt11-version-compatible-refit-runtime-validation.json"));
        JsonElement root = document.RootElement;

        Assert.Equal("passed", root.GetProperty("validationState").GetString());
        Assert.True(root.GetProperty("strict").GetBoolean());
        Assert.Equal(45, root.GetProperty("checkCount").GetInt32());
        Assert.Equal(45, root.GetProperty("passedCount").GetInt32());
        Assert.Equal(0, root.GetProperty("failureCount").GetInt32());
    }

    private static string ReadSource(params string[] parts)
    {
        string current = AppContext.BaseDirectory;
        while (!string.IsNullOrWhiteSpace(current))
        {
            string candidate = Path.Combine(new[] { current }.Concat(parts).ToArray());
            if (File.Exists(candidate))
            {
                return File.ReadAllText(candidate);
            }

            DirectoryInfo? parent = Directory.GetParent(current);
            if (parent == null)
            {
                break;
            }

            current = parent.FullName;
        }

        throw new FileNotFoundException($"Could not locate source file: {Path.Combine(parts)}");
    }
}
