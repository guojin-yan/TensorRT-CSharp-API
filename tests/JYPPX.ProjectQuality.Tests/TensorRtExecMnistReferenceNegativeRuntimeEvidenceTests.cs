using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TensorRtExecMnistReferenceNegativeRuntimeEvidenceTests
{
    [Fact]
    public void FiveControlledCasesFailClosedAfterSourceTreeAndPackageConsumerEnqueue()
    {
        using JsonDocument document = ReadJson(
            "artifacts",
            "interface-coverage",
            "tensorrtexec-mnist-reference-negative-runtime-evidence.json");
        JsonElement root = document.RootElement;
        JsonElement[] cases = root.GetProperty("cases").EnumerateArray().ToArray();

        Assert.Equal("tensorrtexec-mnist-reference-negative-runtime-evidence.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("controlled-negative-runtime", root.GetProperty("evidenceClassification").GetString());
        Assert.Equal(5, root.GetProperty("caseCount").GetInt32());
        Assert.Equal(5, root.GetProperty("sourceTreeFailClosedCount").GetInt32());
        Assert.Equal(5, root.GetProperty("localPackageConsumerFailClosedCount").GetInt32());
        Assert.Equal(
            new[] { "name-mismatch", "shape-mismatch", "value-count-mismatch", "nan-reject", "infinity-reject" },
            cases.Select(static item => item.GetProperty("id").GetString()).ToArray());

        foreach (JsonElement item in cases)
        {
            JsonElement source = item.GetProperty("sourceTree");
            JsonElement consumer = item.GetProperty("localPackageConsumer");
            Assert.Equal(2, source.GetProperty("exitCode").GetInt32());
            Assert.True(source.GetProperty("inferenceRan").GetBoolean());
            Assert.True(source.GetProperty("outputCaptureAvailable").GetBoolean());
            Assert.False(source.GetProperty("outputValidated").GetBoolean());
            Assert.False(source.GetProperty("validationPassed").GetBoolean());
            Assert.Equal(1, consumer.GetProperty("exitCode").GetInt32());
            Assert.True(consumer.GetProperty("enqueueCompleted").GetBoolean());
            Assert.True(consumer.GetProperty("ownerScopeExited").GetBoolean());
            Assert.False(consumer.GetProperty("outputValidated").GetBoolean());
            Assert.False(consumer.GetProperty("validationPassed").GetBoolean());
            Assert.Equal(source.GetProperty("outputSha256").GetString(), consumer.GetProperty("outputSha256").GetString());
            Assert.Equal(source.GetProperty("referenceSha256").GetString(), consumer.GetProperty("referenceSha256").GetString());
        }
    }

    [Fact]
    public void MetadataAndSpecialValueCasesRecordDifferentCompletionSemantics()
    {
        using JsonDocument document = ReadJson(
            "artifacts",
            "interface-coverage",
            "tensorrtexec-mnist-reference-negative-runtime-evidence.json");
        JsonElement[] cases = document.RootElement.GetProperty("cases").EnumerateArray().ToArray();

        foreach (JsonElement item in cases.Take(3))
        {
            Assert.False(item.GetProperty("sourceTree").GetProperty("validationCompleted").GetBoolean());
            Assert.False(item.GetProperty("localPackageConsumer").GetProperty("validationCompleted").GetBoolean());
            Assert.Equal(0, item.GetProperty("sourceTree").GetProperty("comparedElementCount").GetInt32());
        }
        foreach (JsonElement item in cases.Skip(3))
        {
            Assert.True(item.GetProperty("sourceTree").GetProperty("validationCompleted").GetBoolean());
            Assert.True(item.GetProperty("localPackageConsumer").GetProperty("validationCompleted").GetBoolean());
            Assert.Equal(10, item.GetProperty("sourceTree").GetProperty("comparedElementCount").GetInt32());
            Assert.Equal(1, item.GetProperty("sourceTree").GetProperty("mismatchCount").GetInt32());
            Assert.Equal(0, item.GetProperty("sourceTree").GetProperty("firstMismatchIndex").GetInt32());
        }
    }

    [Fact]
    public void PackageConsumerAndStrictValidatorKeepFailClosedAndProofBoundariesExplicit()
    {
        using JsonDocument evidence = ReadJson(
            "artifacts",
            "interface-coverage",
            "tensorrtexec-mnist-reference-negative-runtime-evidence.json");
        using JsonDocument validation = ReadJson(
            "artifacts",
            "interface-coverage",
            "tensorrtexec-mnist-reference-negative-runtime-validation.json");
        JsonElement consumer = evidence.RootElement.GetProperty("localPackageConsumer");
        JsonElement boundary = evidence.RootElement.GetProperty("proofBoundary");

        Assert.True(consumer.GetProperty("usesPackageReferenceOnly").GetBoolean());
        Assert.False(consumer.GetProperty("usesProjectReference").GetBoolean());
        Assert.True(consumer.GetProperty("remoteSourcesCleared").GetBoolean());
        Assert.True(consumer.GetProperty("isolatedRestoreCache").GetBoolean());
        Assert.True(consumer.GetProperty("workspaceRemovedAfterValidation").GetBoolean());
        Assert.True(boundary.GetProperty("provesRealEnqueueBeforeReferenceRejection").GetBoolean());
        Assert.False(boundary.GetProperty("ownerReviewedGolden").GetBoolean());
        Assert.False(boundary.GetProperty("publicPackageProof").GetBoolean());
        Assert.False(boundary.GetProperty("postPublishProof").GetBoolean());
        Assert.False(boundary.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(boundary.GetProperty("canCloseReleaseIssue").GetBoolean());

        Assert.Equal("tensorrtexec-mnist-reference-negative-runtime-validation.v1", validation.RootElement.GetProperty("schemaVersion").GetString());
        Assert.True(validation.RootElement.GetProperty("strict").GetBoolean());
        Assert.True(validation.RootElement.GetProperty("runtimeArtifactChecksRequired").GetBoolean());
        Assert.True(validation.RootElement.GetProperty("checkCount").GetInt32() >= 72);
        Assert.Equal(validation.RootElement.GetProperty("checkCount").GetInt32(), validation.RootElement.GetProperty("passedCount").GetInt32());
        Assert.Equal(0, validation.RootElement.GetProperty("failureCount").GetInt32());

        string program = File.ReadAllText(RepositoryPath("samples", "RefittedPlan.PackageConsumer", "Program.cs"));
        string runner = File.ReadAllText(RepositoryPath("eng", "Test-TensorRtExecMnistReferenceNegativeRuntime.ps1"));
        string validator = File.ReadAllText(RepositoryPath("eng", "Test-TensorRtExecMnistReferenceNegativeRuntimeEvidence.ps1"));
        Assert.Contains("OutputValidated=", program, StringComparison.Ordinal);
        Assert.Contains("MetadataMismatch", program, StringComparison.Ordinal);
        Assert.Contains("OwnerScopeExited=True", program, StringComparison.Ordinal);
        Assert.Contains("value-count-mismatch", runner, StringComparison.Ordinal);
        Assert.Contains("<clear />", runner, StringComparison.Ordinal);
        Assert.Contains("PackageReferenceOnly=", runner, StringComparison.Ordinal);
        Assert.Contains("RequireRuntimeArtifacts", validator, StringComparison.Ordinal);
        Assert.Contains("proof-promotion-boundary", validator, StringComparison.Ordinal);
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
}
