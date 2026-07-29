using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TensorRtExecMnistReferenceEvidenceTests
{
    [Fact]
    public void CompactEvidenceKeepsMnistReferenceAndPromotionBoundariesExplicit()
    {
        using JsonDocument document = ReadJson(
            "artifacts",
            "interface-coverage",
            "tensorrtexec-mnist-reference-validation-evidence.json");
        JsonElement root = document.RootElement;
        JsonElement reference = root.GetProperty("reference");
        JsonElement owner = root.GetProperty("ownerReview");
        JsonElement boundary = root.GetProperty("proofBoundary");

        Assert.Equal("tensorrtexec-mnist-reference-validation-evidence.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("real-model-reference-candidate-runtime", root.GetProperty("evidenceClassification").GetString());
        Assert.Equal("owner-review-required", root.GetProperty("model").GetProperty("licenseReviewState").GetString());
        Assert.False(root.GetProperty("model").GetProperty("redistributionApproved").GetBoolean());
        Assert.Equal("Plus214_Output_0", reference.GetProperty("tensorName").GetString());
        Assert.Equal(new[] { 1, 10 }, reference.GetProperty("shape").EnumerateArray().Select(static item => item.GetInt32()).ToArray());
        Assert.Equal(10, reference.GetProperty("elementCount").GetInt32());
        Assert.Equal("repository-mnist-runtime-output-derived-unreviewed", reference.GetProperty("sourceClassification").GetString());
        Assert.False(reference.GetProperty("independentFrameworkGolden").GetBoolean());
        Assert.False(reference.GetProperty("ownerReviewedGolden").GetBoolean());
        Assert.False(owner.GetProperty("acceptedAsGoldenReference").GetBoolean());
        Assert.False(owner.GetProperty("acceptedForRepositoryRedistribution").GetBoolean());
        Assert.False(owner.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.True(boundary.GetProperty("usesExistingRealOnnxModel").GetBoolean());
        Assert.True(boundary.GetProperty("provesStructuredReferenceRegressionAcrossBuildLoadAndLocalPackageConsumer").GetBoolean());
        Assert.False(boundary.GetProperty("isIndependentNumericalGoldenProof").GetBoolean());
        Assert.False(boundary.GetProperty("isOwnerAcceptedRealModelProof").GetBoolean());
        Assert.False(boundary.GetProperty("isPublicPackageConsumerProof").GetBoolean());
        Assert.False(boundary.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(boundary.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(boundary.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    [Fact]
    public void BuildLoadAndLocalConsumerAllRecordPassingReferenceComparison()
    {
        using JsonDocument document = ReadJson(
            "artifacts",
            "interface-coverage",
            "tensorrtexec-mnist-reference-validation-evidence.json");
        JsonElement root = document.RootElement;
        JsonElement build = root.GetProperty("sourceTreeBuild");
        JsonElement load = root.GetProperty("independentLoadEngine");
        JsonElement consumer = root.GetProperty("localPackageConsumer");
        string referenceSha256 = root.GetProperty("reference").GetProperty("sha256").GetString()!;

        AssertRun(build, "external-onnx-reference-validated-runtime");
        AssertRun(load, "load-engine-reference-validated-runtime");
        Assert.Equal(build.GetProperty("engineSha256").GetString(), load.GetProperty("engineSha256").GetString());
        Assert.Equal(build.GetProperty("rawOutputSha256").GetString(), load.GetProperty("rawOutputSha256").GetString());
        Assert.True(consumer.GetProperty("packageReferenceOnly").GetBoolean());
        Assert.False(consumer.GetProperty("projectReference").GetBoolean());
        Assert.False(consumer.GetProperty("publicFeedEnabled").GetBoolean());
        Assert.True(consumer.GetProperty("referenceValidationCompleted").GetBoolean());
        Assert.True(consumer.GetProperty("referenceValidationPassed").GetBoolean());
        Assert.Equal(10, consumer.GetProperty("referenceComparedElementCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("referenceMismatchCount").GetInt32());
        Assert.Equal(-1, consumer.GetProperty("referenceFirstMismatchIndex").GetInt32());
        Assert.Equal(referenceSha256, consumer.GetProperty("referenceSha256").GetString());
        Assert.False(consumer.GetProperty("isPublicPackageProof").GetBoolean());
    }

    [Fact]
    public void ReferenceAndStrictValidationArtifactsAreCheckedIn()
    {
        string referencePath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "real-case",
            "onnx-to-engine-mnist-trt10-runtime",
            "digit-7",
            "mnist-trt10-7.reference.json");
        string sidecarPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "real-case",
            "onnx-to-engine-mnist-trt10-runtime",
            "digit-7",
            "mnist-trt10-7.reference.sidecar.json");
        string validatorPath = Path.Combine(RepositoryPaths.Root, "eng", "Test-TensorRtExecMnistReferenceEvidence.ps1");

        using JsonDocument reference = JsonDocument.Parse(File.ReadAllText(referencePath));
        using JsonDocument sidecar = JsonDocument.Parse(File.ReadAllText(sidecarPath));
        using JsonDocument validation = ReadJson(
            "artifacts",
            "interface-coverage",
            "tensorrtexec-mnist-reference-validation.json");

        Assert.Equal(1, reference.RootElement.GetProperty("schemaVersion").GetInt32());
        Assert.Equal(10, reference.RootElement.GetProperty("values").GetArrayLength());
        Assert.Equal("repository-mnist-runtime-output-derived-unreviewed", reference.RootElement.GetProperty("sourceClassification").GetString());
        Assert.Equal("not-provided", sidecar.RootElement.GetProperty("ownerReview").GetProperty("status").GetString());
        Assert.False(sidecar.RootElement.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.Equal("tensorrtexec-mnist-reference-validation.v1", validation.RootElement.GetProperty("schemaVersion").GetString());
        Assert.True(validation.RootElement.GetProperty("strict").GetBoolean());
        Assert.True(validation.RootElement.GetProperty("runtimeArtifactChecksRequired").GetBoolean());
        Assert.Equal(validation.RootElement.GetProperty("checkCount").GetInt32(), validation.RootElement.GetProperty("passedCount").GetInt32());
        Assert.Equal(0, validation.RootElement.GetProperty("failureCount").GetInt32());

        string validator = File.ReadAllText(validatorPath);
        Assert.Contains("RequireRuntimeArtifacts", validator, StringComparison.Ordinal);
        Assert.Contains("owner-review-required", validator, StringComparison.Ordinal);
        Assert.Contains("isIndependentNumericalGoldenProof", validator, StringComparison.Ordinal);
        Assert.Contains("package-evidence-cross-check", validator, StringComparison.Ordinal);
        Assert.Equal("07969a96f76f9dc69a777770ca581df64ad9cb2d1ea320f97bdfbb7cc697beef", ComputeSha256(referencePath));
    }

    private static void AssertRun(JsonElement run, string expectedState)
    {
        Assert.Equal(expectedState, run.GetProperty("state").GetString());
        Assert.True(run.GetProperty("success").GetBoolean());
        Assert.True(run.GetProperty("inferenceRan").GetBoolean());
        Assert.True(run.GetProperty("outputMatch").GetBoolean());
        Assert.True(run.GetProperty("outputValidated").GetBoolean());
        Assert.False(run.GetProperty("identityOutputMatch").GetBoolean());
        Assert.Equal("synthetic-input-runtime", run.GetProperty("genericToolProofClassification").GetString());
        Assert.Equal(10, run.GetProperty("comparedElementCount").GetInt32());
        Assert.Equal(0, run.GetProperty("mismatchCount").GetInt32());
        Assert.Equal(-1, run.GetProperty("firstMismatchIndex").GetInt32());
    }

    private static JsonDocument ReadJson(params string[] parts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray());
        Assert.True(File.Exists(path), "Required JSON file is missing: " + path);
        return JsonDocument.Parse(File.ReadAllText(path));
    }

    private static string ComputeSha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }
}
