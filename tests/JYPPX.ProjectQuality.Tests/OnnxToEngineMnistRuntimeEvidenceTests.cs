using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OnnxToEngineMnistRuntimeEvidenceTests
{
    [Fact]
    public void EvidencePinsIndependentLogitsPositiveRuntimeAndControlledNegative()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "onnxtoengine-mnist-real-model-runtime-evidence.json")));
        JsonElement root = document.RootElement;
        JsonElement reference = root.GetProperty("independentReferenceValidation");
        JsonElement runtime = root.GetProperty("runtimeValidation");
        JsonElement negative = root.GetProperty("controlledNegativeValidation");

        Assert.Equal("OnnxToEngine", root.GetProperty("sampleName").GetString());
        Assert.Equal("real-model-runtime", root.GetProperty("proofClassification").GetString());
        Assert.Equal(
            "2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf",
            root.GetProperty("model").GetProperty("sha256").GetString());
        Assert.Equal(10, reference.GetProperty("comparedLogitCount").GetInt32());
        Assert.Equal(0, reference.GetProperty("mismatchCount").GetInt32());
        Assert.True(reference.GetProperty("sameArgmax").GetBoolean());
        Assert.True(reference.GetProperty("passed").GetBoolean());

        Assert.Equal(0, runtime.GetProperty("exitCode").GetInt32());
        Assert.Equal(7, runtime.GetProperty("expectedDigit").GetInt32());
        Assert.Equal(7, runtime.GetProperty("predictedDigit").GetInt32());
        Assert.True(runtime.GetProperty("outputMatch").GetBoolean());
        Assert.True(runtime.GetProperty("isRealModelRuntimeProof").GetBoolean());
        Assert.True(runtime.GetProperty("passed").GetBoolean());

        Assert.Equal(2, negative.GetProperty("exitCode").GetInt32());
        Assert.Equal(6, negative.GetProperty("expectedDigit").GetInt32());
        Assert.Equal(7, negative.GetProperty("predictedDigit").GetInt32());
        Assert.False(negative.GetProperty("outputMatch").GetBoolean());
        Assert.False(negative.GetProperty("isRealModelRuntimeProof").GetBoolean());
        Assert.True(negative.GetProperty("failClosed").GetBoolean());
        Assert.Equal(5, root.GetProperty("additionalReferenceNegativeEvidence").GetProperty("caseCount").GetInt32());

        JsonElement boundary = root.GetProperty("proofBoundary");
        Assert.False(boundary.GetProperty("ownerReviewedGolden").GetBoolean());
        Assert.False(boundary.GetProperty("modelRedistributionApproved").GetBoolean());
        Assert.False(boundary.GetProperty("packageConsumerRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("publicPackageProof").GetBoolean());
        Assert.False(boundary.GetProperty("postPublishProof").GetBoolean());
        Assert.False(boundary.GetProperty("uploadsAssets").GetBoolean());
        Assert.False(boundary.GetProperty("performsPublish").GetBoolean());
    }

    [Fact]
    public void OnnxToEngineDocumentationUsesOuterModelCacheAndNamesEvidenceBoundary()
    {
        string readme = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "OnnxToEngine",
            "README.md"));

        Assert.Contains("E:\\GitSpace\\TensorRT-CSharp-API-4.0\\models\\OnnxToEngine\\MNIST", readme, StringComparison.Ordinal);
        Assert.Contains("onnxtoengine-mnist-real-model-runtime-evidence.json", readme, StringComparison.Ordinal);
        Assert.Contains("--expectedDigit", readme, StringComparison.Ordinal);
        Assert.Contains("State=mnist-output-mismatch", readme, StringComparison.Ordinal);
        Assert.Contains("not package-consumer", readme, StringComparison.Ordinal);
    }
}
