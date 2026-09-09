using System.Security.Cryptography;
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
            "applications", "OnnxToEngine",
            "README.md"));

        Assert.Contains("<workspace>\\models\\OnnxToEngine\\MNIST", readme, StringComparison.Ordinal);
        Assert.DoesNotContain("E:\\GitSpace", readme, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("third_party", readme, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("onnxtoengine-mnist-real-model-runtime-evidence.json", readme, StringComparison.Ordinal);
        Assert.Contains("--expectedDigit", readme, StringComparison.Ordinal);
        Assert.Contains("State=mnist-output-mismatch", readme, StringComparison.Ordinal);
        Assert.Contains("not package-consumer", readme, StringComparison.Ordinal);
    }

    [Fact]
    public void TechnicalArticleEvidencePinsHistoricalRuntimeSourceAndScreenshot()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "onnxtoengine-mnist-article-runtime-evidence.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = document.RootElement;
        JsonElement assets = root.GetProperty("assets");
        JsonElement runtime = root.GetProperty("runtimeValidation");
        JsonElement boundary = root.GetProperty("proofBoundary");

        Assert.Equal("onnxtoengine-mnist-technical-article-runtime-evidence", root.GetProperty("recordKind").GetString());
        Assert.Equal(7, runtime.GetProperty("predictedDigit").GetInt32());
        Assert.True(runtime.GetProperty("outputMatch").GetBoolean());
        Assert.Equal(0, runtime.GetProperty("processExitCode").GetInt32());
        Assert.False(boundary.GetProperty("modelRedistributionApproved").GetBoolean());
        Assert.False(boundary.GetProperty("inputAssetRedistributionApproved").GetBoolean());
        Assert.False(boundary.GetProperty("performsPublish").GetBoolean());

        string servicePath = Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp.Tools",
            "Runtime",
            "MnistOnnxRuntimeService.cs");
        string screenshotPath = Path.Combine(
            RepositoryPaths.Root,
            assets.GetProperty("runtimeScreenshotPath").GetString()!.Replace('/', Path.DirectorySeparatorChar));
        Assert.Matches("^[a-f0-9]{40}$", root.GetProperty("sourceBaseCommit").GetString()!);
        Assert.Matches("^[a-f0-9]{64}$", assets.GetProperty("runtimeServiceSha256").GetString()!);
        Assert.Equal(assets.GetProperty("runtimeScreenshotSha256").GetString(), ComputeSha256(screenshotPath));

        string service = File.ReadAllText(servicePath);
        Assert.Contains("Model={Path.GetFileName(modelPath)}", service, StringComparison.Ordinal);
        Assert.Contains("Path={Path.GetFileName(enginePath)}", service, StringComparison.Ordinal);

        string article = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "onnx-to-engine-quickstart.md"));
        Assert.Contains("onnx-to-engine-mnist-runtime-terminal.png", article, StringComparison.Ordinal);
        Assert.Contains("Predicted=7 Confidence=0.999993", article, StringComparison.Ordinal);
        Assert.DoesNotContain(@"E:\", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(@"C:\Users\", article, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void OwnerGeneratedArticlePinsRuntimeReferenceNegativeAndVisuals()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "onnxtoengine-mnist-owner-generated-runtime-evidence.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = document.RootElement;
        JsonElement runtime = root.GetProperty("runtimeValidation");
        JsonElement reference = root.GetProperty("independentReferenceValidation");
        JsonElement negative = root.GetProperty("controlledNegative");
        JsonElement visuals = root.GetProperty("articleVisuals");

        Assert.Equal("real-model-runtime", root.GetProperty("proofClassification").GetString());
        Assert.Equal("project-generated-deterministic-geometry", root.GetProperty("inputAsset").GetProperty("sourceClassification").GetString());
        Assert.Equal("CC0-1.0", root.GetProperty("inputAsset").GetProperty("license").GetString());
        Assert.Equal(7, runtime.GetProperty("predictedDigit").GetInt32());
        Assert.True(runtime.GetProperty("outputMatch").GetBoolean());
        Assert.Equal(0, reference.GetProperty("mismatchCount").GetInt32());
        Assert.True(reference.GetProperty("passed").GetBoolean());
        Assert.Equal(2, negative.GetProperty("exitCode").GetInt32());
        Assert.False(negative.GetProperty("outputMatch").GetBoolean());
        Assert.True(negative.GetProperty("failClosed").GetBoolean());

        Assert.Equal(
            root.GetProperty("sourceChangesIncluded").GetProperty("generatorSha256").GetString(),
            ComputeSha256(Path.Combine(RepositoryPaths.Root, "eng", "New-MnistOwnerGeneratedDigit.ps1")));
        Assert.Equal(
            root.GetProperty("sourceChangesIncluded").GetProperty("visualizationWriterSha256").GetString(),
            ComputeSha256(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "Runtime", "MnistVisualizationWriter.cs")));
        JsonElement maintenance = root.GetProperty("currentMaintenanceValidation");
        Assert.Equal(
            maintenance.GetProperty("currentSampleProgramSha256").GetString(),
            ComputeSha256(Path.Combine(RepositoryPaths.Root, "applications", "OnnxToEngine", "Program.cs")));
        Assert.False(maintenance.GetProperty("gpuRuntimeScenarioRerun").GetBoolean());
        Assert.True(maintenance.GetProperty("historicalRuntimeEvidenceRetained").GetBoolean());
        Assert.Equal(
            visuals.GetProperty("annotatedResultSha256").GetString(),
            ComputeSha256(Path.Combine(RepositoryPaths.Root, visuals.GetProperty("annotatedResult").GetString()!.Replace('/', Path.DirectorySeparatorChar))));
        Assert.Equal(
            visuals.GetProperty("runtimeTerminalSha256").GetString(),
            ComputeSha256(Path.Combine(RepositoryPaths.Root, visuals.GetProperty("runtimeTerminal").GetString()!.Replace('/', Path.DirectorySeparatorChar))));

        string article = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "onnxtoengine-mnist-owner-generated-tutorial.md"));
        Assert.Contains("模型获取与许可证", article, StringComparison.Ordinal);
        Assert.Contains("ONNX 转换与暂存", article, StringComparison.Ordinal);
        Assert.Contains("终端截图来自本次真实运行的 stdout", article, StringComparison.Ordinal);
        Assert.Contains("两张图都来自同一次真实 TensorRT 执行", article, StringComparison.Ordinal);
        Assert.DoesNotMatch("[A-Za-z]:\\\\", article);
    }

    private static string ComputeSha256(string path)
    {
        return Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
    }
}
