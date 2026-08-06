using System.Security.Cryptography;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class InferenceBindingsTechnicalArticleTests
{
    [Fact]
    public void ArticleUsesARealRuntimeScreenshotAndAvoidsMachineSpecificPaths()
    {
        string article = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "inference-bindings-tutorial.md"));

        Assert.Contains("本文使用的项目与库", article, StringComparison.Ordinal);
        Assert.Contains("没有使用深度学习模型，也不需要 ONNX 文件", article, StringComparison.Ordinal);
        Assert.Contains("终端截图来自本次真实运行的 stdout", article, StringComparison.Ordinal);
        Assert.Contains("../../images/inference-bindings-runtime-terminal.png", article, StringComparison.Ordinal);
        Assert.Contains("ElapsedMs=0.66 OutputMatch=True", article, StringComparison.Ordinal);
        Assert.Contains("ProcessExitCode=0", article, StringComparison.Ordinal);
        Assert.Contains("samples/assets/inference-bindings-article-runtime-evidence.json", article, StringComparison.Ordinal);
        Assert.DoesNotContain(@"E:\", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(@"C:\Users\", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("dotnet nuget push", article, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void RuntimeEvidenceMatchesTrackedSourceAndScreenshot()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "inference-bindings-article-runtime-evidence.json");
        Assert.True(File.Exists(evidencePath));

        using JsonDocument evidence = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = evidence.RootElement;
        JsonElement assets = root.GetProperty("assets");
        JsonElement validation = root.GetProperty("runtimeValidation");
        JsonElement maintenance = root.GetProperty("maintenanceValidation");
        JsonElement boundary = root.GetProperty("proofBoundary");

        Assert.Equal("inference-bindings-technical-article-runtime-evidence", root.GetProperty("recordKind").GetString());
        Assert.False(root.GetProperty("network").GetProperty("modelOrOnnxRequired").GetBoolean());
        Assert.True(validation.GetProperty("bindingReady").GetBoolean());
        Assert.True(validation.GetProperty("readinessReady").GetBoolean());
        Assert.True(validation.GetProperty("allTensorAddressesBound").GetBoolean());
        Assert.True(validation.GetProperty("outputMatch").GetBoolean());
        Assert.Equal(0, validation.GetProperty("processExitCode").GetInt32());
        Assert.False(boundary.GetProperty("performsPublish").GetBoolean());
        Assert.False(boundary.GetProperty("uploadsVendorRuntime").GetBoolean());

        string sourcePath = Path.Combine(RepositoryPaths.Root, "samples", "Inference", "01.Bindings", "Program.cs");
        string screenshotPath = Path.Combine(
            RepositoryPaths.Root,
            assets.GetProperty("runtimeScreenshotPath").GetString()!.Replace('/', Path.DirectorySeparatorChar));
        Assert.Equal(maintenance.GetProperty("currentSourceSha256").GetString(), ComputeSha256(sourcePath));
        Assert.Equal(0, maintenance.GetProperty("processExitCode").GetInt32());
        Assert.True(maintenance.GetProperty("outputMatch").GetBoolean());
        Assert.False(maintenance.GetProperty("runtimeScreenshotRecaptured").GetBoolean());
        Assert.True(maintenance.GetProperty("historicalRuntimeScreenshotRetained").GetBoolean());
        Assert.Equal(assets.GetProperty("runtimeScreenshotSha256").GetString(), ComputeSha256(screenshotPath));

        string sampleReadme = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "samples", "Inference", "01.Bindings",
            "README.md"));
        Assert.Contains("inference-bindings-tutorial.md", sampleReadme, StringComparison.Ordinal);
    }

    private static string ComputeSha256(string path)
    {
        return Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
    }
}
