using System.Security.Cryptography;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TechnicalArticleCampaignSecondBatchBodyTests
{
    [Fact]
    public void CurrentRuntimeTutorialsAreLinkedAndAvoidMachineSpecificPaths()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        foreach ((string articleFile, string imageFile, string sampleReadme) in new[]
        {
            ("dynamic-shape-optimization-profile-tutorial.md", "dynamic-shape-runtime-terminal.png", "samples/Inference/02.DynamicShapes/README.md"),
            ("cuda-stream-event-multistream-tutorial.md", "cuda-multistream-runtime-terminal.png", "samples/Performance/01.MultiStream/README.md")
        })
        {
            string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile));
            string href = "articles/zh-cn/" + articleFile;

            Assert.Contains(href, docsIndex, StringComparison.Ordinal);
            Assert.Contains(href, docsToc, StringComparison.Ordinal);
            Assert.Contains("本文使用的项目与库", article, StringComparison.Ordinal);
            Assert.Contains(imageFile, article, StringComparison.Ordinal);
            Assert.Contains("ProcessExitCode=0", article, StringComparison.Ordinal);
            Assert.DoesNotContain(@"E:\", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain(@"C:\Users\", article, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(articleFile, File.ReadAllText(Path.Combine(
                RepositoryPaths.Root,
                sampleReadme.Replace('/', Path.DirectorySeparatorChar))), StringComparison.Ordinal);
        }
    }

    [Fact]
    public void DynamicShapeArticleEvidenceMatchesTrackedSourceAndScreenshot()
    {
        string evidencePath = Path.Combine(RepositoryPaths.Root, "samples", "assets", "dynamic-shape-article-runtime-evidence.json");
        using JsonDocument evidence = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = evidence.RootElement;
        JsonElement assets = root.GetProperty("assets");
        JsonElement validation = root.GetProperty("runtimeValidation");
        JsonElement maintenance = root.GetProperty("maintenanceValidation");

        Assert.Equal("dynamic-shape-technical-article-runtime-evidence", root.GetProperty("recordKind").GetString());
        Assert.False(root.GetProperty("network").GetProperty("modelOrOnnxRequired").GetBoolean());
        Assert.True(validation.GetProperty("profileValid").GetBoolean());
        Assert.True(validation.GetProperty("allTensorAddressesBound").GetBoolean());
        Assert.True(validation.GetProperty("outputMatch").GetBoolean());
        Assert.Equal(0, validation.GetProperty("processExitCode").GetInt32());

        string sourcePath = Path.Combine(RepositoryPaths.Root, "samples", "Inference", "02.DynamicShapes", "Program.cs");
        string screenshotPath = Path.Combine(
            RepositoryPaths.Root,
            assets.GetProperty("runtimeScreenshotPath").GetString()!.Replace('/', Path.DirectorySeparatorChar));
        Assert.Equal(maintenance.GetProperty("currentSourceSha256").GetString(), ComputeSha256(sourcePath));
        Assert.Equal(assets.GetProperty("runtimeScreenshotSha256").GetString(), ComputeSha256(screenshotPath));
        Assert.True(maintenance.GetProperty("historicalRuntimeScreenshotRetained").GetBoolean());
    }

    private static string ComputeSha256(string path)
    {
        return Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
    }
}
