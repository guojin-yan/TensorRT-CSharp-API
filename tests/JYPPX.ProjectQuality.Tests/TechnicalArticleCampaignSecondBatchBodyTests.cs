using System.Security.Cryptography;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TechnicalArticleCampaignSecondBatchBodyTests
{
    private static readonly string[] ArticleFiles =
    {
        "cuda-memory-wrapper.md",
        "cuda-stream-event-multistream-tutorial.md",
        "dynamic-shape-optimization-profile-tutorial.md",
        "onnx-parser-to-serialized-engine-tutorial.md",
        "network-layer-coverage-guide.md",
        "blog-refit-weights-guide.md",
        "tensorrtexec-cli-parameter-map.md",
        "yolovision-sample-overview.md",
        "yolo-vision-model-matrix.md",
        "local-nuget-feed-consumer.md",
        "linux-runner-evidence-checklist.md",
        "runtime-packages.md",
    };

    [Fact]
    public void SecondCampaignArticleBodyBatchExistsAndKeepsProofBoundaries()
    {
        Assert.InRange(ArticleFiles.Length, 10, 15);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string zhReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));

        foreach (string articleFile in ArticleFiles)
        {
            string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
            string href = "articles/zh-cn/" + articleFile;
            string readmePath = "docs/articles/zh-cn/" + articleFile;

            Assert.True(File.Exists(articlePath), articlePath);

            string article = File.ReadAllText(articlePath);
            if (articleFile == "dynamic-shape-optimization-profile-tutorial.md")
            {
                Assert.Contains("本文使用的项目与库", article, StringComparison.Ordinal);
                Assert.Contains("没有使用深度学习模型，也不需要 ONNX 文件", article, StringComparison.Ordinal);
                Assert.Contains("../../images/dynamic-shape-runtime-terminal.png", article, StringComparison.Ordinal);
                Assert.Contains("截图来自同一次真实运行的 stdout", article, StringComparison.Ordinal);
                Assert.Contains("ElapsedMs=0.629 OutputMatch=True", article, StringComparison.Ordinal);
                Assert.Contains("ProcessExitCode=0", article, StringComparison.Ordinal);
                Assert.Contains("samples/assets/dynamic-shape-article-runtime-evidence.json", article, StringComparison.Ordinal);
                Assert.DoesNotContain("第二批正文门禁", article, StringComparison.Ordinal);
                Assert.DoesNotContain(@"E:\", article, StringComparison.OrdinalIgnoreCase);
                Assert.DoesNotContain(@"C:\Users\", article, StringComparison.OrdinalIgnoreCase);
            }
            else if (articleFile == "cuda-stream-event-multistream-tutorial.md")
            {
                Assert.Contains("本文使用的项目与库", article, StringComparison.Ordinal);
                Assert.Contains("没有使用深度学习模型，也不需要 ONNX 文件", article, StringComparison.Ordinal);
                Assert.Contains("../../images/cuda-multistream-runtime-terminal.png", article, StringComparison.Ordinal);
                Assert.Contains("终端截图来自本次真实运行的 stdout", article, StringComparison.Ordinal);
                Assert.Contains("IndependentStreams=True A=True B=True Bytes=4096", article, StringComparison.Ordinal);
                Assert.Contains("CrossStreamWait=True", article, StringComparison.Ordinal);
                Assert.Contains("ProcessExitCode=0", article, StringComparison.Ordinal);
                Assert.Contains("samples/assets/cuda-multistream-article-runtime-evidence.json", article, StringComparison.Ordinal);
                Assert.DoesNotContain("第二批正文门禁", article, StringComparison.Ordinal);
                Assert.DoesNotContain(@"E:\", article, StringComparison.OrdinalIgnoreCase);
                Assert.DoesNotContain(@"C:\Users\", article, StringComparison.OrdinalIgnoreCase);
            }
            else
            {
                Assert.Contains("适用读者", article, StringComparison.Ordinal);
                Assert.Contains("解决问题", article, StringComparison.Ordinal);
                Assert.Contains("核心思路", article, StringComparison.Ordinal);
                Assert.True(
                    article.Contains("操作路径", StringComparison.Ordinal) ||
                    article.Contains("实现路径", StringComparison.Ordinal),
                    articleFile + " must contain 操作路径 or 实现路径.");
                Assert.Contains("边界说明", article, StringComparison.Ordinal);
                Assert.Contains("下一步", article, StringComparison.Ordinal);
                Assert.Contains("proof", article, StringComparison.OrdinalIgnoreCase);
                Assert.Contains("runtime proof", article, StringComparison.OrdinalIgnoreCase);
                Assert.Contains("build-only", article, StringComparison.OrdinalIgnoreCase);
                Assert.Contains("dry-run", article, StringComparison.OrdinalIgnoreCase);
                Assert.Contains("template", article, StringComparison.OrdinalIgnoreCase);
                Assert.Contains("local feed", article, StringComparison.OrdinalIgnoreCase);
                Assert.Contains("ProjectReference", article, StringComparison.Ordinal);
                Assert.Contains("direct `.nupkg`", article, StringComparison.Ordinal);
                Assert.Contains("TensorRtExec report", article, StringComparison.Ordinal);
                Assert.Contains("YoloVision matrix", article, StringComparison.Ordinal);
                Assert.Contains("OnnxToEngine report", article, StringComparison.Ordinal);
                Assert.Contains("readonly diagnostics", article, StringComparison.Ordinal);
            }
            Assert.DoesNotContain("dotnet nuget push", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("canPublishPublicly=true", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("canCloseReleaseIssue=true", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("canPromoteRuntimeProof=true", article, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("YoloDet", article, StringComparison.OrdinalIgnoreCase);

            Assert.Contains(href, docsIndex, StringComparison.Ordinal);
            Assert.Contains(href, docsToc, StringComparison.Ordinal);
            Assert.Contains(readmePath, readme, StringComparison.Ordinal);
            Assert.Contains(readmePath, zhReadme, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("YoloDet", docsIndex, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("YoloDet", docsToc, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void DynamicShapeArticleEvidenceMatchesTrackedSourceAndScreenshot()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "dynamic-shape-article-runtime-evidence.json");
        string articlePath = Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "dynamic-shape-optimization-profile-tutorial.md");
        string sampleReadmePath = Path.Combine(RepositoryPaths.Root, "samples", "Inference", "02.DynamicShapes", "README.md");

        Assert.True(File.Exists(evidencePath));
        using JsonDocument evidence = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = evidence.RootElement;
        JsonElement assets = root.GetProperty("assets");
        JsonElement validation = root.GetProperty("runtimeValidation");
        JsonElement maintenance = root.GetProperty("maintenanceValidation");

        Assert.Equal("dynamic-shape-technical-article-runtime-evidence", root.GetProperty("recordKind").GetString());
        Assert.False(root.GetProperty("network").GetProperty("modelOrOnnxRequired").GetBoolean());
        Assert.True(validation.GetProperty("profileValid").GetBoolean());
        Assert.True(validation.GetProperty("readinessReady").GetBoolean());
        Assert.True(validation.GetProperty("allTensorAddressesBound").GetBoolean());
        Assert.True(validation.GetProperty("outputMatch").GetBoolean());
        Assert.Equal(0, validation.GetProperty("processExitCode").GetInt32());

        string sourcePath = Path.Combine(RepositoryPaths.Root, "samples", "Inference", "02.DynamicShapes", "Program.cs");
        string screenshotPath = Path.Combine(
            RepositoryPaths.Root,
            assets.GetProperty("runtimeScreenshotPath").GetString()!.Replace('/', Path.DirectorySeparatorChar));
        Assert.Equal(maintenance.GetProperty("currentSourceSha256").GetString(), ComputeSha256(sourcePath));
        Assert.Equal(0, maintenance.GetProperty("processExitCode").GetInt32());
        Assert.True(maintenance.GetProperty("outputMatch").GetBoolean());
        Assert.False(maintenance.GetProperty("runtimeScreenshotRecaptured").GetBoolean());
        Assert.True(maintenance.GetProperty("historicalRuntimeScreenshotRetained").GetBoolean());
        Assert.Equal(assets.GetProperty("runtimeScreenshotSha256").GetString(), ComputeSha256(screenshotPath));

        string article = File.ReadAllText(articlePath);
        string sampleReadme = File.ReadAllText(sampleReadmePath);
        Assert.Contains("dynamic-shape-runtime-terminal.png", article, StringComparison.Ordinal);
        Assert.Contains("dynamic-shape-article-runtime-evidence.json", article, StringComparison.Ordinal);
        Assert.Contains("dynamic-shape-optimization-profile-tutorial.md", sampleReadme, StringComparison.Ordinal);
    }

    private static string ComputeSha256(string path)
    {
        return Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
    }
}
