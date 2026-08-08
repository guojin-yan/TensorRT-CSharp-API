using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PublicPackageArticleMigrationTests
{
    private static readonly string[] CurrentInstallationGuidePaths =
    {
        "README.md",
        "README.zh-CN.md",
        "docs/articles/en/sample-series-overview.md",
        "docs/articles/zh-cn/classification-real-asset-walkthrough.md",
        "docs/articles/zh-cn/sample-series-overview.md",
        "docs/articles/zh-cn/yolovision-yolov10n-real-asset-tutorial.md",
        "docs/articles/zh-cn/yolovision-yolox-official-runtime-tutorial.md",
        "samples/ComputerVision/01.Classification/README.md",
        "samples/ComputerVision/01.Classification/README.zh-CN.md",
        "samples/README.md",
        "samples/README.zh-CN.md"
    };

    private static readonly string[] MigratedArticleNames =
    {
        "debug-listener-local-package-consumer-tutorial.md",
        "gpu-allocator-local-package-consumer-tutorial.md",
        "logger-local-package-consumer-tutorial.md",
        "output-allocator-local-package-consumer-tutorial.md",
        "profiler-local-package-consumer-tutorial.md",
        "progress-monitor-local-package-consumer-tutorial.md",
        "stream-reader-local-package-consumer-tutorial.md",
        "tensorrtexec-refitted-plan-local-package-consumer.md",
        "yolovision-lraspp-semantic-local-package-consumer-tutorial.md",
        "yolovision-yolov8-seg-local-package-consumer-tutorial.md",
        "yolovision-yolov8n-cls-local-package-consumer-tutorial.md",
        "yolovision-yolov8n-det-local-package-consumer-tutorial.md",
        "yolovision-yolov8n-obb-local-package-consumer-tutorial.md",
        "yolovision-yolov8n-pose-local-package-consumer-tutorial.md",
        "yolovision-yolox-local-package-consumer-tutorial.md"
    };

    [Fact]
    public void MigratedArticlesUseThePublicPreviewLineBeforeHistoricalLocalFeedEvidence()
    {
        foreach (string articleName in MigratedArticleNames)
        {
            string article = ReadSource("docs", "articles", "zh-cn", articleName);
            int publicInstall = article.IndexOf(
                "dotnet add package JYPPX.TensorRT.CSharp.API --version \"4.0.0-*\"",
                StringComparison.Ordinal);
            int historicalEvidence = article.IndexOf(
                "### 发布前 local-feed 证据复核",
                StringComparison.Ordinal);
            string publicFlow = historicalEvidence >= 0
                ? article.Substring(0, historicalEvidence)
                : article;

            Assert.True(publicInstall >= 0, "Missing public package command: " + articleName);
            Assert.True(
                historicalEvidence > publicInstall,
                "Historical evidence must follow public installation: " + articleName);
            Assert.Contains(
                "dotnet add package JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge --version \"4.0.0-*\"",
                article,
                StringComparison.Ordinal);
            Assert.Contains("API 不兼容的历史", article, StringComparison.Ordinal);
            Assert.DoesNotContain(
                "JYPPX.TensorRT.CSharp.API --prerelease",
                article,
                StringComparison.Ordinal);
            Assert.DoesNotContain("本地候选包", publicFlow, StringComparison.Ordinal);
            Assert.DoesNotContain("从本地 NuGet 包", publicFlow, StringComparison.Ordinal);
            Assert.DoesNotContain("两个本地包", publicFlow, StringComparison.Ordinal);
            Assert.DoesNotContain("三个本地包", publicFlow, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void MigratedHistoricalFileNamesStayOutOfPublicNavigation()
    {
        string toc = ReadSource("docs", "toc.yml");
        string index = ReadSource("docs", "index.md");

        foreach (string articleName in MigratedArticleNames)
        {
            string relativePath = "articles/zh-cn/" + articleName;
            Assert.Contains("# " + relativePath, toc, StringComparison.Ordinal);
            Assert.Contains("<!-- " + relativePath + " -->", index, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void CurrentInstallationGuidesUseTheMaintainedPreviewLine()
    {
        foreach (string relativePath in CurrentInstallationGuidePaths)
        {
            string guide = ReadSource(relativePath.Split('/'));

            Assert.Contains(
                "dotnet add package JYPPX.TensorRT.CSharp.API --version \"4.0.0-*\"",
                guide,
                StringComparison.Ordinal);
            Assert.Contains(
                "dotnet add package JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge --version \"4.0.0-*\"",
                guide,
                StringComparison.Ordinal);
            Assert.DoesNotContain(
                "dotnet add package JYPPX.TensorRT.CSharp.API --prerelease",
                guide,
                StringComparison.Ordinal);
        }
    }

    [Fact]
    public void KnownIssueTracksTheCompletedMigrationWithoutReclassifyingHistoricalEvidence()
    {
        string knownIssues = ReadSource("docs", "releases", "known-issues.md");

        Assert.Contains("| KI-004 | Closed (15/15) |", knownIssues, StringComparison.Ordinal);
        Assert.Contains("全部 15 篇教程已改为公开包优先", knownIssues, StringComparison.Ordinal);
        Assert.Contains("隔离 local-feed 历史证据", knownIssues, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }
}
