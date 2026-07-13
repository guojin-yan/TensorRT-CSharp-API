using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TechnicalArticleCampaignFourthBatchBodyTests
{
    private static readonly string[] ArticleFiles =
    {
        "yolovision-detection-yolov8n-download-export-run.md",
        "yolovision-segmentation-mask-postprocess-guide.md",
        "yolovision-pose-keypoint-output-guide.md",
        "yolovision-obb-angle-output-guide.md",
        "tensorrtexec-winforms-screenshot-walkthrough.md",
        "tensorrtexec-report-schema-guide.md",
        "runtime-package-windows-linux-install-faq.md",
        "plugin-registry-inventory-user-guide.md",
        "deferred-readonly-api-upgrade-playbook.md",
        "csharp-wrapper-lifetime-design.md",
        "release-evidence-non-substitute-guide.md",
        "project-roadmap-to-public-release.md",
    };

    [Fact]
    public void FourthCampaignArticleBodyBatchExistsAndKeepsProofBoundaries()
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
            Assert.Contains("适用读者", article, StringComparison.Ordinal);
            Assert.Contains("解决问题", article, StringComparison.Ordinal);
            Assert.Contains("背景与场景", article, StringComparison.Ordinal);
            Assert.True(
                article.Contains("操作路径", StringComparison.Ordinal) ||
                article.Contains("实现路径", StringComparison.Ordinal),
                articleFile + " must contain 操作路径 or 实现路径.");
            Assert.Contains("代码与文件入口", article, StringComparison.Ordinal);
            Assert.Contains("图示建议", article, StringComparison.Ordinal);
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
    }
}
