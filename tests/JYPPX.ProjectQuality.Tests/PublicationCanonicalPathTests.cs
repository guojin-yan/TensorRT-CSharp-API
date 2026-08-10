using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PublicationCanonicalPathTests
{
    private static readonly string[] CanonicalPaths =
    {
        "docs/articles/zh-cn/yolovision-yolov8n-detection-tutorial.md",
        "docs/articles/zh-cn/yolovision-lraspp-semantic-segmentation-tutorial.md",
        "docs/articles/zh-cn/yolovision-yolov8n-classification-tutorial.md",
        "docs/articles/zh-cn/yolovision-yolov8n-pose-tutorial.md",
        "docs/articles/zh-cn/yolovision-yolov8n-obb-tutorial.md",
        "docs/articles/zh-cn/yolovision-yolov8n-instance-segmentation-tutorial.md"
    };

    [Fact]
    public void PublicationCatalogUsesStableCanonicalPathsAndRetainsLegacyCompatibility()
    {
        string catalogPath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "publication-catalog.json");
        using JsonDocument catalog = JsonDocument.Parse(File.ReadAllText(catalogPath));
        JsonElement[] articles = catalog.RootElement.GetProperty("articles").EnumerateArray().ToArray();

        foreach (string expectedPath in CanonicalPaths)
        {
            JsonElement article = Assert.Single(articles, item => item.GetProperty("path").GetString() == expectedPath);
            string legacyPath = Assert.IsType<string>(article.GetProperty("legacyPath").GetString());
            Assert.Contains("local-package-consumer", legacyPath, StringComparison.Ordinal);
            Assert.True(File.Exists(Resolve(expectedPath)), expectedPath);
            Assert.True(File.Exists(Resolve(legacyPath)), legacyPath);
            Assert.Equal(File.ReadAllText(Resolve(legacyPath)), File.ReadAllText(Resolve(expectedPath)));
        }

        Assert.DoesNotContain(
            articles,
            article => article.GetProperty("path").GetString()!.Contains("local-package-consumer", StringComparison.Ordinal));
    }

    [Fact]
    public void PublicNavigationUsesCanonicalPathsOnly()
    {
        string toc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string index = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string articleReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "README.md"));

        foreach (string path in CanonicalPaths)
        {
            string tocPath = path["docs/".Length..];
            string readmePath = Path.GetFileName(path);
            Assert.Contains("href: " + tocPath, toc, StringComparison.Ordinal);
            Assert.Contains("(" + tocPath + ")", index, StringComparison.Ordinal);
            Assert.Contains("(" + readmePath + ")", articleReadme, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("href: articles/zh-cn/yolovision-yolov8n-det-local-package-consumer", toc, StringComparison.Ordinal);
        Assert.DoesNotContain("href: articles/zh-cn/yolovision-lraspp-semantic-local-package-consumer", toc, StringComparison.Ordinal);
        Assert.DoesNotContain("href: articles/zh-cn/yolovision-yolov8n-pose-local-package-consumer", toc, StringComparison.Ordinal);
        Assert.DoesNotContain("href: articles/zh-cn/yolovision-yolov8n-cls-local-package-consumer", toc, StringComparison.Ordinal);
        Assert.DoesNotContain("href: articles/zh-cn/yolovision-yolov8n-obb-local-package-consumer", toc, StringComparison.Ordinal);
        Assert.DoesNotContain("href: articles/zh-cn/yolovision-yolov8-seg-local-package-consumer", toc, StringComparison.Ordinal);
    }

    private static string Resolve(string relativePath) =>
        Path.Combine(RepositoryPaths.Root, relativePath.Replace('/', Path.DirectorySeparatorChar));
}
