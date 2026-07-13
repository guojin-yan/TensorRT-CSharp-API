using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseProofArticleFirstBatchTests
{
    private static readonly string[] ArticleFiles =
    {
        "release-proof-sample-article-closure.md",
        "release-proof-owner-input-dashboard.md",
        "tensorrtexec-report-proof-boundary.md",
        "onnx-to-engine-trtexec-proof-boundary.md",
        "yolovision-owner-asset-evidence-guide.md",
        "callback-allocator-listener-readonly-safety-gates.md"
    };

    [Fact]
    public void ReleaseProofArticleBatchExistsAndKeepsNonProofBoundary()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string zhReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));

        foreach (string articleFile in ArticleFiles)
        {
            string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", articleFile);
            string article = File.ReadAllText(articlePath);
            string href = "articles/zh-cn/" + articleFile;
            string readmePath = "docs/" + href;

            Assert.True(File.Exists(articlePath), articleFile);
            Assert.Contains(href, docsIndex, StringComparison.Ordinal);
            Assert.Contains(href, docsToc, StringComparison.Ordinal);
            Assert.Contains(readmePath, readme, StringComparison.Ordinal);
            Assert.Contains(readmePath, zhReadme, StringComparison.Ordinal);

            Assert.Contains("适用读者", article, StringComparison.Ordinal);
            Assert.Contains("解决问题", article, StringComparison.Ordinal);
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
            Assert.DoesNotContain("YoloDet", article, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void RoadmapEntriesThirtyThreeThroughThirtySevenPointToArticleBodiesOrRunnableEvidence()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "publishing",
            "article-roadmap-30plus.json")));

        JsonElement[] entries = document.RootElement.GetProperty("articles")
            .EnumerateArray()
            .Where(static item => item.GetProperty("id").GetInt32() is >= 33 and <= 37)
            .ToArray();

        Assert.Equal(5, entries.Length);
        foreach (JsonElement entry in entries)
        {
            string path = entry.GetProperty("sampleOrCodePath").GetString()!;
            Assert.NotEqual("planned", entry.GetProperty("status").GetString());
            Assert.True(File.Exists(Path.Combine(RepositoryPaths.Root, path.Replace('/', Path.DirectorySeparatorChar))), path);

            if (entry.GetProperty("id").GetInt32() is >= 33 and <= 36)
            {
                Assert.StartsWith("docs/articles/zh-cn/", path, StringComparison.Ordinal);
                Assert.EndsWith(".md", path, StringComparison.Ordinal);
            }
            else
            {
                Assert.StartsWith("smoke/", path, StringComparison.Ordinal);
                Assert.EndsWith(".cs", path, StringComparison.Ordinal);
            }
        }
    }

    [Fact]
    public void ReleaseArticleFirstBatchHandoffCoversFiveConcreteArticleCaseOutlines()
    {
        string articlePath = Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-article-first-batch-handoff.md");
        string article = File.ReadAllText(articlePath);

        Assert.True(File.Exists(articlePath));
        foreach (string topic in new[]
        {
            "YoloVision 六任务真实资产证据链",
            "OnnxToEngine 与 TensorRtExec 转换边界",
            "Package Consumer Proof 分层",
            "Source Build、CMake 与 Runtime Package Readiness",
            "CUDA / TensorRT / cuDNN 版本矩阵与 Runtime Package Key"
        })
        {
            Assert.Contains(topic, article, StringComparison.Ordinal);
        }

        foreach (string path in new[]
        {
            "samples/YoloVision",
            "samples/OnnxToEngine",
            "applications/TensorRtExec",
            "eng\\Test-PackageConsumer.ps1",
            "eng\\Test-ExternalRuntimeProofRecord.ps1",
            "eng\\Generate-Bindings.ps1",
            "eng\\Resolve-RuntimeRoots.ps1",
            "artifacts/package-readiness/runtime-package-readiness-summary.md"
        })
        {
            Assert.Contains(path, article, StringComparison.Ordinal);
        }

        foreach (string boundary in new[]
        {
            "build-only",
            "dry-run",
            "template",
            "local feed",
            "ProjectReference",
            "direct `.nupkg`",
            "TensorRtExec report",
            "YoloVision matrix",
            "OnnxToEngine report",
            "readonly diagnostics",
            "Smoke=not-requested",
            "blocked-by-cuda-driver",
            "-RequireExistingLog -FailOnNotProof"
        })
        {
            Assert.Contains(boundary, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("YoloDet", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canPublishPublicly=true", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canCloseReleaseIssue=true", article, StringComparison.OrdinalIgnoreCase);
    }
}
