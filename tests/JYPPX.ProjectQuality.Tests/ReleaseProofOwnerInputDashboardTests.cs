using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseProofOwnerInputDashboardTests
{
    [Fact]
    public void OwnerDashboardSeparatesProofLaddersAndRemainsBlocked()
    {
        string dashboardPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-proof-owner-input-dashboard.json");
        Assert.True(File.Exists(dashboardPath), dashboardPath);

        string text = File.ReadAllText(dashboardPath);
        using JsonDocument document = JsonDocument.Parse(text);
        JsonElement root = document.RootElement;

        Assert.Equal("release-proof-owner-input-dashboard.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("blocked-owner-real-proof-required", root.GetProperty("dashboardState").GetString());
        Assert.False(root.GetProperty("isRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());

        string[] ladders = root.GetProperty("ownerInputLadders")
            .EnumerateArray()
            .Select(static item => item.GetProperty("ladderId").GetString()!)
            .ToArray();

        Assert.Equal(new[] { "sample-run-evidence", "package-consumer-runtime", "post-publish-verification" }, ladders);
        Assert.Contains("sample-run-evidence cannot replace package-consumer-runtime", text, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime cannot replace post-publish verification", text, StringComparison.Ordinal);
        Assert.Contains("post-publish verification can only be collected after real selected-channel publication", text, StringComparison.Ordinal);
    }

    [Fact]
    public void OwnerDashboardListsForbiddenSubstitutesAndValidators()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-proof-owner-input-dashboard.json")));

        JsonElement root = document.RootElement;
        string[] substitutes = root.GetProperty("forbiddenProofSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray();

        foreach (string expected in new[]
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
            "design gate",
            "blocked-by-cuda-driver"
        })
        {
            Assert.Contains(expected, substitutes);
        }

        foreach (JsonElement ladder in root.GetProperty("ownerInputLadders").EnumerateArray())
        {
            Assert.Equal("owner-action", ladder.GetProperty("state").GetString()![..12], StringComparer.OrdinalIgnoreCase);
            Assert.False(ladder.GetProperty("isProof").GetBoolean());
            Assert.Contains("-FailOnNotProof", ladder.GetProperty("requiredValidator").GetString(), StringComparison.Ordinal);
            Assert.True(ladder.GetProperty("requiredOwnerFields").GetArrayLength() >= 8);
            Assert.True(ladder.GetProperty("relatedArtifacts").GetArrayLength() >= 2);
        }
    }

    [Fact]
    public void OwnerDashboardArticleExplainsBoundaryWithoutPublishingCommands()
    {
        string article = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "release-proof-owner-input-dashboard.md"));

        Assert.Contains("适用读者", article, StringComparison.Ordinal);
        Assert.Contains("解决问题", article, StringComparison.Ordinal);
        Assert.Contains("边界说明", article, StringComparison.Ordinal);
        Assert.Contains("下一步", article, StringComparison.Ordinal);
        Assert.Contains("sample-run-evidence 不能替代 `package-consumer-runtime`", article, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime` 不能替代 `post-publish verification`", article, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec report", article, StringComparison.Ordinal);
        Assert.Contains("YoloVision matrix", article, StringComparison.Ordinal);
        Assert.Contains("OnnxToEngine report", article, StringComparison.Ordinal);
        Assert.Contains("readonly diagnostics", article, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue=false", article, StringComparison.Ordinal);
        Assert.DoesNotContain("dotnet nuget push", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canPublishPublicly=true", article, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void OwnerDashboardAndYoloEvidenceGuidesAreLinkedFromDocsAndReadmes()
    {
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string zhReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));

        foreach (string articlePath in new[]
        {
            "articles/zh-cn/release-proof-owner-input-dashboard.md",
            "articles/zh-cn/release-proof-sample-article-closure.md",
            "articles/zh-cn/yolovision-owner-asset-evidence-guide.md"
        })
        {
            Assert.Contains(articlePath, docsIndex, StringComparison.Ordinal);
            Assert.Contains(articlePath, docsToc, StringComparison.Ordinal);
            Assert.Contains("docs/" + articlePath, readme, StringComparison.Ordinal);
            Assert.Contains("docs/" + articlePath, zhReadme, StringComparison.Ordinal);
        }

        foreach (string artifactPath in new[]
        {
            "artifacts/final-release/release-proof-owner-input-dashboard.json",
            "artifacts/final-release/yolovision-owner-asset-evidence.template.json"
        })
        {
            Assert.Contains(artifactPath, readme, StringComparison.Ordinal);
            Assert.Contains(artifactPath, zhReadme, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("YoloDet", docsIndex + docsToc, StringComparison.OrdinalIgnoreCase);
    }
}
