using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseCandidateFreezeManifestTests
{
    [Fact]
    public void ReleaseCandidateFreezeManifestKeepsBlockedProofLanesFrozen()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-candidate-freeze-manifest.json");
        JsonElement root = document.RootElement;

        Assert.Equal("release-candidate-freeze-manifest.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("blocked-release-candidate-owner-real-proof-required", root.GetProperty("freezeState").GetString());
        AssertFalseProofPublishCloseFlags(root);

        string[] laneIds = root.GetProperty("blockedProofLanes")
            .EnumerateArray()
            .Select(static lane => lane.GetProperty("laneId").GetString()!)
            .ToArray();

        Assert.Equal(new[]
        {
            "sample-run-evidence",
            "package-consumer-runtime",
            "post-publish-verification",
            "release-close-owner-approval"
        }, laneIds);

        foreach (JsonElement lane in root.GetProperty("blockedProofLanes").EnumerateArray())
        {
            Assert.False(lane.GetProperty("isProof").GetBoolean());
            Assert.False(lane.GetProperty("canFreezeAsPassed").GetBoolean());
            Assert.Contains("Owner", lane.GetProperty("blockedReason").GetString(), StringComparison.OrdinalIgnoreCase);
        }

        AssertForbiddenSubstitutes(root.GetProperty("forbiddenProofSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray());
    }

    internal static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    internal static void AssertFalseProofPublishCloseFlags(JsonElement root)
    {
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
    }

    internal static void AssertForbiddenSubstitutes(string[] substitutes)
    {
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
            "blocked-by-cuda-driver"
        })
        {
            Assert.Contains(expected, substitutes);
        }
    }

    internal static void AssertArticleLinked(string fileName)
    {
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", fileName));
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string zhReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string href = "articles/zh-cn/" + fileName;

        Assert.Contains(href, docsIndex, StringComparison.Ordinal);
        Assert.Contains(href, docsToc, StringComparison.Ordinal);
        Assert.Contains("docs/" + href, readme, StringComparison.Ordinal);
        Assert.Contains("docs/" + href, zhReadme, StringComparison.Ordinal);

        foreach (string marker in new[] { "适用读者", "解决问题", "边界说明", "下一步", "runtime proof", "post-publish proof", "TensorRtExec report", "YoloVision matrix", "OnnxToEngine report", "readonly diagnostics" })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("dotnet nuget push", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canPublishPublicly=true", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canCloseReleaseIssue=true", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("samples/YoloDet", article, StringComparison.OrdinalIgnoreCase);
    }
}
