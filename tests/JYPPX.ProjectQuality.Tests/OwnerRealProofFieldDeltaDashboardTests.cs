using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OwnerRealProofFieldDeltaDashboardTests
{
    [Fact]
    public void FieldDeltaDashboardListsBlockedOwnerProofLanes()
    {
        using JsonDocument document = ReadFinalReleaseJson("owner-real-proof-field-delta-dashboard.json");
        JsonElement root = document.RootElement;

        Assert.Equal("owner-real-proof-field-delta-dashboard.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("blocked-owner-real-proof-required", root.GetProperty("dashboardState").GetString());
        Assert.False(root.GetProperty("isRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());

        string[] laneIds = root.GetProperty("fieldDeltaLanes").EnumerateArray().Select(static lane => lane.GetProperty("laneId").GetString()!).ToArray();
        Assert.Equal(new[] { "sample-run-evidence", "package-consumer-runtime", "post-publish-verification", "release-close-owner-approval" }, laneIds);

        foreach (JsonElement lane in root.GetProperty("fieldDeltaLanes").EnumerateArray())
        {
            Assert.False(lane.GetProperty("isProof").GetBoolean());
            Assert.True(lane.GetProperty("missingFields").GetArrayLength() >= 7);
            Assert.Contains("-FailOn", lane.GetProperty("validatorCommand").GetString(), StringComparison.Ordinal);
            Assert.Contains("owner", lane.GetProperty("blockedReason").GetString(), StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void FieldDeltaArticleIsLinkedAndKeepsNonProofBoundary()
    {
        AssertArticleLinked("owner-real-proof-field-delta-dashboard.md");
    }

    internal static void AssertArticleLinked(string fileName)
    {
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", fileName));
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string href = "articles/zh-cn/" + fileName;

        Assert.Contains(href, docsIndex, StringComparison.Ordinal);
        Assert.Contains(href, docsToc, StringComparison.Ordinal);
        foreach (string marker in new[] { "适用读者", "解决问题", "边界说明", "下一步", "runtime proof", "TensorRtExec report", "YoloVision matrix", "OnnxToEngine report", "readonly diagnostics" })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("dotnet nuget push", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("YoloDet", article, StringComparison.OrdinalIgnoreCase);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }
}
