using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OwnerRealProofImportAuditBundleTests
{
    [Fact]
    public void OwnerRealProofImportAuditBundleKeepsLanesBlockedAndSeparate()
    {
        using JsonDocument document = ReadFinalReleaseJson("owner-real-proof-import-audit-bundle.json");
        JsonElement root = document.RootElement;

        Assert.Equal("owner-real-proof-import-audit-bundle.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("blocked-owner-real-proof-required", root.GetProperty("auditState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("importReadiness").GetBoolean());

        string[] laneIds = root.GetProperty("proofLanes")
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

        foreach (JsonElement lane in root.GetProperty("proofLanes").EnumerateArray())
        {
            Assert.False(lane.GetProperty("isProof").GetBoolean());
            Assert.True(lane.GetProperty("requiredOwnerInputs").GetArrayLength() >= 9);
            Assert.Contains("pwsh", lane.GetProperty("strictValidator").GetString(), StringComparison.Ordinal);
            Assert.Contains("-FailOn", lane.GetProperty("strictValidator").GetString(), StringComparison.Ordinal);
        }

        string raw = root.GetRawText();
        Assert.Contains("sample-run-evidence cannot replace package-consumer-runtime", raw, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime cannot replace post-publish verification", raw, StringComparison.Ordinal);
        AssertForbiddenSubstitutes(root.GetProperty("forbiddenProofSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray());
    }

    [Fact]
    public void OwnerRealProofImportAuditBundleArticleIsLinkedAndNonProof()
    {
        AssertArticleLinked("owner-real-proof-import-audit-bundle.md");
    }

    internal static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
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
