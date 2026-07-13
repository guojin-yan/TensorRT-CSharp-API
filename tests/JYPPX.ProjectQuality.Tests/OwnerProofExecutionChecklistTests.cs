using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OwnerProofExecutionChecklistTests
{
    [Fact]
    public void OwnerProofExecutionChecklistKeepsThreeLaddersBlockedAndSeparate()
    {
        string path = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-proof-execution-checklist.json");
        Assert.True(File.Exists(path), path);

        string text = File.ReadAllText(path);
        using JsonDocument document = JsonDocument.Parse(text);
        JsonElement root = document.RootElement;

        Assert.Equal("owner-proof-execution-checklist.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("blocked-owner-real-proof-required", root.GetProperty("checklistState").GetString());
        Assert.False(root.GetProperty("isRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("sample-run-evidence cannot replace package-consumer-runtime", text, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime cannot replace post-publish verification", text, StringComparison.Ordinal);

        string[] ids = root.GetProperty("checklistItems")
            .EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();

        Assert.Equal(new[] { "sample-run-evidence", "package-consumer-runtime", "post-publish-verification" }, ids);

        foreach (JsonElement item in root.GetProperty("checklistItems").EnumerateArray())
        {
            Assert.False(item.GetProperty("isProof").GetBoolean());
            Assert.False(item.GetProperty("canPromoteProof").GetBoolean());
            Assert.Contains("-FailOnNotProof", item.GetProperty("validatorCommand").GetString(), StringComparison.Ordinal);
            Assert.True(item.GetProperty("requiredFields").GetArrayLength() >= 10);
            Assert.True(item.GetProperty("cannotReplace").GetArrayLength() >= 2);
        }
    }

    [Fact]
    public void OwnerProofExecutionChecklistArticleIsLinkedAndNonProof()
    {
        string article = ReadArticle("owner-proof-execution-checklist.md");
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        Assert.Contains("articles/zh-cn/owner-proof-execution-checklist.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/owner-proof-execution-checklist.md", docsToc, StringComparison.Ordinal);

        foreach (string marker in new[] { "适用读者", "解决问题", "边界说明", "下一步", "runtime proof", "TensorRtExec report", "YoloVision matrix", "OnnxToEngine report", "readonly diagnostics" })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("dotnet nuget push", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canPublishPublicly=true", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canCloseReleaseIssue=true", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("YoloDet", article, StringComparison.OrdinalIgnoreCase);
    }

    private static string ReadArticle(string fileName)
    {
        return File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", fileName));
    }
}
