using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PostPublishCleanConsumerOwnerInputTests
{
    [Fact]
    public void PostPublishCleanConsumerOwnerInputTemplateRequiresAfterPublicationFields()
    {
        string path = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "post-publish-clean-consumer-owner-input.template.json");
        Assert.True(File.Exists(path), path);

        string text = File.ReadAllText(path);
        using JsonDocument document = JsonDocument.Parse(text);
        JsonElement root = document.RootElement;

        Assert.Equal("post-publish-clean-consumer-owner-input.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("template-only-not-proof", root.GetProperty("proofClassification").GetString());
        Assert.False(root.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("canBeReplacedByPackageConsumerRuntime").GetBoolean());
        Assert.False(root.GetProperty("canBeReplacedBySampleRunEvidence").GetBoolean());
        Assert.Contains("Package-consumer-runtime and sample-run-evidence cannot replace it", text, StringComparison.OrdinalIgnoreCase);

        JsonElement fields = root.GetProperty("requiredOwnerFields");
        foreach (string field in new[]
        {
            "publishedPackageSource",
            "publishedPackageVersion",
            "cleanConsumerRoot",
            "consumerProjectPath",
            "restoreLogPath",
            "restoreLogSha256",
            "nativeAssetListingPath",
            "nativeAssetListingSha256",
            "dependencyProbeLogPath",
            "dependencyProbeLogSha256",
            "smokeLogPath",
            "smokeLogSha256",
            "ownerVerificationStatus"
        })
        {
            Assert.True(fields.TryGetProperty(field, out JsonElement value), field);
            Assert.Contains("owner-required-after-publication", value.GetString(), StringComparison.Ordinal);
        }
    }

    [Fact]
    public void PostPublishCleanConsumerOwnerInputGuideIsLinkedAndRejectsSubstitutes()
    {
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "post-publish-clean-consumer-owner-input-guide.md"));
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        Assert.Contains("articles/zh-cn/post-publish-clean-consumer-owner-input-guide.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/post-publish-clean-consumer-owner-input-guide.md", docsToc, StringComparison.Ordinal);

        foreach (string marker in new[] { "适用读者", "解决问题", "边界说明", "下一步", "post-publish verification", "package-consumer-runtime", "sample-run-evidence", "TensorRtExec report", "YoloVision matrix", "OnnxToEngine report", "readonly diagnostics" })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("dotnet nuget push", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("YoloDet", article, StringComparison.OrdinalIgnoreCase);
    }
}
