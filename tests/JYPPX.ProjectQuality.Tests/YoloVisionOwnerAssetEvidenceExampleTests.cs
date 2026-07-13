using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionOwnerAssetEvidenceExampleTests
{
    [Fact]
    public void YoloVisionOwnerAssetEvidenceExampleIsPlaceholderOnly()
    {
        string path = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "yolovision-owner-asset-evidence.example.json");
        Assert.True(File.Exists(path), path);

        string text = File.ReadAllText(path);
        using JsonDocument document = JsonDocument.Parse(text);
        JsonElement root = document.RootElement;

        Assert.Equal("yolovision-owner-asset-evidence-example", root.GetProperty("recordKind").GetString());
        Assert.Equal("example-not-proof", root.GetProperty("proofClassification").GetString());
        Assert.Equal("YoloVision", root.GetProperty("sampleName").GetString());
        Assert.False(root.GetProperty("isRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(root.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("owner-to-fill", text, StringComparison.Ordinal);
        Assert.Contains("YoloVision Passed=True", text, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec report", text, StringComparison.Ordinal);
        Assert.Contains("YoloVision matrix", text, StringComparison.Ordinal);
        Assert.Contains("OnnxToEngine report", text, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet", text, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void YoloVisionOwnerAssetEvidenceExampleGuideIsLinkedAndNonProof()
    {
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-owner-asset-evidence-example.md"));
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        Assert.Contains("articles/zh-cn/yolovision-owner-asset-evidence-example.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/yolovision-owner-asset-evidence-example.md", docsToc, StringComparison.Ordinal);

        foreach (string marker in new[] { "适用读者", "解决问题", "边界说明", "下一步", "example-not-proof", "owner-to-fill", "YoloVision Passed=True", "TensorRtExec report", "YoloVision matrix", "OnnxToEngine report", "readonly diagnostics" })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("dotnet nuget push", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("YoloDet", article, StringComparison.OrdinalIgnoreCase);
    }
}
