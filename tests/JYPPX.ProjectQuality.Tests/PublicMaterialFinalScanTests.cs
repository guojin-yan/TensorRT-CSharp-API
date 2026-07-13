using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PublicMaterialFinalScanTests
{
    [Fact]
    public void PublicMaterialFinalScanRecordsRetiredIdentityAndProofPromotionRules()
    {
        using JsonDocument document = ReleaseCandidateFreezeManifestTests.ReadFinalReleaseJson("public-material-final-scan.json");
        JsonElement root = document.RootElement;

        Assert.Equal("public-material-final-scan.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("blocked-owner-real-proof-required", root.GetProperty("scanState").GetString());
        ReleaseCandidateFreezeManifestTests.AssertFalseProofPublishCloseFlags(root);

        string raw = root.GetRawText();
        foreach (string marker in new[]
        {
            "no-retired-sample-live-identity",
            "<retired-sample-directory>",
            "<retired-sample-project-file>",
            "proof-ladder-separation",
            "sample-run-evidence cannot replace package-consumer-runtime",
            "package-consumer-runtime cannot replace post-publish-verification",
            "TensorRtExec report",
            "YoloVision matrix",
            "OnnxToEngine report",
            "readonly diagnostics",
            "blocked-by-cuda-driver"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("samples/YoloDet", raw, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("YoloDet.csproj", raw, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void PublicMaterialFinalScanArticleIsLinkedAndNonProof()
    {
        ReleaseCandidateFreezeManifestTests.AssertArticleLinked("public-material-final-scan.md");
    }

    [Fact]
    public void PublicEntryPointsDoNotContainLiveYoloDetPath()
    {
        string publicEntryText = string.Join(Environment.NewLine, new[]
        {
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md")),
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md")),
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md")),
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"))
        });

        Assert.DoesNotContain("samples/YoloDet", publicEntryText, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("samples\\YoloDet", publicEntryText, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("YoloDet.csproj", publicEntryText, StringComparison.OrdinalIgnoreCase);
    }
}
