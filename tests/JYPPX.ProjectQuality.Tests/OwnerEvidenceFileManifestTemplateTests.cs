using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OwnerEvidenceFileManifestTemplateTests
{
    [Fact]
    public void OwnerEvidenceFileManifestTemplateRequiresPathsHashesAndOwnerFill()
    {
        using JsonDocument document = OwnerRealProofImportAuditBundleTests.ReadFinalReleaseJson("owner-evidence-file-manifest.template.json");
        JsonElement root = document.RootElement;

        Assert.Equal("owner-evidence-file-manifest-template.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("template-only-not-proof", root.GetProperty("templateState").GetString());
        Assert.True(root.GetProperty("ownerToFill").GetBoolean());
        Assert.True(root.GetProperty("pathMustExistBeforeImport").GetBoolean());
        Assert.False(root.GetProperty("canBeImportedAsProof").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());

        JsonElement sha = root.GetProperty("sha256Requirements");
        Assert.Equal("64 lowercase hexadecimal characters", sha.GetProperty("format").GetString());
        string raw = root.GetRawText();
        foreach (string marker in new[]
        {
            "owner-to-fill",
            "owner-required",
            "template-only",
            "example-not-proof",
            "blocked-by-cuda-driver",
            "sampleRunEvidenceFiles",
            "packageConsumerRuntimeFiles",
            "postPublishVerificationFiles",
            "releaseCloseOwnerApprovalFiles"
        })
        {
            Assert.Contains(marker, raw, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void OwnerEvidenceFileManifestTemplateArticleIsLinkedAndNonProof()
    {
        OwnerRealProofImportAuditBundleTests.AssertArticleLinked("owner-evidence-file-manifest-template.md");
    }
}
