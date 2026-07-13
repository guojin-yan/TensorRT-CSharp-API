using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OwnerProofImportPreflightTests
{
    [Fact]
    public void OwnerProofImportPreflightDetectsPlaceholdersHashesAndForbiddenSubstitutes()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "owner-proof-import-preflight.json")));

        JsonElement root = document.RootElement;
        Assert.Equal("owner-proof-import-preflight.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("blocked-placeholder-owner-input-required", root.GetProperty("preflightState").GetString());
        Assert.False(root.GetProperty("isRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());

        string raw = root.GetRawText();
        foreach (string marker in new[] { "owner-to-fill", "owner-required", "template-only", "example-not-proof", "blocked-by-cuda-driver", "64 lowercase hexadecimal characters", "stdoutLogPath", "outputJsonPath", "external-runtime-proof-record.json", "post-publish-verification-record.json" })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        string[] substitutes = root.GetProperty("forbiddenProofSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("TensorRtExec report", substitutes);
        Assert.Contains("YoloVision matrix", substitutes);
        Assert.Contains("OnnxToEngine report", substitutes);
        Assert.Contains("readonly diagnostics", substitutes);
    }

    [Fact]
    public void OwnerProofImportPreflightArticleIsLinkedAndNonProof()
    {
        OwnerRealProofFieldDeltaDashboardTests.AssertArticleLinked("owner-proof-import-preflight.md");
    }
}
