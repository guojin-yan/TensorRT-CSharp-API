using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseIssueCloseOwnerInputFinalChecklistTests
{
    [Fact]
    public void ReleaseIssueCloseOwnerInputFinalChecklistStaysBlockedUntilOwnerEvidenceExists()
    {
        using JsonDocument document = OwnerRealProofImportAuditBundleTests.ReadFinalReleaseJson("release-issue-close-owner-input-final-checklist.json");
        JsonElement root = document.RootElement;

        Assert.Equal("release-issue-close-owner-input-final-checklist.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("blocked-owner-release-close-input-required", root.GetProperty("checklistState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("releaseCloseReady").GetBoolean());
        Assert.True(root.GetProperty("ownerActionRequired").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());

        string raw = root.GetRawText();
        foreach (string expected in new[]
        {
            "releaseEvidenceBundleSha256",
            "releaseClosePreflightSha256",
            "sampleRunEvidenceValidationSha256",
            "packageConsumerRuntimeValidationSha256",
            "postPublishVerificationSha256",
            "rollbackPlan",
            "ownerFinalDecision",
            "ownerSignature",
            "decisionTimestampUtc",
            "Test-ReleaseIssueCloseRecord.ps1",
            "-FailOnNotCloseReady"
        })
        {
            Assert.Contains(expected, raw, StringComparison.Ordinal);
        }

        OwnerRealProofImportAuditBundleTests.AssertForbiddenSubstitutes(root.GetProperty("forbiddenProofSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray());
    }

    [Fact]
    public void ReleaseIssueCloseOwnerInputFinalChecklistArticleIsLinkedAndNonProof()
    {
        OwnerRealProofImportAuditBundleTests.AssertArticleLinked("release-issue-close-owner-input-final-checklist.md");
    }
}
