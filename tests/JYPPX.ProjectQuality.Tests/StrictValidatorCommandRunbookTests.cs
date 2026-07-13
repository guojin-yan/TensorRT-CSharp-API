using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class StrictValidatorCommandRunbookTests
{
    [Fact]
    public void StrictValidatorCommandRunbookListsLaneCommandsWithoutPublishing()
    {
        using JsonDocument document = OwnerRealProofImportAuditBundleTests.ReadFinalReleaseJson("strict-validator-command-runbook.json");
        JsonElement root = document.RootElement;

        Assert.Equal("strict-validator-command-runbook.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("blocked-owner-real-proof-required", root.GetProperty("runbookState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());

        string[] laneIds = root.GetProperty("validatorCommands").EnumerateArray()
            .Select(static item => item.GetProperty("laneId").GetString()!)
            .ToArray();

        foreach (string expected in new[] { "sample-run-evidence", "package-consumer-runtime", "post-publish-verification", "release-close-owner-approval" })
        {
            Assert.Contains(expected, laneIds);
        }

        string raw = root.GetRawText();
        foreach (string expected in new[]
        {
            "Test-SampleRunEvidenceRecord.ps1",
            "Test-ExternalRuntimeProofRecord.ps1",
            "Test-PostPublishVerificationRecord.ps1",
            "Test-ReleaseIssueCloseRecord.ps1",
            "-RequireExistingLog",
            "-FailOnNotProof",
            "-FailOnNotCloseReady",
            "do not execute real publication"
        })
        {
            Assert.Contains(expected, raw, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void StrictValidatorCommandRunbookArticleIsLinkedAndNonProof()
    {
        OwnerRealProofImportAuditBundleTests.AssertArticleLinked("strict-validator-command-runbook.md");
    }
}
