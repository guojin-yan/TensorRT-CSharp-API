using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseIssueCloseFinalOwnerDecisionAndPostPublishAuditTests
{
    [Fact]
    public void FinalOwnerDecisionAndPostPublishAuditStayBlockedUntilRealOwnerProofExists()
    {
        RealExternalProofExecutionAndCloseOwnerInputPipeline.Run();
        RunPowerShell("Export-PublicPackageProofOwnerInputTemplate.ps1");
        RunPowerShell("Test-PublicPackageProofOwnerInput.ps1", "-Strict");
        RunPowerShell("Export-PostPublishProofOwnerConfirmation.ps1");
        RunPowerShell("Test-PostPublishProofOwnerConfirmation.ps1", "-Strict");
        RunPowerShell("Export-ReleaseClosePublicProofBridge.ps1");
        RunPowerShell("Test-ReleaseClosePublicProofBridge.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");
        RunPowerShell("Export-ReleaseIssueCloseFinalOwnerDecisionAudit.ps1");
        RunPowerShell("Test-ReleaseIssueCloseFinalOwnerDecisionAudit.ps1", "-Strict");
        RunPowerShell("Export-FinalPostPublishAuditPack.ps1");
        RunPowerShell("Test-FinalPostPublishAuditPack.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument finalOwnerAuditDocument = ReadFinalReleaseJson("release-issue-close-final-owner-decision-audit.json");
        JsonElement finalOwnerAudit = finalOwnerAuditDocument.RootElement;
        Assert.Equal("blocked-release-issue-close-final-owner-decision-required", finalOwnerAudit.GetProperty("auditState").GetString());
        Assert.Equal(7, finalOwnerAudit.GetProperty("finalOwnerDecisionGateCount").GetInt32());
        Assert.Equal(6, finalOwnerAudit.GetProperty("blockedFinalOwnerDecisionGateCount").GetInt32());
        Assert.Equal(1, finalOwnerAudit.GetProperty("readyFinalOwnerDecisionGateCount").GetInt32());
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", finalOwnerAudit.GetProperty("strictCloseCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("-FailOnNotCloseReady", finalOwnerAudit.GetProperty("strictCloseCommand").GetString(), StringComparison.Ordinal);
        AssertFalseProofPublishCloseFlags(finalOwnerAudit);

        using JsonDocument finalOwnerValidationDocument = ReadFinalReleaseJson("release-issue-close-final-owner-decision-audit-validation.json");
        JsonElement finalOwnerValidation = finalOwnerValidationDocument.RootElement;
        Assert.Equal("blocked-release-issue-close-final-owner-decision-required", finalOwnerValidation.GetProperty("validationState").GetString());
        Assert.Equal(7, finalOwnerValidation.GetProperty("finalOwnerDecisionGateCount").GetInt32());
        Assert.Equal(6, finalOwnerValidation.GetProperty("blockedFinalOwnerDecisionGateCount").GetInt32());
        Assert.Equal(1, finalOwnerValidation.GetProperty("readyFinalOwnerDecisionGateCount").GetInt32());
        Assert.Equal(0, finalOwnerValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, finalOwnerValidation.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFalseProofPublishCloseFlags(finalOwnerValidation);

        using JsonDocument postPublishAuditDocument = ReadFinalReleaseJson("final-post-publish-audit-pack.json");
        JsonElement postPublishAudit = postPublishAuditDocument.RootElement;
        Assert.Equal("blocked-final-post-publish-audit-required", postPublishAudit.GetProperty("auditState").GetString());
        Assert.Equal(7, postPublishAudit.GetProperty("auditLaneCount").GetInt32());
        Assert.Equal(7, postPublishAudit.GetProperty("blockedAuditLaneCount").GetInt32());
        Assert.Equal(0, postPublishAudit.GetProperty("readyAuditLaneCount").GetInt32());
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", postPublishAudit.GetProperty("strictCloseCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("-FailOnNotCloseReady", postPublishAudit.GetProperty("strictCloseCommand").GetString(), StringComparison.Ordinal);
        AssertFalseProofPublishCloseFlags(postPublishAudit);

        using JsonDocument postPublishValidationDocument = ReadFinalReleaseJson("final-post-publish-audit-pack-validation.json");
        JsonElement postPublishValidation = postPublishValidationDocument.RootElement;
        Assert.Equal("blocked-final-post-publish-audit-required", postPublishValidation.GetProperty("validationState").GetString());
        Assert.Equal(7, postPublishValidation.GetProperty("auditLaneCount").GetInt32());
        Assert.Equal(7, postPublishValidation.GetProperty("blockedAuditLaneCount").GetInt32());
        Assert.Equal(0, postPublishValidation.GetProperty("readyAuditLaneCount").GetInt32());
        Assert.Equal(0, postPublishValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(7, postPublishValidation.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFalseProofPublishCloseFlags(postPublishValidation);

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-release-issue-close-final-owner-decision-required", evidence.GetProperty("releaseIssueCloseFinalOwnerDecisionAuditValidationState").GetString());
        Assert.Equal(6, evidence.GetProperty("releaseIssueCloseFinalOwnerDecisionAuditBlockedGateCount").GetInt32());
        Assert.False(evidence.GetProperty("releaseIssueCloseFinalOwnerDecisionAuditCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("releaseIssueCloseFinalOwnerDecisionAuditIsPostPublishProof").GetBoolean());
        Assert.Equal("blocked-final-post-publish-audit-required", evidence.GetProperty("finalPostPublishAuditPackValidationState").GetString());
        Assert.Equal(7, evidence.GetProperty("finalPostPublishAuditPackBlockedLaneCount").GetInt32());
        Assert.False(evidence.GetProperty("finalPostPublishAuditPackCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("finalPostPublishAuditPackIsPostPublishProof").GetBoolean());

        AssertBlockedEvidenceItem(evidence, "release-issue-close-final-owner-decision-audit", "not release close approval");
        AssertBlockedEvidenceItem(evidence, "final-post-publish-audit-pack", "not post-publish proof");

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/release-issue-close-final-owner-decision-audit.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-issue-close-final-owner-decision-audit-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-post-publish-audit-pack.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-post-publish-audit-pack-validation.json", sourceArtifacts);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        AssertAuditedNonProofItem(audit, "release-issue-close-final-owner-decision-audit");
        AssertAuditedNonProofItem(audit, "final-post-publish-audit-pack");

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/release-issue-close-final-owner-decision-audit.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/final-post-publish-audit-pack.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("release-issue-close-final-owner-decision-audit", readme, StringComparison.Ordinal);
        Assert.Contains("final-post-publish-audit-pack", readmeZh, StringComparison.Ordinal);
        Assert.Contains("release-issue-close-final-owner-decision-audit", releaseEvidenceDoc, StringComparison.Ordinal);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static void AssertFalseProofPublishCloseFlags(JsonElement element)
    {
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(element);
        Assert.False(element.GetProperty("isPostPublishProof").GetBoolean());
    }

    private static void AssertBlockedEvidenceItem(JsonElement evidence, string id, string boundaryText)
    {
        JsonElement item = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(item => item.GetProperty("id").GetString() == id);

        Assert.False(item.GetProperty("passed").GetBoolean());
        Assert.Contains(boundaryText, item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertAuditedNonProofItem(JsonElement audit, string id)
    {
        Assert.Contains(audit.GetProperty("auditedItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(scriptPath, arguments);
    }
}
