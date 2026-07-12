using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PublicPublishResultAndStrictCloseConvergenceTests
{
    [Fact]
    public void PublicPublishResultBackfillAndStrictCloseConvergenceStayBlockedNonProof()
    {
        RunPowerShell("Export-PublicPublishResultOwnerInputTemplate.ps1");
        RunPowerShell("Test-PublicPublishResultOwnerInput.ps1", "-Strict");
        RunPowerShell("Import-PublicPublishResultOwnerInput.ps1");
        RunPowerShell("Test-PublicPublishResultImport.ps1", "-Strict");
        RunPowerShell("Export-PostPublishCleanConsumerProofRecordContract.ps1");
        RunPowerShell("Test-PostPublishCleanConsumerProofRecordContract.ps1", "-Strict");
        RunPowerShell("Export-PostPublishCleanConsumerResultConvergence.ps1");
        RunPowerShell("Test-PostPublishCleanConsumerResultConvergence.ps1", "-Strict");
        RunPowerShell("Export-ReleaseIssueCloseOwnerDecisionInput.ps1");
        RunPowerShell("Test-ReleaseIssueCloseOwnerDecisionInput.ps1", "-Strict");
        RunPowerShell("Export-StrictCloseReadyConvergenceDashboard.ps1");
        RunPowerShell("Test-StrictCloseReadyConvergenceDashboard.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument ownerInputDocument = ReadFinalReleaseJson("public-publish-result-owner-input-validation.json");
        JsonElement ownerInput = ownerInputDocument.RootElement;
        Assert.Equal("blocked-public-publish-result-owner-input-required", ownerInput.GetProperty("validationState").GetString());
        Assert.Equal(0, ownerInput.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(ownerInput.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        AssertFalseProofPublishCloseFlags(ownerInput);
        AssertPublicPublishResultFieldCoverage(ownerInput);

        using JsonDocument importDocument = ReadFinalReleaseJson("public-publish-result-import-validation.json");
        JsonElement import = importDocument.RootElement;
        Assert.Equal("blocked-public-publish-result-import-required", import.GetProperty("validationState").GetString());
        Assert.Equal(0, import.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(import.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        AssertFalseProofPublishCloseFlags(import);
        AssertPublicPublishResultFieldCoverage(import);

        using JsonDocument postPublishContractDocument = ReadFinalReleaseJson("post-publish-clean-consumer-proof-record-contract-validation.json");
        JsonElement postPublishContract = postPublishContractDocument.RootElement;
        Assert.Equal("blocked-post-publish-clean-consumer-proof-record-required", postPublishContract.GetProperty("validationState").GetString());
        Assert.True(postPublishContract.GetProperty("requiredFieldCount").GetInt32() >= 40);
        Assert.Equal(0, postPublishContract.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(postPublishContract.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        AssertFalseProofPublishCloseFlags(postPublishContract);

        using JsonDocument convergenceDocument = ReadFinalReleaseJson("post-publish-clean-consumer-result-convergence-validation.json");
        JsonElement convergence = convergenceDocument.RootElement;
        Assert.Equal("blocked-post-publish-clean-consumer-result-required", convergence.GetProperty("validationState").GetString());
        Assert.True(convergence.GetProperty("laneCount").GetInt32() >= 5);
        Assert.True(convergence.GetProperty("blockedLaneCount").GetInt32() > 0);
        Assert.Equal(0, convergence.GetProperty("failedBlockerCount").GetInt32());
        AssertFalseProofPublishCloseFlags(convergence);

        using JsonDocument ownerDecisionDocument = ReadFinalReleaseJson("release-issue-close-owner-decision-input-validation.json");
        JsonElement ownerDecision = ownerDecisionDocument.RootElement;
        Assert.Equal("blocked-release-issue-close-owner-decision-input-required", ownerDecision.GetProperty("validationState").GetString());
        Assert.Equal(0, ownerDecision.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(ownerDecision.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        AssertFalseProofPublishCloseFlags(ownerDecision);

        using JsonDocument strictDocument = ReadFinalReleaseJson("strict-close-ready-convergence-dashboard-validation.json");
        JsonElement strict = strictDocument.RootElement;
        Assert.Equal("blocked-strict-close-ready-owner-action-required", strict.GetProperty("validationState").GetString());
        Assert.True(strict.GetProperty("laneCount").GetInt32() >= 9);
        Assert.True(strict.GetProperty("blockedLaneCount").GetInt32() > 0);
        Assert.Equal(0, strict.GetProperty("failedBlockerCount").GetInt32());
        AssertFalseProofPublishCloseFlags(strict);

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-public-publish-result-owner-input-required", evidence.GetProperty("publicPublishResultOwnerInputValidationState").GetString());
        Assert.Equal("blocked-public-publish-result-import-required", evidence.GetProperty("publicPublishResultImportValidationState").GetString());
        Assert.Equal("blocked-post-publish-clean-consumer-proof-record-required", evidence.GetProperty("postPublishCleanConsumerProofRecordContractValidationState").GetString());
        Assert.Equal("blocked-post-publish-clean-consumer-result-required", evidence.GetProperty("postPublishCleanConsumerResultConvergenceValidationState").GetString());
        Assert.Equal("blocked-release-issue-close-owner-decision-input-required", evidence.GetProperty("releaseIssueCloseOwnerDecisionInputValidationState").GetString());
        Assert.Equal("blocked-strict-close-ready-owner-action-required", evidence.GetProperty("strictCloseReadyConvergenceDashboardValidationState").GetString());
        Assert.False(evidence.GetProperty("publicPublishResultOwnerInputCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("publicPublishResultImportCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("postPublishCleanConsumerProofRecordContractCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("postPublishCleanConsumerResultConvergenceCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("releaseIssueCloseOwnerDecisionInputCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("strictCloseReadyConvergenceDashboardCanCloseReleaseIssue").GetBoolean());

        AssertBlockedEvidenceItem(evidence, "public-publish-result-owner-input", "not package push");
        AssertBlockedEvidenceItem(evidence, "public-publish-result-import", "not post-publish proof");
        AssertBlockedEvidenceItem(evidence, "post-publish-clean-consumer-proof-record-contract", "not post-publish proof");
        AssertBlockedEvidenceItem(evidence, "post-publish-clean-consumer-result-convergence", "not release close approval");
        AssertBlockedEvidenceItem(evidence, "release-issue-close-owner-decision-input", "not release close approval");
        AssertBlockedEvidenceItem(evidence, "strict-close-ready-convergence-dashboard", "not publish approval");

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/public-publish-result-owner-input.template.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-publish-result-owner-input-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-publish-result-import.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-publish-result-import-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-clean-consumer-proof-record-contract.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-clean-consumer-result-convergence.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-clean-consumer-result-convergence-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-issue-close-owner-decision-input.template.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-issue-close-owner-decision-input-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/strict-close-ready-convergence-dashboard.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json", sourceArtifacts);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        AssertAuditedNonProofItem(audit, "public-publish-result-owner-input");
        AssertAuditedNonProofItem(audit, "public-publish-result-import");
        AssertAuditedNonProofItem(audit, "post-publish-clean-consumer-proof-record-contract");
        AssertAuditedNonProofItem(audit, "post-publish-clean-consumer-result-convergence");
        AssertAuditedNonProofItem(audit, "release-issue-close-owner-decision-input");
        AssertAuditedNonProofItem(audit, "strict-close-ready-convergence-dashboard");

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));
        string publicPublishOwnerInputDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "public-publish-result-owner-input.md"));

        Assert.Contains("articles/zh-cn/public-publish-result-owner-input.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/strict-close-ready-convergence-dashboard.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("post-publish-clean-consumer-result-convergence", readme, StringComparison.Ordinal);
        Assert.Contains("public-publish-result-import", readmeZh, StringComparison.Ordinal);
        Assert.Contains("strict-close-ready-convergence-dashboard", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("nugetPackageSource", publicPublishOwnerInputDoc, StringComparison.Ordinal);
        Assert.Contains("githubRelease.managedAssetSha256", publicPublishOwnerInputDoc, StringComparison.Ordinal);
        Assert.Contains("managedPackage.publicDownloadSha256", publicPublishOwnerInputDoc, StringComparison.Ordinal);
        Assert.Contains("runtimePackage.publicDownloadSha256", publicPublishOwnerInputDoc, StringComparison.Ordinal);
        Assert.Contains("rollbackReview.rollbackPlanSha256", publicPublishOwnerInputDoc, StringComparison.Ordinal);
        Assert.Contains("finalCloseDecision.releaseIssueUrl", publicPublishOwnerInputDoc, StringComparison.Ordinal);
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
        Assert.True(element.GetProperty("notExecutedByAutomation").GetBoolean());
        Assert.False(element.GetProperty("performsPublish").GetBoolean());
        Assert.False(element.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(element.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseProof").GetBoolean());
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

    private static void AssertPublicPublishResultFieldCoverage(JsonElement validation)
    {
        string[] ids = validation.GetProperty("validationItems")
            .EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();

        string[] required =
        [
            "field-nugetPackageSource",
            "github-release-releaseUrl",
            "github-release-tagName",
            "github-release-managedAssetPath",
            "github-release-managedAssetSha256",
            "github-release-runtimeAssetPath",
            "github-release-runtimeAssetSha256",
            "managed-public-download-sha256",
            "runtime-public-download-sha256",
            "owner-review-reviewer",
            "owner-review-reviewedAtUtc",
            "owner-review-approvalState",
            "rollback-review-reviewedBy",
            "rollback-review-reviewedAtUtc",
            "rollback-review-rollbackPlanSha256",
            "rollback-review-decision",
            "final-close-decision-decision",
            "final-close-decision-decidedAtUtc",
            "final-close-decision-ownerReviewer",
            "final-close-decision-releaseIssueUrl",
        ];

        foreach (string id in required)
        {
            Assert.Contains(id, ids);
        }
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(scriptPath, arguments);
    }
}
