using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseCandidateFinalFreezeAndOwnerPublishHandoffTests
{
    [Fact]
    public void FinalFreezePublishHandoffAndCloseDashboardStayNonProof()
    {
        RealExternalProofExecutionAndCloseOwnerInputPipeline.Run();
        RunPowerShell("Export-PublicPackageProofOwnerInputTemplate.ps1");
        RunPowerShell("Test-PublicPackageProofOwnerInput.ps1", "-Strict");
        RunPowerShell("Export-PostPublishProofOwnerConfirmation.ps1");
        RunPowerShell("Test-PostPublishProofOwnerConfirmation.ps1", "-Strict");
        RunPowerShell("Export-ReleaseClosePublicProofBridge.ps1");
        RunPowerShell("Test-ReleaseClosePublicProofBridge.ps1", "-Strict");
        RunPowerShell("Export-ReleaseIssueCloseFinalOwnerDecisionAudit.ps1");
        RunPowerShell("Test-ReleaseIssueCloseFinalOwnerDecisionAudit.ps1", "-Strict");
        RunPowerShell("Export-FinalPostPublishAuditPack.ps1");
        RunPowerShell("Test-FinalPostPublishAuditPack.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCandidateFinalFreezeManifest.ps1");
        RunPowerShell("Test-ReleaseCandidateFinalFreezeManifest.ps1", "-Strict");
        RunPowerShell("Export-PublicPublishOwnerManualCommandHandoff.ps1");
        RunPowerShell("Test-PublicPublishOwnerManualCommandHandoff.ps1", "-Strict");
        RunPowerShell("Export-FinalReleaseCloseBlockerDashboard.ps1");
        RunPowerShell("Test-FinalReleaseCloseBlockerDashboard.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument freezeDocument = ReadFinalReleaseJson("release-candidate-final-freeze-manifest-validation.json");
        JsonElement freeze = freezeDocument.RootElement;
        Assert.Equal("release-candidate-final-freeze-manifest-ready-for-owner-handoff", freeze.GetProperty("validationState").GetString());
        Assert.True(freeze.GetProperty("artifactCount").GetInt32() >= 8);
        Assert.Equal(0, freeze.GetProperty("failedBlockerCount").GetInt32());
        AssertFalseProofPublishCloseFlags(freeze);

        using JsonDocument handoffDocument = ReadFinalReleaseJson("public-publish-owner-manual-command-handoff.json");
        JsonElement handoff = handoffDocument.RootElement;
        Assert.Equal("blocked-owner-public-publish-required", handoff.GetProperty("handoffState").GetString());
        Assert.True(handoff.GetProperty("notExecutedByAutomation").GetBoolean());
        Assert.True(handoff.GetProperty("commandCount").GetInt32() >= 5);
        Assert.Contains(handoff.GetProperty("manualCommands").EnumerateArray(), command =>
            command.GetProperty("command").GetString()!.Contains("dotnet nuget push", StringComparison.Ordinal) &&
            command.GetProperty("notExecutedByAutomation").GetBoolean() &&
            command.GetProperty("performsPublish").GetBoolean() == false);
        AssertFalseProofPublishCloseFlags(handoff);

        using JsonDocument handoffValidationDocument = ReadFinalReleaseJson("public-publish-owner-manual-command-handoff-validation.json");
        JsonElement handoffValidation = handoffValidationDocument.RootElement;
        Assert.Equal("public-publish-owner-manual-command-handoff-ready", handoffValidation.GetProperty("validationState").GetString());
        Assert.True(handoffValidation.GetProperty("notExecutedByAutomation").GetBoolean());
        Assert.Equal(0, handoffValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertFalseProofPublishCloseFlags(handoffValidation);

        using JsonDocument dashboardDocument = ReadFinalReleaseJson("final-release-close-blocker-dashboard-validation.json");
        JsonElement dashboard = dashboardDocument.RootElement;
        Assert.Equal("final-release-close-blocker-dashboard-ready", dashboard.GetProperty("validationState").GetString());
        Assert.True(dashboard.GetProperty("blockerCount").GetInt32() >= 9);
        Assert.True(dashboard.GetProperty("blockedBlockerCount").GetInt32() > 0);
        Assert.Equal(0, dashboard.GetProperty("failedBlockerCount").GetInt32());
        AssertFalseProofPublishCloseFlags(dashboard);

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("release-candidate-final-freeze-manifest-ready-for-owner-handoff", evidence.GetProperty("releaseCandidateFinalFreezeManifestValidationState").GetString());
        Assert.Equal("public-publish-owner-manual-command-handoff-ready", evidence.GetProperty("publicPublishOwnerManualCommandHandoffValidationState").GetString());
        Assert.Equal("final-release-close-blocker-dashboard-ready", evidence.GetProperty("finalReleaseCloseBlockerDashboardValidationState").GetString());
        Assert.False(evidence.GetProperty("releaseCandidateFinalFreezeManifestCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("publicPublishOwnerManualCommandHandoffCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("finalReleaseCloseBlockerDashboardCanCloseReleaseIssue").GetBoolean());

        AssertBlockedEvidenceItem(evidence, "release-candidate-final-freeze-manifest", "not package push");
        AssertBlockedEvidenceItem(evidence, "public-publish-owner-manual-command-handoff", "not publish approval");
        AssertBlockedEvidenceItem(evidence, "final-release-close-blocker-dashboard", "not release close approval");

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/release-candidate-final-freeze-manifest.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-candidate-final-freeze-manifest-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-publish-owner-manual-command-handoff.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-publish-owner-manual-command-handoff-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-release-close-blocker-dashboard.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-release-close-blocker-dashboard-validation.json", sourceArtifacts);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        AssertAuditedNonProofItem(audit, "release-candidate-final-freeze-manifest");
        AssertAuditedNonProofItem(audit, "public-publish-owner-manual-command-handoff");
        AssertAuditedNonProofItem(audit, "final-release-close-blocker-dashboard");

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/release-candidate-final-freeze-manifest.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/public-publish-owner-manual-command-handoff.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("final-release-close-blocker-dashboard", readme, StringComparison.Ordinal);
        Assert.Contains("public-publish-owner-manual-command-handoff", readmeZh, StringComparison.Ordinal);
        Assert.Contains("release-candidate-final-freeze-manifest", releaseEvidenceDoc, StringComparison.Ordinal);
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
