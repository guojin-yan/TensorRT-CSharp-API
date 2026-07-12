using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalPrepublishQualityFreezeDashboardTests
{
    [Fact]
    public void FinalPrepublishGatesStayBlockedNonProofAndReachEvidenceBundle()
    {
        RunPipeline();

        using JsonDocument dashboardDocument = ReadFinalReleaseJson("final-prepublish-quality-freeze-dashboard.json");
        JsonElement dashboard = dashboardDocument.RootElement;
        Assert.Equal("final-prepublish-quality-freeze-dashboard", dashboard.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-prepublish-quality-freeze-owner-action-required", dashboard.GetProperty("freezeState").GetString());
        Assert.True(dashboard.GetProperty("laneCount").GetInt32() >= 12);
        Assert.True(dashboard.GetProperty("blockedLaneCount").GetInt32() > 0);
        Assert.Equal(0, dashboard.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(dashboard.GetProperty("readyForOwnerPublicPublishExecution").GetBoolean());
        AssertFlagsStayNonProof(dashboard, expectGitHubActionsFlag: true);

        using JsonDocument dashboardValidationDocument = ReadFinalReleaseJson("final-prepublish-quality-freeze-dashboard-validation.json");
        JsonElement dashboardValidation = dashboardValidationDocument.RootElement;
        Assert.Equal("final-prepublish-quality-freeze-dashboard-validation", dashboardValidation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-prepublish-quality-freeze-owner-action-required", dashboardValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, dashboardValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(0, dashboardValidation.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFlagsStayNonProof(dashboardValidation, expectGitHubActionsFlag: true);

        using JsonDocument consistencyDocument = ReadFinalReleaseJson("owner-public-publish-execution-consistency-gate.json");
        JsonElement consistency = consistencyDocument.RootElement;
        Assert.Equal("owner-public-publish-execution-consistency-gate", consistency.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-public-publish-execution-consistency-owner-input-required", consistency.GetProperty("gateState").GetString());
        Assert.True(consistency.GetProperty("consistencyItemCount").GetInt32() >= 20);
        Assert.Equal(0, consistency.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(consistency.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        AssertFlagsStayNonProof(consistency, expectGitHubActionsFlag: false);

        using JsonDocument consistencyValidationDocument = ReadFinalReleaseJson("owner-public-publish-execution-consistency-gate-validation.json");
        JsonElement consistencyValidation = consistencyValidationDocument.RootElement;
        Assert.Equal("owner-public-publish-execution-consistency-gate-validation", consistencyValidation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-public-publish-execution-consistency-owner-input-required", consistencyValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, consistencyValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(0, consistencyValidation.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFlagsStayNonProof(consistencyValidation, expectGitHubActionsFlag: false);

        using JsonDocument githubDocument = ReadFinalReleaseJson("github-publish-and-ci-status-snapshot.json");
        JsonElement github = githubDocument.RootElement;
        Assert.Equal("github-publish-and-ci-status-snapshot", github.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-github-actions-and-public-publish-proof-required", github.GetProperty("snapshotState").GetString());
        Assert.Equal("missing-github-actions-proof", github.GetProperty("githubActionsProofState").GetString());
        Assert.Equal("blocked-real-github-actions-package-publish-proof-required", github.GetProperty("packagePublishOnGitHubState").GetString());
        AssertFlagsStayNonProof(github, expectGitHubActionsFlag: true);

        using JsonDocument githubValidationDocument = ReadFinalReleaseJson("github-publish-and-ci-status-snapshot-validation.json");
        JsonElement githubValidation = githubValidationDocument.RootElement;
        Assert.Equal("github-publish-and-ci-status-snapshot-validation", githubValidation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-github-actions-and-public-publish-proof-required", githubValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, githubValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(0, githubValidation.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFlagsStayNonProof(githubValidation, expectGitHubActionsFlag: true);

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-final-prepublish-quality-freeze-owner-action-required", evidence.GetProperty("finalPrepublishQualityFreezeDashboardValidationState").GetString());
        Assert.Equal("blocked-owner-public-publish-execution-consistency-owner-input-required", evidence.GetProperty("ownerPublicPublishExecutionConsistencyGateValidationState").GetString());
        Assert.Equal("blocked-github-actions-and-public-publish-proof-required", evidence.GetProperty("githubPublishAndCiStatusSnapshotValidationState").GetString());
        Assert.False(evidence.GetProperty("finalPrepublishQualityFreezeDashboardCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("ownerPublicPublishExecutionConsistencyGateCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("githubPublishAndCiStatusSnapshotCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("githubPublishAndCiStatusSnapshotIsGitHubActionsProof").GetBoolean());

        AssertBlockedEvidenceItem(evidence, "final-prepublish-quality-freeze-dashboard", "not GitHub Actions proof");
        AssertBlockedEvidenceItem(evidence, "owner-public-publish-execution-consistency-gate", "not package push");
        AssertBlockedEvidenceItem(evidence, "github-publish-and-ci-status-snapshot", "not GitHub Actions proof");

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        foreach (string artifact in RequiredSourceArtifacts)
        {
            Assert.Contains(artifact, sourceArtifacts);
        }

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        AssertAuditedNonProofItem(audit, "final-prepublish-quality-freeze-dashboard");
        AssertAuditedNonProofItem(audit, "owner-public-publish-execution-consistency-gate");
        AssertAuditedNonProofItem(audit, "github-publish-and-ci-status-snapshot");
    }

    private static readonly string[] RequiredSourceArtifacts =
    [
        "artifacts/final-release/final-prepublish-quality-freeze-dashboard.json",
        "artifacts/final-release/final-prepublish-quality-freeze-dashboard.md",
        "artifacts/final-release/final-prepublish-quality-freeze-dashboard-validation.json",
        "artifacts/final-release/final-prepublish-quality-freeze-dashboard-validation.md",
        "artifacts/final-release/owner-public-publish-execution-consistency-gate.json",
        "artifacts/final-release/owner-public-publish-execution-consistency-gate.md",
        "artifacts/final-release/owner-public-publish-execution-consistency-gate-validation.json",
        "artifacts/final-release/owner-public-publish-execution-consistency-gate-validation.md",
        "artifacts/final-release/github-publish-and-ci-status-snapshot.json",
        "artifacts/final-release/github-publish-and-ci-status-snapshot.md",
        "artifacts/final-release/github-publish-and-ci-status-snapshot-validation.json",
        "artifacts/final-release/github-publish-and-ci-status-snapshot-validation.md",
    ];

    private static void RunPipeline()
    {
        RunPowerShell("Export-ReleaseScriptReferenceIndex.ps1");
        RunPowerShell("Test-ReleaseScriptReferenceIndex.ps1", "-Strict");
        RunPowerShell("Export-OwnerPublicPublishAuthorizationInputTemplate.ps1");
        RunPowerShell("Import-OwnerPublicPublishAuthorizationInput.ps1");
        RunPowerShell("Test-OwnerPublicPublishAuthorizationInput.ps1", "-Strict");
        RunPowerShell("Export-OwnerPublicPublishAuthorizationGate.ps1");
        RunPowerShell("Test-OwnerPublicPublishAuthorizationGate.ps1", "-Strict");
        RunPowerShell("Export-PublicPublishResultAuthorizationConvergenceGate.ps1");
        RunPowerShell("Test-PublicPublishResultAuthorizationConvergenceGate.ps1", "-Strict");
        RunPowerShell("Export-PublicPublishFinalOwnerExecutionPack.ps1");
        RunPowerShell("Test-PublicPublishFinalOwnerExecutionPack.ps1", "-Strict");
        RunPowerShell("Export-PublicPublishCommandCrossCheck.ps1");
        RunPowerShell("Test-PublicPublishCommandCrossCheck.ps1", "-Strict");
        RunPowerShell("Export-FinalEvidenceFreezeNonProofAudit.ps1");
        RunPowerShell("Test-FinalEvidenceFreezeNonProofAudit.ps1", "-Strict");
        RunPowerShell("Export-FinalQualityFreezeDashboard.ps1");
        RunPowerShell("Test-FinalQualityFreezeDashboard.ps1", "-Strict");
        RunPowerShell("Export-FinalPrepublishQualityFreezeDashboard.ps1");
        RunPowerShell("Test-FinalPrepublishQualityFreezeDashboard.ps1", "-Strict");
        RunPowerShell("Export-OwnerPublicPublishExecutionConsistencyGate.ps1");
        RunPowerShell("Test-OwnerPublicPublishExecutionConsistencyGate.ps1", "-Strict");
        RunPowerShell("Export-GitHubPublishAndCiStatusSnapshot.ps1");
        RunPowerShell("Test-GitHubPublishAndCiStatusSnapshot.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static void AssertFlagsStayNonProof(JsonElement element, bool expectGitHubActionsFlag)
    {
        Assert.True(element.GetProperty("notExecutedByAutomation").GetBoolean());
        Assert.False(element.GetProperty("performsPublish").GetBoolean());
        Assert.False(element.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(element.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseProof").GetBoolean());
        if (expectGitHubActionsFlag)
        {
            Assert.False(element.GetProperty("isGitHubActionsProof").GetBoolean());
        }
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
