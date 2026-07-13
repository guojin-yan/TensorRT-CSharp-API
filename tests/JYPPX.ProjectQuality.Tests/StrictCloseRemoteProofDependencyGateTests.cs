using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class StrictCloseRemoteProofDependencyGateTests
{
    [Fact]
    public void StrictCloseDashboardRequiresRemotePublicAndPostPublishProofLanes()
    {
        RunPipeline();

        using JsonDocument dashboardDocument = ReadFinalReleaseJson("strict-close-ready-convergence-dashboard.json");
        JsonElement dashboard = dashboardDocument.RootElement;
        Assert.Equal("strict-close-ready-convergence-dashboard", dashboard.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-strict-close-ready-owner-action-required", dashboard.GetProperty("dashboardState").GetString());
        Assert.Equal(13, dashboard.GetProperty("laneCount").GetInt32());
        Assert.True(dashboard.GetProperty("blockedLaneCount").GetInt32() >= RequiredRemoteProofLaneIds.Length);
        Assert.Equal("blocked-remote-ci-and-public-publish-proof-backfill-required", dashboard.GetProperty("remoteCiAndPublicPublishProofBackfillGateState").GetString());
        AssertFlagsStayNonProof(dashboard);

        string[] requiredRemoteIds = dashboard.GetProperty("remoteProofRequiredLaneIds")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        foreach (string laneId in RequiredRemoteProofLaneIds)
        {
            Assert.Contains(laneId, requiredRemoteIds);
        }

        JsonElement[] lanes = dashboard.GetProperty("closeReadinessLanes").EnumerateArray().ToArray();
        foreach (string laneId in RequiredRemoteProofLaneIds)
        {
            JsonElement lane = lanes.Single(lane => lane.GetProperty("laneId").GetString() == laneId);
            Assert.False(lane.GetProperty("ready").GetBoolean());
            Assert.True(lane.GetProperty("blocksStrictClose").GetBoolean());
            Assert.Equal(laneId, lane.GetProperty("remoteProofGateLaneId").GetString());
            AssertFlagsStayNonProof(lane);
        }

        JsonElement postPublishLane = lanes.Single(static lane => lane.GetProperty("laneId").GetString() == "post-publish-clean-consumer-proof");
        Assert.True(postPublishLane.GetProperty("remoteGateStateReady").GetBoolean());
        Assert.True(postPublishLane.GetProperty("requireProofReady").GetBoolean());
        Assert.Equal("proofCandidateReady", postPublishLane.GetProperty("proofReadyProperty").GetString());
        Assert.False(postPublishLane.GetProperty("proofReady").GetBoolean());
        Assert.False(postPublishLane.GetProperty("ready").GetBoolean());

        string[] sourceArtifacts = dashboard.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string artifact in RequiredRemoteProofSourceArtifacts)
        {
            Assert.Contains(artifact, sourceArtifacts);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("strict-close-ready-convergence-dashboard-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("strict-close-ready-convergence-dashboard-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-strict-close-ready-owner-action-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        AssertValidationItemPassed(validation, "remote-proof-lanes-present");
        AssertValidationItemPassed(validation, "remote-proof-lanes-block-close");
        AssertValidationItemPassed(validation, "post-publish-proof-lane-requires-proof-candidate-ready");
        AssertFlagsStayNonProof(validation);

        using JsonDocument finalCloseDocument = ReadFinalReleaseJson("final-close-gate-convergence.json");
        JsonElement finalClose = finalCloseDocument.RootElement;
        Assert.Equal("final-close-gate-convergence", finalClose.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-close-gate-owner-proof-required", finalClose.GetProperty("convergenceState").GetString());
        Assert.Equal(16, finalClose.GetProperty("laneCount").GetInt32());
        Assert.Equal(16, finalClose.GetProperty("blockedLaneCount").GetInt32());
        Assert.Equal(2, finalClose.GetProperty("dualPackageRouteCount").GetInt32());
        Assert.False(finalClose.GetProperty("dualPackageAcceptsSubstituteProof").GetBoolean());
        Assert.Equal("blocked-remote-ci-and-public-publish-proof-backfill-required", finalClose.GetProperty("remoteCiAndPublicPublishProofBackfillGateState").GetString());
        AssertFlagsStayNonProof(finalClose);

        JsonElement[] finalCloseLanes = finalClose.GetProperty("gateLanes").EnumerateArray().ToArray();
        foreach (string laneId in RequiredRemoteProofLaneIds)
        {
            JsonElement lane = finalCloseLanes.Single(lane => lane.GetProperty("id").GetString() == laneId);
            Assert.False(lane.GetProperty("ready").GetBoolean());
            Assert.Equal(laneId, lane.GetProperty("remoteProofGateLaneId").GetString());
            AssertFlagsStayNonProof(lane);
        }

        foreach (string laneId in RequiredDualPackageLaneIds)
        {
            JsonElement lane = finalCloseLanes.Single(lane => lane.GetProperty("id").GetString() == laneId);
            AssertDualPackageLaneBlocksClose(lane);
        }

        JsonElement finalClosePostPublishLane = finalCloseLanes.Single(static lane => lane.GetProperty("id").GetString() == "post-publish-clean-consumer-proof");
        Assert.True(finalClosePostPublishLane.GetProperty("remoteGateStateReady").GetBoolean());
        Assert.True(finalClosePostPublishLane.GetProperty("requireProofReady").GetBoolean());
        Assert.Equal("proofCandidateReady", finalClosePostPublishLane.GetProperty("proofReadyProperty").GetString());
        Assert.False(finalClosePostPublishLane.GetProperty("proofReady").GetBoolean());
        Assert.False(finalClosePostPublishLane.GetProperty("ready").GetBoolean());
        JsonElement finalClosePublicDownloadLane = finalCloseLanes.Single(static lane => lane.GetProperty("id").GetString() == "public-package-download-proof");
        Assert.False(finalClosePublicDownloadLane.GetProperty("ready").GetBoolean());
        Assert.Contains("Repository-external clean consumer", finalClosePostPublishLane.GetProperty("requiredEvidence").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("public package download proof only", finalClosePostPublishLane.GetProperty("requiredEvidence").GetString(), StringComparison.OrdinalIgnoreCase);

        using JsonDocument finalCloseValidationDocument = ReadFinalReleaseJson("final-close-gate-convergence-validation.json");
        JsonElement finalCloseValidation = finalCloseValidationDocument.RootElement;
        Assert.Equal("blocked-final-close-gate-owner-proof-required", finalCloseValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, finalCloseValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(2, finalCloseValidation.GetProperty("dualPackageRouteCount").GetInt32());
        Assert.Equal(2, finalCloseValidation.GetProperty("dualPackageBlockedLaneCount").GetInt32());
        Assert.False(finalCloseValidation.GetProperty("dualPackageAcceptsSubstituteProof").GetBoolean());
        AssertValidationItemPassed(finalCloseValidation, "remote-proof-lanes-present");
        AssertValidationItemPassed(finalCloseValidation, "remote-proof-lanes-block-close");
        AssertValidationItemPassed(finalCloseValidation, "dual-package-lanes-present");
        AssertValidationItemPassed(finalCloseValidation, "dual-package-lanes-block-close");
        AssertValidationItemPassed(finalCloseValidation, "post-publish-proof-lane-requires-proof-candidate-ready");
        AssertFlagsStayNonProof(finalCloseValidation);

        using JsonDocument ownerApprovalDocument = ReadFinalReleaseJson("final-release-close-owner-approval-contract.json");
        JsonElement ownerApproval = ownerApprovalDocument.RootElement;
        Assert.Equal("final-release-close-owner-approval-contract", ownerApproval.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-release-close-owner-approval-required", ownerApproval.GetProperty("contractState").GetString());
        Assert.Equal("blocked-remote-ci-and-public-publish-proof-backfill-required", ownerApproval.GetProperty("sourceRemoteCiAndPublicPublishProofBackfillGateValidationState").GetString());
        Assert.True(ownerApproval.GetProperty("requiredOwnerInputFieldCount").GetInt32() >= 120);
        string ownerApprovalRaw = ownerApproval.GetRawText();
        foreach (string marker in RequiredRemoteProofLaneIds)
        {
            Assert.Contains(marker, ownerApprovalRaw, StringComparison.Ordinal);
        }

        Assert.Contains("githubActionsRunProofPath", ownerApprovalRaw, StringComparison.Ordinal);
        Assert.Contains("ownerPublicPublishResultPath", ownerApprovalRaw, StringComparison.Ordinal);
        Assert.Contains("publicPackageDownloadProofPath", ownerApprovalRaw, StringComparison.Ordinal);
        Assert.Contains("postPublishCleanConsumerProofResultPath", ownerApprovalRaw, StringComparison.Ordinal);
        AssertFlagsStayNonProof(ownerApproval);

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal(16, evidence.GetProperty("finalCloseGateConvergenceLaneCount").GetInt32());
        Assert.Equal(16, evidence.GetProperty("finalCloseGateConvergenceBlockedLaneCount").GetInt32());
        Assert.Equal(2, evidence.GetProperty("finalCloseGateConvergenceDualPackageRouteCount").GetInt32());
        Assert.Equal(2, evidence.GetProperty("finalCloseGateConvergenceDualPackageBlockedLaneCount").GetInt32());
        Assert.False(evidence.GetProperty("finalCloseGateConvergenceDualPackageAcceptsSubstituteProof").GetBoolean());
    }

    private static void RunPipeline()
    {
        RunPowerShell("Export-GitHubPublishAndCiStatusSnapshot.ps1");
        RunPowerShell("Test-GitHubPublishAndCiStatusSnapshot.ps1", "-Strict");
        RunPowerShell("Export-FinalPrepublishQualityFreezeDashboard.ps1");
        RunPowerShell("Test-FinalPrepublishQualityFreezeDashboard.ps1", "-Strict");
        RunPowerShell("Test-GitHubActionsRunEvidenceImport.ps1", "-Strict");
        RunPowerShell("Export-OwnerPublicPublishExecutionResultInputContract.ps1");
        RunPowerShell("Export-OwnerPublicPublishExecutionResultInputTemplate.ps1");
        RunPowerShell("Test-OwnerPublicPublishExecutionResultPreflight.ps1", "-Strict");
        RunPowerShell("Import-OwnerPublicPublishExecutionResultCandidate.ps1");
        RunPowerShell("Test-OwnerPublicPublishExecutionResultCandidate.ps1", "-Strict");
        RunPowerShell("Export-PublicPackageDownloadProofInputTemplate.ps1");
        RunPowerShell("Test-PublicPackageDownloadProofInput.ps1", "-Strict");
        RunPowerShell("Import-PublicPackageDownloadProofCandidate.ps1");
        RunPowerShell("Test-PublicPackageDownloadProofCandidate.ps1", "-Strict");
        RunPowerShell("Import-PostPublishCleanConsumerProofResult.ps1");
        RunPowerShell("Test-PostPublishCleanConsumerProofResult.ps1", "-Strict");
        RunPowerShell("Export-RemoteCiAndPublicPublishProofBackfillGate.ps1");
        RunPowerShell("Test-RemoteCiAndPublicPublishProofBackfillGate.ps1", "-Strict");
        RunPowerShell("Export-DualPackagePublishPreflightMatrix.ps1");
        RunPowerShell("Test-DualPackagePublishPreflightMatrix.ps1", "-Strict");
        RunPowerShell("Export-StrictCloseReadyConvergenceDashboard.ps1");
        RunPowerShell("Test-StrictCloseReadyConvergenceDashboard.ps1", "-Strict");
        RunPowerShell("Export-FinalCloseGateConvergence.ps1");
        RunPowerShell("Test-FinalCloseGateConvergence.ps1", "-Strict");
        RunPowerShell("Export-FinalReleaseCloseOwnerApprovalContract.ps1");
        RunPowerShell("Test-FinalReleaseCloseOwnerApprovalContract.ps1", "-Strict");
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

    private static void AssertValidationItemPassed(JsonElement validation, string id)
    {
        Assert.Contains(validation.GetProperty("validationItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            item.GetProperty("passed").GetBoolean());
    }

    private static void AssertFlagsStayNonProof(JsonElement element)
    {
        Assert.False(element.GetProperty("performsPublish").GetBoolean());
        Assert.False(element.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(element.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    private static void AssertDualPackageLaneBlocksClose(JsonElement lane)
    {
        Assert.False(lane.GetProperty("ready").GetBoolean());
        Assert.True(lane.GetProperty("ownerActionRequired").GetBoolean());
        Assert.True(lane.GetProperty("externalProofRequired").GetBoolean());
        Assert.True(lane.GetProperty("postPublishProofRequired").GetBoolean());
        Assert.False(lane.GetProperty("acceptsSubstituteProof").GetBoolean());
        Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("nextOwnerAction").GetString()));
        Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("externalProofMissingReason").GetString()));
        Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("postPublishProofMissingReason").GetString()));
        AssertFlagsStayNonProof(lane);
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(scriptPath, arguments);
    }

    private static readonly string[] RequiredRemoteProofLaneIds =
    [
        "github-actions-run-proof",
        "owner-public-publish-result",
        "public-package-download-proof",
        "post-publish-clean-consumer-proof",
    ];

    private static readonly string[] RequiredDualPackageLaneIds =
    [
        "dual-package-nuget-small-bridge-core",
        "dual-package-github-packages-full-runtime",
    ];

    private static readonly string[] RequiredRemoteProofSourceArtifacts =
    [
        "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate.json",
        "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate.md",
        "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate-validation.json",
        "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate-validation.md",
    ];
}
