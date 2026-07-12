using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalOwnerPublishAndArticleActionDashboardTests
{
    [Fact]
    public void FinalOwnerPublishAndArticleActionDashboardAggregatesCurrentBlockedGates()
    {
        RunActionDashboardPipeline();

        using JsonDocument dashboardDocument = ReadFinalReleaseJson("final-owner-publish-and-article-action-dashboard.json");
        JsonElement dashboard = dashboardDocument.RootElement;
        Assert.Equal("final-owner-publish-and-article-action-dashboard", dashboard.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-publish-and-article-action-dashboard-owner-action-required", dashboard.GetProperty("dashboardState").GetString());
        Assert.Equal(9, dashboard.GetProperty("gateCount").GetInt32());
        Assert.Equal(9, dashboard.GetProperty("blockedGateCount").GetInt32());
        AssertNonProofFlags(dashboard);

        foreach (string gate in new[] { "final-readonly-audit", "owner-one-screen-manual", "publish-replay-checklist", "evidence-import-runbook", "owner-intake-dry-run", "post-publish-cross-check", "article-readiness-matrix", "post-publish-article-proof-gate", "release-close-strict-closure" })
        {
            AssertDashboardGate(dashboard, gate);
        }

        string raw = dashboard.GetRawText();
        Assert.Contains("ProjectReference", raw, StringComparison.Ordinal);
        Assert.Contains("direct nupkg", raw, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec report", raw, StringComparison.Ordinal);

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-owner-publish-and-article-action-dashboard-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-owner-publish-and-article-action-dashboard-validation-ready-non-proof", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(9, validation.GetProperty("gateCount").GetInt32());
        AssertNonProofFlags(validation);
    }

    [Fact]
    public void MissingEvidenceRepairPackAndArticleBoundaryScanStayNonProofAndActionable()
    {
        RunActionDashboardPipeline();

        using JsonDocument repairDocument = ReadFinalReleaseJson("owner-missing-real-publish-evidence-repair-pack.json");
        JsonElement repair = repairDocument.RootElement;
        Assert.Equal("owner-missing-real-publish-evidence-repair-pack", repair.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-missing-real-publish-evidence-repair-required", repair.GetProperty("repairState").GetString());
        Assert.True(repair.GetProperty("contractRequiredFieldCount").GetInt32() >= 178);
        Assert.True(repair.GetProperty("ownerPublishMissingFieldCount").GetInt32() >= 178);
        Assert.True(repair.GetProperty("postPublishMissingFieldCount").GetInt32() >= 10);
        Assert.True(repair.GetProperty("failedPostPublishCrossCheckCount").GetInt32() >= 1);
        Assert.True(repair.GetProperty("releaseCloseBlockedLaneCount").GetInt32() >= 1);
        Assert.Equal(4, repair.GetProperty("repairGroupCount").GetInt32());
        AssertNonProofFlags(repair);

        foreach (string group in new[] { "owner-public-publish-contract", "post-publish-owner-input", "post-publish-strict-cross-checks", "release-close-blocked-lanes" })
        {
            AssertRepairGroup(repair, group);
        }

        using JsonDocument repairValidationDocument = ReadFinalReleaseJson("owner-missing-real-publish-evidence-repair-pack-validation.json");
        JsonElement repairValidation = repairValidationDocument.RootElement;
        Assert.Equal("owner-missing-real-publish-evidence-repair-pack-validation-ready-non-proof", repairValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, repairValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertNonProofFlags(repairValidation);

        using JsonDocument scanDocument = ReadFinalReleaseJson("public-article-draft-boundary-scan.json");
        JsonElement scan = scanDocument.RootElement;
        Assert.Equal("public-article-draft-boundary-scan", scan.GetProperty("recordKind").GetString());
        Assert.Equal("docs/articles/zh-cn", scan.GetProperty("articlesRoot").GetString());
        Assert.True(scan.GetProperty("scannedFileCount").GetInt32() >= 1);
        Assert.True(scan.GetProperty("scanPatternCount").GetInt32() >= 8);
        AssertNonProofFlags(scan);

        string scanRaw = scan.GetRawText();
        Assert.Contains("nuget-published", scanRaw, StringComparison.Ordinal);
        Assert.Contains("clean-consumer-passed", scanRaw, StringComparison.Ordinal);
        Assert.Contains("release-closed", scanRaw, StringComparison.Ordinal);
        Assert.Contains("local-feed-proof", scanRaw, StringComparison.Ordinal);

        using JsonDocument scanValidationDocument = ReadFinalReleaseJson("public-article-draft-boundary-scan-validation.json");
        JsonElement scanValidation = scanValidationDocument.RootElement;
        Assert.Equal("public-article-draft-boundary-scan-validation-ready-non-proof", scanValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, scanValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertNonProofFlags(scanValidation);
    }

    private static void RunActionDashboardPipeline()
    {
        RunPowerShell("Export-FinalReadonlyPublishAuditPack.ps1");
        RunPowerShell("Test-FinalReadonlyPublishAuditPack.ps1", "-Strict");
        RunPowerShell("Export-FinalOwnerOneScreenExecutionManual.ps1");
        RunPowerShell("Test-FinalOwnerOneScreenExecutionManual.ps1", "-Strict");
        RunPowerShell("Export-OwnerPublicPublishExecutionResultInputContract.ps1");
        RunPowerShell("Export-OwnerRealPublishEvidenceIntakeDryRunPack.ps1");
        RunPowerShell("Test-OwnerRealPublishEvidenceIntakeDryRunPack.ps1", "-Strict");
        RunPowerShell("Export-PostPublishStrictCrossCheckPack.ps1");
        RunPowerShell("Test-PostPublishStrictCrossCheckPack.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCloseStrictEvidenceClosure.ps1");
        RunPowerShell("Test-ReleaseCloseStrictEvidenceClosure.ps1", "-Strict");
        RunPowerShell("Export-ArticleRoadmap30Plus.ps1");
        RunPowerShell("Test-ArticleRoadmap30Plus.ps1", "-Strict");
        RunPowerShell("Export-TechnicalArticlePublicationMatrix.ps1");
        RunPowerShell("Export-ReleaseCloseStrictProofExecutionOrder.ps1");
        RunPowerShell("Export-ArticlePublishingReadinessMap.ps1");
        RunPowerShell("Test-ArticlePublishingReadinessMap.ps1", "-Strict");
        RunPowerShell("Export-FinalOwnerPublishExecutionReplayChecklistPack.ps1");
        RunPowerShell("Test-FinalOwnerPublishExecutionReplayChecklistPack.ps1", "-Strict");
        RunPowerShell("Export-FinalOwnerPublishEvidenceImportRunbook.ps1");
        RunPowerShell("Test-FinalOwnerPublishEvidenceImportRunbook.ps1", "-Strict");
        RunPowerShell("Export-PublicArticleReadinessMatrix.ps1");
        RunPowerShell("Test-PublicArticleReadinessMatrix.ps1", "-Strict");
        RunPowerShell("Export-PostPublishArticleProofGate.ps1");
        RunPowerShell("Test-PostPublishArticleProofGate.ps1", "-Strict");
        RunPowerShell("Export-FinalOwnerPublishAndArticleActionDashboard.ps1");
        RunPowerShell("Test-FinalOwnerPublishAndArticleActionDashboard.ps1", "-Strict");
        RunPowerShell("Export-OwnerMissingRealPublishEvidenceRepairPack.ps1");
        RunPowerShell("Test-OwnerMissingRealPublishEvidenceRepairPack.ps1", "-Strict");
        RunPowerShell("Export-PublicArticleDraftBoundaryScan.ps1");
        RunPowerShell("Test-PublicArticleDraftBoundaryScan.ps1", "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static void AssertDashboardGate(JsonElement dashboard, string id)
    {
        Assert.Contains(dashboard.GetProperty("gates").EnumerateArray(), gate =>
            gate.GetProperty("id").GetString() == id &&
            gate.GetProperty("ownerActionRequired").GetBoolean() &&
            gate.GetProperty("blocked").GetBoolean() &&
            !gate.GetProperty("performsPublish").GetBoolean() &&
            !gate.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    private static void AssertRepairGroup(JsonElement repair, string id)
    {
        Assert.Contains(repair.GetProperty("repairGroups").EnumerateArray(), group =>
            group.GetProperty("id").GetString() == id &&
            group.GetProperty("ownerActionRequired").GetBoolean() &&
            group.GetProperty("blocked").GetBoolean() &&
            group.GetProperty("missingItemCount").GetInt32() >= 1 &&
            !group.GetProperty("performsPublish").GetBoolean() &&
            !group.GetProperty("isProof").GetBoolean());
    }

    private static void AssertNonProofFlags(JsonElement element)
    {
        Assert.False(element.GetProperty("performsPublish").GetBoolean());
        Assert.False(element.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(element.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", scriptName), arguments);
    }
}
