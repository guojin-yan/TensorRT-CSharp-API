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

    [Fact]
    public void ArticleOwnerReviewRewriteDraftAndFinalActionPackStayArtifactOnly()
    {
        RunActionDashboardPipeline();
        RunPowerShell("Export-PublicArticleBlockedClaimOwnerReviewList.ps1");
        RunPowerShell("Test-PublicArticleBlockedClaimOwnerReviewList.ps1", "-Strict");
        RunPowerShell("Export-PublicArticleSafeRewriteDraftPack.ps1");
        RunPowerShell("Test-PublicArticleSafeRewriteDraftPack.ps1", "-Strict");
        RunPowerShell("Export-OwnerFinalMissingActionOneScreenPack.ps1");
        RunPowerShell("Test-OwnerFinalMissingActionOneScreenPack.ps1", "-Strict");

        using JsonDocument reviewDocument = ReadFinalReleaseJson("public-article-blocked-claim-owner-review-list.json");
        JsonElement review = reviewDocument.RootElement;
        Assert.Equal("public-article-blocked-claim-owner-review-list", review.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-public-article-claim-owner-review-required", review.GetProperty("reviewState").GetString());
        Assert.True(review.GetProperty("blockedClaimCount").GetInt32() >= 1);
        Assert.Equal(review.GetProperty("blockedClaimCount").GetInt32(), review.GetProperty("reviewItemCount").GetInt32());
        AssertNonProofFlags(review);

        string reviewRaw = review.GetRawText();
        Assert.Contains("remove", reviewRaw, StringComparison.Ordinal);
        Assert.Contains("downgrade-to-roadmap", reviewRaw, StringComparison.Ordinal);
        Assert.Contains("wait-for-real-proof", reviewRaw, StringComparison.Ordinal);
        Assert.Contains("bind-to-evidence-field", reviewRaw, StringComparison.Ordinal);
        Assert.Contains("post-publish-owner-input", reviewRaw, StringComparison.Ordinal);
        Assert.Contains("forbidden-substitute-scan", reviewRaw, StringComparison.Ordinal);

        using JsonDocument reviewValidationDocument = ReadFinalReleaseJson("public-article-blocked-claim-owner-review-list-validation.json");
        JsonElement reviewValidation = reviewValidationDocument.RootElement;
        Assert.Equal("public-article-blocked-claim-owner-review-list-validation-ready-non-proof", reviewValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, reviewValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertNonProofFlags(reviewValidation);

        using JsonDocument rewriteDocument = ReadFinalReleaseJson("public-article-safe-rewrite-draft-pack.json");
        JsonElement rewrite = rewriteDocument.RootElement;
        Assert.Equal("public-article-safe-rewrite-draft-pack", rewrite.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-public-article-safe-rewrite-draft-owner-review-required", rewrite.GetProperty("draftState").GetString());
        Assert.True(rewrite.GetProperty("draftItemCount").GetInt32() >= 1);
        Assert.False(rewrite.GetProperty("writesSourceArticles").GetBoolean());
        AssertNonProofFlags(rewrite);

        string rewriteRaw = rewrite.GetRawText();
        Assert.Contains("artifact-only-no-source-overwrite", rewriteRaw, StringComparison.Ordinal);
        Assert.Contains("不能作为 proof", rewriteRaw, StringComparison.Ordinal);
        Assert.Contains("不得宣称已关闭", rewriteRaw, StringComparison.Ordinal);
        Assert.Contains("post-publish-owner-input", rewriteRaw, StringComparison.Ordinal);

        using JsonDocument rewriteValidationDocument = ReadFinalReleaseJson("public-article-safe-rewrite-draft-pack-validation.json");
        JsonElement rewriteValidation = rewriteValidationDocument.RootElement;
        Assert.Equal("public-article-safe-rewrite-draft-pack-validation-ready-non-proof", rewriteValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, rewriteValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertNonProofFlags(rewriteValidation);

        using JsonDocument oneScreenDocument = ReadFinalReleaseJson("owner-final-missing-action-one-screen-pack.json");
        JsonElement oneScreen = oneScreenDocument.RootElement;
        Assert.Equal("owner-final-missing-action-one-screen-pack", oneScreen.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-final-missing-actions-required", oneScreen.GetProperty("oneScreenState").GetString());
        Assert.True(oneScreen.GetProperty("missingActionCount").GetInt32() >= 7);
        AssertNonProofFlags(oneScreen);

        string oneScreenRaw = oneScreen.GetRawText();
        Assert.Contains("owner-authorize-real-publish", oneScreenRaw, StringComparison.Ordinal);
        Assert.Contains("run-real-public-publish", oneScreenRaw, StringComparison.Ordinal);
        Assert.Contains("run-external-clean-consumer", oneScreenRaw, StringComparison.Ordinal);
        Assert.Contains("clear-article-proof-gate", oneScreenRaw, StringComparison.Ordinal);
        Assert.Contains("close-release-strictly", oneScreenRaw, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec", oneScreenRaw, StringComparison.Ordinal);

        using JsonDocument oneScreenValidationDocument = ReadFinalReleaseJson("owner-final-missing-action-one-screen-pack-validation.json");
        JsonElement oneScreenValidation = oneScreenValidationDocument.RootElement;
        Assert.Equal("owner-final-missing-action-one-screen-pack-validation-ready-non-proof", oneScreenValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, oneScreenValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertNonProofFlags(oneScreenValidation);
    }

    [Fact]
    public void ArticlePatchProposalAndAuthorizationCommandGuardStayNonProof()
    {
        RunActionDashboardPipeline();
        RunPowerShell("Export-PublicArticleBlockedClaimOwnerReviewList.ps1");
        RunPowerShell("Test-PublicArticleBlockedClaimOwnerReviewList.ps1", "-Strict");
        RunPowerShell("Export-PublicArticleSafeRewriteDraftPack.ps1");
        RunPowerShell("Test-PublicArticleSafeRewriteDraftPack.ps1", "-Strict");
        RunPowerShell("Export-PublicArticleSourcePatchProposalPack.ps1");
        RunPowerShell("Test-PublicArticleSourcePatchProposalPack.ps1", "-Strict");
        RunPowerShell("Export-OwnerAuthorizationCommandGuardPack.ps1");
        RunPowerShell("Test-OwnerAuthorizationCommandGuardPack.ps1", "-Strict");

        using JsonDocument proposalDocument = ReadFinalReleaseJson("public-article-source-patch-proposal-pack.json");
        JsonElement proposal = proposalDocument.RootElement;
        Assert.Equal("public-article-source-patch-proposal-pack", proposal.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-public-article-source-patch-proposal-owner-review-required", proposal.GetProperty("proposalState").GetString());
        Assert.True(proposal.GetProperty("proposalCount").GetInt32() >= 64);
        Assert.False(proposal.GetProperty("writesSourceArticles").GetBoolean());
        AssertNonProofFlags(proposal);

        string proposalRaw = proposal.GetRawText();
        Assert.Contains("artifact-only-proposal", proposalRaw, StringComparison.Ordinal);
        Assert.Contains("originalMatchedText", proposalRaw, StringComparison.Ordinal);
        Assert.Contains("safeRewriteZh", proposalRaw, StringComparison.Ordinal);
        Assert.Contains("requiredProofFieldOrGate", proposalRaw, StringComparison.Ordinal);
        Assert.Contains("patchRisk", proposalRaw, StringComparison.Ordinal);
        Assert.Contains("ProjectReference", proposalRaw, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec", proposalRaw, StringComparison.Ordinal);

        using JsonDocument proposalValidationDocument = ReadFinalReleaseJson("public-article-source-patch-proposal-pack-validation.json");
        JsonElement proposalValidation = proposalValidationDocument.RootElement;
        Assert.Equal("public-article-source-patch-proposal-pack-validation-ready-non-proof", proposalValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, proposalValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertNonProofFlags(proposalValidation);

        using JsonDocument guardDocument = ReadFinalReleaseJson("owner-authorization-command-guard-pack.json");
        JsonElement guard = guardDocument.RootElement;
        Assert.Equal("owner-authorization-command-guard-pack", guard.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-authorization-required-for-publish-close-and-article-release", guard.GetProperty("guardState").GetString());
        Assert.True(guard.GetProperty("commandCount").GetInt32() >= 8);
        Assert.True(guard.GetProperty("blockedCommandCount").GetInt32() >= 5);
        Assert.True(guard.GetProperty("allowedReadonlyCommandCount").GetInt32() >= 3);
        AssertNonProofFlags(guard);

        string guardRaw = guard.GetRawText();
        Assert.Contains("dotnet nuget push", guardRaw, StringComparison.Ordinal);
        Assert.Contains("nuget-org-push", guardRaw, StringComparison.Ordinal);
        Assert.Contains("github-packages-push", guardRaw, StringComparison.Ordinal);
        Assert.Contains("workflow-dispatch-publish", guardRaw, StringComparison.Ordinal);
        Assert.Contains("blocked-until-owner-authorization-and-real-proof", guardRaw, StringComparison.Ordinal);
        Assert.Contains("allowed-before-owner-authorization-readonly-non-proof", guardRaw, StringComparison.Ordinal);
        Assert.Contains("deprecation", guardRaw, StringComparison.Ordinal);

        using JsonDocument guardValidationDocument = ReadFinalReleaseJson("owner-authorization-command-guard-pack-validation.json");
        JsonElement guardValidation = guardValidationDocument.RootElement;
        Assert.Equal("owner-authorization-command-guard-pack-validation-ready-non-proof", guardValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, guardValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertNonProofFlags(guardValidation);
    }

    [Fact]
    public void ArticlePatchApplyReadinessAndFinalAuthorizationSummaryStayNonProof()
    {
        RunActionDashboardPipeline();
        RunPowerShell("Export-PublicArticleBlockedClaimOwnerReviewList.ps1");
        RunPowerShell("Test-PublicArticleBlockedClaimOwnerReviewList.ps1", "-Strict");
        RunPowerShell("Export-PublicArticleSafeRewriteDraftPack.ps1");
        RunPowerShell("Test-PublicArticleSafeRewriteDraftPack.ps1", "-Strict");
        RunPowerShell("Export-PublicArticleSourcePatchProposalPack.ps1");
        RunPowerShell("Test-PublicArticleSourcePatchProposalPack.ps1", "-Strict");
        RunPowerShell("Export-PublicArticleSourcePatchApplyReadinessPack.ps1");
        RunPowerShell("Test-PublicArticleSourcePatchApplyReadinessPack.ps1", "-Strict");
        RunPowerShell("Export-OwnerAuthorizationCommandGuardPack.ps1");
        RunPowerShell("Test-OwnerAuthorizationCommandGuardPack.ps1", "-Strict");
        RunPowerShell("Export-OwnerFinalMissingActionOneScreenPack.ps1");
        RunPowerShell("Test-OwnerFinalMissingActionOneScreenPack.ps1", "-Strict");
        RunPowerShell("Export-OwnerFinalAuthorizationRequestSummary.ps1");
        RunPowerShell("Test-OwnerFinalAuthorizationRequestSummary.ps1", "-Strict");

        using JsonDocument readinessDocument = ReadFinalReleaseJson("public-article-source-patch-apply-readiness-pack.json");
        JsonElement readiness = readinessDocument.RootElement;
        Assert.Equal("public-article-source-patch-apply-readiness-pack", readiness.GetProperty("recordKind").GetString());
        Assert.True(readiness.GetProperty("readinessItemCount").GetInt32() >= 64);
        Assert.False(readiness.GetProperty("writesSourceArticles").GetBoolean());
        AssertNonProofFlags(readiness);

        string readinessRaw = readiness.GetRawText();
        Assert.Contains("lineExists", readinessRaw, StringComparison.Ordinal);
        Assert.Contains("matchedTextStillPresent", readinessRaw, StringComparison.Ordinal);
        Assert.Contains("nearbyMatchFound", readinessRaw, StringComparison.Ordinal);
        Assert.Contains("canApplyAfterOwnerApproval", readinessRaw, StringComparison.Ordinal);
        Assert.Contains("证据导入前", readinessRaw, StringComparison.Ordinal);

        using JsonDocument readinessValidationDocument = ReadFinalReleaseJson("public-article-source-patch-apply-readiness-pack-validation.json");
        JsonElement readinessValidation = readinessValidationDocument.RootElement;
        Assert.Equal("public-article-source-patch-apply-readiness-pack-validation-ready-non-proof", readinessValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, readinessValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertNonProofFlags(readinessValidation);

        using JsonDocument summaryDocument = ReadFinalReleaseJson("owner-final-authorization-request-summary.json");
        JsonElement summary = summaryDocument.RootElement;
        Assert.Equal("owner-final-authorization-request-summary", summary.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-final-authorization-required", summary.GetProperty("requestState").GetString());
        Assert.True(summary.GetProperty("blockedCommandCount").GetInt32() >= 5);
        Assert.True(summary.GetProperty("allowedReadonlyCommandCount").GetInt32() >= 3);
        Assert.True(summary.GetProperty("articleProposalCount").GetInt32() >= 64);
        AssertNonProofFlags(summary);

        string summaryRaw = summary.GetRawText();
        Assert.Contains("Authorize real public publish", summaryRaw, StringComparison.Ordinal);
        Assert.Contains("keep all publish/article/release-close commands blocked", summaryRaw, StringComparison.Ordinal);
        Assert.Contains("owner-public-publish-evidence", summaryRaw, StringComparison.Ordinal);
        Assert.Contains("post-publish-owner-input", summaryRaw, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec report", summaryRaw, StringComparison.Ordinal);

        using JsonDocument summaryValidationDocument = ReadFinalReleaseJson("owner-final-authorization-request-summary-validation.json");
        JsonElement summaryValidation = summaryValidationDocument.RootElement;
        Assert.Equal("owner-final-authorization-request-summary-validation-ready-non-proof", summaryValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, summaryValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertNonProofFlags(summaryValidation);
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
