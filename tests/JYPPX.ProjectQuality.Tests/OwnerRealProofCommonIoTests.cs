using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OwnerRealProofCommonIoTests
{
    [Fact]
    public void OwnerRealProofCommonUsesAtomicWritesAndRetryReads()
    {
        string commonPath = Path.Combine(RepositoryPaths.Root, "eng", "OwnerRealProofCommon.ps1");
        string common = File.ReadAllText(commonPath);

        Assert.Contains("function Write-Utf8FileAtomic", common, StringComparison.Ordinal);
        Assert.Contains("[System.IO.File]::Replace", common, StringComparison.Ordinal);
        Assert.Contains("for ($attempt = 1; $attempt -le 10; $attempt++)", common, StringComparison.Ordinal);
        Assert.Contains("for ($attempt = 1; $attempt -le 8; $attempt++)", common, StringComparison.Ordinal);
        Assert.Contains("ConvertFrom-Json", common, StringComparison.Ordinal);
        Assert.DoesNotContain("Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json", common, StringComparison.Ordinal);
    }

    [Fact]
    public void OwnerRealPublishEvidenceIntakeDryRunPackUsesCommonAtomicWriterForJson()
    {
        foreach (string scriptName in new[]
        {
            "Export-OwnerRealPublishEvidenceIntakeDryRunPack.ps1",
            "Export-FinalReadonlyPublishAuditPack.ps1",
            "Test-FinalReadonlyPublishAuditPack.ps1",
            "Export-FinalOwnerOneScreenExecutionManual.ps1",
            "Test-FinalOwnerOneScreenExecutionManual.ps1",
            "Test-OwnerRealPublishEvidenceIntakeDryRunPack.ps1",
            "Export-PostPublishStrictCrossCheckPack.ps1",
            "Test-PostPublishStrictCrossCheckPack.ps1",
            "Export-ReleaseCloseStrictEvidenceClosure.ps1",
            "Test-ReleaseCloseStrictEvidenceClosure.ps1",
            "Export-FinalOwnerPublishExecutionReplayChecklistPack.ps1",
            "Test-FinalOwnerPublishExecutionReplayChecklistPack.ps1",
            "Export-PublicArticleReadinessMatrix.ps1",
            "Test-PublicArticleReadinessMatrix.ps1",
            "Export-FinalOwnerPublishEvidenceImportRunbook.ps1",
            "Test-FinalOwnerPublishEvidenceImportRunbook.ps1",
            "Export-PostPublishArticleProofGate.ps1",
            "Test-PostPublishArticleProofGate.ps1"
        })
        {
            string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
            string script = File.ReadAllText(scriptPath);

            Assert.Contains("Write-Utf8File -LiteralPath $jsonPath -InputObject", script, StringComparison.Ordinal);
            Assert.DoesNotContain("Set-Content -LiteralPath $jsonPath", script, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void OwnerCloseAndPublicArticleProofScriptsUseCommonAtomicWriterForJson()
    {
        foreach (string scriptName in new[]
        {
            "Export-FinalOwnerHandoffIndex.ps1",
            "Export-FinalOwnerNextDecisionGate.ps1",
            "Export-FinalOwnerPublishAndArticleActionDashboard.ps1",
            "Export-GitHubActionsRunnerNonProofGuardPack.ps1",
            "Export-OwnerAuthorizationCommandGuardPack.ps1",
            "Export-OwnerFinalAuthorizationRequestSummary.ps1",
            "Export-OwnerFinalMissingActionOneScreenPack.ps1",
            "Export-OwnerMissingRealPublishEvidenceRepairPack.ps1",
            "Export-OwnerRealProofEvidenceBackfillPackage.ps1",
            "Export-OwnerRealProofStagingWorkspaceContract.ps1",
            "Export-PackageConsumerPreflight.ps1",
            "Export-PublicArticleBlockedClaimOwnerReviewList.ps1",
            "Export-PublicArticleDraftBoundaryScan.ps1",
            "Export-PublicArticleSafeRewriteDraftPack.ps1",
            "Export-PublicArticleSourcePatchApplyReadinessPack.ps1",
            "Export-PublicArticleSourcePatchProposalPack.ps1",
            "Export-ReleaseCloseStrictEvidenceClosureDashboard.ps1",
            "Import-FinalOwnerCloseDecision.ps1",
            "Import-FinalOwnerRollbackReview.ps1",
            "Test-FinalOwnerCloseDecision.ps1",
            "Test-FinalOwnerHandoffIndex.ps1",
            "Test-FinalOwnerNextDecisionGate.ps1",
            "Test-FinalOwnerPublishAndArticleActionDashboard.ps1",
            "Test-FinalOwnerRollbackReview.ps1",
            "Test-GitHubActionsRunnerNonProofGuardPack.ps1",
            "Test-OwnerAuthorizationCommandGuardPack.ps1",
            "Test-OwnerFinalAuthorizationRequestSummary.ps1",
            "Test-OwnerFinalMissingActionOneScreenPack.ps1",
            "Test-OwnerMissingRealPublishEvidenceRepairPack.ps1",
            "Test-OwnerRealProofEvidenceBackfillPackage.ps1",
            "Test-OwnerRealProofStagingWorkspace.ps1",
            "Test-OwnerRealProofStagingWorkspaceContract.ps1",
            "Test-PackageConsumerPreflight.ps1",
            "Test-PublicArticleBlockedClaimOwnerReviewList.ps1",
            "Test-PublicArticleDraftBoundaryScan.ps1",
            "Test-PublicArticleSafeRewriteDraftPack.ps1",
            "Test-PublicArticleSourcePatchApplyReadinessPack.ps1",
            "Test-PublicArticleSourcePatchProposalPack.ps1",
            "Test-ReleaseCloseStrictEvidenceClosureDashboard.ps1"
        })
        {
            string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
            string script = File.ReadAllText(scriptPath);

            Assert.Contains("Write-Utf8File -LiteralPath $", script, StringComparison.Ordinal);
            Assert.DoesNotContain("Set-Content -LiteralPath $jsonPath", script, StringComparison.Ordinal);
            Assert.DoesNotContain("Set-Content -LiteralPath $templatePath", script, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void OwnerPublicPublishExecutionResultCommonUsesAtomicWritesAndRetryReads()
    {
        string commonPath = Path.Combine(RepositoryPaths.Root, "eng", "OwnerPublicPublishExecutionResultCommon.ps1");
        string common = File.ReadAllText(commonPath);

        Assert.Contains("function Write-OwnerUtf8File", common, StringComparison.Ordinal);
        Assert.Contains("[IO.File]::Replace", common, StringComparison.Ordinal);
        Assert.Contains("for ($attempt = 1; $attempt -le 10; $attempt++)", common, StringComparison.Ordinal);
        Assert.Contains("for ($attempt = 1; $attempt -le 8; $attempt++)", common, StringComparison.Ordinal);
        Assert.Contains("[IO.File]::ReadAllText", common, StringComparison.Ordinal);
        Assert.DoesNotContain("[IO.File]::WriteAllText($LiteralPath", common, StringComparison.Ordinal);
        Assert.DoesNotContain("Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8 | ConvertFrom-Json", common, StringComparison.Ordinal);
    }

    [Fact]
    public void FinalOwnerExecutionAndCloseProofScriptsUseAtomicLocalWritersForJson()
    {
        foreach (string scriptName in new[]
        {
            "Export-FinalOwnerExecutionBlockerLedger.ps1",
            "Export-FinalOwnerExecutionChecklist.ps1",
            "Export-FinalOwnerExecutionExternalResultInputContract.ps1",
            "Export-FinalOwnerExecutionInputSkeleton.ps1",
            "Export-FinalOwnerExecutionOwnerInputDraft.ps1",
            "Export-FinalOwnerExecutionRealInputTemplate.ps1",
            "Export-FinalOwnerExecutionRepairChecklist.ps1",
            "Export-FinalOwnerExecutionRepairInputSkeleton.ps1",
            "Export-FinalOwnerRealProofExecutionPackage.ps1",
            "Export-FinalOwnerRealProofGapMatrix.ps1",
            "Export-FinalPostPublishCleanConsumerProofRecordContract.ps1",
            "Export-FinalReleaseCloseOwnerApprovalContract.ps1",
            "Import-FinalOwnerExecutionExternalResultCandidate.ps1",
            "Import-FinalOwnerExecutionRealInput.ps1",
            "Import-FinalPostPublishCleanConsumerProofCandidate.ps1",
            "Import-FinalReleaseCloseOwnerApprovalCandidate.ps1",
            "Test-FinalOwnerExecutionBlockerLedger.ps1",
            "Test-FinalOwnerExecutionChecklist.ps1",
            "Test-FinalOwnerExecutionCloseReadinessFromRealInput.ps1",
            "Test-FinalOwnerExecutionExternalResultCandidate.ps1",
            "Test-FinalOwnerExecutionExternalResultInputContract.ps1",
            "Test-FinalOwnerExecutionExternalResultInputPreflight.ps1",
            "Test-FinalOwnerExecutionInputPreflight.ps1",
            "Test-FinalOwnerExecutionInputSkeleton.ps1",
            "Test-FinalOwnerExecutionOwnerInputDraft.ps1",
            "Test-FinalOwnerExecutionRealInputImport.ps1",
            "Test-FinalOwnerExecutionRealInputStrictPreflight.ps1",
            "Test-FinalOwnerExecutionRealInputTemplate.ps1",
            "Test-FinalOwnerExecutionRepairChecklist.ps1",
            "Test-FinalOwnerExecutionRepairInputSkeleton.ps1",
            "Test-FinalOwnerRealProofConvergenceGate.ps1",
            "Test-FinalOwnerRealProofExecutionPackage.ps1",
            "Test-FinalOwnerRealProofGapMatrix.ps1",
            "Test-FinalPostPublishCleanConsumerProofCandidate.ps1",
            "Test-FinalPostPublishCleanConsumerProofPreflight.ps1",
            "Test-FinalPostPublishCleanConsumerProofRecordContract.ps1",
            "Test-FinalReleaseCloseOwnerApprovalCandidate.ps1",
            "Test-FinalReleaseCloseOwnerApprovalContract.ps1",
            "Test-FinalReleaseCloseOwnerApprovalPreflight.ps1"
        })
        {
            string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
            string script = File.ReadAllText(scriptPath);

            Assert.Contains("function Write-Utf8File", script, StringComparison.Ordinal);
            Assert.Contains("[System.IO.File]::Replace", script, StringComparison.Ordinal);
            Assert.Contains("for ($attempt = 1; $attempt -le 10; $attempt++)", script, StringComparison.Ordinal);
            Assert.DoesNotContain("WriteAllText($LiteralPath", script, StringComparison.Ordinal);
            Assert.DoesNotContain("Set-Content -LiteralPath $jsonPath", script, StringComparison.Ordinal);
            Assert.DoesNotContain("Set-Content -LiteralPath $templateJsonPath", script, StringComparison.Ordinal);
            Assert.DoesNotContain("Set-Content -LiteralPath $exampleJsonPath", script, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void OwnerProofAndRealProofScriptsUseCommonAtomicWriterForJson()
    {
        foreach (string scriptName in new[]
        {
            "Export-OwnerInputPreflightBundle.ps1",
            "Export-OwnerAuthorizedPublishCommandPlan.ps1",
            "Export-OwnerExternalProofBackfillOrchestrator.ps1",
            "Export-OwnerExternalExecutionResultBackfillKit.ps1",
            "Export-OwnerInputContractConvergence.ps1",
            "Export-OwnerExternalProofExecutionResultInputTemplate.ps1",
            "Export-OwnerInputCrossHashAudit.ps1",
            "Export-OwnerExternalProofExecutionBundle.ps1",
            "Export-OwnerExternalProofInputPreflight.ps1",
            "Test-OwnerExternalExecutionResultBackfillKit.ps1",
            "Export-OwnerRealProofFinalActionWorklist.ps1",
            "Test-OwnerInputContractConvergence.ps1",
            "Test-OwnerProofInputDraftPack.ps1",
            "Test-OwnerAuthorizedPublishCommandPlan.ps1",
            "Export-OwnerRealProofReportPack.ps1",
            "Export-OwnerRealProofFieldDeltaPack.ps1",
            "Export-OwnerRuntimeProofResultInputTemplate.ps1",
            "Test-OwnerOnlyPublishExecutionCandidate.ps1",
            "Export-OwnerRealProofExecutionClosurePack.ps1",
            "Test-OwnerExternalProofExecutionResultInputTemplate.ps1",
            "Export-OwnerRuntimeProofExecutionRunbook.ps1",
            "Export-OwnerReleaseExecutionPackage.ps1",
            "Test-OwnerExternalProofExecutionBundle.ps1",
            "Test-OwnerExternalProofBackfillOrchestrator.ps1",
            "Test-OwnerExternalProofExecutionResultImport.ps1",
            "Test-OwnerInputCrossHashAudit.ps1",
            "Export-OwnerProofBackfillExecutionPack.ps1",
            "Export-OwnerProofExecutionHandoff.ps1",
            "Export-OwnerProofInputReadiness.ps1",
            "Export-OwnerProofInputDraftPack.ps1",
            "Export-OwnerProofRealBackfillExecutionPack.ps1",
            "Export-OwnerProofInputRepairPack.ps1",
            "Export-OwnerProofRealInputConvergence.ps1",
            "Test-OwnerProofRealInputConvergence.ps1",
            "Export-OwnerRealEvidenceImportPacket.ps1",
            "Test-OwnerProofInputReadiness.ps1",
            "Test-OwnerProofRealBackfillExecutionPack.ps1",
            "Test-OwnerRealEvidenceImportPacket.ps1",
            "Test-OwnerRealProofExecutionClosurePack.ps1",
            "Test-OwnerRealProofReportPack.ps1",
            "Test-OwnerRealProofFieldDeltaPack.ps1",
            "Test-OwnerReleaseExecutionPackage.ps1",
            "Test-OwnerRuntimeProofExecutionRunbook.ps1",
            "Test-OwnerRuntimeProofResultInput.ps1"
        })
        {
            string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
            string script = File.ReadAllText(scriptPath);

            Assert.Contains("OwnerRealProofCommon.ps1", script, StringComparison.Ordinal);
            Assert.Contains("Write-Utf8File -LiteralPath $", script, StringComparison.Ordinal);
            Assert.DoesNotContain("Set-Content -LiteralPath $jsonPath", script, StringComparison.Ordinal);
            Assert.DoesNotContain("Set-Content -LiteralPath $outputFullPath", script, StringComparison.Ordinal);
            Assert.DoesNotContain("Set-Content -LiteralPath $resolvedOutputPath", script, StringComparison.Ordinal);
            Assert.DoesNotContain("Set-Content -LiteralPath $templatePath", script, StringComparison.Ordinal);
        }
    }
}
