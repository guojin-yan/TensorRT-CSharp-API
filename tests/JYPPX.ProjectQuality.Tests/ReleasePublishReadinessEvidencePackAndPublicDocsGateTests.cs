using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleasePublishReadinessEvidencePackAndPublicDocsGateTests
{
    [Fact]
    public void PublicDocsAndPackageMetadataGateRejectsOverClaimsWithoutPromotion()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicDocsAndPackageMetadataGate.ps1"), "-Strict");

        using JsonDocument document = ReadJson("artifacts", "final-release", "public-docs-package-metadata-gate.json");
        JsonElement gate = document.RootElement;

        Assert.Equal("public-docs-package-metadata-gate", gate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-public-postpublish-proof-required", gate.GetProperty("gateState").GetString());
        Assert.Equal(0, gate.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(gate.GetProperty("scannedPathCount").GetInt32() > 0);
        Assert.False(gate.GetProperty("performsPublish").GetBoolean());
        Assert.False(gate.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(gate.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(gate.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.Empty(gate.GetProperty("blockedMatches").EnumerateArray());
        Assert.Equal("inline-plus-markdown-heading-and-table-header", gate.GetProperty("boundaryContextMode").GetString());
        JsonElement[] allowedMatches = gate.GetProperty("allowedBoundaryMatches").EnumerateArray().ToArray();
        Assert.Contains(allowedMatches, static item =>
            item.GetProperty("file").GetString() == "README.md" &&
            item.GetProperty("text").GetString()!.Contains("without `ProjectReference`", StringComparison.Ordinal));
        Assert.Contains(allowedMatches, static item =>
            item.GetProperty("file").GetString()!.Replace('\\', '/') == "docs/articles/zh-cn/why-not-plain-pinvoke.md" &&
            item.GetProperty("text").GetString()!.Contains("用 ProjectReference 证明包可用", StringComparison.Ordinal) &&
            item.GetProperty("structuredContext").GetString()!.Contains("不应采用的修复", StringComparison.Ordinal));

        string raw = gate.GetRawText();
        foreach (string marker in new[]
        {
            "no-yolodet-live",
            "no-tensorrt-layer-tensor-info-live",
            "no-publication-overclaim",
            "no-runtime-proof-overclaim",
            "no-forbidden-substitute-as-proof",
            "public-boundary-markers-present",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "not proof"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        string markdown = ReadText("artifacts", "final-release", "public-docs-package-metadata-gate.md");
        Assert.Contains("Public Docs And Package Metadata Gate", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleasePublishReadinessEvidencePackAggregatesGatesWithoutPromotion()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionOwnerProofFieldDeltaRepairPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseProofLaneWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-TensorRtExecAdvancedProofReadinessChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseProofDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseProofDashboard.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerInputPreflightBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicDocsAndPackageMetadataGate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPublishProofGate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerProofActionWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerProofActionWorklist.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionPackage.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CleanExternalPackageConsumerOwnerRunbook.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-CleanExternalPackageConsumerOwnerRunbook.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishOwnerVerificationRunbook.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishOwnerVerificationRunbook.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePublishReadinessEvidencePack.ps1"));

        using JsonDocument document = ReadJson("artifacts", "final-release", "release-publish-readiness-evidence-pack.json");
        JsonElement pack = document.RootElement;

        Assert.Equal("release-publish-readiness-evidence-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-public-postpublish-proof-required", pack.GetProperty("packState").GetString());
        Assert.False(pack.GetProperty("performsPublish").GetBoolean());
        Assert.False(pack.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(pack.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(pack.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.Equal("blocked-owner-input-required", pack.GetProperty("ownerInputPreflightState").GetString());
        Assert.Equal("blocked-owner-proof-required", pack.GetProperty("dashboardValidationState").GetString());
        Assert.Equal("blocked-final-publish-real-proof-required", pack.GetProperty("finalGateState").GetString());
        Assert.Equal("blocked-final-owner-proof-action-required", pack.GetProperty("finalOwnerProofActionWorklistState").GetString());
        Assert.Equal("blocked-final-owner-proof-action-required", pack.GetProperty("finalOwnerProofActionWorklistValidationState").GetString());
        Assert.Equal(8, pack.GetProperty("finalOwnerProofActionCount").GetInt32());
        Assert.Equal(8, pack.GetProperty("finalOwnerProofBlockedActionCount").GetInt32());
        Assert.Equal(0, pack.GetProperty("finalOwnerProofMissingActionRequiredIdCount").GetInt32());
        Assert.Equal("blocked-final-owner-execution-required", pack.GetProperty("finalOwnerExecutionPackageState").GetString());
        Assert.Equal("blocked-final-owner-execution-required", pack.GetProperty("finalOwnerExecutionPackageValidationState").GetString());
        Assert.Equal(8, pack.GetProperty("finalOwnerExecutionStepCount").GetInt32());
        Assert.Equal(8, pack.GetProperty("finalOwnerBlockedExecutionStepCount").GetInt32());
        Assert.Equal(0, pack.GetProperty("finalOwnerExecutionPackageFailedBlockerCount").GetInt32());
        Assert.Equal("blocked-owner-external-proof-execution-result-required", pack.GetProperty("ownerExternalProofExecutionResultImportState").GetString());
        Assert.Equal("blocked-owner-external-proof-execution-result-required", pack.GetProperty("ownerExternalProofExecutionResultImportValidationState").GetString());
        Assert.Equal(0, pack.GetProperty("ownerExternalProofExecutionResultImportReadyForStrictValidatorLaneCount").GetInt32());
        Assert.Equal(6, pack.GetProperty("ownerExternalProofExecutionResultImportBlockedLaneCount").GetInt32());
        Assert.Equal(0, pack.GetProperty("ownerExternalProofExecutionResultImportPromotableLaneCount").GetInt32());
        Assert.Equal("blocked-real-external-proof-record-import-required", pack.GetProperty("realExternalProofRecordImportValidatorState").GetString());
        Assert.Equal("blocked-real-external-proof-record-import-required", pack.GetProperty("realExternalProofRecordImportValidatorValidationState").GetString());
        Assert.Equal(0, pack.GetProperty("realExternalProofRecordImportValidatorReadyForStrictValidatorContractCount").GetInt32());
        Assert.Equal(6, pack.GetProperty("realExternalProofRecordImportValidatorBlockedContractCount").GetInt32());
        Assert.Equal("blocked-owner-result-import-candidate-required", pack.GetProperty("realProofRecordCandidateFromOwnerResultImportState").GetString());
        Assert.Equal("blocked-owner-result-import-candidate-required", pack.GetProperty("realProofRecordCandidateFromOwnerResultImportValidationState").GetString());
        Assert.Equal(0, pack.GetProperty("realProofRecordCandidateFromOwnerResultImportCandidateCount").GetInt32());
        Assert.Equal(0, pack.GetProperty("realProofRecordCandidateFromOwnerResultImportStrictValidatorReadyCandidateCount").GetInt32());
        Assert.Equal("blocked-release-close-real-proof-import-required", pack.GetProperty("releaseCloseRealProofImportBridgeState").GetString());
        Assert.Equal("blocked-release-close-real-proof-import-required", pack.GetProperty("releaseCloseRealProofImportBridgeValidationState").GetString());
        Assert.Equal(6, pack.GetProperty("releaseCloseRealProofImportBridgeBlockedLaneCount").GetInt32());
        Assert.Equal(0, pack.GetProperty("releaseCloseRealProofImportBridgeFailedBlockerCount").GetInt32());
        Assert.Equal("blocked-owner-clean-external-package-consumer-execution-required", pack.GetProperty("cleanExternalPackageConsumerOwnerRunbookState").GetString());
        Assert.Equal("blocked-owner-clean-external-package-consumer-execution-required", pack.GetProperty("cleanExternalPackageConsumerOwnerRunbookValidationState").GetString());
        Assert.Equal(9, pack.GetProperty("cleanExternalPackageConsumerOwnerRunbookStepCount").GetInt32());
        Assert.Equal(9, pack.GetProperty("cleanExternalPackageConsumerOwnerRunbookBlockedStepCount").GetInt32());
        Assert.Equal(0, pack.GetProperty("cleanExternalPackageConsumerOwnerRunbookFailedBlockerCount").GetInt32());
        Assert.Equal("blocked-owner-post-publish-verification-required", pack.GetProperty("postPublishOwnerVerificationRunbookState").GetString());
        Assert.Equal("blocked-owner-post-publish-verification-required", pack.GetProperty("postPublishOwnerVerificationRunbookValidationState").GetString());
        Assert.Equal(6, pack.GetProperty("postPublishOwnerVerificationRunbookStepCount").GetInt32());
        Assert.Equal(6, pack.GetProperty("postPublishOwnerVerificationRunbookBlockedStepCount").GetInt32());
        Assert.Equal(0, pack.GetProperty("postPublishOwnerVerificationRunbookFailedBlockerCount").GetInt32());
        Assert.Equal("blocked-owner-public-postpublish-proof-required", pack.GetProperty("publicDocsPackageMetadataGateState").GetString());
        Assert.Equal(5, pack.GetProperty("readinessLaneCount").GetInt32());
        Assert.Equal(5, pack.GetProperty("blockedReadinessLaneCount").GetInt32());

        JsonElement[] lanes = pack.GetProperty("readinessLanes").EnumerateArray().ToArray();
        string[] laneIds = lanes.Select(static lane => lane.GetProperty("id").GetString()!).ToArray();
        Assert.Contains("real-model-runtime", laneIds);
        Assert.Contains("package-consumer-runtime", laneIds);
        Assert.Contains("post-publish-verification", laneIds);
        Assert.Contains("public-owner-confirmation", laneIds);
        Assert.Contains("public-docs-and-package-metadata", laneIds);
        Assert.All(lanes, lane =>
        {
            Assert.True(lane.GetProperty("blocked").GetBoolean());
            Assert.False(lane.GetProperty("performsPublish").GetBoolean());
            Assert.False(lane.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(lane.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(lane.GetProperty("canCloseReleaseIssue").GetBoolean());
        });

        string raw = pack.GetRawText();
        foreach (string marker in new[]
        {
            "owner-input-preflight-bundle.json",
            "final-owner-proof-action-worklist.json",
            "final-owner-proof-action-worklist-validation.json",
            "final-owner-execution-package.json",
            "final-owner-execution-package-validation.json",
            "owner-external-proof-execution-result-import.json",
            "real-external-proof-record-import-validator.json",
            "real-proof-record-candidate-from-owner-result-import.json",
            "release-close-real-proof-import-bridge-validation.json",
            "clean-external-package-consumer-owner-runbook.json",
            "clean-external-package-consumer-owner-runbook-validation.json",
            "post-publish-owner-verification-runbook.json",
            "post-publish-owner-verification-runbook-validation.json",
            "release-proof-dashboard-validation.json",
            "public-docs-package-metadata-gate.json",
            "final-publish-proof-gate-report.json",
            "package-consumer-runtime-proof-owner-input-validation.json",
            "post-publish-verification-validation.json",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "sample-run-evidence",
            "final-owner-proof-action-worklist",
            "final-owner-execution-package",
            "owner-result-candidate-bridge",
            "owner executable clean external consumer guidance only",
            "owner executable post-publish guidance only",
            "strict-validator input only",
            "nonProofEvidenceCatalog",
            "non-publishing, non-proof aggregator"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        string markdown = ReadText("artifacts", "final-release", "release-publish-readiness-evidence-pack.md");
        Assert.Contains("Release Publish Readiness Evidence Pack", markdown, StringComparison.Ordinal);
        Assert.Contains("Readiness Lanes", markdown, StringComparison.Ordinal);
        Assert.Contains("Non-Proof Evidence Catalog", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseEvidenceBundleCarriesFinalOwnerProofActionWorklistWithoutPromotion()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerProofActionWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerProofActionWorklist.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionPackage.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CleanExternalPackageConsumerOwnerRunbook.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-CleanExternalPackageConsumerOwnerRunbook.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishOwnerVerificationRunbook.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishOwnerVerificationRunbook.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument document = ReadJson("artifacts", "final-release", "release-evidence-bundle.json");
        JsonElement bundle = document.RootElement;

        Assert.Equal("blocked-final-owner-proof-action-required", bundle.GetProperty("finalOwnerProofActionWorklistState").GetString());
        Assert.Equal("blocked-final-owner-proof-action-required", bundle.GetProperty("finalOwnerProofActionWorklistValidationState").GetString());
        Assert.Equal(8, bundle.GetProperty("finalOwnerProofActionWorklistActionCount").GetInt32());
        Assert.Equal(8, bundle.GetProperty("finalOwnerProofActionWorklistBlockedActionCount").GetInt32());
        Assert.Equal(0, bundle.GetProperty("finalOwnerProofActionWorklistMissingActionRequiredIdCount").GetInt32());
        Assert.False(bundle.GetProperty("finalOwnerProofActionWorklistPerformsPublish").GetBoolean());
        Assert.False(bundle.GetProperty("finalOwnerProofActionWorklistCanPromoteRuntimeProof").GetBoolean());
        Assert.False(bundle.GetProperty("finalOwnerProofActionWorklistCanPublishPublicly").GetBoolean());
        Assert.False(bundle.GetProperty("finalOwnerProofActionWorklistCanCloseReleaseIssue").GetBoolean());
        Assert.Equal("blocked-final-owner-execution-required", bundle.GetProperty("finalOwnerExecutionPackageState").GetString());
        Assert.Equal("blocked-final-owner-execution-required", bundle.GetProperty("finalOwnerExecutionPackageValidationState").GetString());
        Assert.Equal(8, bundle.GetProperty("finalOwnerExecutionPackageExecutionStepCount").GetInt32());
        Assert.Equal(8, bundle.GetProperty("finalOwnerExecutionPackageBlockedExecutionStepCount").GetInt32());
        Assert.Equal(0, bundle.GetProperty("finalOwnerExecutionPackageFailedBlockerCount").GetInt32());
        Assert.Equal(8, bundle.GetProperty("finalOwnerExecutionPackageReleaseCloseRealInputChainCount").GetInt32());
        Assert.True(bundle.GetProperty("finalOwnerExecutionPackageReleaseCloseRealInputChainRequiredFieldCount").GetInt32() >= 100);
        Assert.True(bundle.GetProperty("finalOwnerExecutionPackageReleaseCloseRealInputChainRejectedSubstituteCount").GetInt32() >= 30);
        Assert.True(bundle.GetProperty("finalOwnerExecutionPackageReleaseCloseRealInputChainSourceReadinessSignalCount").GetInt32() >= 18);
        Assert.True(bundle.GetProperty("finalOwnerExecutionPackageReleaseCloseRealInputChainBlockedRealInputCount").GetInt32() > 0);
        Assert.Equal(8, bundle.GetProperty("finalOwnerExecutionPackageOwnerReleaseCloseHardGateCount").GetInt32());
        Assert.Equal(8, bundle.GetProperty("finalOwnerExecutionPackageBlockedOwnerReleaseCloseHardGateCount").GetInt32());
        Assert.True(bundle.GetProperty("finalOwnerExecutionPackageSourceArtifactEvidenceCount").GetInt32() >= 14);
        Assert.Equal(0, bundle.GetProperty("finalOwnerExecutionPackageSourceArtifactEvidenceMissingCount").GetInt32());
        Assert.Equal(
            bundle.GetProperty("finalOwnerExecutionPackageSourceArtifactEvidenceCount").GetInt32(),
            bundle.GetProperty("finalOwnerExecutionPackageSourceArtifactEvidenceSha256Count").GetInt32());
        Assert.Equal(
            bundle.GetProperty("finalOwnerExecutionPackageSourceArtifactEvidenceCount").GetInt32(),
            bundle.GetProperty("finalOwnerExecutionPackageSourceArtifactEvidenceNonProofBoundaryCount").GetInt32());
        Assert.False(bundle.GetProperty("finalOwnerExecutionPackagePublicPackageDownloadProofCandidateReady").GetBoolean());
        Assert.False(bundle.GetProperty("finalOwnerExecutionPackagePostPublishProofCandidateReady").GetBoolean());
        Assert.False(bundle.GetProperty("finalOwnerExecutionPackagePostPublishProofSourceLinkageReady").GetBoolean());
        Assert.False(string.IsNullOrWhiteSpace(bundle.GetProperty("finalOwnerExecutionPackageReleaseEvidenceBundleSha256").GetString()));
        Assert.Equal("blocked-release-issue-close-owner-decision-input-required", bundle.GetProperty("finalOwnerExecutionPackageReleaseIssueCloseOwnerDecisionValidationState").GetString());
        Assert.Equal("blocked-final-close-gate-owner-proof-required", bundle.GetProperty("finalOwnerExecutionPackageFinalCloseStrictValidatorOutputState").GetString());
        Assert.True(bundle.GetProperty("finalOwnerExecutionPackagePublicDownloadCannotSubstitutePostPublishProof").GetBoolean());
        Assert.True(bundle.GetProperty("finalOwnerExecutionPackageBundleHashCannotSubstituteFinalCloseDecision").GetBoolean());
        Assert.True(bundle.GetProperty("finalOwnerExecutionPackageStrictCloseOutputCannotCloseIssue").GetBoolean());
        Assert.Equal("blocked-owner-external-proof-execution-result-required", bundle.GetProperty("ownerExternalProofExecutionResultImportState").GetString());
        Assert.Equal("blocked-owner-external-proof-execution-result-required", bundle.GetProperty("ownerExternalProofExecutionResultImportValidationState").GetString());
        Assert.Equal(6, bundle.GetProperty("ownerExternalProofExecutionResultImportStrictBlockedLaneCount").GetInt32());
        Assert.Equal(0, bundle.GetProperty("ownerExternalProofExecutionResultImportReadyForStrictValidatorLaneCount").GetInt32());
        Assert.Equal(0, bundle.GetProperty("ownerExternalProofExecutionResultImportPromotableLaneCount").GetInt32());
        Assert.Equal("blocked-real-external-proof-record-import-required", bundle.GetProperty("realExternalProofRecordImportValidatorState").GetString());
        Assert.Equal("blocked-real-external-proof-record-import-required", bundle.GetProperty("realExternalProofRecordImportValidatorValidationState").GetString());
        Assert.Equal(6, bundle.GetProperty("realExternalProofRecordImportValidatorStrictBlockedContractCount").GetInt32());
        Assert.Equal(0, bundle.GetProperty("realExternalProofRecordImportValidatorReadyForStrictValidatorContractCount").GetInt32());
        Assert.Equal("blocked-owner-result-import-candidate-required", bundle.GetProperty("realProofRecordCandidateFromOwnerResultImportState").GetString());
        Assert.Equal("blocked-owner-result-import-candidate-required", bundle.GetProperty("realProofRecordCandidateFromOwnerResultImportValidationState").GetString());
        Assert.Equal(0, bundle.GetProperty("realProofRecordCandidateFromOwnerResultImportCandidateCount").GetInt32());
        Assert.Equal(0, bundle.GetProperty("realProofRecordCandidateFromOwnerResultImportStrictValidatorReadyCandidateCount").GetInt32());
        Assert.True(bundle.GetProperty("realProofRecordCandidateFromOwnerResultImportStrictValidatorInputOnly").GetBoolean());
        Assert.Equal("blocked-release-close-real-proof-import-required", bundle.GetProperty("releaseCloseRealProofImportBridgeState").GetString());
        Assert.Equal("blocked-release-close-real-proof-import-required", bundle.GetProperty("releaseCloseRealProofImportBridgeValidationState").GetString());
        Assert.Equal(6, bundle.GetProperty("releaseCloseRealProofImportBridgeBlockedLaneCount").GetInt32());
        Assert.Equal(0, bundle.GetProperty("releaseCloseRealProofImportBridgeFailedBlockerCount").GetInt32());
        Assert.True(bundle.GetProperty("releaseCloseRealProofImportBridgeInputOnly").GetBoolean());
        Assert.False(bundle.GetProperty("finalOwnerExecutionPackagePerformsPublish").GetBoolean());
        Assert.False(bundle.GetProperty("finalOwnerExecutionPackageCanPromoteRuntimeProof").GetBoolean());
        Assert.False(bundle.GetProperty("finalOwnerExecutionPackageCanPublishPublicly").GetBoolean());
        Assert.False(bundle.GetProperty("finalOwnerExecutionPackageCanCloseReleaseIssue").GetBoolean());
        Assert.Equal("blocked-owner-clean-external-package-consumer-execution-required", bundle.GetProperty("cleanExternalPackageConsumerOwnerRunbookState").GetString());
        Assert.Equal("blocked-owner-clean-external-package-consumer-execution-required", bundle.GetProperty("cleanExternalPackageConsumerOwnerRunbookValidationState").GetString());
        Assert.Equal(9, bundle.GetProperty("cleanExternalPackageConsumerOwnerRunbookStepCount").GetInt32());
        Assert.Equal(9, bundle.GetProperty("cleanExternalPackageConsumerOwnerRunbookBlockedStepCount").GetInt32());
        Assert.Equal(0, bundle.GetProperty("cleanExternalPackageConsumerOwnerRunbookFailedBlockerCount").GetInt32());
        Assert.False(bundle.GetProperty("cleanExternalPackageConsumerOwnerRunbookPerformsPublish").GetBoolean());
        Assert.False(bundle.GetProperty("cleanExternalPackageConsumerOwnerRunbookCanPromoteRuntimeProof").GetBoolean());
        Assert.False(bundle.GetProperty("cleanExternalPackageConsumerOwnerRunbookCanPublishPublicly").GetBoolean());
        Assert.False(bundle.GetProperty("cleanExternalPackageConsumerOwnerRunbookCanCloseReleaseIssue").GetBoolean());
        Assert.Equal("blocked-owner-post-publish-verification-required", bundle.GetProperty("postPublishOwnerVerificationRunbookState").GetString());
        Assert.Equal("blocked-owner-post-publish-verification-required", bundle.GetProperty("postPublishOwnerVerificationRunbookValidationState").GetString());
        Assert.Equal(6, bundle.GetProperty("postPublishOwnerVerificationRunbookStepCount").GetInt32());
        Assert.Equal(6, bundle.GetProperty("postPublishOwnerVerificationRunbookBlockedStepCount").GetInt32());
        Assert.Equal(0, bundle.GetProperty("postPublishOwnerVerificationRunbookFailedBlockerCount").GetInt32());
        Assert.False(bundle.GetProperty("postPublishOwnerVerificationRunbookPerformsPublish").GetBoolean());
        Assert.False(bundle.GetProperty("postPublishOwnerVerificationRunbookCanPromoteRuntimeProof").GetBoolean());
        Assert.False(bundle.GetProperty("postPublishOwnerVerificationRunbookCanPublishPublicly").GetBoolean());
        Assert.False(bundle.GetProperty("postPublishOwnerVerificationRunbookCanCloseReleaseIssue").GetBoolean());

        JsonElement item = bundle.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static candidate => candidate.GetProperty("id").GetString() == "final-owner-proof-action-worklist");
        Assert.False(item.GetProperty("passed").GetBoolean());

        JsonElement executionItem = bundle.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static candidate => candidate.GetProperty("id").GetString() == "final-owner-execution-package");
        Assert.False(executionItem.GetProperty("passed").GetBoolean());
        string executionItemState = executionItem.GetProperty("state").GetString()!;
        Assert.Contains("hardGates=8", executionItemState, StringComparison.Ordinal);
        Assert.Contains("sourceArtifactEvidence=", executionItemState, StringComparison.Ordinal);
        Assert.Contains("sourceArtifactMissing=0", executionItemState, StringComparison.Ordinal);
        Assert.Contains("sourceArtifactSha256=", executionItemState, StringComparison.Ordinal);
        Assert.Contains("sourceArtifactNonProofBoundary=", executionItemState, StringComparison.Ordinal);
        Assert.Contains("publicDownloadCannotSubstitutePostPublish=True", executionItemState, StringComparison.Ordinal);
        Assert.Contains("bundleHashCannotSubstituteFinalCloseDecision=True", executionItemState, StringComparison.Ordinal);
        Assert.Contains("strictCloseOutputCannotCloseIssue=True", executionItemState, StringComparison.Ordinal);
        Assert.Contains("strictCloseOutput=blocked-final-close-gate-owner-proof-required", executionItemState, StringComparison.Ordinal);

        foreach (string id in new[]
        {
            "owner-external-proof-execution-result-import",
            "real-external-proof-record-import-validator",
            "real-proof-record-candidate-from-owner-result-import",
            "release-close-real-proof-import-bridge",
            "clean-external-package-consumer-owner-runbook",
            "post-publish-owner-verification-runbook"
        })
        {
            JsonElement proofImportItem = bundle.GetProperty("evidenceItems")
                .EnumerateArray()
                .Single(candidate => candidate.GetProperty("id").GetString() == id);
            Assert.False(proofImportItem.GetProperty("passed").GetBoolean());
            Assert.Contains("not", proofImportItem.GetProperty("boundary").GetString()!, StringComparison.OrdinalIgnoreCase);
        }

        string itemBoundary = item.GetProperty("boundary").GetString()!;
        foreach (string marker in new[]
        {
            "not runtime proof",
            "not post-publish proof",
            "not package push",
            "not release close"
        })
        {
            Assert.Contains(marker, itemBoundary, StringComparison.OrdinalIgnoreCase);
        }

        string raw = bundle.GetRawText();
        foreach (string marker in new[]
        {
            "final-owner-proof-action-worklist.json",
            "final-owner-proof-action-worklist-validation.json",
            "final-owner-execution-package.json",
            "final-owner-execution-package-validation.json",
            "owner-external-proof-execution-result-import.json",
            "owner-external-proof-execution-result-import-validation.json",
            "real-external-proof-record-import-validator.json",
            "real-proof-record-candidate-from-owner-result-import.json",
            "release-close-real-proof-import-bridge-validation.json",
            "clean-external-package-consumer-owner-runbook.json",
            "clean-external-package-consumer-owner-runbook-validation.json",
            "post-publish-owner-verification-runbook.json",
            "post-publish-owner-verification-runbook-validation.json",
            "repository-external projects",
            "does not run dotnet nuget push",
            "owner-result-candidate-bridge",
            "strict-validator input only",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "build-only",
            "dry-run",
            "blocked-by-driver"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        string markdown = ReadText("artifacts", "final-release", "release-evidence-bundle.md");
        Assert.Contains("final owner proof action worklist", markdown, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("final owner proof action worklist blocked actions", markdown, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("final owner execution package", markdown, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("final owner execution package blocked steps", markdown, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("clean external package consumer owner runbook", markdown, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("post-publish owner verification runbook", markdown, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FinalPublishProofGateConsumesPublicDocsGate()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicDocsAndPackageMetadataGate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPublishProofGate.ps1"), "-Strict");

        using JsonDocument gateDocument = ReadJson("artifacts", "final-release", "final-publish-proof-gate-report.json");
        JsonElement gate = gateDocument.RootElement;
        JsonElement[] items = gate.GetProperty("validationItems").EnumerateArray().ToArray();

        Assert.Contains(items, item => item.GetProperty("id").GetString() == "public-docs-package-metadata-gate-passed" && item.GetProperty("passed").GetBoolean());
        Assert.Contains("public-docs-package-metadata-gate.json", gate.GetRawText(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("public docs/package metadata gate", gate.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    private static JsonDocument ReadJson(params string[] pathParts)
    {
        return JsonDocument.Parse(ReadText(pathParts));
    }

    private static string ReadText(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        using Process process = new();
        process.StartInfo.FileName = "pwsh";
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.StartInfo.WorkingDirectory = RepositoryPaths.Root;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.UseShellExecute = false;

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptPath}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}
