using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseCandidateReadinessTests
{
    [Fact]
    public void ReleaseCandidateScriptsDefineLocalFeedAndAggregateReadinessGates()
    {
        string localFeed = ReadSource("eng", "Test-LocalNuGetFeedConsumer.ps1");
        string checklist = ReadSource("eng", "Export-ReleaseCandidateChecklist.ps1");
        string readiness = ReadSource("eng", "Test-ReleaseCandidateReadiness.ps1");
        string finalRelease = ReadSource("eng", "Test-FinalReleaseDryRun.ps1");
        string bilingualBacklog = ReadSource("eng", "Export-PublicApiBilingualDocumentationBacklog.ps1");
        string releaseOwnerDecision = ReadSource("eng", "Export-ReleaseOwnerDecisionTemplate.ps1");
        string releaseOwnerApprovalInput = ReadSource("eng", "Export-ReleaseOwnerApprovalInputTemplate.ps1");
        string releaseOwnerApprovalInputExample = ReadSource("eng", "Export-ReleaseOwnerApprovalInputExample.ps1");
        string releaseOwnerApprovalInputValidator = ReadSource("eng", "Test-ReleaseOwnerApprovalInput.ps1");
        string releaseOwnerDecisionRecord = ReadSource("eng", "Export-ReleaseOwnerDecisionRecord.ps1");
        string externalRuntimeProofRecord = ReadSource("eng", "Export-ExternalRuntimeProofRecordTemplate.ps1");
        string externalRuntimeProofInputTemplate = ReadSource("eng", "Export-ExternalRuntimeProofRecordInputTemplate.ps1");
        string externalRuntimeProofExample = ReadSource("eng", "Export-ExternalRuntimeProofRecordExample.ps1");
        string externalRuntimeProofValidator = ReadSource("eng", "Test-ExternalRuntimeProofRecord.ps1");
        string externalRuntimeProofOwnerHandoff = ReadSource("eng", "Export-ExternalRuntimeProofOwnerHandoff.ps1");
        string compatibleHostRunbook = ReadSource("eng", "Export-CompatibleHostRuntimeProofRunbook.ps1");
        string externalRuntimeProofBackfillPlan = ReadSource("eng", "Export-ExternalRuntimeProofBackfillPlan.ps1");
        string externalRuntimeProofCollectionPackage = ReadSource("eng", "Export-ExternalRuntimeProofCollectionPackage.ps1");
        string postPublishVerificationRecord = ReadSource("eng", "Export-PostPublishVerificationRecordTemplate.ps1");
        string postPublishVerificationValidator = ReadSource("eng", "Test-PostPublishVerificationRecord.ps1");
        string postPublishVerificationBackfillPlan = ReadSource("eng", "Export-PostPublishVerificationBackfillPlan.ps1");
        string postPublishVerificationCollectionPackage = ReadSource("eng", "Export-PostPublishVerificationCollectionPackage.ps1");
        string finalPackageReviewBundle = ReadSource("eng", "Export-FinalPackageReviewBundle.ps1");
        string releasePackageProofBundle = ReadSource("eng", "Export-ReleasePackageProofBundle.ps1");
        string docsPublishReadinessBundle = ReadSource("eng", "Export-DocsPublishReadinessBundle.ps1");
        string releaseEvidenceBundle = ReadSource("eng", "Export-ReleaseEvidenceBundle.ps1");
        string releasePublishExecutionChecklist = ReadSource("eng", "Export-ReleasePublishExecutionChecklist.ps1");
        string releasePromotionIssueRecord = ReadSource("eng", "Export-ReleasePromotionIssueRecord.ps1");
        string releaseCandidateFreezeSummary = ReadSource("eng", "Export-ReleaseCandidateFreezeSummary.ps1");
        string releaseCandidateFreezeSummaryValidator = ReadSource("eng", "Test-ReleaseCandidateFreezeSummary.ps1");
        string releaseCandidateFreezeChecklist = ReadSource("eng", "Export-ReleaseCandidateFreezeChecklist.ps1");
        string ownerAuthorizedPublishCommandPlan = ReadSource("eng", "Export-OwnerAuthorizedPublishCommandPlan.ps1");
        string ownerAuthorizedPublishCommandPlanValidator = ReadSource("eng", "Test-OwnerAuthorizedPublishCommandPlan.ps1");
        string releaseCandidateFullAcceptanceSummary = ReadSource("eng", "Export-ReleaseCandidateFullAcceptanceSummary.ps1");
        string linuxEvidenceTemplate = ReadSource("eng", "Export-LinuxRunnerEvidenceTemplate.ps1");
        string linuxEvidenceRecordTemplate = ReadSource("eng", "Export-LinuxRunnerEvidenceRecordTemplate.ps1");
        string linuxEvidenceRecordValidator = ReadSource("eng", "Test-LinuxRunnerEvidenceRecord.ps1");
        string sampleAssetManifest = ReadSource("eng", "Test-SampleAssetManifest.ps1");
        string sampleAssetAcquisitionPlan = ReadSource("eng", "Export-SampleAssetAcquisitionPlan.ps1");
        string onnxEngineBuildEvidenceSidecarValidator = ReadSource("eng", "Test-OnnxEngineBuildEvidenceSidecar.ps1");
        string onnxEngineBuildEvidenceSidecarTemplate = ReadSource("eng", "Export-OnnxEngineBuildEvidenceSidecarTemplate.ps1");
        string sampleRunEvidenceRecordTemplate = ReadSource("eng", "Export-SampleRunEvidenceRecordTemplate.ps1");
        string sampleRunEvidenceRecordValidator = ReadSource("eng", "Test-SampleRunEvidenceRecord.ps1");
        string realModelOwnerHandoff = ReadSource("eng", "Export-RealModelOwnerHandoff.ps1");
        string deferredReadOnlyPlan = ReadSource("eng", "Export-DeferredReadOnlyApiCandidatePlan.ps1");
        string deferredManualDesignGroupsDoc = ReadSource("docs", "articles", "zh-cn", "deferred-manual-design-groups.md");
        string userAcceptanceCatalog = ReadSource("eng", "Export-UserAcceptanceSampleCatalog.ps1");
        string staleClaims = ReadSource("eng", "Test-StaleReleaseClaims.ps1");

        Assert.Contains("local-release-candidate-feed", localFeed, StringComparison.Ordinal);
        Assert.Contains("RestoreSourceMode = \"local-feed-only\"", localFeed, StringComparison.Ordinal);
        Assert.Contains("UsesProjectReference", localFeed, StringComparison.Ordinal);
        Assert.Contains("ProjectReference", localFeed, StringComparison.Ordinal);
        Assert.Contains("DependencyProbe BridgeInitialized", localFeed, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", localFeed, StringComparison.Ordinal);
        Assert.Contains("local-nuget-feed-consumer-summary.json", localFeed, StringComparison.Ordinal);
        Assert.Contains("lastWriteTimeUtc", localFeed, StringComparison.Ordinal);
        Assert.Contains("Expression = { $_.lastWriteTimeUtc }; Descending = $true", localFeed, StringComparison.Ordinal);
        Assert.Contains("CreateHighLevelWrapperSurfaceSummary", localFeed, StringComparison.Ordinal);
        Assert.Contains("nameof(TensorRtOnnxParser.IsSubgraphSupported)", localFeed, StringComparison.Ordinal);
        Assert.Contains("nameof(TensorRtPluginRegistryInventory.FindCreator)", localFeed, StringComparison.Ordinal);
        Assert.Contains("nameof(TensorRtPluginRegistryInventory.TryFindCreator)", localFeed, StringComparison.Ordinal);
        Assert.Contains("nameof(TensorRtBuilderConfig.GetDlaCore)", localFeed, StringComparison.Ordinal);
        Assert.Contains("nameof(TensorRtBuilderConfig.GetMaxTactics)", localFeed, StringComparison.Ordinal);
        Assert.Contains("no-public-borrowed-plugin-creator-pointer", localFeed, StringComparison.Ordinal);
        Assert.Contains("local-feed-is-not-post-publish-proof", localFeed, StringComparison.Ordinal);
        Assert.Contains("final-package-review-bundle", finalPackageReviewBundle, StringComparison.Ordinal);
        Assert.Contains("Local package review is not public channel proof", finalPackageReviewBundle, StringComparison.Ordinal);
        Assert.Contains("does not publish packages", finalPackageReviewBundle, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", finalPackageReviewBundle, StringComparison.Ordinal);

        Assert.Contains("release-candidate-checklist.json", checklist, StringComparison.Ordinal);
        Assert.Contains("Runtime package readiness summary", checklist, StringComparison.Ordinal);
        Assert.Contains("Local NuGet feed consumer validation", checklist, StringComparison.Ordinal);
        Assert.Contains("Runtime package matrix", checklist, StringComparison.Ordinal);
        Assert.Contains("Deferred rows are intentional boundary records", checklist, StringComparison.Ordinal);
        Assert.Contains("Public API bilingual documentation backlog", checklist, StringComparison.Ordinal);
        Assert.Contains("Compatible host runtime proof collection bundle", checklist, StringComparison.Ordinal);
        Assert.Contains("compatible-host-runtime-proof-collection-bundle.json", checklist, StringComparison.Ordinal);
        Assert.Contains("Collection bundle is executable guidance, not runtime proof or publish approval", checklist, StringComparison.Ordinal);
        Assert.Contains("TRT10 compatible bridge package runtime proof", checklist, StringComparison.Ordinal);
        Assert.Contains("compatible-host-bridge-package-runtime", checklist, StringComparison.Ordinal);
        Assert.Contains("isPackageConsumerRuntimeProof=false", checklist, StringComparison.Ordinal);
        Assert.Contains("TRT10 compatible bridge package runtime proof", readiness, StringComparison.Ordinal);
        Assert.Contains("does not clear the TRT11 public release blocker", readiness, StringComparison.Ordinal);
        Assert.Contains("compatibleBridgeRuntimeProofStatus", readiness, StringComparison.Ordinal);
        Assert.Contains("quickStart=", checklist, StringComparison.Ordinal);
        Assert.Contains("preflight=", checklist, StringComparison.Ordinal);
        Assert.Contains("recordKind = \"external-runtime-proof-collection-package\"", externalRuntimeProofCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("packageState = \"owner-action-required\"", externalRuntimeProofCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("canPromoteRuntimeProof = $false", externalRuntimeProofCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", externalRuntimeProofCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("isRuntimeExecutionEvidence = $false", externalRuntimeProofCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("Test-PackageConsumer.ps1", externalRuntimeProofCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1", externalRuntimeProofCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("-RequireExistingLog -FailOnNotProof", externalRuntimeProofCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("recordKind = \"post-publish-verification-collection-package\"", postPublishVerificationCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("packageState = \"blocked-real-publication-required\"", postPublishVerificationCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("postPublishRequiredEvidence", postPublishVerificationCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("publishedPackageUrl", postPublishVerificationCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("cleanConsumerRootOutsideRepository", postPublishVerificationCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("runtimeSmokeLogSha256", postPublishVerificationCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("hostMetadata", postPublishVerificationCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("isPostPublishVerificationProof = $false", postPublishVerificationCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", postPublishVerificationCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("Test-PostPublishVerificationRecord.ps1", postPublishVerificationCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("-RequireExistingLog -FailOnNotProof", postPublishVerificationCollectionPackage, StringComparison.Ordinal);
        Assert.Contains("Collection packages are copyable owner guidance only", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-collection-package", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-collection-package", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("Backfill plans and collection packages are guidance only", releaseCandidateFreezeSummary, StringComparison.Ordinal);
        Assert.Contains("Collection packages are copyable owner guidance only", releaseCandidateFreezeChecklist, StringComparison.Ordinal);
        Assert.Contains("release-candidate-full-acceptance-summary", releaseCandidateFullAcceptanceSummary, StringComparison.Ordinal);
        Assert.Contains("ready-for-owner-proof-collection", releaseCandidateFullAcceptanceSummary, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $canCloseReleaseIssue", releaseCandidateFullAcceptanceSummary, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue remains false", releaseCandidateFullAcceptanceSummary, StringComparison.Ordinal);
        Assert.Contains("YoloVision sample asset candidates are not sample smoke passes", releaseCandidateFullAcceptanceSummary, StringComparison.Ordinal);
        Assert.Contains("release-candidate-full-acceptance-summary", docsPublishReadinessBundle, StringComparison.Ordinal);
        Assert.Contains("copyableExecutionOrder=", checklist, StringComparison.Ordinal);
        Assert.Contains("First command:", checklist, StringComparison.Ordinal);
        Assert.Contains("Last command:", checklist, StringComparison.Ordinal);
        Assert.Contains("`pending` means the release candidate needs another local or self-hosted runner pass", checklist, StringComparison.Ordinal);

        Assert.Contains("release-candidate-readiness-summary.json", readiness, StringComparison.Ordinal);
        Assert.Contains("runtime-package-matrix.json", readiness, StringComparison.Ordinal);
        Assert.Contains("Local NuGet feed consumer", readiness, StringComparison.Ordinal);
        Assert.Contains("DebugListener real callback runtime proof", readiness, StringComparison.Ordinal);
        Assert.Contains("overall package readiness does not imply real callback runtime proof", readiness, StringComparison.Ordinal);
        Assert.Contains("AllowRuntimeSmokeBlocked", readiness, StringComparison.Ordinal);
        Assert.Contains("sampleSmokeCatalog", readiness, StringComparison.Ordinal);
        Assert.Contains("unsigned-or-not-requested", readiness, StringComparison.Ordinal);
        Assert.Contains("packageConsumerEvidenceKind", readiness, StringComparison.Ordinal);
        Assert.Contains("runtimeSmokeClassification", readiness, StringComparison.Ordinal);
        Assert.Contains("runtimeProofStatus", readiness, StringComparison.Ordinal);
        Assert.Contains("runtimeProofDiagnostic", readiness, StringComparison.Ordinal);
        Assert.Contains("runtimeProofRequiredForRelease", readiness, StringComparison.Ordinal);
        Assert.Contains("Full package runtime proof", readiness, StringComparison.Ordinal);
        Assert.Contains("runtimeProofSeverity", readiness, StringComparison.Ordinal);
        Assert.Contains("runtime-execution-evidence", readiness, StringComparison.Ordinal);
        Assert.Contains("isRuntimeExecutionEvidence", readiness, StringComparison.Ordinal);
        Assert.Contains("isDependencyProbeOnly", readiness, StringComparison.Ordinal);
        Assert.Contains("compatibleHostRuntimeProofCollectionBundleState", readiness, StringComparison.Ordinal);
        Assert.Contains("compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof", readiness, StringComparison.Ordinal);
        Assert.Contains("compatible-host-runtime-proof-collection-bundle is guidance only", readiness, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record.json", readiness, StringComparison.Ordinal);

        Assert.Contains("final-release-dry-run-summary.json", finalRelease, StringComparison.Ordinal);
        Assert.Contains("ready-needs-manual-approval", finalRelease, StringComparison.Ordinal);
        Assert.Contains("bilingualDocumentationFindingCount", finalRelease, StringComparison.Ordinal);
        Assert.Contains("Manual Approval Items", finalRelease, StringComparison.Ordinal);
        Assert.Contains("InvocationCount>0 callback proof", finalRelease, StringComparison.Ordinal);
        Assert.Contains("User acceptance sample catalog", finalRelease, StringComparison.Ordinal);
        Assert.Contains("Signing and release channel docs", finalRelease, StringComparison.Ordinal);
        Assert.Contains("bilingualDocumentationBacklogStatus", finalRelease, StringComparison.Ordinal);
        Assert.Contains("AllowRuntimeSmokeBlocked", finalRelease, StringComparison.Ordinal);
        Assert.Contains("allowRuntimeSmokeBlocked", finalRelease, StringComparison.Ordinal);
        Assert.Contains("packageConsumerEvidenceKind", finalRelease, StringComparison.Ordinal);
        Assert.Contains("runtimeSmokeClassification", finalRelease, StringComparison.Ordinal);
        Assert.Contains("runtimeProofStatus", finalRelease, StringComparison.Ordinal);
        Assert.Contains("runtimeProofDiagnostic", finalRelease, StringComparison.Ordinal);
        Assert.Contains("runtimeProofRequiredForRelease", finalRelease, StringComparison.Ordinal);
        Assert.Contains("runtimeProofBlockerOwnerActionStatus", finalRelease, StringComparison.Ordinal);
        Assert.Contains("runtimeProofBlockerCategory", finalRelease, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofState", finalRelease, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofClassification", finalRelease, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofRuntimePackageKeyMatches", finalRelease, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofPackageSourceRuntimePackageKeyMatches", finalRelease, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofConsumerProjectIdentityReady", finalRelease, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofSmokeCommandRuntimeKeyReady", finalRelease, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofHostReady", finalRelease, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofCommandsReady", finalRelease, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofManagedNupkgSha256Ready", finalRelease, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofRuntimeNupkgSha256Ready", finalRelease, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofFailedProofItemCount", finalRelease, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofLogSha256FormatReady", finalRelease, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofLogSha256Matches", finalRelease, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofOwnerActionStatus", finalRelease, StringComparison.Ordinal);
        Assert.Contains("External runtime proof record", finalRelease, StringComparison.Ordinal);
        Assert.Contains("releaseReadinessKnownRuntimeProofBlocker", finalRelease, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver-owner-action", finalRelease, StringComparison.Ordinal);
        Assert.Contains("Full package runtime proof", finalRelease, StringComparison.Ordinal);
        Assert.Contains("isRuntimeExecutionEvidence", finalRelease, StringComparison.Ordinal);
        Assert.Contains("isDependencyProbeOnly", finalRelease, StringComparison.Ordinal);
        Assert.Contains("IsDependencyProbeOnly", finalRelease, StringComparison.Ordinal);
        Assert.Contains("Compatible host runtime proof collection bundle", finalRelease, StringComparison.Ordinal);
        Assert.Contains("compatibleHostRuntimeProofCollectionBundleState", finalRelease, StringComparison.Ordinal);
        Assert.Contains("compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof", finalRelease, StringComparison.Ordinal);
        Assert.Contains("compatible-host-runtime-proof-collection-bundle is executable owner guidance, not runtime proof", finalRelease, StringComparison.Ordinal);
        Assert.Contains("Post-publish verification record", finalRelease, StringComparison.Ordinal);
        Assert.Contains("postPublishVerificationState", finalRelease, StringComparison.Ordinal);
        Assert.Contains("postPublishConsumerProjectIdentityReady", finalRelease, StringComparison.Ordinal);
        Assert.Contains("postPublishSmokeCommandRuntimeKeyReady", finalRelease, StringComparison.Ordinal);
        Assert.Contains("postPublishHostReady", finalRelease, StringComparison.Ordinal);
        Assert.Contains("postPublishCommandsReady", finalRelease, StringComparison.Ordinal);
        Assert.Contains("postPublishStdoutStderrSummaryReady", finalRelease, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record.json", finalRelease, StringComparison.Ordinal);

        Assert.Contains("public-api-bilingual-documentation-audit.json", bilingualBacklog, StringComparison.Ordinal);
        Assert.Contains("public-api-bilingual-documentation-backlog.json", bilingualBacklog, StringComparison.Ordinal);
        Assert.Contains("backlogFindingCount", bilingualBacklog, StringComparison.Ordinal);
        Assert.Contains("recommendedBatch", bilingualBacklog, StringComparison.Ordinal);
        Assert.Contains("sourceHint", bilingualBacklog, StringComparison.Ordinal);

        Assert.Contains("pending-release-owner-approval", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("Linux handoff is not Linux runner proof", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver is not API proof", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("runtimeProofBlockerOwnerActionStatus", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("runtimeProofBlockerCategory", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofState", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofClassification", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofRuntimePackageKeyMatches", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofPackageSourceRuntimePackageKeyMatches", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofManagedNupkgSha256Ready", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofRuntimeNupkgSha256Ready", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofFailedProofItemCount", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofLogSha256FormatReady", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofLogSha256Matches", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofOwnerActionStatus", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("compatible-host-runtime-proof-runbook.json", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("compatibleHostRuntimeProofRunbookState", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("compatibleHostRuntimeProofRunbookCanPromoteRuntimeProof", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("compatibleHostRuntimeProofRunbookRuntimeExecutionEvidence", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("compatibleHostRuntimeProofRunbookRunPackageConsumerSmokeCommand", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("compatible-host-runtime-proof-runbook is an owner-action runbook, not runtime proof", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("runtime-deserialization-dependency-diagnostics", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("allowRuntimeSmokeBlocked", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("runtimeProofStatus", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("runtimeProofRequiredForRelease", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("allowRuntimeSmokeBlocked=true records dry-run intent only", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("postPublishVerificationState", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("postPublishConsumerProjectIdentityReady", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("postPublishSmokeCommandRuntimeKeyReady", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("postPublishHostReady", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("postPublishCommandsReady", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("postPublishStdoutStderrSummaryReady", releaseOwnerDecision, StringComparison.Ordinal);
        Assert.Contains("release-owner-approval-input-template.json", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("release-owner-approval-input-record", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("pending-release-owner-input", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("runtimeProofBlockerOwnerActionStatus", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("runtimeProofBlockerCategory", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofState", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofClassification", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofRuntimePackageKeyMatches", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofPackageSourceRuntimePackageKeyMatches", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofManagedNupkgSha256Ready", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofRuntimeNupkgSha256Ready", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofFailedProofItemCount", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofLogSha256FormatReady", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofLogSha256Matches", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofOwnerActionStatus", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("compatible-host-runtime-proof-runbook.json", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("compatibleHostRuntimeProofRunbookState", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("compatibleHostRuntimeProofRunbookCanPromoteRuntimeProof", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("compatibleHostRuntimeProofRunbookRuntimeExecutionEvidence", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("compatibleHostRuntimeProofRunbookRunPackageConsumerSmokeCommand", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("compatible-host-runtime-proof-runbook is an owner-action runbook, not runtime proof", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofDraftState", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("draftManagedNupkgSha256Ready", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("draftRuntimeNupkgSha256Ready", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("draftSmokeLogSha256Ready", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("draftNoProjectReference", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("draftSmokeStatus", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("compatibleHostRequired", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("requiredHostAction", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("promotionBlockedReason", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("postPublishVerificationState", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("postPublishConsumerProjectIdentityReady", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("postPublishSmokeCommandRuntimeKeyReady", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("postPublishHostReady", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("postPublishCommandsReady", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("postPublishStdoutStderrSummaryReady", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("runtime-deserialization-dependency-diagnostics", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("approved-known-limitation-for-rc", releaseOwnerApprovalInput, StringComparison.Ordinal);
        Assert.Contains("release-owner-approval-input-record.example.json", releaseOwnerApprovalInputExample, StringComparison.Ordinal);
        Assert.Contains("example-not-for-publication", releaseOwnerApprovalInputExample, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly = $false", releaseOwnerApprovalInputExample, StringComparison.Ordinal);
        Assert.Contains("release-owner-approval-input-validation.json", releaseOwnerApprovalInputValidator, StringComparison.Ordinal);
        Assert.Contains("blocked-owner-input-required", releaseOwnerApprovalInputValidator, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly=true requires an explicit non-template owner input record", releaseOwnerApprovalInputValidator, StringComparison.Ordinal);
        Assert.Contains("release-owner-decision-record.json", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("recordKind = \"release-owner-decision-record\"", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly = $false", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("pending-release-owner-approval", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("Windows-generated handoff and template-only files are not Linux runner proof", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("allowRuntimeSmokeBlocked", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("runtimeProofStatus", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("runtimeProofRequiredForRelease", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("runtimeProofBlockerOwnerActionStatus", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("runtimeProofBlockerCategory", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofState", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofClassification", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofRuntimePackageKeyMatches", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofPackageSourceRuntimePackageKeyMatches", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofManagedNupkgSha256Ready", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofRuntimeNupkgSha256Ready", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofFailedProofItemCount", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofLogSha256FormatReady", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofLogSha256Matches", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofOwnerActionStatus", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("postPublishVerificationState", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("postPublishConsumerProjectIdentityReady", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("postPublishSmokeCommandRuntimeKeyReady", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("postPublishHostReady", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("postPublishCommandsReady", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("postPublishStdoutStderrSummaryReady", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("runtime-deserialization-dependency-diagnostics", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("ownerApprovalInputValidationStatus", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("ownerApprovalCanPublishPublicly", releaseOwnerDecisionRecord, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record-template.json", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("proofClassification = \"template-only\"", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("evidenceClassifications", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("\"precheck\"", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("cudaDriverSupportedRuntime", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("cudnnVersion", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("consumerProjectName", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("consumerProjectPath", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("managedNupkgSha256", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("runtimeNupkgSha256", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("stdoutSummary", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("stderrSummary", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("logSha256", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("stdoutSummary must summarize reviewed stdout from the real smoke log", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("stderrSummary must summarize reviewed stderr", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("no-stderr-emitted", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("runtimePackageKey must match the release target runtime package key", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("packageSource.runtimePackageKey must match the release target runtime package key", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("managedNupkgSha256 and runtimeNupkgSha256 must be 64-character SHA256 hashes", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("logSha256 must be a 64-character SHA256 hash", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("isRuntimeExecutionEvidence = $false", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("isDependencyProbeOnly = $true", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("DependencyProbe output is useful diagnostics but is not runtime execution proof", externalRuntimeProofRecord, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record.input-template.json", externalRuntimeProofInputTemplate, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record.input-template.md", externalRuntimeProofInputTemplate, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record-input-template", externalRuntimeProofInputTemplate, StringComparison.Ordinal);
        Assert.Contains("requiredEvidenceSummary", externalRuntimeProofInputTemplate, StringComparison.Ordinal);
        Assert.Contains("proofClassification=package-consumer-runtime", externalRuntimeProofInputTemplate, StringComparison.Ordinal);
        Assert.Contains("Keep runtimePackageKey equal to the release target runtime package key", externalRuntimeProofInputTemplate, StringComparison.Ordinal);
        Assert.Contains("packageSource.runtimePackageKey", externalRuntimeProofInputTemplate, StringComparison.Ordinal);
        Assert.Contains("consumerProjectName", externalRuntimeProofInputTemplate, StringComparison.Ordinal);
        Assert.Contains("consumerProjectPath", externalRuntimeProofInputTemplate, StringComparison.Ordinal);
        Assert.Contains("managedNupkgSha256", externalRuntimeProofInputTemplate, StringComparison.Ordinal);
        Assert.Contains("runtimeNupkgSha256", externalRuntimeProofInputTemplate, StringComparison.Ordinal);
        Assert.Contains("logSha256", externalRuntimeProofInputTemplate, StringComparison.Ordinal);
        Assert.Contains("real-model-runtime", externalRuntimeProofInputTemplate, StringComparison.Ordinal);
        Assert.Contains("modelSha256", externalRuntimeProofInputTemplate, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record.example.json", externalRuntimeProofExample, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record.example.md", externalRuntimeProofExample, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record-example", externalRuntimeProofExample, StringComparison.Ordinal);
        Assert.Contains("example-not-for-publication", externalRuntimeProofExample, StringComparison.Ordinal);
        Assert.Contains("proofClassification = \"dependency-probe-only\"", externalRuntimeProofExample, StringComparison.Ordinal);
        Assert.Contains("A real record must keep runtimePackageKey equal to the release target runtime package key", externalRuntimeProofExample, StringComparison.Ordinal);
        Assert.Contains("logSha256", externalRuntimeProofExample, StringComparison.Ordinal);
        Assert.Contains("isRuntimeExecutionEvidence = $false", externalRuntimeProofExample, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-validation.json", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-validation.md", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("OutputRoot", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("RequireExistingLog", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("proofClassificationKnown", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("proofClassificationPromotable", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("runtimePackageKeyMatches", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("consumerProjectIdentityReady", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("smokeCommandRuntimeKeyReady", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("logSha256FormatReady", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("logSha256Matches", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("stdoutSummaryReady", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("stderrSummaryReady", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("New-ValidationItem -Id \"runtime-package-key\"", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("package-source-runtime-package-key", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("consumer-project-identity", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("smoke-command-runtime-package-key", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("package-sha256", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("New-ValidationItem -Id \"log-sha256\"", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("New-ValidationItem -Id \"log-sha256-match\"", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("New-ValidationItem -Id \"stdout-summary\"", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("New-ValidationItem -Id \"stderr-summary\"", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("proof-classification-promotable", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("stdout-stderr-summary", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("Both stdoutSummary and stderrSummary must be reviewed", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("real-model-evidence", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("failedProofItemCount", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("failedBlockerCount", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("requiredEvidenceSummary", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("example-not-for-publication", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record-draft", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("draft-blocked-by-cuda-driver", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("draft-rich-but-not-proof", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("isDraftRecord", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("noProjectReference", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("packageSourceReady", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("commandsReady", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("hostReady", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("validationState", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("isRuntimeExecutionEvidence", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("canPromoteRuntimeProof", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("bridge-only package consumer log", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("bridge-only wrapper surface", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("Skipped=True", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("WrapperSurfaceEvidenceKind=compile-surface-proof", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("IsRuntimeExecutionProof=False", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("Parser/ParserRefitter diagnostic snapshots", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("copied managed diagnostic snapshot", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("FailOnNotProof", externalRuntimeProofValidator, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-owner-handoff.json", externalRuntimeProofOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-owner-handoff.md", externalRuntimeProofOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("owner-action-required", externalRuntimeProofOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("Get-FileHash", externalRuntimeProofOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("packageSourceRuntimePackageKeyMatches", externalRuntimeProofOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("managedNupkgSha256Ready", externalRuntimeProofOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("runtimeNupkgSha256Ready", externalRuntimeProofOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("failedProofItemCount", externalRuntimeProofOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("computePackageNupkgSha256", externalRuntimeProofOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("RequireExistingLog", externalRuntimeProofOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("stdoutSummaryReady", externalRuntimeProofOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("stderrSummaryReady", externalRuntimeProofOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("review-stdout-stderr", externalRuntimeProofOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("no-stderr-emitted", externalRuntimeProofOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("The handoff record does not publish packages and does not approve public release", externalRuntimeProofOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("compatible-host-runtime-proof-runbook.json", compatibleHostRunbook, StringComparison.Ordinal);
        Assert.Contains("compatible-host-runtime-proof-runbook.md", compatibleHostRunbook, StringComparison.Ordinal);
        Assert.Contains("Test-PackageConsumer.ps1", compatibleHostRunbook, StringComparison.Ordinal);
        Assert.Contains("FailOnNotProof", compatibleHostRunbook, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofStdoutSummaryReady", compatibleHostRunbook, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofStderrSummaryReady", compatibleHostRunbook, StringComparison.Ordinal);
        Assert.Contains("results.stdoutSummary", compatibleHostRunbook, StringComparison.Ordinal);
        Assert.Contains("results.stderrSummary", compatibleHostRunbook, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", compatibleHostRunbook, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-record-template.json", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("isPostPublishVerificationProof = $false", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("postPublishProofClassification = \"template-only\"", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("allowedPostPublishProofClassifications", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("packageIdentity", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("managedNupkgSha256", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("runtimeNupkgSha256", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("consumerProjectName", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("consumerProjectPath", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("cudnnVersion", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("restoreCommand", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("buildCommand", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("smokeCommand", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("--runtime-package-key", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("stdoutSummary", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("stderrSummary", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("expectedRuntimePackageKey", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("executionSteps", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("capture-host-metadata", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("run-compatible-host-smoke", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("validate-record", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("channelSourceUri", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("cleanConsumerRoot", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("restoreLogSha256", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("nativeAssetListingSha256", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("dependencyProbeLogSha256", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("smokeLogSha256", postPublishVerificationRecord, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-validation.json", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-validation.md", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("validationState", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("postPublishProofClassification", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("post-publish-proof-classification-promotable", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("published-package-version", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("runtime-package-key", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("runtimePackageKeyReady", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("expectedRuntimePackageKey", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("consumerProjectIdentityReady", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("hostReady", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("smokeCommandRuntimeKeyReady", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("RequireExistingLog", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("stdoutSummaryReady", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("stderrSummaryReady", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("stdoutStderrSummaryReady", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("restore-log-sha256-match", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("native-asset-listing-sha256-match", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("dependency-probe-log-sha256-match", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("smoke-log-sha256-match", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("isPostPublishVerificationProof", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("failedBlockerCount", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("failedProofItemCount", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("notPassedCount", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("missingEvidenceCount", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("channel-source", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("clean-consumer-root", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("clean-consumer-project-identity", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("host-runtime-metadata", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("runtime-key-smoke-command", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("stdout-stderr-summary", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("runtime-smoke-log", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("execution-steps-present", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("missingExecutionStepIds", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver is not smoke passed", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("bridge-only package consumer log", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("bridge-only wrapper surface", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("Skipped=True", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("WrapperSurfaceEvidenceKind=compile-surface-proof", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("IsRuntimeExecutionProof=False", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("Parser/ParserRefitter diagnostic snapshots", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("copied managed diagnostic snapshot", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("FailOnNotProof", postPublishVerificationValidator, StringComparison.Ordinal);
        Assert.Contains("release-package-proof-bundle.json", releasePackageProofBundle, StringComparison.Ordinal);
        Assert.Contains("canUseAsPublicPackageProof", releasePackageProofBundle, StringComparison.Ordinal);
        Assert.Contains("isRuntimeExecutionProof", releasePackageProofBundle, StringComparison.Ordinal);
        Assert.Contains("isDependencyProbeOnly", releasePackageProofBundle, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", releasePackageProofBundle, StringComparison.Ordinal);
        Assert.Contains("runtime-package-matrix.json", releasePackageProofBundle, StringComparison.Ordinal);
        Assert.Contains("runtime-packages.manifest.json", releasePackageProofBundle, StringComparison.Ordinal);
        Assert.Contains("split-runtime-packages.manifest.json", releasePackageProofBundle, StringComparison.Ordinal);
        Assert.Contains("local-nuget-feed-consumer-summary.json", releasePackageProofBundle, StringComparison.Ordinal);
        Assert.Contains("postPublishVerificationState", releasePackageProofBundle, StringComparison.Ordinal);
        Assert.Contains("postPublishConsumerProjectIdentityReady", releasePackageProofBundle, StringComparison.Ordinal);
        Assert.Contains("postPublishSmokeCommandRuntimeKeyReady", releasePackageProofBundle, StringComparison.Ordinal);
        Assert.Contains("postPublishHostReady", releasePackageProofBundle, StringComparison.Ordinal);
        Assert.Contains("postPublishCommandsReady", releasePackageProofBundle, StringComparison.Ordinal);
        Assert.Contains("postPublishStdoutSummaryReady", releasePackageProofBundle, StringComparison.Ordinal);
        Assert.Contains("postPublishStderrSummaryReady", releasePackageProofBundle, StringComparison.Ordinal);
        Assert.Contains("postPublishStdoutStderrSummaryReady", releasePackageProofBundle, StringComparison.Ordinal);
        Assert.Contains("postPublishAllLogSha256Matches", releasePackageProofBundle, StringComparison.Ordinal);
        Assert.Contains("docs-publish-readiness-bundle.json", docsPublishReadinessBundle, StringComparison.Ordinal);
        Assert.Contains("articleCount", docsPublishReadinessBundle, StringComparison.Ordinal);
        Assert.Contains("highQualityArticleCount", docsPublishReadinessBundle, StringComparison.Ordinal);
        Assert.Contains("sampleBackedArticleCount", docsPublishReadinessBundle, StringComparison.Ordinal);
        Assert.Contains("canPublishDocsExternally", docsPublishReadinessBundle, StringComparison.Ordinal);
        Assert.Contains("coreArticleCoverage", docsPublishReadinessBundle, StringComparison.Ordinal);
        Assert.Contains("coreArticleCoveredCount", docsPublishReadinessBundle, StringComparison.Ordinal);
        Assert.Contains("ownerActions", docsPublishReadinessBundle, StringComparison.Ordinal);
        Assert.Contains("external-publishing-plan", docsPublishReadinessBundle, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-record.md", docsPublishReadinessBundle, StringComparison.Ordinal);
        Assert.Contains("technical-article-roadmap.md", docsPublishReadinessBundle, StringComparison.Ordinal);
        Assert.Contains("docs/_site/index.html", docsPublishReadinessBundle, StringComparison.Ordinal);
        Assert.Contains("release-evidence-bundle.json", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("release-evidence-bundle.md", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("release-package-proof-bundle.json", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("docs-publish-readiness-bundle.json", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofPackageSourceRuntimePackageKeyMatches", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofManagedNupkgSha256Ready", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofRuntimeNupkgSha256Ready", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofFailedProofItemCount", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("postPublishProofClassificationPromotable", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("postPublishManagedPackageId", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("postPublishRuntimePackageVersion", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("postPublishRuntimeNupkgSha256Ready", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("postPublishConsumerProjectIdentityReady", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("postPublishSmokeCommandRuntimeKeyReady", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("postPublishHostReady", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("postPublishCommandsReady", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("postPublishStdoutStderrSummaryReady", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("Get-PackageConsumerStatus", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("NativeAssetsExpected", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("packageConsumerNativeAssetsReady", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("packageConsumerSmokeResult", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("packageConsumerEvidenceKind", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("packageConsumerIsRuntimeExecutionEvidence", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("packageConsumerIsDependencyProbeOnly", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("runtimeDeserializationDependencyDiagnosticsState", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("runtimeDeserializationDependencyDiagnosticsDependencyProbeOnly", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("runtimeDeserializationDependencyDiagnosticsBlockedByCudaDriver", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("runtimeProofBlockerOwnerActionStatus", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("runtimeProofBlockerOwnerActionCategory", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("runtime proof blocker owner action", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("runtime-deserialization-dependency-diagnostics", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.DoesNotContain("-Name \"status\"", releaseEvidenceBundle);
        Assert.Contains("canPublishPublicly", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("canExecutePublicPublish", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("canPromoteRuntimeProof", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofClassification", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofRuntimePackageKeyMatches", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofLogSha256FormatReady", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofLogSha256Matches", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("proofClassification=package-consumer-runtime", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("isReleaseEvidenceComplete", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("owner-action-required", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/owner-action-required.md", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("Owner action checklist is an execution handoff only", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("realModelOwnerHandoffState", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("real-model-owner-handoff.json", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("blocked-evidence-incomplete", releaseEvidenceBundle, StringComparison.Ordinal);
        Assert.Contains("manualReviewDesignGroups", deferredReadOnlyPlan, StringComparison.Ordinal);
        Assert.Contains("runtime-execution-boundary", deferredReadOnlyPlan, StringComparison.Ordinal);
        Assert.Contains("runtime-deserialization-boundary", deferredReadOnlyPlan, StringComparison.Ordinal);
        Assert.Contains("dimension-expression-snapshot-design", deferredReadOnlyPlan, StringComparison.Ordinal);
        Assert.Contains("calibrator-callback-metadata-design", deferredReadOnlyPlan, StringComparison.Ordinal);
        Assert.Contains("Do not promote", deferredReadOnlyPlan, StringComparison.Ordinal);
        Assert.Contains("Deferred 人工设计分组指南", deferredManualDesignGroupsDoc, StringComparison.Ordinal);
        Assert.Contains("manual review design groups", deferredManualDesignGroupsDoc, StringComparison.Ordinal);
        Assert.Contains("runtime-execution-boundary", deferredManualDesignGroupsDoc, StringComparison.Ordinal);
        Assert.Contains("runtime-deserialization-boundary", deferredManualDesignGroupsDoc, StringComparison.Ordinal);
        Assert.Contains("dimension-expression-snapshot-design", deferredManualDesignGroupsDoc, StringComparison.Ordinal);
        Assert.Contains("calibrator-callback-metadata-design", deferredManualDesignGroupsDoc, StringComparison.Ordinal);
        Assert.Contains("不允许把 `-IncludeMediumRisk` 的输出当成低风险提升清单", deferredManualDesignGroupsDoc, StringComparison.Ordinal);
        Assert.Contains("release-publish-execution-checklist.json", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("canExecutePublicPublish", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("release-evidence-bundle.json", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("release-package-proof-bundle.json", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("docs-publish-readiness-bundle.json", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("isReleaseEvidenceComplete", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofState", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofClassification", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofRuntimePackageKeyMatches", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofPackageSourceRuntimePackageKeyMatches", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofConsumerProjectIdentityReady", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofSmokeCommandRuntimeKeyReady", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofHostReady", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofCommandsReady", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofManagedNupkgSha256Ready", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofRuntimeNupkgSha256Ready", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofFailedProofItemCount", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofLogSha256FormatReady", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofLogSha256Matches", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofOwnerActionStatus", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofDraftState", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("draftManagedNupkgSha256Ready", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("draftRuntimeNupkgSha256Ready", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("draftSmokeLogSha256Ready", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("draftNoProjectReference", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("draftSmokeStatus", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("compatibleHostRequired", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("requiredHostAction", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("promotionBlockedReason", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("postPublishVerificationState", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("postPublishProofClassificationPromotable", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("postPublishManagedPackageId", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("postPublishRuntimeNupkgSha256Ready", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("postPublishConsumerProjectIdentityReady", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("postPublishSmokeCommandRuntimeKeyReady", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("postPublishHostReady", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("postPublishCommandsReady", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("postPublishStdoutStderrSummaryReady", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-validation.json", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-validation.json", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("dotnet nuget push <package>.nupkg", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("gh release upload <tag>", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("DependencyProbe output is not runtime execution proof", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("runtimeProofBlockerOwnerActionStatus", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("runtimeProofBlockerCategory", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("runtime-deserialization-dependency-diagnostics", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("runtime-proof-owner-action", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("template-only and handoff-only are not Linux runner proof", releasePublishExecutionChecklist, StringComparison.Ordinal);
        Assert.Contains("release-promotion-issue-record.json", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("promotionState = \"pending-release-owner-approval\"", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly = $false", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("release-package-proof-bundle.json", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("docs-publish-readiness-bundle.json", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("publishExecutionChecklistState", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("canExecutePublicPublish", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("release-evidence-bundle.json", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("isReleaseEvidenceComplete", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofState", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofClassification", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofPackageSourceRuntimePackageKeyMatches", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofConsumerProjectIdentityReady", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofSmokeCommandRuntimeKeyReady", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofHostReady", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofCommandsReady", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofManagedNupkgSha256Ready", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofRuntimeNupkgSha256Ready", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofFailedProofItemCount", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("postPublishProofClassificationPromotable", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("postPublishManagedPackageId", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("postPublishRuntimeNupkgSha256Ready", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("postPublishConsumerProjectIdentityReady", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("postPublishSmokeCommandRuntimeKeyReady", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("postPublishHostReady", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("postPublishCommandsReady", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("postPublishStdoutStderrSummaryReady", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofRuntimePackageKeyMatches", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofLogSha256FormatReady", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofLogSha256Matches", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofOwnerActionStatus", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofDraftState", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("draftManagedNupkgSha256Ready", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("draftRuntimeNupkgSha256Ready", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("draftSmokeLogSha256Ready", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("draftNoProjectReference", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("draftSmokeStatus", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("compatibleHostRequired", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("requiredHostAction", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("promotionBlockedReason", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("postPublishVerificationState", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-validation.json", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-validation.json", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver is not smoke passed", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("runtimeProofBlockerOwnerActionStatus", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("runtimeProofBlockerCategory", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("runtime-deserialization-dependency-diagnostics", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("allowRuntimeSmokeBlocked", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("runtimeProofStatus", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("runtimeProofRequiredForRelease", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("ownerApprovalInputValidationStatus", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("ownerApprovalCanPublishPublicly", releasePromotionIssueRecord, StringComparison.Ordinal);
        Assert.Contains("release-candidate-freeze-summary.json", releaseCandidateFreezeSummary, StringComparison.Ordinal);
        Assert.Contains("release-candidate-freeze-summary", releaseCandidateFreezeSummary, StringComparison.Ordinal);
        Assert.Contains("realExternalRuntimeProofReady", releaseCandidateFreezeSummary, StringComparison.Ordinal);
        Assert.Contains("realPostPublishVerificationReady", releaseCandidateFreezeSummary, StringComparison.Ordinal);
        Assert.Contains("closeReadinessConsistent", releaseCandidateFreezeSummary, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver is not smoke passed", releaseCandidateFreezeSummary, StringComparison.Ordinal);
        Assert.Contains("does not execute dotnet nuget push", releaseCandidateFreezeSummary, StringComparison.Ordinal);
        Assert.Contains("release-candidate-freeze-validation.json", releaseCandidateFreezeSummaryValidator, StringComparison.Ordinal);
        Assert.Contains("no-close-without-post-publish-proof", releaseCandidateFreezeSummaryValidator, StringComparison.Ordinal);
        Assert.Contains("no-publish-without-external-runtime-proof", releaseCandidateFreezeSummaryValidator, StringComparison.Ordinal);
        Assert.Contains("release-candidate-freeze-checklist.json", releaseCandidateFreezeChecklist, StringComparison.Ordinal);
        Assert.Contains("ownerDecisionItems", releaseCandidateFreezeChecklist, StringComparison.Ordinal);
        Assert.Contains("publishPlaceholders", releaseCandidateFreezeChecklist, StringComparison.Ordinal);
        Assert.Contains("postPublishActions", releaseCandidateFreezeChecklist, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver is not smoke passed", releaseCandidateFreezeChecklist, StringComparison.Ordinal);
        Assert.Contains("owner-authorized-publish-command-plan.json", ownerAuthorizedPublishCommandPlan, StringComparison.Ordinal);
        Assert.Contains("owner-authorized-publish-command-plan.md", ownerAuthorizedPublishCommandPlan, StringComparison.Ordinal);
        Assert.Contains("recordKind = \"owner-authorized-publish-command-plan\"", ownerAuthorizedPublishCommandPlan, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-backfill-plan.json", ownerAuthorizedPublishCommandPlan, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-backfill-plan.json", ownerAuthorizedPublishCommandPlan, StringComparison.Ordinal);
        Assert.Contains("externalRuntimeProofBackfillPlanState", ownerAuthorizedPublishCommandPlan, StringComparison.Ordinal);
        Assert.Contains("postPublishVerificationBackfillPlanState", ownerAuthorizedPublishCommandPlan, StringComparison.Ordinal);
        Assert.Contains("Backfill plans and collection packages are guidance only", ownerAuthorizedPublishCommandPlan, StringComparison.Ordinal);
        Assert.Contains("requiresExplicitOwnerAuthorization = $true", ownerAuthorizedPublishCommandPlan, StringComparison.Ordinal);
        Assert.Contains("canMaterializeExecutableCommands = $canMaterializeExecutableCommands", ownerAuthorizedPublishCommandPlan, StringComparison.Ordinal);
        Assert.Contains("authorized = $false", ownerAuthorizedPublishCommandPlan, StringComparison.Ordinal);
        Assert.Contains("executable = $false", ownerAuthorizedPublishCommandPlan, StringComparison.Ordinal);
        Assert.Contains("dotnet nuget push", ownerAuthorizedPublishCommandPlan, StringComparison.Ordinal);
        Assert.Contains("gh release upload", ownerAuthorizedPublishCommandPlan, StringComparison.Ordinal);
        Assert.Contains("postPublishCommandPlan", ownerAuthorizedPublishCommandPlan, StringComparison.Ordinal);
        Assert.Contains("run-runtime-smoke", ownerAuthorizedPublishCommandPlan, StringComparison.Ordinal);
        Assert.Contains("validate-post-publish-record", ownerAuthorizedPublishCommandPlan, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver is not smoke passed", ownerAuthorizedPublishCommandPlan, StringComparison.Ordinal);
        Assert.Contains("owner-authorized-publish-command-plan-validation.json", ownerAuthorizedPublishCommandPlanValidator, StringComparison.Ordinal);
        Assert.Contains("owner-authorized-publish-command-plan-validation.md", ownerAuthorizedPublishCommandPlanValidator, StringComparison.Ordinal);
        Assert.Contains("no-materialized-executable-commands", ownerAuthorizedPublishCommandPlanValidator, StringComparison.Ordinal);
        Assert.Contains("publish-commands-safe", ownerAuthorizedPublishCommandPlanValidator, StringComparison.Ordinal);
        Assert.Contains("backfill-plan-source-evidence-visible", ownerAuthorizedPublishCommandPlanValidator, StringComparison.Ordinal);
        Assert.Contains("external-backfill-plan-blocked", ownerAuthorizedPublishCommandPlanValidator, StringComparison.Ordinal);
        Assert.Contains("post-publish-backfill-plan-blocked", ownerAuthorizedPublishCommandPlanValidator, StringComparison.Ordinal);
        Assert.Contains("post-publish-smoke-plan-visible", ownerAuthorizedPublishCommandPlanValidator, StringComparison.Ordinal);
        Assert.Contains("blocked-driver-not-promoted", ownerAuthorizedPublishCommandPlanValidator, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-backfill-plan.json", externalRuntimeProofBackfillPlan, StringComparison.Ordinal);
        Assert.Contains("recordKind = \"external-runtime-proof-backfill-plan\"", externalRuntimeProofBackfillPlan, StringComparison.Ordinal);
        Assert.Contains("blocked-compatible-host-proof-required", externalRuntimeProofBackfillPlan, StringComparison.Ordinal);
        Assert.Contains("canPromoteRuntimeProof = $false", externalRuntimeProofBackfillPlan, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver is not smoke passed", externalRuntimeProofBackfillPlan, StringComparison.Ordinal);
        Assert.Contains("RequireExistingLog -FailOnNotProof", externalRuntimeProofBackfillPlan, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-backfill-plan.json", postPublishVerificationBackfillPlan, StringComparison.Ordinal);
        Assert.Contains("recordKind = \"post-publish-verification-backfill-plan\"", postPublishVerificationBackfillPlan, StringComparison.Ordinal);
        Assert.Contains("blocked-real-post-publish-proof-required", postPublishVerificationBackfillPlan, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", postPublishVerificationBackfillPlan, StringComparison.Ordinal);
        Assert.Contains("ProjectReference", postPublishVerificationBackfillPlan, StringComparison.Ordinal);
        Assert.Contains("RequireExistingLog -FailOnNotProof", postPublishVerificationBackfillPlan, StringComparison.Ordinal);
        Assert.Contains("linux-runner-evidence-template.json", linuxEvidenceTemplate, StringComparison.Ordinal);
        Assert.Contains("linux-runner-issue-template.md", linuxEvidenceTemplate, StringComparison.Ordinal);
        Assert.Contains("template-only", linuxEvidenceTemplate, StringComparison.Ordinal);
        Assert.Contains("InvocationCount>0", linuxEvidenceTemplate, StringComparison.Ordinal);
        Assert.Contains("linux-runner-evidence-record-template.json", linuxEvidenceRecordTemplate, StringComparison.Ordinal);
        Assert.Contains("recordKind = \"linux-runner-evidence-record-template\"", linuxEvidenceRecordTemplate, StringComparison.Ordinal);
        Assert.Contains("templateOnly = $true", linuxEvidenceRecordTemplate, StringComparison.Ordinal);
        Assert.Contains("runnerOwner", linuxEvidenceRecordTemplate, StringComparison.Ordinal);
        Assert.Contains("evidenceRationale", linuxEvidenceRecordTemplate, StringComparison.Ordinal);
        Assert.Contains("isRealLinuxRunnerProof = $false", linuxEvidenceRecordTemplate, StringComparison.Ordinal);
        Assert.Contains("canPromoteLinuxPackage = $false", linuxEvidenceRecordTemplate, StringComparison.Ordinal);
        Assert.Contains("linux-runner-evidence-validation.json", linuxEvidenceRecordValidator, StringComparison.Ordinal);
        Assert.Contains("FailOnNotProof", linuxEvidenceRecordValidator, StringComparison.Ordinal);
        Assert.Contains("template-only", linuxEvidenceRecordValidator, StringComparison.Ordinal);
        Assert.Contains("isRealLinuxRunnerProof", linuxEvidenceRecordValidator, StringComparison.Ordinal);
        Assert.Contains("canPromoteLinuxPackage", linuxEvidenceRecordValidator, StringComparison.Ordinal);
        Assert.Contains("requiredCommandIds", linuxEvidenceRecordValidator, StringComparison.Ordinal);
        Assert.Contains("package-consumer-copy", linuxEvidenceRecordValidator, StringComparison.Ordinal);
        Assert.Contains("GPU smoke is separate from packaging proof", linuxEvidenceRecordValidator, StringComparison.Ordinal);
        Assert.Contains("sample-asset-manifest-audit.json", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("candidate-not-downloaded", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("isSmokePassed", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("proofClassification", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("evidenceSidecar", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("sampleRunEvidenceRecord", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("sampleRunEvidenceValidation", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("sampleProjectCrossCheckRules", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("sampleProjectExists", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("sampleProjectNameMatches", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("sample-project-missing", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("sidecarCrossCheckRules", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("sampleRunEvidenceCrossCheckRules", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("sidecar-model-sha256-mismatch", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("sidecar-input-sha256-mismatch", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("sample-run-evidence-model-sha256-mismatch", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("sample-run-evidence-package-consumer-proof", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("proof-classification-enum", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("sample-package-consumer-proof", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("real-model-runtime-without-smoke", sampleAssetManifest, StringComparison.Ordinal);
        Assert.Contains("sample-asset-acquisition-plan.json", sampleAssetAcquisitionPlan, StringComparison.Ordinal);
        Assert.Contains("owner-action-required", sampleAssetAcquisitionPlan, StringComparison.Ordinal);
        Assert.Contains("build-only-classification", sampleAssetAcquisitionPlan, StringComparison.Ordinal);
        Assert.Contains("evidence-sidecar", sampleAssetAcquisitionPlan, StringComparison.Ordinal);
        Assert.Contains("proofClassification", sampleAssetAcquisitionPlan, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime belongs to release proof records", sampleAssetAcquisitionPlan, StringComparison.Ordinal);
        Assert.Contains("performsDownload = $false", sampleAssetAcquisitionPlan, StringComparison.Ordinal);
        Assert.Contains("performsSampleRun = $false", sampleAssetAcquisitionPlan, StringComparison.Ordinal);
        Assert.Contains("onnx-engine-build-evidence-sidecar-audit.json", onnxEngineBuildEvidenceSidecarValidator, StringComparison.Ordinal);
        Assert.Contains("owner-action-required", onnxEngineBuildEvidenceSidecarValidator, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime", onnxEngineBuildEvidenceSidecarValidator, StringComparison.Ordinal);
        Assert.Contains("real-model-runtime", onnxEngineBuildEvidenceSidecarValidator, StringComparison.Ordinal);
        Assert.Contains("modelSha256", onnxEngineBuildEvidenceSidecarValidator, StringComparison.Ordinal);
        Assert.Contains("preprocessedInputTensorName", onnxEngineBuildEvidenceSidecarValidator, StringComparison.Ordinal);
        Assert.Contains("preprocessedInputTensorSha256", onnxEngineBuildEvidenceSidecarValidator, StringComparison.Ordinal);
        Assert.Contains("preprocessed-input-tensor-sha256-missing", onnxEngineBuildEvidenceSidecarValidator, StringComparison.Ordinal);
        Assert.Contains("stdoutSummary", onnxEngineBuildEvidenceSidecarValidator, StringComparison.Ordinal);
        Assert.Contains("stderrSummary", onnxEngineBuildEvidenceSidecarValidator, StringComparison.Ordinal);
        Assert.Contains("canPromotePackageConsumerRuntime = $false", onnxEngineBuildEvidenceSidecarValidator, StringComparison.Ordinal);
        Assert.Contains("onnx-engine-build-evidence-sidecar.template.json", onnxEngineBuildEvidenceSidecarTemplate, StringComparison.Ordinal);
        Assert.Contains("onnx-engine-build-evidence-sidecar.classification.template.json", onnxEngineBuildEvidenceSidecarTemplate, StringComparison.Ordinal);
        Assert.Contains("onnx-engine-build-evidence-sidecar.yolovision.template.json", onnxEngineBuildEvidenceSidecarTemplate, StringComparison.Ordinal);
        Assert.Contains("onnx-engine-build-evidence-sidecar.yolox-s.template.json", onnxEngineBuildEvidenceSidecarTemplate, StringComparison.Ordinal);
        Assert.Contains("proofClassification = \"template-only\"", onnxEngineBuildEvidenceSidecarTemplate, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime belongs to release proof records", onnxEngineBuildEvidenceSidecarTemplate, StringComparison.Ordinal);
        Assert.Contains("sample-run-evidence-record.template.json", sampleRunEvidenceRecordTemplate, StringComparison.Ordinal);
        Assert.Contains("sample-run-evidence-record.classification.template.json", sampleRunEvidenceRecordTemplate, StringComparison.Ordinal);
        Assert.Contains("sample-run-evidence-record.yolovision.template.json", sampleRunEvidenceRecordTemplate, StringComparison.Ordinal);
        Assert.Contains("sample-run-evidence-record.yolox-s.template.json", sampleRunEvidenceRecordTemplate, StringComparison.Ordinal);
        Assert.Contains("canPromoteRealModelRuntime = $false", sampleRunEvidenceRecordTemplate, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime is forbidden in sample run evidence records", sampleRunEvidenceRecordTemplate, StringComparison.Ordinal);
        Assert.Contains("sample-run-evidence-record-validation.json", sampleRunEvidenceRecordValidator, StringComparison.Ordinal);
        Assert.Contains("owner-action-required", sampleRunEvidenceRecordValidator, StringComparison.Ordinal);
        Assert.Contains("real-model-runtime", sampleRunEvidenceRecordValidator, StringComparison.Ordinal);
        Assert.Contains("preprocessedInputTensorPath", sampleRunEvidenceRecordValidator, StringComparison.Ordinal);
        Assert.Contains("preprocessedInputTensorSha256", sampleRunEvidenceRecordValidator, StringComparison.Ordinal);
        Assert.Contains("yolovision-external-input-evidence-line", sampleRunEvidenceRecordValidator, StringComparison.Ordinal);
        Assert.Contains("InputSource=external", sampleRunEvidenceRecordValidator, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime-forbidden", sampleRunEvidenceRecordValidator, StringComparison.Ordinal);
        Assert.Contains("canPromoteRealModelRuntime", sampleRunEvidenceRecordValidator, StringComparison.Ordinal);
        Assert.Contains("FailOnNotProof", sampleRunEvidenceRecordValidator, StringComparison.Ordinal);
        Assert.Contains("real-model-owner-handoff.json", realModelOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("real-model-owner-handoff.md", realModelOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("samples/assets/*-example.json", realModelOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("performsDownload = $false", realModelOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("performsSampleRun = $false", realModelOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("packageConsumerRuntimeForbidden", realModelOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("Sample evidence can promote only to real-model-runtime", realModelOwnerHandoff, StringComparison.Ordinal);
        Assert.Contains("deferred-readonly-api-candidate-plan.json", deferredReadOnlyPlan, StringComparison.Ordinal);
        Assert.Contains("IncludeMediumRisk", deferredReadOnlyPlan, StringComparison.Ordinal);
        Assert.Contains("highRiskDeferredRowCount", deferredReadOnlyPlan, StringComparison.Ordinal);
        Assert.Contains("Deduplicate by interface and TensorRT major line", deferredReadOnlyPlan, StringComparison.Ordinal);
        Assert.Contains("active deferred-only low-risk rows only", deferredReadOnlyPlan, StringComparison.Ordinal);
        Assert.Contains("algorithm selector callback", deferredReadOnlyPlan, StringComparison.Ordinal);
        Assert.Contains("candidate-for-readonly-diagnostic-promotion", deferredReadOnlyPlan, StringComparison.Ordinal);
        Assert.Contains("Do not delete deferred records", deferredReadOnlyPlan, StringComparison.Ordinal);
        Assert.Contains("sampleAssetManifestAuditStatus", userAcceptanceCatalog, StringComparison.Ordinal);
        Assert.Contains("sampleAssetAcquisitionPlanState", userAcceptanceCatalog, StringComparison.Ordinal);
        Assert.Contains("sampleRunEvidenceValidationState", userAcceptanceCatalog, StringComparison.Ordinal);
        Assert.Contains("sampleRunEvidenceCanPromoteRealModelRuntime", userAcceptanceCatalog, StringComparison.Ordinal);
        Assert.Contains("realModelOwnerHandoffState", userAcceptanceCatalog, StringComparison.Ordinal);
        Assert.Contains("real-model-owner-handoff.json", userAcceptanceCatalog, StringComparison.Ordinal);
        Assert.Contains("runner evidence state:", userAcceptanceCatalog, StringComparison.Ordinal);
        Assert.Contains("proof classification:", userAcceptanceCatalog, StringComparison.Ordinal);
        Assert.Contains("Candidate manifests, build-only records, acquisition plans, owner handoffs, and owner-action-required runner evidence are not sample smoke passes", userAcceptanceCatalog, StringComparison.Ordinal);
        Assert.Contains("stale-release-claims-audit.json", staleClaims, StringComparison.Ordinal);
        Assert.Contains("old-bilingual-281", staleClaims, StringComparison.Ordinal);
        Assert.Contains("allow-runtime-smoke-blocked-ready", staleClaims, StringComparison.Ordinal);
        Assert.Contains("dependency-probe-runtime-proof", staleClaims, StringComparison.Ordinal);
        Assert.Contains("compatible-host-runbook-runtime-proof", staleClaims, StringComparison.Ordinal);
        Assert.Contains("compatible-host-collection-runtime-proof", staleClaims, StringComparison.Ordinal);
        Assert.Contains("compatible-host-collection-public-release", staleClaims, StringComparison.Ordinal);
        Assert.Contains("collection-bundle-can-promote-true", staleClaims, StringComparison.Ordinal);
        Assert.Contains("collection-bundle-runtime-evidence-true", staleClaims, StringComparison.Ordinal);
        Assert.Contains("collection-bundle-performs-publish-true", staleClaims, StringComparison.Ordinal);
        Assert.Contains("collection-bundle-approves-release-true", staleClaims, StringComparison.Ordinal);
        Assert.Contains("release-owner-decision-record.md", staleClaims, StringComparison.Ordinal);
        Assert.Contains("release-evidence-bundle.md", staleClaims, StringComparison.Ordinal);
        Assert.Contains("release-package-proof-bundle.md", staleClaims, StringComparison.Ordinal);
        Assert.Contains("docs-publish-readiness-bundle.md", staleClaims, StringComparison.Ordinal);
        Assert.Contains("release-owner-approval-input-template.md", staleClaims, StringComparison.Ordinal);
        Assert.Contains("release-owner-approval-input-validation.md", staleClaims, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record-template.md", staleClaims, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-validation.md", staleClaims, StringComparison.Ordinal);
        Assert.Contains("release-publish-execution-checklist.md", staleClaims, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-record-template.md", staleClaims, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-validation.md", staleClaims, StringComparison.Ordinal);
        Assert.Contains("release-promotion-issue-record.md", staleClaims, StringComparison.Ordinal);
        Assert.Contains("artifacts\\linux-dry-run", staleClaims, StringComparison.Ordinal);
        Assert.Contains("runtime-proof-complete", staleClaims, StringComparison.Ordinal);
        Assert.Contains("release-ready-to-publish", staleClaims, StringComparison.Ordinal);
        Assert.Contains("release-ready-to-publish-hyphen", staleClaims, StringComparison.Ordinal);
        Assert.Contains("post-publish-verified", staleClaims, StringComparison.Ordinal);
        Assert.Contains("post-publish-cn-verified", staleClaims, StringComparison.Ordinal);
        Assert.Contains("published-to-nuget", staleClaims, StringComparison.Ordinal);
        Assert.Contains("nuget-published-en", staleClaims, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime-passed", staleClaims, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime-cn-passed", staleClaims, StringComparison.Ordinal);
        Assert.Contains("real-model-runtime-passed", staleClaims, StringComparison.Ordinal);
        Assert.Contains("real-model-runtime-cn-passed", staleClaims, StringComparison.Ordinal);
        Assert.Contains("build-only-release-proof", staleClaims, StringComparison.Ordinal);
        Assert.Contains("parse-only-implemented", staleClaims, StringComparison.Ordinal);
        Assert.Contains("sidecar-only-runtime-proof", staleClaims, StringComparison.Ordinal);
        Assert.Contains("sidecar-cn-close-release-issue", staleClaims, StringComparison.Ordinal);
        Assert.Contains("sidecar-en-close-release-issue", staleClaims, StringComparison.Ordinal);
        Assert.Contains("classification-passed-without-boundary", staleClaims, StringComparison.Ordinal);
        Assert.Contains("yolovision-passed-without-boundary", staleClaims, StringComparison.Ordinal);
        Assert.Contains("yolodet-project-name", staleClaims, StringComparison.Ordinal);
        Assert.Contains("README.zh-CN.md", staleClaims, StringComparison.Ordinal);
        Assert.Contains("\"samples\"", staleClaims, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseCandidateDocsExposeLocalFeedMatrixAndRcChecklist()
    {
        string toc = ReadSource("docs", "toc.yml");
        string index = ReadSource("docs", "index.md");
        string localFeedDoc = ReadSource("docs", "articles", "zh-cn", "local-nuget-feed-consumer.md");
        string matrixDoc = ReadSource("docs", "articles", "zh-cn", "runtime-package-matrix.md");
        string releaseNotes = ReadSource("docs", "articles", "zh-cn", "release-notes-4.0.0-rc.md");
        string limitations = ReadSource("docs", "articles", "zh-cn", "known-limitations-4.0.0-rc.md");
        string releaseCandidateGate = ReadSource("docs", "articles", "zh-cn", "release-candidate-gate.md");
        string checklist = ReadSource("docs", "articles", "zh-cn", "release-checklist-4.0.0-rc.md");
        string finalReleaseDryRun = ReadSource("docs", "articles", "zh-cn", "final-release-dry-run.md");
        string signingPolicy = ReadSource("docs", "articles", "zh-cn", "signing-and-trust-policy.md");
        string releaseGuide = ReadSource("docs", "articles", "zh-cn", "nuget-and-github-packages-release-guide.md");
        string packageConsumerValidationFlow = ReadSource("docs", "articles", "zh-cn", "nuget-package-consumer-validation-flow.md");
        string userAcceptance = ReadSource("docs", "articles", "zh-cn", "user-acceptance-samples.md");
        string linuxHandoff = ReadSource("docs", "articles", "zh-cn", "linux-runtime-handoff.md");
        string releaseOwnerGuide = ReadSource("docs", "articles", "zh-cn", "release-owner-approval-guide.md");
        string releaseOwnerRecord = ReadSource("docs", "articles", "zh-cn", "release-owner-decision-record.md");
        string releaseEvidenceBundleDoc = ReadSource("docs", "articles", "zh-cn", "release-evidence-bundle.md");
        string releasePackageProofBundleDoc = ReadSource("docs", "articles", "zh-cn", "release-package-proof-bundle.md");
        string docsPublishReadinessBundleDoc = ReadSource("docs", "articles", "zh-cn", "docs-publish-readiness-bundle.md");
        string releaseOwnerApprovalInputDoc = ReadSource("docs", "articles", "zh-cn", "release-owner-approval-input.md");
        string releaseOwnerDryRun = ReadSource("docs", "articles", "zh-cn", "release-owner-dry-run-to-decision.md");
        string releaseChannelPreflight = ReadSource("docs", "articles", "zh-cn", "release-channel-preflight-and-rollback.md");
        string releasePublishExecutionChecklistDoc = ReadSource("docs", "articles", "zh-cn", "release-publish-execution-checklist.md");
        string releaseCandidateFreezeDoc = ReadSource("docs", "articles", "zh-cn", "release-candidate-freeze.md");
        string ownerAuthorizedPublishCommandPlanDoc = ReadSource("docs", "articles", "zh-cn", "owner-authorized-publish-command-plan.md");
        string externalRuntimeProofRecordDoc = ReadSource("docs", "articles", "zh-cn", "external-runtime-proof-record.md");
        string externalRuntimeProofBackfillPlanDoc = ReadSource("docs", "articles", "zh-cn", "external-runtime-proof-backfill-plan.md");
        string compatibleHostRunbookDoc = ReadSource("docs", "articles", "zh-cn", "compatible-host-runtime-proof-runbook.md");
        string postPublishVerificationRecordDoc = ReadSource("docs", "articles", "zh-cn", "post-publish-verification-record.md");
        string postPublishVerificationBackfillPlanDoc = ReadSource("docs", "articles", "zh-cn", "post-publish-verification-backfill-plan.md");
        string bilingualProgress = ReadSource("docs", "articles", "zh-cn", "public-api-bilingual-documentation-progress.md");
        string linuxRunnerEvidenceChecklist = ReadSource("docs", "articles", "zh-cn", "linux-runner-evidence-checklist.md");
        string linuxRunnerEvidenceRecordSchema = ReadSource("docs", "articles", "zh-cn", "linux-runner-evidence-record-schema.md");
        string classificationAssets = ReadSource("docs", "articles", "zh-cn", "classification-model-assets.md");
        string classificationAssetCandidates = ReadSource("docs", "articles", "zh-cn", "classification-asset-candidates.md");
        string yoloAssets = ReadSource("docs", "articles", "zh-cn", "yolovision-model-assets.md");
        string yoloAssetCandidates = ReadSource("docs", "articles", "zh-cn", "yolovision-asset-candidates.md");
        string sampleAssetManifestGuide = ReadSource("docs", "articles", "zh-cn", "sample-asset-manifest-guide.md");
        string sampleAssetAcquisitionPlanDoc = ReadSource("docs", "articles", "zh-cn", "sample-asset-acquisition-plan.md");
        string tensorRtObjectModel = ReadSource("docs", "articles", "zh-cn", "tensorrt-object-model.md");
        string pluginInventory = ReadSource("docs", "articles", "zh-cn", "plugin-inventory-readonly-api.md");
        string pluginOwnershipBoundary = ReadSource("docs", "articles", "zh-cn", "plugin-ownership-boundary.md");
        string pluginSerialization = ReadSource("docs", "articles", "zh-cn", "plugin-serialization-paths.md");
        string cudaGraphArticle = ReadSource("docs", "articles", "zh-cn", "cuda-graph-capabilities-boundary.md");
        string cudaMemoryWrapper = ReadSource("docs", "articles", "zh-cn", "cuda-memory-wrapper.md");
        string cudaMemoryRange = ReadSource("docs", "articles", "zh-cn", "cuda-memory-range-apis.md");
        string refitWeights = ReadSource("docs", "articles", "zh-cn", "refit-weights-guide.md");
        string trt11ModernLayers = ReadSource("docs", "articles", "zh-cn", "trt11-modern-layers-guide.md");
        string networkLayerCoverage = ReadSource("docs", "articles", "zh-cn", "network-layer-coverage-guide.md");
        string errorRecorderSnapshot = ReadSource("docs", "articles", "zh-cn", "error-recorder-snapshot-guide.md");
        string managedCallbackGuide = ReadSource("docs", "articles", "zh-cn", "managed-logger-profiler-progress-monitor.md");
        string localNuGetDeepDive = ReadSource("docs", "articles", "zh-cn", "local-nuget-feed-deep-dive.md");
        string runtimeMatrixGuide = ReadSource("docs", "articles", "zh-cn", "runtime-package-matrix-reading-guide.md");
        string readinessSummaryGuide = ReadSource("docs", "articles", "zh-cn", "readiness-summary-guide.md");
        string packageReadinessCurrentState = ReadSource("docs", "articles", "zh-cn", "package-readiness-current-state.md");
        string callbackAllocatorGuide = ReadSource("docs", "articles", "zh-cn", "callback-allocator-boundary-guide.md");
        string deferredManualDesignGroupsDoc = ReadSource("docs", "articles", "zh-cn", "deferred-manual-design-groups.md");
        string blogProjectIntroduction = ReadSource("docs", "articles", "zh-cn", "blog-project-introduction.md");
        string blogPackageConsumerEvidence = ReadSource("docs", "articles", "zh-cn", "blog-package-consumer-evidence-chain.md");
        string blogLinuxRunnerEvidence = ReadSource("docs", "articles", "zh-cn", "blog-linux-runner-evidence-guide.md");
        string blogDynamicShape = ReadSource("docs", "articles", "zh-cn", "blog-dynamic-shape-optimization-profile.md");
        string blogInferenceBindings = ReadSource("docs", "articles", "zh-cn", "blog-inference-bindings-identity-network.md");
        string blogOnnxParser = ReadSource("docs", "articles", "zh-cn", "blog-onnx-parser-engine-roundtrip.md");
        string blogMultiStream = ReadSource("docs", "articles", "zh-cn", "blog-multistream-cuda-stream-event.md");
        string blogPluginInventory = ReadSource("docs", "articles", "zh-cn", "blog-plugin-inventory-readonly-api.md");
        string blogCudaMemory = ReadSource("docs", "articles", "zh-cn", "blog-cuda-memory-wrapper.md");
        string blogRefitWeights = ReadSource("docs", "articles", "zh-cn", "blog-refit-weights-guide.md");
        string blogNetworkLayerCoverage = ReadSource("docs", "articles", "zh-cn", "blog-network-layer-coverage-guide.md");
        string technicalArticleRoadmap = ReadSource("docs", "articles", "zh-cn", "technical-article-roadmap.md");
        string onnxToEngineGuide = ReadSource("docs", "articles", "zh-cn", "onnx-to-engine-trtexec-conversion-guide.md");
        string tensorRtExecToolGuide = ReadSource("docs", "articles", "zh-cn", "tensorrtexec-tool-getting-started.md");
        string tensorRtExecGui = ReadSource("docs", "articles", "zh-cn", "tensorrtexec-gui-user-guide.md");
        string tensorRtExecExternalReport = ReadSource("docs", "articles", "zh-cn", "tensorrtexec-external-onnx-build-report.md");
        string classificationRealAssetWalkthrough = ReadSource("docs", "articles", "zh-cn", "classification-real-asset-walkthrough.md");
        string realModelOwnerBackfillChecklist = ReadSource("docs", "articles", "zh-cn", "real-model-owner-backfill-checklist.md");
        string classificationReadme = ReadSource("samples", "Classification", "README.md");
        string yoloReadme = ReadSource("samples", "YoloVision", "README.md");
        string sampleAssetsReadme = ReadSource("samples", "assets", "README.md");
        string classificationManifest = ReadSource("samples", "assets", "classification-assets.template.json");
        string yoloManifest = ReadSource("samples", "assets", "yolovision-assets.template.json");

        Assert.Contains("local-nuget-feed-consumer.md", toc, StringComparison.Ordinal);
        Assert.Contains("runtime-package-matrix.md", toc, StringComparison.Ordinal);
        Assert.Contains("release-notes-4.0.0-rc.md", toc, StringComparison.Ordinal);
        Assert.Contains("known-limitations-4.0.0-rc.md", toc, StringComparison.Ordinal);
        Assert.Contains("release-checklist-4.0.0-rc.md", toc, StringComparison.Ordinal);
        Assert.Contains("final-release-dry-run.md", toc, StringComparison.Ordinal);
        Assert.Contains("signing-and-trust-policy.md", toc, StringComparison.Ordinal);
        Assert.Contains("nuget-and-github-packages-release-guide.md", toc, StringComparison.Ordinal);
        Assert.Contains("user-acceptance-samples.md", toc, StringComparison.Ordinal);
        Assert.Contains("linux-runtime-handoff.md", toc, StringComparison.Ordinal);
        Assert.Contains("release-owner-approval-guide.md", toc, StringComparison.Ordinal);
        Assert.Contains("release-owner-decision-record.md", toc, StringComparison.Ordinal);
        Assert.Contains("release-evidence-bundle.md", toc, StringComparison.Ordinal);
        Assert.Contains("release-package-proof-bundle.md", toc, StringComparison.Ordinal);
        Assert.Contains("docs-publish-readiness-bundle.md", toc, StringComparison.Ordinal);
        Assert.Contains("release-owner-approval-input.md", toc, StringComparison.Ordinal);
        Assert.Contains("release-owner-dry-run-to-decision.md", toc, StringComparison.Ordinal);
        Assert.Contains("linux-runner-evidence-checklist.md", toc, StringComparison.Ordinal);
        Assert.Contains("linux-runner-evidence-record-schema.md", toc, StringComparison.Ordinal);
        Assert.Contains("release-channel-preflight-and-rollback.md", toc, StringComparison.Ordinal);
        Assert.Contains("release-publish-execution-checklist.md", toc, StringComparison.Ordinal);
        Assert.Contains("release-candidate-freeze.md", toc, StringComparison.Ordinal);
        Assert.Contains("owner-authorized-publish-command-plan.md", toc, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record.md", toc, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-backfill-plan.md", toc, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-record.md", toc, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-backfill-plan.md", toc, StringComparison.Ordinal);
        Assert.Contains("classification-model-assets.md", toc, StringComparison.Ordinal);
        Assert.Contains("classification-asset-candidates.md", toc, StringComparison.Ordinal);
        Assert.Contains("yolovision-model-assets.md", toc, StringComparison.Ordinal);
        Assert.Contains("yolovision-asset-candidates.md", toc, StringComparison.Ordinal);
        Assert.Contains("sample-asset-manifest-guide.md", toc, StringComparison.Ordinal);
        Assert.Contains("sample-asset-acquisition-plan.md", toc, StringComparison.Ordinal);
        Assert.Contains("real-model-owner-backfill-checklist.md", toc, StringComparison.Ordinal);
        Assert.Contains("tensorrt-object-model.md", toc, StringComparison.Ordinal);
        Assert.Contains("cuda-memory-wrapper.md", toc, StringComparison.Ordinal);
        Assert.Contains("cuda-memory-range-apis.md", toc, StringComparison.Ordinal);
        Assert.Contains("refit-weights-guide.md", toc, StringComparison.Ordinal);
        Assert.Contains("trt11-modern-layers-guide.md", toc, StringComparison.Ordinal);
        Assert.Contains("network-layer-coverage-guide.md", toc, StringComparison.Ordinal);
        Assert.Contains("error-recorder-snapshot-guide.md", toc, StringComparison.Ordinal);
        Assert.Contains("managed-logger-profiler-progress-monitor.md", toc, StringComparison.Ordinal);
        Assert.Contains("local-nuget-feed-deep-dive.md", toc, StringComparison.Ordinal);
        Assert.Contains("public-api-bilingual-documentation-progress.md", toc, StringComparison.Ordinal);
        Assert.Contains("plugin-inventory-readonly-api.md", toc, StringComparison.Ordinal);
        Assert.Contains("plugin-ownership-boundary.md", toc, StringComparison.Ordinal);
        Assert.Contains("plugin-serialization-paths.md", toc, StringComparison.Ordinal);
        Assert.Contains("cuda-graph-capabilities-boundary.md", toc, StringComparison.Ordinal);
        Assert.Contains("runtime-package-matrix-reading-guide.md", toc, StringComparison.Ordinal);
        Assert.Contains("callback-allocator-boundary-guide.md", toc, StringComparison.Ordinal);
        Assert.Contains("deferred-manual-design-groups.md", toc, StringComparison.Ordinal);
        Assert.Contains("blog-project-introduction.md", toc, StringComparison.Ordinal);
        Assert.Contains("blog-package-consumer-evidence-chain.md", toc, StringComparison.Ordinal);
        Assert.Contains("blog-linux-runner-evidence-guide.md", toc, StringComparison.Ordinal);
        Assert.Contains("blog-dynamic-shape-optimization-profile.md", toc, StringComparison.Ordinal);
        Assert.Contains("blog-inference-bindings-identity-network.md", toc, StringComparison.Ordinal);
        Assert.Contains("blog-onnx-parser-engine-roundtrip.md", toc, StringComparison.Ordinal);
        Assert.Contains("blog-multistream-cuda-stream-event.md", toc, StringComparison.Ordinal);
        Assert.Contains("blog-plugin-inventory-readonly-api.md", toc, StringComparison.Ordinal);
        Assert.Contains("blog-cuda-memory-wrapper.md", toc, StringComparison.Ordinal);
        Assert.Contains("blog-refit-weights-guide.md", toc, StringComparison.Ordinal);
        Assert.Contains("blog-network-layer-coverage-guide.md", toc, StringComparison.Ordinal);
        Assert.Contains("Public API Bilingual Documentation Progress", index, StringComparison.Ordinal);
        Assert.Contains("Plugin Inventory Readonly API", index, StringComparison.Ordinal);
        Assert.Contains("Plugin Ownership Boundary", index, StringComparison.Ordinal);
        Assert.Contains("Plugin Serialization Paths", index, StringComparison.Ordinal);
        Assert.Contains("CUDA Graph Capabilities And Boundary", index, StringComparison.Ordinal);
        Assert.Contains("Runtime Package Matrix Reading Guide", index, StringComparison.Ordinal);
        Assert.Contains("Callback And Allocator Boundary Guide", index, StringComparison.Ordinal);
        Assert.Contains("Deferred Manual Design Groups", index, StringComparison.Ordinal);
        Assert.Contains("Release owner decision template", index, StringComparison.Ordinal);
        Assert.Contains("Linux runner evidence template", index, StringComparison.Ordinal);
        Assert.Contains("Stale release claims audit", index, StringComparison.Ordinal);
        Assert.Contains("Local NuGet Feed Consumer", index, StringComparison.Ordinal);
        Assert.Contains("Runtime Package Matrix", index, StringComparison.Ordinal);
        Assert.Contains("Final Release Dry Run", index, StringComparison.Ordinal);
        Assert.Contains("Linux Runtime Handoff", index, StringComparison.Ordinal);
        Assert.Contains("Release Owner Approval Guide", index, StringComparison.Ordinal);
        Assert.Contains("Release Owner Decision Record", index, StringComparison.Ordinal);
        Assert.Contains("Release Evidence Bundle", index, StringComparison.Ordinal);
        Assert.Contains("Release Package Proof Bundle", index, StringComparison.Ordinal);
        Assert.Contains("Docs Publish Readiness Bundle", index, StringComparison.Ordinal);
        Assert.Contains("Release Owner Approval Input", index, StringComparison.Ordinal);
        Assert.Contains("Release Owner From Dry Run To Decision", index, StringComparison.Ordinal);
        Assert.Contains("Linux Runner Evidence Checklist", index, StringComparison.Ordinal);
        Assert.Contains("Linux Runner Evidence Record Schema", index, StringComparison.Ordinal);
        Assert.Contains("Release Channel Preflight And Rollback", index, StringComparison.Ordinal);
        Assert.Contains("Release Publish Execution Checklist", index, StringComparison.Ordinal);
        Assert.Contains("Release Candidate Freeze", index, StringComparison.Ordinal);
        Assert.Contains("Release Candidate Full Acceptance Summary", index, StringComparison.Ordinal);
        Assert.Contains("External Runtime Proof Record", index, StringComparison.Ordinal);
        Assert.Contains("Post Publish Verification Record", index, StringComparison.Ordinal);
        Assert.Contains("Classification Model Assets", index, StringComparison.Ordinal);
        Assert.Contains("Classification Asset Candidates", index, StringComparison.Ordinal);
        Assert.Contains("Classification Real Asset Walkthrough", index, StringComparison.Ordinal);
        Assert.Contains("YoloVision Model Assets", index, StringComparison.Ordinal);
        Assert.Contains("YoloVision Asset Candidates", index, StringComparison.Ordinal);
        Assert.Contains("Sample Asset Manifest Guide", index, StringComparison.Ordinal);
        Assert.Contains("Sample Asset Acquisition Plan", index, StringComparison.Ordinal);
        Assert.Contains("Real Model Owner Backfill Checklist", index, StringComparison.Ordinal);
        Assert.Contains("TensorRT Builder Runtime Engine Object Model", index, StringComparison.Ordinal);
        Assert.Contains("CUDA Memory Wrapper", index, StringComparison.Ordinal);
        Assert.Contains("CUDA Memory Range APIs", index, StringComparison.Ordinal);
        Assert.Contains("Refit Weights Guide", index, StringComparison.Ordinal);
        Assert.Contains("TensorRT 11 Modern Layers Guide", index, StringComparison.Ordinal);
        Assert.Contains("Network Layer Coverage Guide", index, StringComparison.Ordinal);
        Assert.Contains("ErrorRecorder Snapshot Guide", index, StringComparison.Ordinal);
        Assert.Contains("Managed Logger Profiler Progress Monitor", index, StringComparison.Ordinal);
        Assert.Contains("Local NuGet Feed Deep Dive", index, StringComparison.Ordinal);
        Assert.Contains("Release owner decision record", index, StringComparison.Ordinal);
        Assert.Contains("Release evidence bundle", index, StringComparison.Ordinal);
        Assert.Contains("Release package proof bundle", index, StringComparison.Ordinal);
        Assert.Contains("Docs publish readiness bundle", index, StringComparison.Ordinal);
        Assert.Contains("Release publish execution checklist", index, StringComparison.Ordinal);
        Assert.Contains("External runtime proof record template", index, StringComparison.Ordinal);
        Assert.Contains("Post publish verification record template", index, StringComparison.Ordinal);
        Assert.Contains("Release promotion issue record", index, StringComparison.Ordinal);
        Assert.Contains("Linux runner evidence record template", index, StringComparison.Ordinal);
        Assert.Contains("Signing And Trust Policy", index, StringComparison.Ordinal);
        Assert.Contains("User Acceptance Samples", index, StringComparison.Ordinal);
        Assert.Contains("Blog Project Introduction", index, StringComparison.Ordinal);
        Assert.Contains("Blog Package Consumer Evidence Chain", index, StringComparison.Ordinal);
        Assert.Contains("Blog Linux Runner Evidence Guide", index, StringComparison.Ordinal);
        Assert.Contains("Blog Dynamic Shape Optimization Profile", index, StringComparison.Ordinal);
        Assert.Contains("Blog InferenceBindings Identity Network", index, StringComparison.Ordinal);
        Assert.Contains("Blog ONNX Parser Engine RoundTrip", index, StringComparison.Ordinal);
        Assert.Contains("Blog MultiStream CUDA Stream Event", index, StringComparison.Ordinal);
        Assert.Contains("Blog Plugin Inventory Readonly API", index, StringComparison.Ordinal);
        Assert.Contains("Blog CUDA Memory Wrapper", index, StringComparison.Ordinal);
        Assert.Contains("Blog Refit Weights Guide", index, StringComparison.Ordinal);
        Assert.Contains("Blog Network Layer Coverage Guide", index, StringComparison.Ordinal);

        Assert.Contains("不允许 `ProjectReference`", localFeedDoc, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", localFeedDoc, StringComparison.Ordinal);
        Assert.Contains("Evidence Classification", packageConsumerValidationFlow, StringComparison.Ordinal);
        Assert.Contains("IsDependencyProbeOnly", packageConsumerValidationFlow, StringComparison.Ordinal);
        Assert.Contains("InvocationCount>0", matrixDoc, StringComparison.Ordinal);
        Assert.Contains("Runtime proof", matrixDoc, StringComparison.Ordinal);
        Assert.Contains("CUDA error 35", releaseNotes, StringComparison.Ordinal);
        Assert.Contains("IDebugListener::processDebugTensor", limitations, StringComparison.Ordinal);
        Assert.Contains("packageConsumerEvidenceKind", releaseCandidateGate, StringComparison.Ordinal);
        Assert.Contains("runtimeSmokeClassification", releaseCandidateGate, StringComparison.Ordinal);
        Assert.Contains("runtimeProofStatus", releaseCandidateGate, StringComparison.Ordinal);
        Assert.Contains("isRuntimeExecutionEvidence", releaseCandidateGate, StringComparison.Ordinal);
        Assert.Contains("isDependencyProbeOnly", releaseCandidateGate, StringComparison.Ordinal);
        Assert.Contains("isRealCallbackRuntimeProof", releaseCandidateGate, StringComparison.Ordinal);
        Assert.Contains("Export-ReleaseCandidateChecklist.ps1", checklist, StringComparison.Ordinal);
        Assert.Contains("Test-ReleaseCandidateReadiness.ps1", checklist, StringComparison.Ordinal);
        Assert.Contains("ready-needs-manual-approval", finalReleaseDryRun, StringComparison.Ordinal);
        Assert.Contains("AllowRuntimeSmokeBlocked", finalReleaseDryRun, StringComparison.Ordinal);
        Assert.Contains("InvocationCount>0", finalReleaseDryRun, StringComparison.Ordinal);
        Assert.Contains("Export-PublicApiBilingualDocumentationBacklog.ps1", finalReleaseDryRun, StringComparison.Ordinal);
        Assert.Contains("packageConsumerEvidenceKind", finalReleaseDryRun, StringComparison.Ordinal);
        Assert.Contains("runtimeSmokeClassification", finalReleaseDryRun, StringComparison.Ordinal);
        Assert.Contains("runtimeProofStatus", finalReleaseDryRun, StringComparison.Ordinal);
        Assert.Contains("runtimeProofRequiredForRelease", finalReleaseDryRun, StringComparison.Ordinal);
        Assert.Contains("postPublishCommandsReady", finalReleaseDryRun, StringComparison.Ordinal);
        Assert.Contains("postPublishStdoutStderrSummaryReady", finalReleaseDryRun, StringComparison.Ordinal);
        Assert.Contains("isRuntimeExecutionEvidence", finalReleaseDryRun, StringComparison.Ordinal);
        Assert.Contains("isDependencyProbeOnly", finalReleaseDryRun, StringComparison.Ordinal);
        Assert.Contains("isRealCallbackRuntimeProof", finalReleaseDryRun, StringComparison.Ordinal);
        Assert.Contains("unsigned-or-not-requested", signingPolicy, StringComparison.Ordinal);
        Assert.Contains("NVIDIA", signingPolicy, StringComparison.Ordinal);
        Assert.Contains("Test-FinalReleaseDryRun.ps1", releaseGuide, StringComparison.Ordinal);
        Assert.Contains("AllowRuntimeSmokeBlocked", releaseGuide, StringComparison.Ordinal);
        Assert.Contains("asset-required", userAcceptance, StringComparison.Ordinal);
        Assert.Contains("不是 Linux runner proof", linuxHandoff, StringComparison.Ordinal);
        Assert.Contains("dry-run-only", linuxHandoff, StringComparison.Ordinal);
        Assert.Contains("ready-needs-manual-approval", releaseOwnerGuide, StringComparison.Ordinal);
        Assert.Contains("runtimeProofStatus=blocked-by-cuda-driver", releaseOwnerGuide, StringComparison.Ordinal);
        Assert.Contains("runtimeProofRequiredForRelease=true", releaseOwnerGuide, StringComparison.Ordinal);
        Assert.Contains("bilingualDocumentationFindingCount=0", releaseOwnerGuide, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly=false", releaseOwnerRecord, StringComparison.Ordinal);
        Assert.Contains("pending-release-owner-approval", releaseOwnerRecord, StringComparison.Ordinal);
        Assert.Contains("runtimeProofStatus", releaseOwnerRecord, StringComparison.Ordinal);
        Assert.Contains("runtimeProofRequiredForRelease", releaseOwnerRecord, StringComparison.Ordinal);
        Assert.Contains("release-evidence-bundle.json", releaseEvidenceBundleDoc, StringComparison.Ordinal);
        Assert.Contains("bundleState=blocked-evidence-incomplete", releaseEvidenceBundleDoc, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly=false", releaseEvidenceBundleDoc, StringComparison.Ordinal);
        Assert.Contains("canExecutePublicPublish=false", releaseEvidenceBundleDoc, StringComparison.Ordinal);
        Assert.Contains("canPromoteRuntimeProof=false", releaseEvidenceBundleDoc, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue=false", releaseEvidenceBundleDoc, StringComparison.Ordinal);
        Assert.Contains("isReleaseEvidenceComplete=false", releaseEvidenceBundleDoc, StringComparison.Ordinal);
        Assert.Contains("postPublishCommandsReady=false", releaseEvidenceBundleDoc, StringComparison.Ordinal);
        Assert.Contains("postPublishStdoutStderrSummaryReady=false", releaseEvidenceBundleDoc, StringComparison.Ordinal);
        Assert.Contains("DependencyProbe 不是 runtime execution proof", releaseEvidenceBundleDoc, StringComparison.Ordinal);
        Assert.Contains("release-package-proof-bundle.json", releaseEvidenceBundleDoc, StringComparison.Ordinal);
        Assert.Contains("docs-publish-readiness-bundle.json", releaseEvidenceBundleDoc, StringComparison.Ordinal);
        Assert.Contains("compatible-host-runtime-proof-runbook.json", releaseEvidenceBundleDoc, StringComparison.Ordinal);
        Assert.Contains("package proof bundle 是本地包", releaseEvidenceBundleDoc, StringComparison.Ordinal);
        Assert.Contains("docs publish readiness bundle 是文档本地材料", releaseEvidenceBundleDoc, StringComparison.Ordinal);
        Assert.Contains("release-package-proof-bundle.json", releasePackageProofBundleDoc, StringComparison.Ordinal);
        Assert.Contains("canUseAsPublicPackageProof=false", releasePackageProofBundleDoc, StringComparison.Ordinal);
        Assert.Contains("isRuntimeExecutionProof=false", releasePackageProofBundleDoc, StringComparison.Ordinal);
        Assert.Contains("isDependencyProbeOnly=true", releasePackageProofBundleDoc, StringComparison.Ordinal);
        Assert.Contains("runtimeProofStatus=blocked-by-cuda-driver", releasePackageProofBundleDoc, StringComparison.Ordinal);
        Assert.Contains("postPublishCommandsReady=false", releasePackageProofBundleDoc, StringComparison.Ordinal);
        Assert.Contains("postPublishStdoutStderrSummaryReady=true", releasePackageProofBundleDoc, StringComparison.Ordinal);
        Assert.Contains("docs-publish-readiness-bundle.json", docsPublishReadinessBundleDoc, StringComparison.Ordinal);
        Assert.Contains("canPublishDocsExternally=false", docsPublishReadinessBundleDoc, StringComparison.Ordinal);
        Assert.Contains("coreArticleCoverage", docsPublishReadinessBundleDoc, StringComparison.Ordinal);
        Assert.Contains("ownerActionCount", docsPublishReadinessBundleDoc, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-record.md", docsPublishReadinessBundleDoc, StringComparison.Ordinal);
        Assert.Contains("ready-for-owner-review", docsPublishReadinessBundleDoc, StringComparison.Ordinal);
        Assert.Contains("DocFX 本地输出不是外部站点发布证明", docsPublishReadinessBundleDoc, StringComparison.Ordinal);
        Assert.Contains("release-owner-approval-input-template.json", releaseOwnerApprovalInputDoc, StringComparison.Ordinal);
        Assert.Contains("Export-ReleaseOwnerApprovalInputExample.ps1", releaseOwnerApprovalInputDoc, StringComparison.Ordinal);
        Assert.Contains("example-not-for-publication", releaseOwnerApprovalInputDoc, StringComparison.Ordinal);
        Assert.Contains("blocked-owner-input-required", releaseOwnerApprovalInputDoc, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly=false", releaseOwnerApprovalInputDoc, StringComparison.Ordinal);
        Assert.Contains("approved-known-limitation-for-rc", releaseOwnerApprovalInputDoc, StringComparison.Ordinal);
        Assert.Contains("postPublishCommandsReady=false", releaseOwnerApprovalInputDoc, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-disposition", releaseOwnerApprovalInputDoc, StringComparison.Ordinal);
        Assert.Contains("Linux handoff 不是 Linux runner proof", releaseOwnerRecord, StringComparison.Ordinal);
        Assert.Contains("postPublishCommandsReady=false", releaseOwnerRecord, StringComparison.Ordinal);
        Assert.Contains("stdout/stderr summary", releaseOwnerRecord, StringComparison.Ordinal);
        Assert.Contains("Export-ReleaseOwnerDecisionRecord.ps1", releaseOwnerDryRun, StringComparison.Ordinal);
        Assert.Contains("Test-ReleaseOwnerApprovalInput.ps1", releaseOwnerDryRun, StringComparison.Ordinal);
        Assert.Contains("release-owner-decision-record.md", releaseOwnerDryRun, StringComparison.Ordinal);
        Assert.Contains("Export-ReleasePublishExecutionChecklist.ps1", releaseChannelPreflight, StringComparison.Ordinal);
        Assert.Contains("Export-ReleasePromotionIssueRecord.ps1", releaseChannelPreflight, StringComparison.Ordinal);
        Assert.Contains("Export-ReleaseEvidenceBundle.ps1", releaseChannelPreflight, StringComparison.Ordinal);
        Assert.Contains("Export-ReleasePackageProofBundle.ps1", releaseChannelPreflight, StringComparison.Ordinal);
        Assert.Contains("Export-DocsPublishReadinessBundle.ps1", releaseChannelPreflight, StringComparison.Ordinal);
        Assert.Contains("Export-ExternalRuntimeProofRecordInputTemplate.ps1", releaseChannelPreflight, StringComparison.Ordinal);
        Assert.Contains("Export-ExternalRuntimeProofRecordDraft.ps1", releaseChannelPreflight, StringComparison.Ordinal);
        Assert.Contains("Export-ExternalRuntimeProofRecordExample.ps1", releaseChannelPreflight, StringComparison.Ordinal);
        Assert.Contains("Export-ExternalRuntimeProofOwnerHandoff.ps1", releaseChannelPreflight, StringComparison.Ordinal);
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1", releaseChannelPreflight, StringComparison.Ordinal);
        Assert.Contains("Test-PostPublishVerificationRecord.ps1", releaseChannelPreflight, StringComparison.Ordinal);
        Assert.Contains("rollback", releaseChannelPreflight, StringComparison.Ordinal);
        Assert.Contains("owner approval input validation", releaseChannelPreflight, StringComparison.Ordinal);
        Assert.Contains("runtimeProofRequiredForRelease", releaseChannelPreflight, StringComparison.Ordinal);
        Assert.Contains("不允许 `ProjectReference`", releaseChannelPreflight, StringComparison.Ordinal);
        Assert.Contains("release-publish-execution-checklist.json", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("release-evidence-bundle.json", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("release-package-proof-bundle.json", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("docs-publish-readiness-bundle.json", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("canUseAsPublicPackageProof=false", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("canPublishDocsExternally=false", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("release evidence bundle 不是发布批准", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record-template.json", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-validation.json", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-record-template.json", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-validation.json", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("canExecutePublicPublish=false", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("performsPublish=false", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("publish placeholder", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("DependencyProbe BridgeInitialized", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("release-candidate-freeze-summary.json", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("freezeState=blocked-freeze-owner-action-required", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("owner-authorized-publish-command-plan.json", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("owner-authorized-publish-command-plan-validation.json", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("canMaterializeExecutableCommands=false", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("authorized=false", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("executable=false", releasePublishExecutionChecklistDoc, StringComparison.Ordinal);
        Assert.Contains("release-candidate-freeze-summary.json", releaseCandidateFreezeDoc, StringComparison.Ordinal);
        Assert.Contains("release-candidate-freeze-checklist.json", releaseCandidateFreezeDoc, StringComparison.Ordinal);
        Assert.Contains("release-candidate-freeze-validation.json", releaseCandidateFreezeDoc, StringComparison.Ordinal);
        Assert.Contains("owner-authorized-publish-command-plan.json", releaseCandidateFreezeDoc, StringComparison.Ordinal);
        Assert.Contains("owner-authorized-publish-command-plan-validation.json", releaseCandidateFreezeDoc, StringComparison.Ordinal);
        Assert.Contains("release candidate freeze 不是 publish", releaseCandidateFreezeDoc, StringComparison.Ordinal);
        Assert.Contains("ValidationState=blocked-freeze-owner-action-required", releaseCandidateFreezeDoc, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver 不是 smoke passed", releaseCandidateFreezeDoc, StringComparison.Ordinal);
        Assert.Contains("realExternalRuntimeProofReady=false", releaseCandidateFreezeDoc, StringComparison.Ordinal);
        Assert.Contains("realPostPublishVerificationReady=false", releaseCandidateFreezeDoc, StringComparison.Ordinal);
        Assert.Contains("owner-authorized-publish-command-plan.json", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("owner-authorized-publish-command-plan-validation.json", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-backfill-plan.json", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-backfill-plan.json", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("planState=blocked-owner-authorization-required", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("performsPublish=false", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("requiresExplicitOwnerAuthorization=true", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("canMaterializeExecutableCommands=false", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("Owner Authorization Proof Gate", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("ownerAuthorizationProofGate", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("requiredFieldCount=14", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("missingOwnerInputCount=14", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("approvedCommandPlanSha256", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("approvedProofBundleSha256", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("manualMaterializationPrerequisites", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("postPublishRequiredEvidence", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("runtimeSmokeLogSha256", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("host metadata", ownerAuthorizedPublishCommandPlanDoc, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("backfill plan 不是 proof", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("authorized=false", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("executable=false", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("dotnet nuget push", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("gh release upload", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("Test-PostPublishVerificationRecord.ps1 -FailOnNotProof", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", ownerAuthorizedPublishCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record-template.json", externalRuntimeProofRecordDoc, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record.input-template.json", externalRuntimeProofRecordDoc, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record.draft.json", externalRuntimeProofRecordDoc, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record.example.json", externalRuntimeProofRecordDoc, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-owner-handoff.json", externalRuntimeProofRecordDoc, StringComparison.Ordinal);
        Assert.Contains("compatible-host-runtime-proof-runbook.json", externalRuntimeProofRecordDoc, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-backfill-plan.json", externalRuntimeProofRecordDoc, StringComparison.Ordinal);
        Assert.Contains("backfill plan 不是 runtime proof", externalRuntimeProofRecordDoc, StringComparison.Ordinal);
        Assert.Contains("recordKind=external-runtime-proof-backfill-plan", externalRuntimeProofBackfillPlanDoc, StringComparison.Ordinal);
        Assert.Contains("blocked-compatible-host-proof-required", externalRuntimeProofBackfillPlanDoc, StringComparison.Ordinal);
        Assert.Contains("canPromoteRuntimeProof=false", externalRuntimeProofBackfillPlanDoc, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", externalRuntimeProofBackfillPlanDoc, StringComparison.Ordinal);
        Assert.Contains("-RequireExistingLog -FailOnNotProof", externalRuntimeProofBackfillPlanDoc, StringComparison.Ordinal);
        Assert.Contains("ExternalRuntimeProofOwnerHandoff", externalRuntimeProofRecordDoc, StringComparison.Ordinal);
        Assert.Contains("example-not-for-publication", externalRuntimeProofRecordDoc, StringComparison.Ordinal);
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1", externalRuntimeProofRecordDoc, StringComparison.Ordinal);
        Assert.Contains("RequireExistingLog", externalRuntimeProofRecordDoc, StringComparison.Ordinal);
        Assert.Contains("validationState=template-only", externalRuntimeProofRecordDoc, StringComparison.Ordinal);
        Assert.Contains("isRuntimeExecutionEvidence=false", externalRuntimeProofRecordDoc, StringComparison.Ordinal);
        Assert.Contains("canPromoteRuntimeProof=false", externalRuntimeProofRecordDoc, StringComparison.Ordinal);
        Assert.Contains("DependencyProbe BridgeInitialized", externalRuntimeProofRecordDoc, StringComparison.Ordinal);
        Assert.Contains("Export-CompatibleHostRuntimeProofRunbook.ps1", compatibleHostRunbookDoc, StringComparison.Ordinal);
        Assert.Contains("runbook 不是 runtime proof", compatibleHostRunbookDoc, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", compatibleHostRunbookDoc, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-record-template.json", postPublishVerificationRecordDoc, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-backfill-plan.json", postPublishVerificationRecordDoc, StringComparison.Ordinal);
        Assert.Contains("post-publish backfill plan 不是 post-publish proof", postPublishVerificationRecordDoc, StringComparison.Ordinal);
        Assert.Contains("recordKind=post-publish-verification-backfill-plan", postPublishVerificationBackfillPlanDoc, StringComparison.Ordinal);
        Assert.Contains("blocked-real-post-publish-proof-required", postPublishVerificationBackfillPlanDoc, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue=false", postPublishVerificationBackfillPlanDoc, StringComparison.Ordinal);
        Assert.Contains("ProjectReference", postPublishVerificationBackfillPlanDoc, StringComparison.Ordinal);
        Assert.Contains("-RequireExistingLog -FailOnNotProof", postPublishVerificationBackfillPlanDoc, StringComparison.Ordinal);
        Assert.Contains("Test-PostPublishVerificationRecord.ps1", postPublishVerificationRecordDoc, StringComparison.Ordinal);
        Assert.Contains("validationState=template-only", postPublishVerificationRecordDoc, StringComparison.Ordinal);
        Assert.Contains("isPostPublishVerificationProof=false", postPublishVerificationRecordDoc, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue=false", postPublishVerificationRecordDoc, StringComparison.Ordinal);
        Assert.Contains("no ProjectReference", postPublishVerificationRecordDoc, StringComparison.Ordinal);
        Assert.Contains("channelSourceUri", postPublishVerificationRecordDoc, StringComparison.Ordinal);
        Assert.Contains("smokeLogSha256", postPublishVerificationRecordDoc, StringComparison.Ordinal);
        Assert.Contains("owner-authorized-publish-command-plan", postPublishVerificationRecordDoc, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", postPublishVerificationRecordDoc, StringComparison.Ordinal);
        Assert.Contains("linux-runner-issue-template.md", linuxRunnerEvidenceChecklist, StringComparison.Ordinal);
        Assert.Contains("template-only", linuxRunnerEvidenceChecklist, StringComparison.Ordinal);
        Assert.Contains("InvocationCount>0", linuxRunnerEvidenceChecklist, StringComparison.Ordinal);
        Assert.Contains("linux-runner-evidence-record-template.json", linuxRunnerEvidenceRecordSchema, StringComparison.Ordinal);
        Assert.Contains("templateOnly=true", linuxRunnerEvidenceRecordSchema, StringComparison.Ordinal);
        Assert.Contains("runnerOwner", linuxRunnerEvidenceRecordSchema, StringComparison.Ordinal);
        Assert.Contains("evidenceRationale", linuxRunnerEvidenceRecordSchema, StringComparison.Ordinal);
        Assert.Contains("isRealLinuxRunnerProof=false", linuxRunnerEvidenceRecordSchema, StringComparison.Ordinal);
        Assert.Contains("canPromoteLinuxPackage=false", linuxRunnerEvidenceRecordSchema, StringComparison.Ordinal);
        Assert.Contains("Test-LinuxRunnerEvidenceRecord.ps1", linuxRunnerEvidenceRecordSchema, StringComparison.Ordinal);
        Assert.Contains("validationState=template-only", linuxRunnerEvidenceChecklist, StringComparison.Ordinal);
        Assert.Contains("validationState=real-linux-runner-proof", linuxHandoff, StringComparison.Ordinal);
        Assert.Contains("TensorRtSharp4.0：把 TensorRT 和 CUDA 带进 .NET 工程化部署", blogProjectIntroduction, StringComparison.Ordinal);
        Assert.Contains("manifest/source 100% 匹配，不等于", blogProjectIntroduction, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", blogProjectIntroduction, StringComparison.Ordinal);
        Assert.Contains("发布摘要", blogProjectIntroduction, StringComparison.Ordinal);
        Assert.Contains("公众号封面建议", blogProjectIntroduction, StringComparison.Ordinal);
        Assert.Contains("从 NuGet 包到真实消费端", blogPackageConsumerEvidence, StringComparison.Ordinal);
        Assert.Contains("EvidenceKind=full-runtime-package-consumer-smoke-driver-blocked", blogPackageConsumerEvidence, StringComparison.Ordinal);
        Assert.Contains("IsDependencyProbeOnly=True", blogPackageConsumerEvidence, StringComparison.Ordinal);
        Assert.Contains("RuntimeSmokeClassification", blogPackageConsumerEvidence, StringComparison.Ordinal);
        Assert.Contains("Linux Runner Evidence 回填指南", blogLinuxRunnerEvidence, StringComparison.Ordinal);
        Assert.Contains("validationState=template-only", blogLinuxRunnerEvidence, StringComparison.Ordinal);
        Assert.Contains("validationState=real-linux-runner-proof", blogLinuxRunnerEvidence, StringComparison.Ordinal);
        Assert.Contains("Dynamic Shape 博客版", blogDynamicShape, StringComparison.Ordinal);
        Assert.Contains("samples/DynamicShape", blogDynamicShape, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", blogDynamicShape, StringComparison.Ordinal);
        Assert.Contains("InferenceBindings 博客版", blogInferenceBindings, StringComparison.Ordinal);
        Assert.Contains("samples/InferenceBindings", blogInferenceBindings, StringComparison.Ordinal);
        Assert.Contains("不是 allocator/debug listener callback runtime proof", blogInferenceBindings, StringComparison.Ordinal);
        Assert.Contains("ONNX Parser 博客版", blogOnnxParser, StringComparison.Ordinal);
        Assert.Contains("samples/OnnxToEngine", blogOnnxParser, StringComparison.Ordinal);
        Assert.Contains("不宣称所有真实模型都无需修改即可运行", blogOnnxParser, StringComparison.Ordinal);
        Assert.Contains("MultiStream 博客版", blogMultiStream, StringComparison.Ordinal);
        Assert.Contains("samples/MultiStream", blogMultiStream, StringComparison.Ordinal);
        Assert.Contains("IndependentStreams=True", blogMultiStream, StringComparison.Ordinal);
        Assert.Contains("CrossStreamWait=True", blogMultiStream, StringComparison.Ordinal);
        Assert.Contains("TensorRT callback runtime proof 已完成", blogMultiStream, StringComparison.Ordinal);
        Assert.Contains("Plugin Inventory 博客版", blogPluginInventory, StringComparison.Ordinal);
        Assert.Contains("PluginRegistryInventorySmokeRunner", blogPluginInventory, StringComparison.Ordinal);
        Assert.Contains("public API 不返回裸 `IntPtr` plugin creator", blogPluginInventory, StringComparison.Ordinal);
        Assert.Contains("ownership 不清楚的 borrowed pointer", blogPluginInventory, StringComparison.Ordinal);
        Assert.Contains("CUDA Memory Wrapper 博客版", blogCudaMemory, StringComparison.Ordinal);
        Assert.Contains("smoke/CudaSmokeRunner", blogCudaMemory, StringComparison.Ordinal);
        Assert.Contains("PinnedAsyncRoundTrip=True", blogCudaMemory, StringComparison.Ordinal);
        Assert.Contains("Memory wrapper ready 不等于 allocator callback proof ready", blogCudaMemory, StringComparison.Ordinal);
        Assert.Contains("Refit Weights 博客版", blogRefitWeights, StringComparison.Ordinal);
        Assert.Contains("smoke/RefitWeightsSmokeRunner", blogRefitWeights, StringComparison.Ordinal);
        Assert.Contains("OutputChanged=True", blogRefitWeights, StringComparison.Ordinal);
        Assert.Contains("Linux runner proof", blogRefitWeights, StringComparison.Ordinal);
        Assert.Contains("Network Layer Coverage 博客版", blogNetworkLayerCoverage, StringComparison.Ordinal);
        Assert.Contains("Network*SmokeRunner", blogNetworkLayerCoverage, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", blogNetworkLayerCoverage, StringComparison.Ordinal);
        Assert.Contains("real callback runtime proof complete", blogNetworkLayerCoverage, StringComparison.Ordinal);
        Assert.Contains("blog-package-consumer-evidence-chain.md", technicalArticleRoadmap, StringComparison.Ordinal);
        Assert.Contains("blog-linux-runner-evidence-guide.md", technicalArticleRoadmap, StringComparison.Ordinal);
        Assert.Contains("blog-dynamic-shape-optimization-profile.md", technicalArticleRoadmap, StringComparison.Ordinal);
        Assert.Contains("blog-inference-bindings-identity-network.md", technicalArticleRoadmap, StringComparison.Ordinal);
        Assert.Contains("blog-onnx-parser-engine-roundtrip.md", technicalArticleRoadmap, StringComparison.Ordinal);
        Assert.Contains("blog-multistream-cuda-stream-event.md", technicalArticleRoadmap, StringComparison.Ordinal);
        Assert.Contains("blog-plugin-inventory-readonly-api.md", technicalArticleRoadmap, StringComparison.Ordinal);
        Assert.Contains("blog-cuda-memory-wrapper.md", technicalArticleRoadmap, StringComparison.Ordinal);
        Assert.Contains("blog-refit-weights-guide.md", technicalArticleRoadmap, StringComparison.Ordinal);
        Assert.Contains("blog-network-layer-coverage-guide.md", technicalArticleRoadmap, StringComparison.Ordinal);
        Assert.Contains("ProofClassification", onnxToEngineGuide, StringComparison.Ordinal);
        Assert.Contains("EvidenceClassifications", onnxToEngineGuide, StringComparison.Ordinal);
        Assert.Contains("BuildEvidenceOnly=true", onnxToEngineGuide, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime", onnxToEngineGuide, StringComparison.Ordinal);
        Assert.Contains("ProofClassification", tensorRtExecToolGuide, StringComparison.Ordinal);
        Assert.Contains("EvidenceClassifications", tensorRtExecToolGuide, StringComparison.Ordinal);
        Assert.Contains("--evidenceSidecar", tensorRtExecToolGuide, StringComparison.Ordinal);
        Assert.Contains("IsRealModelRuntimeProof", tensorRtExecToolGuide, StringComparison.Ordinal);
        Assert.Contains("IsPackageConsumerRuntimeProof", tensorRtExecToolGuide, StringComparison.Ordinal);
        Assert.Contains("ModelEvidence", tensorRtExecToolGuide, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime is tracked by release proof records", tensorRtExecToolGuide, StringComparison.Ordinal);
        Assert.Contains("ProofClassification", tensorRtExecGui, StringComparison.Ordinal);
        Assert.Contains("IsRealModelRuntimeProof", tensorRtExecGui, StringComparison.Ordinal);
        Assert.Contains("IsPackageConsumerRuntimeProof", tensorRtExecGui, StringComparison.Ordinal);
        Assert.Contains("不会声明 `package-consumer-runtime`", tensorRtExecGui, StringComparison.Ordinal);
        Assert.Contains("ProofClassification", tensorRtExecExternalReport, StringComparison.Ordinal);
        Assert.Contains("EvidenceClassifications", tensorRtExecExternalReport, StringComparison.Ordinal);
        Assert.Contains("Evidence sidecar", tensorRtExecExternalReport, StringComparison.Ordinal);
        Assert.Contains("Test-OnnxEngineBuildEvidenceSidecar.ps1", tensorRtExecExternalReport, StringComparison.Ordinal);
        Assert.Contains("Export-OnnxEngineBuildEvidenceSidecarTemplate.ps1", tensorRtExecExternalReport, StringComparison.Ordinal);
        Assert.Contains("EvidenceSidecarDiagnostics", tensorRtExecExternalReport, StringComparison.Ordinal);
        Assert.Contains("StdoutSummary", tensorRtExecExternalReport, StringComparison.Ordinal);
        Assert.Contains("StderrSummary", tensorRtExecExternalReport, StringComparison.Ordinal);
        Assert.Contains("ModelEvidence", tensorRtExecExternalReport, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec 报告不会声明该级别", tensorRtExecExternalReport, StringComparison.Ordinal);
        Assert.Contains("sample run evidence record", tensorRtExecExternalReport, StringComparison.Ordinal);
        Assert.Contains("Export-ReleaseEvidenceBundle.ps1", tensorRtExecExternalReport, StringComparison.Ordinal);
        Assert.Contains("owner-action-required", tensorRtExecExternalReport, StringComparison.Ordinal);
        Assert.Contains("Classification 真实资产接入教程", classificationRealAssetWalkthrough, StringComparison.Ordinal);
        Assert.Contains("TorchVision ResNet18", classificationRealAssetWalkthrough, StringComparison.Ordinal);
        Assert.Contains("classifier-sample-run-evidence.json", classificationRealAssetWalkthrough, StringComparison.Ordinal);
        Assert.Contains("Test-SampleRunEvidenceRecord.ps1", classificationRealAssetWalkthrough, StringComparison.Ordinal);
        Assert.Contains("Export-ReleaseEvidenceBundle.ps1", classificationRealAssetWalkthrough, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime", classificationRealAssetWalkthrough, StringComparison.Ordinal);
        Assert.Contains("Deferred 人工设计分组指南", deferredManualDesignGroupsDoc, StringComparison.Ordinal);
        Assert.Contains("manual review design groups", deferredManualDesignGroupsDoc, StringComparison.Ordinal);
        Assert.Contains("runtime-execution-boundary", deferredManualDesignGroupsDoc, StringComparison.Ordinal);
        Assert.Contains("runtime-deserialization-boundary", deferredManualDesignGroupsDoc, StringComparison.Ordinal);
        Assert.Contains("dimension-expression-snapshot-design", deferredManualDesignGroupsDoc, StringComparison.Ordinal);
        Assert.Contains("calibrator-callback-metadata-design", deferredManualDesignGroupsDoc, StringComparison.Ordinal);
        Assert.Contains("不允许把 `-IncludeMediumRisk` 的输出当成低风险提升清单", deferredManualDesignGroupsDoc, StringComparison.Ordinal);
        Assert.Contains("真实模型 Owner 回填 Checklist", realModelOwnerBackfillChecklist, StringComparison.Ordinal);
        Assert.Contains("Classification 回填", realModelOwnerBackfillChecklist, StringComparison.Ordinal);
        Assert.Contains("YoloVision 回填", realModelOwnerBackfillChecklist, StringComparison.Ordinal);
        Assert.Contains("YOLOX-S 回填", realModelOwnerBackfillChecklist, StringComparison.Ordinal);
        Assert.Contains("classifier-sample-run-evidence.json", realModelOwnerBackfillChecklist, StringComparison.Ordinal);
        Assert.Contains("yolovision-sample-run-evidence.json", realModelOwnerBackfillChecklist, StringComparison.Ordinal);
        Assert.Contains("yolox_s-sample-run-evidence.json", realModelOwnerBackfillChecklist, StringComparison.Ordinal);
        Assert.Contains("sampleRunEvidenceCanPromoteRealModelRuntime", realModelOwnerBackfillChecklist, StringComparison.Ordinal);
        Assert.Contains("Export-RealModelOwnerHandoff.ps1", realModelOwnerBackfillChecklist, StringComparison.Ordinal);
        Assert.Contains("real-model-owner-handoff.json", realModelOwnerBackfillChecklist, StringComparison.Ordinal);
        Assert.Contains("handoffState=owner-action-required", realModelOwnerBackfillChecklist, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime", realModelOwnerBackfillChecklist, StringComparison.Ordinal);
        Assert.Contains("不要伪造 SHA256", realModelOwnerBackfillChecklist, StringComparison.Ordinal);
        Assert.Contains("Sample Assets", sampleAssetsReadme, StringComparison.Ordinal);
        Assert.Contains("classifier-sample-run-evidence.json", sampleAssetsReadme, StringComparison.Ordinal);
        Assert.Contains("yolovision-sample-run-evidence.json", sampleAssetsReadme, StringComparison.Ordinal);
        Assert.Contains("Get-FileHash", sampleAssetsReadme, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime", sampleAssetsReadme, StringComparison.Ordinal);
        Assert.Contains("findingCount=0", bilingualProgress, StringComparison.Ordinal);
        Assert.Contains("backlogFindingCount=0", bilingualProgress, StringComparison.Ordinal);
        Assert.Contains("模型来源 URL 和许可证", classificationAssets, StringComparison.Ordinal);
        Assert.Contains("synthetic input", classificationAssets, StringComparison.Ordinal);
        Assert.Contains("TorchVision ResNet18", classificationAssetCandidates, StringComparison.Ordinal);
        Assert.Contains("BSD-3-Clause", classificationAssetCandidates, StringComparison.Ordinal);
        Assert.Contains("未实跑 sample smoke", classificationAssetCandidates, StringComparison.Ordinal);
        Assert.Contains("sample-asset-acquisition-plan.md", classificationAssetCandidates, StringComparison.Ordinal);
        Assert.Contains("输出 layout", yoloAssets, StringComparison.Ordinal);
        Assert.Contains("pipeline evidence", yoloAssets, StringComparison.Ordinal);
        Assert.Contains("YOLOX", yoloAssetCandidates, StringComparison.Ordinal);
        Assert.Contains("Apache-2.0", yoloAssetCandidates, StringComparison.Ordinal);
        Assert.Contains("AGPL-3.0/Enterprise", yoloAssetCandidates, StringComparison.Ordinal);
        Assert.Contains("sample-asset-acquisition-plan.md", yoloAssetCandidates, StringComparison.Ordinal);
        Assert.Contains("candidate-not-downloaded", sampleAssetManifestGuide, StringComparison.Ordinal);
        Assert.Contains("sample-asset-manifest-audit.json", sampleAssetManifestGuide, StringComparison.Ordinal);
        Assert.Contains("isSmokePassed=false", sampleAssetManifestGuide, StringComparison.Ordinal);
        Assert.Contains("proofClassification=template-only", sampleAssetManifestGuide, StringComparison.Ordinal);
        Assert.Contains("evidence sidecar", sampleAssetManifestGuide, StringComparison.Ordinal);
        Assert.Contains("Test-OnnxEngineBuildEvidenceSidecar.ps1", sampleAssetManifestGuide, StringComparison.Ordinal);
        Assert.Contains("sample run evidence record", sampleAssetManifestGuide, StringComparison.Ordinal);
        Assert.Contains("sampleRunEvidenceRecord", sampleAssetManifestGuide, StringComparison.Ordinal);
        Assert.Contains("labelsSha256", sampleAssetManifestGuide, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime", sampleAssetManifestGuide, StringComparison.Ordinal);
        Assert.Contains("Export-SampleAssetAcquisitionPlan.ps1", sampleAssetManifestGuide, StringComparison.Ordinal);
        Assert.Contains("owner-action-required", sampleAssetAcquisitionPlanDoc, StringComparison.Ordinal);
        Assert.Contains("performsDownload=false", sampleAssetAcquisitionPlanDoc, StringComparison.Ordinal);
        Assert.Contains("canPromoteSamples=false", sampleAssetAcquisitionPlanDoc, StringComparison.Ordinal);
        Assert.Contains("proofClassification=template-only", sampleAssetAcquisitionPlanDoc, StringComparison.Ordinal);
        Assert.Contains("evidence sidecar backfill", sampleAssetAcquisitionPlanDoc, StringComparison.Ordinal);
        Assert.Contains("sample run evidence record", sampleAssetAcquisitionPlanDoc, StringComparison.Ordinal);
        Assert.Contains("sampleRunEvidenceValidationState", sampleAssetAcquisitionPlanDoc, StringComparison.Ordinal);
        Assert.Contains("canPromoteRealModelRuntime=false", sampleAssetAcquisitionPlanDoc, StringComparison.Ordinal);
        Assert.Contains("Export-RealModelOwnerHandoff.ps1", sampleAssetAcquisitionPlanDoc, StringComparison.Ordinal);
        Assert.Contains("real-model-owner-handoff.json", sampleAssetAcquisitionPlanDoc, StringComparison.Ordinal);
        Assert.Contains("BuilderConfig", tensorRtObjectModel, StringComparison.Ordinal);
        Assert.Contains("不把 `IntPtr` creator 暴露给 public API", pluginInventory, StringComparison.Ordinal);
        Assert.Contains("PluginRegistryInventorySmokeRunner", pluginInventory, StringComparison.Ordinal);
        Assert.Contains("Plugin Registry copied metadata inventory", pluginOwnershipBoundary, StringComparison.Ordinal);
        Assert.Contains("不能从 candidate plan 中机械提升", pluginOwnershipBoundary, StringComparison.Ordinal);
        Assert.Contains("不暴露 `public IntPtr` / `public nint` plugin creator", pluginOwnershipBoundary, StringComparison.Ordinal);
        Assert.Contains("real runtime smoke", pluginOwnershipBoundary, StringComparison.Ordinal);
        Assert.Contains("PluginSerializationPathsSmokeRunner", pluginSerialization, StringComparison.Ordinal);
        Assert.Contains("TensorRT 8/10/11", pluginSerialization, StringComparison.Ordinal);
        Assert.DoesNotContain("PluginSerializationPathsRequireTensorRt10Or11", pluginSerialization, StringComparison.Ordinal);
        Assert.Contains("CudaGraphSmokeRunner", cudaGraphArticle, StringComparison.Ordinal);
        Assert.Contains("CudaGraphCaptureRoundTrip=True", cudaGraphArticle, StringComparison.Ordinal);
        Assert.Contains("IGpuAllocator", cudaMemoryWrapper, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", cudaMemoryRange, StringComparison.Ordinal);
        Assert.Contains("RefitWeightsSmokeRunner", refitWeights, StringComparison.Ordinal);
        Assert.Contains("TRT11", trt11ModernLayers, StringComparison.Ordinal);
        Assert.Contains("version guard", trt11ModernLayers, StringComparison.Ordinal);
        Assert.Contains("network-layer-coverage.json", networkLayerCoverage, StringComparison.Ordinal);
        Assert.Contains("caller buffer", errorRecorderSnapshot, StringComparison.Ordinal);
        Assert.Contains("InvocationCount>0", managedCallbackGuide, StringComparison.Ordinal);
        Assert.Contains("ProjectReference", localNuGetDeepDive, StringComparison.Ordinal);
        Assert.Contains("planned-dry-run", runtimeMatrixGuide, StringComparison.Ordinal);
        Assert.Contains("Linux runner proof", runtimeMatrixGuide, StringComparison.Ordinal);
        Assert.Contains("Runtime proof", runtimeMatrixGuide, StringComparison.Ordinal);
        Assert.Contains("packageConsumerEvidenceKind", readinessSummaryGuide, StringComparison.Ordinal);
        Assert.Contains("runtimeSmokeClassification", readinessSummaryGuide, StringComparison.Ordinal);
        Assert.Contains("isRuntimeExecutionEvidence", readinessSummaryGuide, StringComparison.Ordinal);
        Assert.Contains("isDependencyProbeOnly", readinessSummaryGuide, StringComparison.Ordinal);
        Assert.Contains("isRealCallbackRuntimeProof", readinessSummaryGuide, StringComparison.Ordinal);
        Assert.Contains("packageConsumerEvidenceKind", packageReadinessCurrentState, StringComparison.Ordinal);
        Assert.Contains("runtimeSmokeClassification", packageReadinessCurrentState, StringComparison.Ordinal);
        Assert.Contains("isRuntimeExecutionEvidence", packageReadinessCurrentState, StringComparison.Ordinal);
        Assert.Contains("isDependencyProbeOnly", packageReadinessCurrentState, StringComparison.Ordinal);
        Assert.Contains("isRealCallbackRuntimeProof", packageReadinessCurrentState, StringComparison.Ordinal);
        Assert.Contains("InvocationCount>0", callbackAllocatorGuide, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", callbackAllocatorGuide, StringComparison.Ordinal);
        Assert.Contains("Asset Metadata Checklist", classificationReadme, StringComparison.Ordinal);
        Assert.Contains("Real Model Evidence Backfill", classificationReadme, StringComparison.Ordinal);
        Assert.Contains("Export-OnnxEngineBuildEvidenceSidecarTemplate.ps1", classificationReadme, StringComparison.Ordinal);
        Assert.Contains("classification-model-assets.md", classificationReadme, StringComparison.Ordinal);
        Assert.Contains("classification-assets.template.json", classificationReadme, StringComparison.Ordinal);
        Assert.Contains("Asset Metadata Checklist", yoloReadme, StringComparison.Ordinal);
        Assert.Contains("Real Model Evidence Backfill", yoloReadme, StringComparison.Ordinal);
        Assert.Contains("Test-OnnxEngineBuildEvidenceSidecar.ps1", yoloReadme, StringComparison.Ordinal);
        Assert.Contains("yolovision-model-assets.md", yoloReadme, StringComparison.Ordinal);
        Assert.Contains("yolovision-assets.template.json", yoloReadme, StringComparison.Ordinal);
        Assert.Contains("\"sampleName\": \"Classification\"", classificationManifest, StringComparison.Ordinal);
        Assert.Contains("\"status\": \"candidate-not-downloaded\"", classificationManifest, StringComparison.Ordinal);
        Assert.Contains("\"proofClassification\": \"template-only\"", classificationManifest, StringComparison.Ordinal);
        Assert.Contains("\"evidenceSidecar\"", classificationManifest, StringComparison.Ordinal);
        Assert.Contains("\"sampleRunEvidenceRecord\"", classificationManifest, StringComparison.Ordinal);
        Assert.Contains("\"sampleRunEvidenceValidation\"", classificationManifest, StringComparison.Ordinal);
        Assert.Contains("\"buildOnlyCommand\"", classificationManifest, StringComparison.Ordinal);
        Assert.Contains("--evidenceSidecar", classificationManifest, StringComparison.Ordinal);
        Assert.Contains("\"stdoutSummary\"", classificationManifest, StringComparison.Ordinal);
        Assert.Contains("\"isSmokePassed\": false", classificationManifest, StringComparison.Ordinal);
        Assert.Contains("\"sampleName\": \"YoloVision\"", yoloManifest, StringComparison.Ordinal);
        Assert.Contains("\"status\": \"candidate-not-downloaded\"", yoloManifest, StringComparison.Ordinal);
        Assert.Contains("\"proofClassification\": \"template-only\"", yoloManifest, StringComparison.Ordinal);
        Assert.Contains("\"evidenceSidecar\"", yoloManifest, StringComparison.Ordinal);
        Assert.Contains("\"sampleRunEvidenceRecord\"", yoloManifest, StringComparison.Ordinal);
        Assert.Contains("\"sampleRunEvidenceValidation\"", yoloManifest, StringComparison.Ordinal);
        Assert.Contains("\"buildOnlyCommand\"", yoloManifest, StringComparison.Ordinal);
        Assert.Contains("--evidenceSidecar", yoloManifest, StringComparison.Ordinal);
        Assert.Contains("\"stdoutSummary\"", yoloManifest, StringComparison.Ordinal);
        Assert.Contains("\"isSmokePassed\": false", yoloManifest, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseCandidateReadinessScriptGeneratesMatrixWithoutLosingCallbackProofBoundary()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseCandidateReadiness.ps1");
        string output = RunPowerShell(
            script,
            "-RuntimePackageKey", "win-x64-trt11.0-cuda13.2-cudnn9.22",
            "-AllowRuntimeSmokeBlocked",
            "-WarnOnly");

        Assert.Contains("Release candidate readiness summary written", output, StringComparison.Ordinal);

        string summaryPath = Path.Combine(RepositoryPaths.Root, "artifacts", "release-candidate", "release-candidate-readiness-summary.json");
        string matrixPath = Path.Combine(RepositoryPaths.Root, "artifacts", "release-candidate", "runtime-package-matrix.json");
        using JsonDocument summary = JsonDocument.Parse(File.ReadAllText(summaryPath));
        using JsonDocument matrix = JsonDocument.Parse(File.ReadAllText(matrixPath));

        JsonElement root = summary.RootElement;
        Assert.Equal("win-x64-trt11.0-cuda13.2-cudnn9.22", root.GetProperty("runtimePackageKey").GetString());
        Assert.False(root.GetProperty("realCallbackRuntimeProof").GetBoolean());
        Assert.Contains(root.GetProperty("runtimeProofStatus").GetString(), new[] { "not-requested", "blocked-by-cuda-driver" });
        Assert.True(root.GetProperty("runtimeProofRequiredForRelease").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.Contains(root.GetProperty("overallStatus").GetString(), new[] { "blocked", "ready-with-warnings", "ready" });

        JsonElement[] entries = matrix.RootElement.EnumerateArray().ToArray();
        Assert.Contains(entries, static entry => entry.GetProperty("key").GetString() == "win-x64-trt11.0-cuda13.2-cudnn9.22");
        Assert.Contains(entries, static entry =>
            entry.GetProperty("key").GetString() == "win-x64-trt11.0-cuda13.2-cudnn9.22" &&
            entry.GetProperty("runtimeProofStatus").GetString() == "blocked-by-cuda-driver");
        Assert.Contains(entries, static entry => entry.GetProperty("key").GetString() == "win-x64-trt8.6-cuda11.8-cudnn8.9");
        Assert.Contains(entries, static entry => entry.GetProperty("key").GetString() == "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22");
    }

    [Fact]
    public void ReleaseCandidateChecklistExporterKeepsPendingItemsVisible()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCandidateChecklist.ps1");
        string output = RunPowerShell(script);

        Assert.Contains("Release candidate checklist written", output, StringComparison.Ordinal);

        string checklistPath = Path.Combine(RepositoryPaths.Root, "artifacts", "release-candidate", "release-candidate-checklist.json");
        using JsonDocument checklist = JsonDocument.Parse(File.ReadAllText(checklistPath));
        JsonElement[] items = checklist.RootElement.EnumerateArray().ToArray();

        Assert.Contains(items, static item => item.GetProperty("name").GetString() == "Local NuGet feed consumer validation");
        Assert.Contains(items, static item => item.GetProperty("name").GetString() == "Aggregated release candidate readiness");
        Assert.Contains(items, static item => item.GetProperty("name").GetString() == "Runtime package matrix");
        Assert.Contains(items, static item => item.GetProperty("notes").GetString()!.Contains("Warnings such as CUDA error 35", StringComparison.Ordinal));
        Assert.Contains(items, static item => item.GetProperty("notes").GetString()!.Contains("Deferred rows are intentional boundary records", StringComparison.Ordinal));
        Assert.Contains(items, static item =>
            item.GetProperty("name").GetString() == "Public API bilingual documentation audit" &&
            item.GetProperty("notes").GetString()!.Contains("findingCount=", StringComparison.Ordinal));
        Assert.Contains(items, static item => item.GetProperty("name").GetString() == "Public API bilingual documentation backlog");
        Assert.Contains(items, static item =>
            item.GetProperty("name").GetString() == "Compatible host runtime proof collection bundle" &&
            item.GetProperty("ready").GetBoolean() &&
            item.GetProperty("notes").GetString()!.Contains("quickStart=", StringComparison.Ordinal) &&
            item.GetProperty("notes").GetString()!.Contains("preflight=", StringComparison.Ordinal) &&
            item.GetProperty("notes").GetString()!.Contains("copyableExecutionOrder=", StringComparison.Ordinal) &&
            item.GetProperty("notes").GetString()!.Contains("Test-PackageConsumer.ps1", StringComparison.Ordinal) &&
            item.GetProperty("notes").GetString()!.Contains("-FailOnNotProof", StringComparison.Ordinal));
    }

    [Fact]
    public void FinalReleaseDryRunAggregatesManualApprovalsWithoutClaimingCallbackProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCandidateChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalRuntimeProofRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishVerificationRecord.ps1"));

        string script = Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalReleaseDryRun.ps1");
        string output = RunPowerShell(
            script,
            "-RuntimePackageKey", "win-x64-trt11.0-cuda13.2-cudnn9.22",
            "-AllowRuntimeSmokeBlocked",
            "-WarnOnly");

        Assert.Contains("Final release dry run summary written", output, StringComparison.Ordinal);

        string summaryPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "final-release-dry-run-summary.json");
        using JsonDocument summary = JsonDocument.Parse(File.ReadAllText(summaryPath));
        JsonElement root = summary.RootElement;

        Assert.Equal("win-x64-trt11.0-cuda13.2-cudnn9.22", root.GetProperty("runtimePackageKey").GetString());
        Assert.False(root.GetProperty("realCallbackRuntimeProof").GetBoolean());
        Assert.Contains(root.GetProperty("packageConsumerSmokeStatus").GetString(), new[] { "not-requested", "blocked-by-cuda-driver" });
        Assert.Contains(root.GetProperty("runtimeProofStatus").GetString(), new[] { "not-requested", "blocked-by-cuda-driver" });
        Assert.True(root.GetProperty("runtimeProofRequiredForRelease").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("runtimeProofBlockerOwnerActionStatus").GetString());
        Assert.Contains(root.GetProperty("runtimeProofBlockerCategory").GetString(), new[] { "runtime-smoke-not-requested", "cuda-driver-runtime-compatibility" });
        Assert.False(root.GetProperty("externalRuntimeExecutionEvidence").GetBoolean());
        Assert.Contains(root.GetProperty("externalRuntimeProofState").GetString(), new[] { "template-only", "draft-blocked-by-cuda-driver" });
        Assert.Contains(root.GetProperty("externalRuntimeProofClassification").GetString(), new[] { "template-only", "dependency-probe-only" });
        Assert.True(root.GetProperty("externalRuntimeProofRuntimePackageKeyMatches").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofLogSha256FormatReady").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofLogSha256Matches").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("externalRuntimeProofOwnerActionStatus").GetString());
        Assert.False(root.GetProperty("externalRuntimeProofCanPromoteRuntimeProof").GetBoolean());
        Assert.Equal("template-only", root.GetProperty("postPublishVerificationState").GetString());
        Assert.Equal("template-only", root.GetProperty("postPublishProofClassification").GetString());
        Assert.False(root.GetProperty("postPublishProofClassificationPromotable").GetBoolean());
        Assert.False(root.GetProperty("postPublishManagedNupkgSha256Ready").GetBoolean());
        Assert.False(root.GetProperty("postPublishRuntimeNupkgSha256Ready").GetBoolean());
        Assert.False(root.GetProperty("postPublishConsumerProjectIdentityReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishSmokeCommandRuntimeKeyReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishHostReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishCommandsReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishStdoutStderrSummaryReady").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.True(root.GetProperty("allowRuntimeSmokeBlocked").GetBoolean());
        Assert.Equal("unsigned-or-not-requested", root.GetProperty("signingStatus").GetString());
        Assert.Equal(0, root.GetProperty("bilingualDocumentationFindingCount").GetInt32());
        Assert.Equal("ready", root.GetProperty("bilingualDocumentationBacklogStatus").GetString());
        Assert.True(root.GetProperty("manualApprovalCount").GetInt32() >= 1);
        Assert.Contains(root.GetProperty("overallStatus").GetString(), new[] { "ready-needs-manual-approval", "blocked", "ready-with-warnings", "ready" });
        Assert.True(root.GetProperty("userAcceptanceCatalogItemCount").GetInt32() >= 1);

        JsonElement[] gates = root.GetProperty("gates").EnumerateArray().ToArray();
        Assert.Contains(gates, static gate => gate.GetProperty("name").GetString() == "Public API bilingual documentation");
        Assert.Contains(gates, static gate => gate.GetProperty("name").GetString() == "Public API bilingual documentation backlog");
        Assert.Contains(gates, static gate => gate.GetProperty("name").GetString() == "Full package runtime smoke");
        Assert.Contains(gates, static gate =>
            gate.GetProperty("name").GetString() == "Full package runtime proof" &&
            (gate.GetProperty("status").GetString() == "not-requested" ||
             gate.GetProperty("status").GetString() == "blocked-by-cuda-driver") &&
            gate.GetProperty("severity").GetString() == "manual-approval");
        Assert.Contains(gates, static gate =>
            gate.GetProperty("name").GetString() == "Release candidate readiness" &&
            (gate.GetProperty("status").GetString() == "ready" ||
             (gate.GetProperty("status").GetString() == "blocked-by-cuda-driver-owner-action" &&
              gate.GetProperty("severity").GetString() == "manual-approval") ||
             (gate.GetProperty("status").GetString() == "runtime-smoke-not-requested-owner-action" &&
              gate.GetProperty("severity").GetString() == "manual-approval")));
        Assert.Contains(gates, static gate =>
            gate.GetProperty("name").GetString() == "Full package runtime smoke" &&
            gate.GetProperty("detail").GetString()!.Contains("does not promote blocked smoke to ready", StringComparison.Ordinal));
        Assert.Contains(gates, static gate =>
            gate.GetProperty("name").GetString() == "External runtime proof record" &&
            gate.GetProperty("status").GetString() == "template-only" &&
            gate.GetProperty("detail").GetString()!.Contains("LogSha256FormatReady=False", StringComparison.Ordinal) &&
            gate.GetProperty("detail").GetString()!.Contains("ManagedNupkgSha256Ready=False", StringComparison.Ordinal) &&
            gate.GetProperty("detail").GetString()!.Contains("RuntimeNupkgSha256Ready=False", StringComparison.Ordinal));
        Assert.Contains(gates, static gate =>
            gate.GetProperty("name").GetString() == "Post-publish verification record" &&
            gate.GetProperty("status").GetString() == "template-only" &&
            gate.GetProperty("detail").GetString()!.Contains("CommandsReady=False", StringComparison.Ordinal) &&
            gate.GetProperty("detail").GetString()!.Contains("StdoutStderrSummaryReady=False", StringComparison.Ordinal));
        Assert.Contains(gates, static gate => gate.GetProperty("detail").GetString()!.Contains("RuntimeProofRequiredForRelease=True", StringComparison.Ordinal));
        Assert.Contains(gates, static gate => gate.GetProperty("detail").GetString()!.Contains("InvocationCount>0 callback proof", StringComparison.Ordinal));
        Assert.Contains(gates, static gate => gate.GetProperty("name").GetString() == "User acceptance sample catalog");
        Assert.Contains(gates, static gate => gate.GetProperty("name").GetString() == "Signing and release channel docs");
        Assert.Contains(gates, static gate =>
            gate.GetProperty("name").GetString() == "Linux dry-run handoff" &&
            gate.GetProperty("detail").GetString()!.Contains("handoff/dry-run", StringComparison.Ordinal));
    }

    [Fact]
    public void ReleaseOwnerDecisionTemplateKeepsManualApprovalBoundariesExplicit()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalRuntimeProofRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishVerificationRecord.ps1"));

        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseOwnerDecisionTemplate.ps1");
        string output = RunPowerShell(script);

        Assert.Contains("Release owner decision template written", output, StringComparison.Ordinal);

        string templatePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-owner-decision-template.json");
        using JsonDocument template = JsonDocument.Parse(File.ReadAllText(templatePath));
        JsonElement root = template.RootElement;

        Assert.Equal("pending-release-owner-approval", root.GetProperty("approvalState").GetString());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.Equal("blocked-by-cuda-driver", root.GetProperty("packageConsumerSmokeStatus").GetString());
        Assert.Equal("blocked-by-cuda-driver", root.GetProperty("runtimeProofStatus").GetString());
        Assert.True(root.GetProperty("runtimeProofRequiredForRelease").GetBoolean());
        Assert.True(root.GetProperty("allowRuntimeSmokeBlocked").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeExecutionEvidence").GetBoolean());
        Assert.Contains(root.GetProperty("externalRuntimeProofState").GetString(), new[] { "template-only", "draft-blocked-by-cuda-driver" });
        Assert.Contains(root.GetProperty("externalRuntimeProofClassification").GetString(), new[] { "template-only", "dependency-probe-only" });
        Assert.True(root.GetProperty("externalRuntimeProofRuntimePackageKeyMatches").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofLogSha256FormatReady").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofLogSha256Matches").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("externalRuntimeProofOwnerActionStatus").GetString());
        Assert.False(root.GetProperty("externalRuntimeProofCanPromoteRuntimeProof").GetBoolean());
        Assert.Equal("template-only", root.GetProperty("postPublishVerificationState").GetString());
        Assert.False(root.GetProperty("postPublishConsumerProjectIdentityReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishSmokeCommandRuntimeKeyReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishHostReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishCommandsReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishStdoutStderrSummaryReady").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("realCallbackRuntimeProof").GetBoolean());
        Assert.Equal("unsigned-or-not-requested", root.GetProperty("signingStatus").GetString());
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note => note.GetString() == "Linux handoff is not Linux runner proof.");
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note => note.GetString()!.Contains("runtimeProofRequiredForRelease=true", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note => note.GetString()!.Contains("allowRuntimeSmokeBlocked=true records dry-run intent only", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("decisions").EnumerateArray(), static decision => decision.GetProperty("id").GetString() == "nvidia-redistribution");
        Assert.Contains(root.GetProperty("decisions").EnumerateArray(), static decision =>
            decision.GetProperty("id").GetString() == "runtime-smoke" &&
            decision.GetProperty("currentStatus").GetString()!.Contains("runtimeProofStatus=blocked-by-cuda-driver", StringComparison.Ordinal) &&
            decision.GetProperty("currentStatus").GetString()!.Contains("allowRuntimeSmokeBlocked=True", StringComparison.Ordinal) &&
            decision.GetProperty("currentStatus").GetString()!.Contains("externalRuntimeProofLogSha256FormatReady=False", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("decisions").EnumerateArray(), static decision =>
            decision.GetProperty("id").GetString() == "post-publish-verification" &&
            decision.GetProperty("currentStatus").GetString()!.Contains("commandsReady=False", StringComparison.Ordinal) &&
            decision.GetProperty("cannotClaim").GetString()!.Contains("stdout/stderr summaries", StringComparison.Ordinal));
    }

    [Fact]
    public void ReleaseOwnerApprovalInputTemplateRequiresExplicitOwnerRecord()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalRuntimeProofRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CompatibleHostRuntimeProofRunbook.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishVerificationRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPackageReviewBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofBackfillPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationBackfillPlan.ps1"));

        string exportScript = Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseOwnerApprovalInputTemplate.ps1");
        string validationScript = Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseOwnerApprovalInput.ps1");
        string exportOutput = RunPowerShell(exportScript);
        string validationOutput = RunPowerShell(validationScript);

        Assert.Contains("Release owner approval input template written", exportOutput, StringComparison.Ordinal);
        Assert.Contains("Release owner approval input validation written", validationOutput, StringComparison.Ordinal);

        string templatePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-owner-approval-input-template.json");
        string validationPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-owner-approval-input-validation.json");
        using JsonDocument template = JsonDocument.Parse(File.ReadAllText(templatePath));
        using JsonDocument validation = JsonDocument.Parse(File.ReadAllText(validationPath));

        JsonElement templateRoot = template.RootElement;
        Assert.Equal("release-owner-approval-input-template", templateRoot.GetProperty("recordKind").GetString());
        Assert.True(templateRoot.GetProperty("templateOnly").GetBoolean());
        Assert.False(templateRoot.GetProperty("canPublishPublicly").GetBoolean());
        Assert.Equal("blocked-by-cuda-driver", templateRoot.GetProperty("runtimeProofStatus").GetString());
        Assert.True(templateRoot.GetProperty("runtimeProofRequiredForRelease").GetBoolean());
        Assert.False(templateRoot.GetProperty("externalRuntimeExecutionEvidence").GetBoolean());
        Assert.Contains(templateRoot.GetProperty("externalRuntimeProofState").GetString(), new[] { "template-only", "draft-blocked-by-cuda-driver" });
        Assert.Contains(templateRoot.GetProperty("externalRuntimeProofClassification").GetString(), new[] { "template-only", "dependency-probe-only" });
        Assert.True(templateRoot.GetProperty("externalRuntimeProofRuntimePackageKeyMatches").GetBoolean());
        Assert.Equal("owner-action-required", templateRoot.GetProperty("externalRuntimeProofOwnerActionStatus").GetString());
        Assert.False(templateRoot.GetProperty("externalRuntimeProofCanPromoteRuntimeProof").GetBoolean());
        Assert.True(templateRoot.GetProperty("externalRuntimeProofFailedProofItemCount").GetInt32() > 0);
        Assert.Equal("owner-action-required", templateRoot.GetProperty("compatibleHostRuntimeProofRunbookState").GetString());
        Assert.True(templateRoot.GetProperty("compatibleHostRuntimeProofRunbookCompatibleHostRequired").GetBoolean());
        Assert.False(templateRoot.GetProperty("compatibleHostRuntimeProofRunbookPerformsPublish").GetBoolean());
        Assert.False(templateRoot.GetProperty("compatibleHostRuntimeProofRunbookApprovesPublicRelease").GetBoolean());
        Assert.False(templateRoot.GetProperty("compatibleHostRuntimeProofRunbookCanPromoteRuntimeProof").GetBoolean());
        Assert.False(templateRoot.GetProperty("compatibleHostRuntimeProofRunbookRuntimeExecutionEvidence").GetBoolean());
        Assert.Equal("blocked-by-cuda-driver is not smoke passed", templateRoot.GetProperty("compatibleHostRuntimeProofRunbookPromotionBlockedReason").GetString());
        Assert.Contains("Test-PackageConsumer.ps1", templateRoot.GetProperty("compatibleHostRuntimeProofRunbookRunPackageConsumerSmokeCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("-FailOnNotProof", templateRoot.GetProperty("compatibleHostRuntimeProofRunbookValidateFilledRecordCommand").GetString(), StringComparison.Ordinal);
        Assert.Equal("template-only", templateRoot.GetProperty("postPublishVerificationState").GetString());
        Assert.False(templateRoot.GetProperty("postPublishConsumerProjectIdentityReady").GetBoolean());
        Assert.False(templateRoot.GetProperty("postPublishSmokeCommandRuntimeKeyReady").GetBoolean());
        Assert.False(templateRoot.GetProperty("postPublishHostReady").GetBoolean());
        Assert.False(templateRoot.GetProperty("postPublishCommandsReady").GetBoolean());
        Assert.False(templateRoot.GetProperty("postPublishStdoutStderrSummaryReady").GetBoolean());
        Assert.False(templateRoot.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(templateRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal("owner-review-required", templateRoot.GetProperty("finalPackageReviewState").GetString());
        Assert.True(templateRoot.GetProperty("finalPackageReviewPackageCount").GetInt32() >= 1);
        Assert.True(templateRoot.GetProperty("finalPackageReviewNativeAssetCount").GetInt32() > 0);
        Assert.False(templateRoot.GetProperty("finalPackageReviewCanUseAsPublicPackageProof").GetBoolean());
        Assert.Equal("blocked-compatible-host-proof-required", templateRoot.GetProperty("externalRuntimeProofBackfillPlanState").GetString());
        Assert.True(templateRoot.GetProperty("externalRuntimeProofBackfillStepCount").GetInt32() >= 7);
        Assert.False(templateRoot.GetProperty("externalRuntimeProofBackfillCanPromoteRuntimeProof").GetBoolean());
        Assert.Equal("blocked-real-post-publish-proof-required", templateRoot.GetProperty("postPublishVerificationBackfillPlanState").GetString());
        Assert.True(templateRoot.GetProperty("postPublishVerificationBackfillStepCount").GetInt32() >= 9);
        Assert.False(templateRoot.GetProperty("postPublishVerificationBackfillCanCloseReleaseIssue").GetBoolean());
        Assert.Contains(templateRoot.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/compatible-host-runtime-proof-runbook.json");
        Assert.Contains(templateRoot.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/final-package-review-bundle.json");
        Assert.Contains(templateRoot.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/post-publish-verification-validation.json");
        Assert.Contains(templateRoot.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/external-runtime-proof-backfill-plan.json");
        Assert.Contains(templateRoot.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/post-publish-verification-backfill-plan.json");
        Assert.Contains(templateRoot.GetProperty("safetyNotes").EnumerateArray(), static note => note.GetString()!.Contains("compatible-host-runtime-proof-runbook is an owner-action runbook, not runtime proof", StringComparison.Ordinal));
        Assert.Contains(templateRoot.GetProperty("safetyNotes").EnumerateArray(), static note => note.GetString()!.Contains("final-package-review-bundle is local package inventory", StringComparison.Ordinal));
        Assert.Contains(templateRoot.GetProperty("safetyNotes").EnumerateArray(), static note => note.GetString()!.Contains("postPublishCommandsReady=false", StringComparison.Ordinal));
        Assert.Contains(templateRoot.GetProperty("safetyNotes").EnumerateArray(), static note => note.GetString()!.Contains("Backfill plans are guidance only", StringComparison.Ordinal));
        Assert.Contains(templateRoot.GetProperty("decisionInputs").EnumerateArray(), static decision =>
            decision.GetProperty("id").GetString() == "final-package-review-acknowledgement" &&
            decision.GetProperty("currentStatus").GetString()!.Contains("canUseAsPublicPackageProof=False", StringComparison.Ordinal) &&
            decision.GetProperty("requiredEvidence").GetString()!.Contains("final-package-review-bundle.json", StringComparison.Ordinal));
        Assert.Contains(templateRoot.GetProperty("decisionInputs").EnumerateArray(), static decision =>
            decision.GetProperty("id").GetString() == "post-publish-verification-disposition");
        Assert.Contains(templateRoot.GetProperty("decisionInputs").EnumerateArray(), static decision =>
            decision.GetProperty("id").GetString() == "backfill-plan-boundary-acknowledgement" &&
            decision.GetProperty("currentStatus").GetString()!.Contains("externalRuntimeProofBackfillPlanState=blocked-compatible-host-proof-required", StringComparison.Ordinal) &&
            decision.GetProperty("requiredEvidence").GetString()!.Contains("post-publish-verification-backfill-plan.json", StringComparison.Ordinal));

        JsonElement validationRoot = validation.RootElement;
        Assert.Equal("release-owner-approval-input-validation", validationRoot.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-input-required", validationRoot.GetProperty("overallStatus").GetString());
        Assert.False(validationRoot.GetProperty("canPublishPublicly").GetBoolean());
        Assert.True(validationRoot.GetProperty("blockingIssueCount").GetInt32() >= 1);
        Assert.Equal("owner-review-required", validationRoot.GetProperty("finalPackageReviewState").GetString());
        Assert.True(validationRoot.GetProperty("finalPackageReviewPackageCount").GetInt32() >= 1);
        Assert.True(validationRoot.GetProperty("finalPackageReviewNativeAssetCount").GetInt32() > 0);
        Assert.False(validationRoot.GetProperty("finalPackageReviewCanUseAsPublicPackageProof").GetBoolean());
        Assert.Equal("blocked-compatible-host-proof-required", validationRoot.GetProperty("externalRuntimeProofBackfillPlanState").GetString());
        Assert.True(validationRoot.GetProperty("externalRuntimeProofBackfillStepCount").GetInt32() >= 7);
        Assert.False(validationRoot.GetProperty("externalRuntimeProofBackfillCanPromoteRuntimeProof").GetBoolean());
        Assert.Equal("blocked-real-post-publish-proof-required", validationRoot.GetProperty("postPublishVerificationBackfillPlanState").GetString());
        Assert.True(validationRoot.GetProperty("postPublishVerificationBackfillStepCount").GetInt32() >= 9);
        Assert.False(validationRoot.GetProperty("postPublishVerificationBackfillCanCloseReleaseIssue").GetBoolean());
        Assert.Contains(validationRoot.GetProperty("issues").EnumerateArray(), static issue =>
            issue.GetProperty("id").GetString() == "final-package-review-bundle" &&
            issue.GetProperty("status").GetString() == "ready");
        Assert.Contains(validationRoot.GetProperty("issues").EnumerateArray(), static issue =>
            issue.GetProperty("id").GetString() == "external-runtime-proof-backfill-plan" &&
            issue.GetProperty("status").GetString() == "ready");
        Assert.Contains(validationRoot.GetProperty("issues").EnumerateArray(), static issue =>
            issue.GetProperty("id").GetString() == "post-publish-verification-backfill-plan" &&
            issue.GetProperty("status").GetString() == "ready");
        Assert.Contains(validationRoot.GetProperty("issues").EnumerateArray(), static issue =>
            issue.GetProperty("id").GetString() == "template-only" &&
            issue.GetProperty("status").GetString() == "blocked");
        Assert.Contains(validationRoot.GetProperty("decisionResults").EnumerateArray(), static decision =>
            decision.GetProperty("id").GetString() == "runtime-proof-disposition" &&
            decision.GetProperty("status").GetString() == "blocked");
        Assert.Contains(validationRoot.GetProperty("decisionResults").EnumerateArray(), static decision =>
            decision.GetProperty("id").GetString() == "final-package-review-acknowledgement" &&
            decision.GetProperty("status").GetString() == "blocked");
        Assert.Contains(validationRoot.GetProperty("decisionResults").EnumerateArray(), static decision =>
            decision.GetProperty("id").GetString() == "post-publish-verification-disposition" &&
            decision.GetProperty("status").GetString() == "blocked");
        Assert.Contains(validationRoot.GetProperty("decisionResults").EnumerateArray(), static decision =>
            decision.GetProperty("id").GetString() == "backfill-plan-boundary-acknowledgement" &&
            decision.GetProperty("status").GetString() == "blocked");
    }

    [Fact]
    public void ReleaseOwnerApprovalInputExampleCannotApprovePublication()
    {
        string exportScript = Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseOwnerApprovalInputExample.ps1");
        string validationScript = Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseOwnerApprovalInput.ps1");
        string exportOutput = RunPowerShell(exportScript);

        Assert.Contains("Release owner approval input example written", exportOutput, StringComparison.Ordinal);

        string examplePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-owner-approval-input-record.example.json");
        string validationOutput = RunPowerShell(validationScript, "-InputPath", "artifacts/final-release/release-owner-approval-input-record.example.json");

        Assert.Contains("Release owner approval input validation written", validationOutput, StringComparison.Ordinal);

        using JsonDocument example = JsonDocument.Parse(File.ReadAllText(examplePath));
        JsonElement exampleRoot = example.RootElement;
        Assert.Equal("release-owner-approval-input-example", exampleRoot.GetProperty("recordKind").GetString());
        Assert.Equal("example-not-for-publication", exampleRoot.GetProperty("approvalState").GetString());
        Assert.False(exampleRoot.GetProperty("canPublishPublicly").GetBoolean());

        string validationPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-owner-approval-input-validation.json");
        using JsonDocument validation = JsonDocument.Parse(File.ReadAllText(validationPath));
        JsonElement validationRoot = validation.RootElement;
        Assert.Equal("blocked-owner-input-required", validationRoot.GetProperty("overallStatus").GetString());
        Assert.False(validationRoot.GetProperty("canPublishPublicly").GetBoolean());
        Assert.Equal("release-owner-approval-input-example", validationRoot.GetProperty("inputRecordKind").GetString());
    }

    [Fact]
    public void ExternalRuntimeAndPostPublishTemplatesAreNotProof()
    {
        string externalOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordTemplate.ps1"));
        string externalValidationOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalRuntimeProofRecord.ps1"));
        string externalHandoffOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofOwnerHandoff.ps1"));
        string postPublishOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationRecordTemplate.ps1"));
        string postPublishValidationOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishVerificationRecord.ps1"));

        Assert.Contains("External runtime proof record template written", externalOutput, StringComparison.Ordinal);
        Assert.Contains("External runtime proof validation written", externalValidationOutput, StringComparison.Ordinal);
        Assert.Contains("ValidationState=template-only", externalValidationOutput, StringComparison.Ordinal);
        Assert.Contains("IsRuntimeExecutionEvidence=False", externalValidationOutput, StringComparison.Ordinal);
        Assert.Contains("CanPromoteRuntimeProof=False", externalValidationOutput, StringComparison.Ordinal);
        Assert.Contains("External runtime proof owner handoff written", externalHandoffOutput, StringComparison.Ordinal);
        Assert.Contains("OwnerActionStatus=owner-action-required", externalHandoffOutput, StringComparison.Ordinal);
        Assert.Contains("CanPromoteRuntimeProof=False", externalHandoffOutput, StringComparison.Ordinal);
        Assert.Contains("Post-publish verification record template written", postPublishOutput, StringComparison.Ordinal);
        Assert.Contains("Post-publish verification validation written", postPublishValidationOutput, StringComparison.Ordinal);
        Assert.Contains("ValidationState=template-only", postPublishValidationOutput, StringComparison.Ordinal);
        Assert.Contains("IsPostPublishVerificationProof=False", postPublishValidationOutput, StringComparison.Ordinal);
        Assert.Contains("CanCloseReleaseIssue=False", postPublishValidationOutput, StringComparison.Ordinal);

        string externalPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "external-runtime-proof-record-template.json");
        using JsonDocument external = JsonDocument.Parse(File.ReadAllText(externalPath));
        JsonElement externalRoot = external.RootElement;
        Assert.Equal("external-runtime-proof-record-template", externalRoot.GetProperty("recordKind").GetString());
        Assert.True(externalRoot.GetProperty("templateOnly").GetBoolean());
        Assert.Equal("template-only", externalRoot.GetProperty("proofState").GetString());
        Assert.Equal("template-only", externalRoot.GetProperty("proofClassification").GetString());
        Assert.Contains(externalRoot.GetProperty("evidenceClassifications").EnumerateArray(), static item =>
            item.GetString() == "package-consumer-runtime");
        Assert.Equal("", externalRoot.GetProperty("command").GetProperty("logSha256").GetString());
        Assert.False(externalRoot.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.True(externalRoot.GetProperty("isDependencyProbeOnly").GetBoolean());
        Assert.False(externalRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());

        string externalValidationPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "external-runtime-proof-validation.json");
        using JsonDocument externalValidation = JsonDocument.Parse(File.ReadAllText(externalValidationPath));
        JsonElement externalValidationRoot = externalValidation.RootElement;
        Assert.Equal("external-runtime-proof-validation", externalValidationRoot.GetProperty("validationKind").GetString());
        Assert.Equal("external-runtime-proof-record-template", externalValidationRoot.GetProperty("inputRecordKind").GetString());
        Assert.Equal("win-x64-trt11.0-cuda13.2-cudnn9.22", externalValidationRoot.GetProperty("runtimePackageKey").GetString());
        Assert.True(externalValidationRoot.GetProperty("runtimePackageKeyMatches").GetBoolean());
        Assert.False(externalValidationRoot.GetProperty("logSha256FormatReady").GetBoolean());
        Assert.False(externalValidationRoot.GetProperty("logSha256Matches").GetBoolean());
        Assert.Equal("template-only", externalValidationRoot.GetProperty("validationState").GetString());
        Assert.Equal("template-only", externalValidationRoot.GetProperty("proofClassification").GetString());
        Assert.False(externalValidationRoot.GetProperty("proofClassificationPromotable").GetBoolean());
        Assert.Contains(externalValidationRoot.GetProperty("classificationRules").EnumerateArray(), static rule =>
            rule.GetString()!.Contains("package-consumer-runtime is the required classification", StringComparison.Ordinal));
        Assert.True(externalValidationRoot.GetProperty("isTemplateOnly").GetBoolean());
        Assert.False(externalValidationRoot.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.True(externalValidationRoot.GetProperty("isDependencyProbeOnly").GetBoolean());
        Assert.False(externalValidationRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.Contains(externalValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "declared-runtime-proof" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(externalValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "proof-classification-promotable" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(externalValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "log-sha256" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(externalValidationRoot.GetProperty("requiredEvidenceSummary").EnumerateArray(), static item =>
            item.GetString()!.Contains("runtimePackageKey matches", StringComparison.Ordinal));
        Assert.Contains(externalValidationRoot.GetProperty("requiredEvidenceSummary").EnumerateArray(), static item =>
            item.GetString()!.Contains("managed/runtime package sources", StringComparison.Ordinal));
        Assert.Contains(externalValidationRoot.GetProperty("requiredEvidenceSummary").EnumerateArray(), static item =>
            item.GetString()!.Contains("clean consumer project identity", StringComparison.Ordinal));
        Assert.Contains(externalValidationRoot.GetProperty("requiredEvidenceSummary").EnumerateArray(), static item =>
            item.GetString()!.Contains("cuDNN", StringComparison.Ordinal));
        Assert.Contains(externalValidationRoot.GetProperty("requiredEvidenceSummary").EnumerateArray(), static item =>
            item.GetString()!.Contains("smokeCommand runtime package key", StringComparison.Ordinal));
        Assert.Contains(externalValidationRoot.GetProperty("requiredEvidenceSummary").EnumerateArray(), static item =>
            item.GetString()!.Contains("logSha256", StringComparison.Ordinal));
        Assert.Contains(externalValidationRoot.GetProperty("promotionRules").EnumerateArray(), static rule =>
            rule.GetString()!.Contains("Template-only records", StringComparison.Ordinal));

        string externalHandoffPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "external-runtime-proof-owner-handoff.json");
        using JsonDocument externalHandoff = JsonDocument.Parse(File.ReadAllText(externalHandoffPath));
        JsonElement externalHandoffRoot = externalHandoff.RootElement;
        Assert.Equal("external-runtime-proof-owner-handoff", externalHandoffRoot.GetProperty("recordKind").GetString());
        Assert.Equal("owner-action-required", externalHandoffRoot.GetProperty("handoffState").GetString());
        Assert.Equal("owner-action-required", externalHandoffRoot.GetProperty("ownerActionStatus").GetString());
        Assert.Equal("win-x64-trt11.0-cuda13.2-cudnn9.22", externalHandoffRoot.GetProperty("runtimePackageKey").GetString());
        Assert.True(externalHandoffRoot.GetProperty("runtimePackageKeyMatches").GetBoolean());
        Assert.True(externalHandoffRoot.GetProperty("packageSourceRuntimePackageKeyMatches").GetBoolean());
        Assert.False(externalHandoffRoot.GetProperty("managedNupkgSha256Ready").GetBoolean());
        Assert.False(externalHandoffRoot.GetProperty("runtimeNupkgSha256Ready").GetBoolean());
        Assert.True(externalHandoffRoot.GetProperty("failedProofItemCount").GetInt32() > 0);
        Assert.False(externalHandoffRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(externalHandoffRoot.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.Contains("Get-FileHash", externalHandoffRoot.GetProperty("commands").GetProperty("computeSmokeLogSha256").GetString()!, StringComparison.Ordinal);
        Assert.Contains("Get-FileHash", externalHandoffRoot.GetProperty("commands").GetProperty("computePackageNupkgSha256").GetString()!, StringComparison.Ordinal);
        Assert.Contains("RequireExistingLog", externalHandoffRoot.GetProperty("commands").GetProperty("validateFilledRecord").GetString()!, StringComparison.Ordinal);
        Assert.Contains(externalHandoffRoot.GetProperty("nonProofBoundaries").EnumerateArray(), static item =>
            item.GetString()!.Contains("handoff record does not publish", StringComparison.Ordinal));

        string postPublishPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "post-publish-verification-record-template.json");
        using JsonDocument postPublish = JsonDocument.Parse(File.ReadAllText(postPublishPath));
        JsonElement postRoot = postPublish.RootElement;
        Assert.Equal("post-publish-verification-record-template", postRoot.GetProperty("recordKind").GetString());
        Assert.True(postRoot.GetProperty("templateOnly").GetBoolean());
        Assert.Equal("template-only", postRoot.GetProperty("verificationState").GetString());
        Assert.False(postRoot.GetProperty("performsPublish").GetBoolean());
        Assert.False(postRoot.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(postRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal("", postRoot.GetProperty("channelSourceUri").GetString());
        Assert.Equal("", postRoot.GetProperty("cleanConsumerRoot").GetString());
        Assert.Equal("", postRoot.GetProperty("consumerProjectName").GetString());
        Assert.Equal("", postRoot.GetProperty("consumerProjectPath").GetString());
        Assert.Equal("", postRoot.GetProperty("host").GetProperty("cudnnVersion").GetString());
        Assert.Equal("", postRoot.GetProperty("restoreCommand").GetString());
        Assert.Equal("", postRoot.GetProperty("buildCommand").GetString());
        Assert.Equal("", postRoot.GetProperty("smokeCommand").GetString());
        Assert.Equal("", postRoot.GetProperty("stdoutSummary").GetString());
        Assert.Equal("", postRoot.GetProperty("stderrSummary").GetString());
        Assert.Equal("", postRoot.GetProperty("restoreLogSha256").GetString());
        Assert.Equal("", postRoot.GetProperty("nativeAssetListingSha256").GetString());
        Assert.Equal("", postRoot.GetProperty("dependencyProbeLogSha256").GetString());
        Assert.Equal("", postRoot.GetProperty("smokeLogSha256").GetString());
        Assert.Contains(postRoot.GetProperty("verificationItems").EnumerateArray(), static item => item.GetProperty("id").GetString() == "clean-consumer-project-identity");
        Assert.Contains(postRoot.GetProperty("verificationItems").EnumerateArray(), static item => item.GetProperty("id").GetString() == "host-runtime-metadata");
        Assert.Contains(postRoot.GetProperty("verificationItems").EnumerateArray(), static item => item.GetProperty("id").GetString() == "runtime-key-smoke-command");
        Assert.Contains(postRoot.GetProperty("verificationItems").EnumerateArray(), static item => item.GetProperty("id").GetString() == "stdout-summary");
        Assert.Contains(postRoot.GetProperty("verificationItems").EnumerateArray(), static item => item.GetProperty("id").GetString() == "stderr-summary");
        Assert.Contains(postRoot.GetProperty("verificationItems").EnumerateArray(), static item => item.GetProperty("id").GetString() == "stdout-stderr-summary");
        Assert.Contains(postRoot.GetProperty("verificationItems").EnumerateArray(), static item => item.GetProperty("id").GetString() == "no-project-reference");
        Assert.True(postRoot.GetProperty("packageIdentity").TryGetProperty("managedPackageSha256Source", out _));
        Assert.True(postRoot.GetProperty("packageIdentity").TryGetProperty("runtimePackageSha256Source", out _));
        Assert.True(postRoot.GetProperty("packageIdentity").TryGetProperty("managedPackageDownloadTimestampUtc", out _));
        Assert.True(postRoot.GetProperty("packageIdentity").TryGetProperty("runtimePackageDownloadTimestampUtc", out _));
        Assert.Contains(postRoot.GetProperty("executionSteps").EnumerateArray(), static step => step.GetProperty("id").GetString() == "capture-host-metadata");
        Assert.Contains(postRoot.GetProperty("executionSteps").EnumerateArray(), static step =>
            step.GetProperty("id").GetString() == "run-compatible-host-smoke" &&
            step.GetProperty("command").GetString()!.Contains("--runtime-package-key", StringComparison.Ordinal));
        Assert.Contains(postRoot.GetProperty("executionSteps").EnumerateArray(), static step => step.GetProperty("id").GetString() == "run-compatible-host-smoke");
        Assert.Contains(postRoot.GetProperty("executionSteps").EnumerateArray(), static step => step.GetProperty("id").GetString() == "validate-record");

        string postPublishValidationPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "post-publish-verification-validation.json");
        using JsonDocument postPublishValidation = JsonDocument.Parse(File.ReadAllText(postPublishValidationPath));
        JsonElement postPublishValidationRoot = postPublishValidation.RootElement;
        Assert.Equal("post-publish-verification-validation", postPublishValidationRoot.GetProperty("validationKind").GetString());
        Assert.Equal("post-publish-verification-record-template", postPublishValidationRoot.GetProperty("inputRecordKind").GetString());
        Assert.Equal("template-only", postPublishValidationRoot.GetProperty("validationState").GetString());
        Assert.True(postPublishValidationRoot.GetProperty("isTemplateOnly").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("consumerProjectIdentityReady").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("hostReady").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("commandsReady").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("smokeCommandRuntimeKeyReady").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("stdoutSummaryReady").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("stderrSummaryReady").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("stdoutStderrSummaryReady").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("restoreLogSha256Matches").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("nativeAssetListingSha256Matches").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("dependencyProbeLogSha256Matches").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("smokeLogSha256Matches").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("restoreLogSha256Ready").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("nativeAssetListingSha256Ready").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("dependencyProbeLogSha256Ready").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("smokeLogSha256Ready").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("managedPackageUrlReady").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("runtimePackageUrlReady").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("packageSha256SourceReady").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("cleanConsumerOutsideRepository").GetBoolean());
        Assert.Contains(postPublishValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "declared-proof" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(postPublishValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "channel-source" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(postPublishValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "published-package-url" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(postPublishValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "published-package-sha256-source" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(postPublishValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "clean-consumer-outside-repository" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(postPublishValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "clean-consumer-project-identity" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(postPublishValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "host-runtime-metadata" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(postPublishValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "runtime-key-smoke-command" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(postPublishValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "stdout-summary" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("detail").GetString()!.Contains("stdoutSummary", StringComparison.Ordinal));
        Assert.Contains(postPublishValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "stderr-summary" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("detail").GetString()!.Contains("no-stderr-emitted", StringComparison.Ordinal));
        Assert.Contains(postPublishValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "stdout-stderr-summary" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(postPublishValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "smoke-log-sha256-match" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(postPublishValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "runtime-smoke-log" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(postPublishValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "execution-steps-present" &&
            item.GetProperty("passed").GetBoolean());
        Assert.Equal(0, postPublishValidationRoot.GetProperty("missingExecutionStepIds").GetArrayLength());
        Assert.Contains(postPublishValidationRoot.GetProperty("promotionRules").EnumerateArray(), static rule =>
            rule.GetString()!.Contains("Template-only records", StringComparison.Ordinal));
    }

    [Fact]
    public void ExternalRuntimeProofInputExampleAndEvidenceBundleStayBlockedByDefault()
    {
        string inputTemplateOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordInputTemplate.ps1"));
        string draftOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordDraft.ps1"));
        string exampleOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordExample.ps1"));
        string exampleValidationOutput = RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalRuntimeProofRecord.ps1"),
            "-InputPath",
            "artifacts/final-release/external-runtime-proof-record.example.json",
            "-OutputRoot",
            "artifacts/final-release/example-validation");

        Assert.Contains("External runtime proof record input template written", inputTemplateOutput, StringComparison.Ordinal);
        Assert.Contains("External runtime proof draft written", draftOutput, StringComparison.Ordinal);
        Assert.Contains("External runtime proof record example written", exampleOutput, StringComparison.Ordinal);
        Assert.Contains("ValidationState=example-not-for-publication", exampleValidationOutput, StringComparison.Ordinal);
        Assert.Contains("IsRuntimeExecutionEvidence=False", exampleValidationOutput, StringComparison.Ordinal);
        Assert.Contains("CanPromoteRuntimeProof=False", exampleValidationOutput, StringComparison.Ordinal);

        string inputTemplatePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "external-runtime-proof-record.input-template.json");
        using JsonDocument inputTemplate = JsonDocument.Parse(File.ReadAllText(inputTemplatePath));
        JsonElement inputTemplateRoot = inputTemplate.RootElement;
        Assert.Equal("external-runtime-proof-record-input-template", inputTemplateRoot.GetProperty("recordKind").GetString());
        Assert.True(inputTemplateRoot.GetProperty("templateOnly").GetBoolean());
        Assert.Equal("template-only", inputTemplateRoot.GetProperty("proofClassification").GetString());
        Assert.Contains(inputTemplateRoot.GetProperty("evidenceClassifications").EnumerateArray(), static item =>
            item.GetString() == "real-model-runtime");
        Assert.False(inputTemplateRoot.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.False(inputTemplateRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.Equal("", inputTemplateRoot.GetProperty("host").GetProperty("cudaDriverSupportedRuntime").GetString());
        Assert.Equal("", inputTemplateRoot.GetProperty("host").GetProperty("cudnnVersion").GetString());
        Assert.Equal("", inputTemplateRoot.GetProperty("host").GetProperty("tensorRtLine").GetString());
        Assert.Equal("", inputTemplateRoot.GetProperty("packageSource").GetProperty("consumerProjectName").GetString());
        Assert.Equal("", inputTemplateRoot.GetProperty("packageSource").GetProperty("consumerProjectPath").GetString());
        Assert.Equal("", inputTemplateRoot.GetProperty("command").GetProperty("logSha256").GetString());
        Assert.Contains(inputTemplateRoot.GetProperty("requiredEvidenceSummary").EnumerateArray(), static item =>
            item.GetString()!.Contains("proofClassification=package-consumer-runtime", StringComparison.Ordinal));
        Assert.Contains(inputTemplateRoot.GetProperty("requiredEvidenceSummary").EnumerateArray(), static item =>
            item.GetString()!.Contains("runtimePackageKey", StringComparison.Ordinal));
        Assert.Contains(inputTemplateRoot.GetProperty("requiredEvidenceSummary").EnumerateArray(), static item =>
            item.GetString()!.Contains("consumerProjectName", StringComparison.Ordinal));
        Assert.Contains(inputTemplateRoot.GetProperty("requiredEvidenceSummary").EnumerateArray(), static item =>
            item.GetString()!.Contains("cuDNN", StringComparison.Ordinal));
        Assert.Contains(inputTemplateRoot.GetProperty("requiredEvidenceSummary").EnumerateArray(), static item =>
            item.GetString()!.Contains("--runtime-package-key", StringComparison.Ordinal));
        Assert.Contains(inputTemplateRoot.GetProperty("requiredEvidenceSummary").EnumerateArray(), static item =>
            item.GetString()!.Contains("logSha256", StringComparison.Ordinal));

        string draftPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "external-runtime-proof-record.draft.json");
        using JsonDocument draftDocument = JsonDocument.Parse(File.ReadAllText(draftPath));
        JsonElement draftRoot = draftDocument.RootElement;
        Assert.Equal("external-runtime-proof-record-draft", draftRoot.GetProperty("recordKind").GetString());
        Assert.False(draftRoot.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.False(draftRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.Equal("win-x64-trt11.0-cuda13.2-cudnn9.22", draftRoot.GetProperty("packageSource").GetProperty("runtimePackageKey").GetString());
        Assert.True(draftRoot.GetProperty("packageSource").TryGetProperty("consumerProjectName", out _));
        Assert.True(draftRoot.GetProperty("packageSource").TryGetProperty("consumerProjectPath", out _));
        string draftMarkdown = File.ReadAllText(Path.ChangeExtension(draftPath, ".md"));
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1", draftMarkdown, StringComparison.Ordinal);
        Assert.Contains("-RequireExistingLog", draftMarkdown, StringComparison.Ordinal);
        Assert.Contains("-FailOnNotProof", draftMarkdown, StringComparison.Ordinal);

        string examplePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "external-runtime-proof-record.example.json");
        using JsonDocument example = JsonDocument.Parse(File.ReadAllText(examplePath));
        JsonElement exampleRoot = example.RootElement;
        Assert.Equal("external-runtime-proof-record-example", exampleRoot.GetProperty("recordKind").GetString());
        Assert.True(exampleRoot.GetProperty("exampleOnly").GetBoolean());
        Assert.Equal("example-not-for-publication", exampleRoot.GetProperty("publicationState").GetString());
        Assert.Equal("dependency-probe-only", exampleRoot.GetProperty("proofClassification").GetString());
        Assert.Equal("example-cudnn-runtime", exampleRoot.GetProperty("host").GetProperty("cudnnVersion").GetString());
        Assert.Equal("win-x64-trt11.0-cuda13.2-cudnn9.22", exampleRoot.GetProperty("packageSource").GetProperty("runtimePackageKey").GetString());
        Assert.Equal("ExampleConsumer", exampleRoot.GetProperty("packageSource").GetProperty("consumerProjectName").GetString());
        Assert.Contains("--runtime-package-key win-x64-trt11.0-cuda13.2-cudnn9.22", exampleRoot.GetProperty("command").GetProperty("smokeCommand").GetString(), StringComparison.Ordinal);
        Assert.Equal("example-model-not-for-publication", exampleRoot.GetProperty("modelEvidence").GetProperty("modelName").GetString());
        Assert.Equal("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", exampleRoot.GetProperty("command").GetProperty("logSha256").GetString());
        Assert.False(exampleRoot.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.False(exampleRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());

        string exampleValidationPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "example-validation", "external-runtime-proof-validation.json");
        using JsonDocument exampleValidation = JsonDocument.Parse(File.ReadAllText(exampleValidationPath));
        JsonElement exampleValidationRoot = exampleValidation.RootElement;
        Assert.Equal("example-not-for-publication", exampleValidationRoot.GetProperty("validationState").GetString());
        Assert.Equal("dependency-probe-only", exampleValidationRoot.GetProperty("proofClassification").GetString());
        Assert.True(exampleValidationRoot.GetProperty("runtimePackageKeyMatches").GetBoolean());
        Assert.True(exampleValidationRoot.GetProperty("logSha256FormatReady").GetBoolean());
        Assert.True(exampleValidationRoot.GetProperty("logSha256Matches").GetBoolean());
        Assert.False(exampleValidationRoot.GetProperty("proofClassificationPromotable").GetBoolean());
        Assert.True(exampleValidationRoot.GetProperty("isExampleOnly").GetBoolean());
        Assert.False(exampleValidationRoot.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.False(exampleValidationRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.True(exampleValidationRoot.GetProperty("failedProofItemCount").GetInt32() > 0);
        Assert.Contains(exampleValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "proof-classification-promotable" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(exampleValidationRoot.GetProperty("requiredEvidenceSummary").EnumerateArray(), static item =>
            item.GetString()!.Contains("smokeStatus=passed", StringComparison.Ordinal));

        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalRuntimeProofRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishVerificationRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseOwnerApprovalInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseOwnerApprovalInput.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePackageProofBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DocsPublishReadinessBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OnnxEngineBuildEvidenceSidecarTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OnnxEngineBuildEvidenceSidecar.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-SampleRunEvidenceRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-SampleRunEvidenceRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-SampleAssetManifest.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-UserAcceptanceSampleCatalog.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RuntimePackageReadiness.ps1"));

        string bundleOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        Assert.Contains("Release evidence bundle written", bundleOutput, StringComparison.Ordinal);

        string bundlePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.json");
        using JsonDocument bundle = JsonDocument.Parse(File.ReadAllText(bundlePath));
        JsonElement root = bundle.RootElement;
        Assert.Equal("release-evidence-bundle", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-evidence-incomplete", root.GetProperty("bundleState").GetString());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canExecutePublicPublish").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isReleaseEvidenceComplete").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeExecutionEvidence").GetBoolean());
        Assert.Contains(root.GetProperty("externalRuntimeProofState").GetString(), new[] { "template-only", "draft-blocked-by-cuda-driver" });
        Assert.Contains(root.GetProperty("externalRuntimeProofClassification").GetString(), new[] { "template-only", "dependency-probe-only" });
        Assert.False(root.GetProperty("externalRuntimeProofClassificationPromotable").GetBoolean());
        Assert.True(root.GetProperty("externalRuntimeProofRuntimePackageKeyMatches").GetBoolean());
        Assert.True(root.GetProperty("externalRuntimeProofPackageSourceRuntimePackageKeyMatches").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofManagedNupkgSha256Ready").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofRuntimeNupkgSha256Ready").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofLogSha256FormatReady").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofLogSha256Matches").GetBoolean());
        Assert.True(root.GetProperty("externalRuntimeProofFailedProofItemCount").GetInt32() > 0);
        Assert.Equal("template-only", root.GetProperty("postPublishVerificationState").GetString());
        Assert.Equal("template-only", root.GetProperty("postPublishProofClassification").GetString());
        Assert.False(root.GetProperty("postPublishProofClassificationPromotable").GetBoolean());
        Assert.Equal(string.Empty, root.GetProperty("postPublishManagedPackageId").GetString());
        Assert.Equal(string.Empty, root.GetProperty("postPublishManagedPackageVersion").GetString());
        Assert.Equal(string.Empty, root.GetProperty("postPublishRuntimePackageId").GetString());
        Assert.Equal(string.Empty, root.GetProperty("postPublishRuntimePackageVersion").GetString());
        Assert.False(root.GetProperty("postPublishManagedNupkgSha256Ready").GetBoolean());
        Assert.False(root.GetProperty("postPublishRuntimeNupkgSha256Ready").GetBoolean());
        Assert.False(root.GetProperty("postPublishConsumerProjectIdentityReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishSmokeCommandRuntimeKeyReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishHostReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishCommandsReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishStdoutStderrSummaryReady").GetBoolean());
        Assert.Equal("package-evidence-owner-review-required", root.GetProperty("releasePackageProofState").GetString());
        Assert.False(root.GetProperty("canUseAsPublicPackageProof").GetBoolean());
        Assert.False(root.GetProperty("packageProofIsRuntimeExecutionProof").GetBoolean());
        Assert.Equal("ready-for-owner-review", root.GetProperty("docsPublishReadinessState").GetString());
        Assert.False(root.GetProperty("canPublishDocsExternally").GetBoolean());
        Assert.True(root.GetProperty("docsArticleCount").GetInt32() >= 30);
        Assert.Equal("owner-action-required", root.GetProperty("realModelAndPackageProofInputPackageState").GetString());
        Assert.False(root.GetProperty("realModelAndPackageProofInputPackagePerformsPublish").GetBoolean());
        Assert.False(root.GetProperty("realModelAndPackageProofInputPackageCanPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("realModelAndPackageProofInputPackageCanCloseReleaseIssue").GetBoolean());
        Assert.True(root.GetProperty("realModelAndPackageProofInputChecklistCount").GetInt32() >= 3);
        Assert.True(root.GetProperty("realModelAndPackageProofInputExecutionOrderCount").GetInt32() >= 10);
        Assert.Equal("owner-action-required", root.GetProperty("sampleRunEvidenceValidationState").GetString());
        Assert.False(root.GetProperty("sampleRunEvidenceCanPromoteRealModelRuntime").GetBoolean());
        Assert.Equal("template-only", root.GetProperty("sampleRunEvidenceProofClassification").GetString());
        Assert.Equal("owner-action-required", root.GetProperty("onnxEngineBuildEvidenceSidecarAuditState").GetString());
        Assert.Equal(0, root.GetProperty("onnxEngineBuildEvidenceSidecarErrorCount").GetInt32());
        Assert.Equal("owner-action-required", root.GetProperty("runtimeProofBlockerOwnerActionStatus").GetString());
        Assert.Equal("cuda-driver-runtime-compatibility", root.GetProperty("runtimeProofBlockerOwnerActionCategory").GetString());
        Assert.True(root.GetProperty("runtimeProofBlockerOwnerActionExternalInputRequired").GetBoolean());
        Assert.Contains("not runtime execution proof", root.GetProperty("runtimeProofBlockerOwnerActionWhyNotSmokePassed").GetString(), StringComparison.Ordinal);
        Assert.False(root.GetProperty("runtimeDeserializationDependencyDiagnosticsCanPromoteRuntimeProof").GetBoolean());
        Assert.True(root.GetProperty("runtimeDeserializationDependencyDiagnosticsDependencyProbeOnly").GetBoolean());
        Assert.Equal("runtime-smoke-driver-blocked", root.GetProperty("runtimeDeserializationDependencyDiagnosticsPackageConsumerEvidenceClassification").GetString());
        Assert.Equal("cuda-driver-runtime-compatibility", root.GetProperty("runtimeDeserializationDependencyDiagnosticsRuntimeProofBlockerCategory").GetString());
        Assert.True(root.GetProperty("runtimeDeserializationDependencyDiagnosticsDriverRuntimeMismatchClassified").GetBoolean());
        Assert.True(root.GetProperty("runtimeDeserializationDependencyDiagnosticsRuntimeProofOwnerActionRequired").GetBoolean());
        Assert.True(root.GetProperty("runtimeDeserializationDependencyDiagnosticsExternalRuntimeProofRequired").GetBoolean());
        Assert.False(root.GetProperty("runtimeDeserializationDependencyDiagnosticsPackageConsumerRuntimeProofPresent").GetBoolean());
        Assert.Contains("not runtime execution proof", root.GetProperty("runtimeDeserializationDependencyDiagnosticsWhyNotRuntimeProof").GetString(), StringComparison.Ordinal);
        Assert.Contains("compatible NVIDIA driver", root.GetProperty("runtimeDeserializationDependencyDiagnosticsNextOwnerAction").GetString(), StringComparison.Ordinal);
        Assert.Contains(root.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "external-runtime-proof-validation" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(root.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "runtime-deserialization-dependency-diagnostics" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(root.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "runtime-proof-blocker-owner-action" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(root.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "sample-run-evidence-validation" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(root.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "onnx-engine-build-evidence-sidecar-audit" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(root.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-verification" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(root.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "release-package-proof-bundle" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(root.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "docs-publish-readiness-bundle" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(root.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "real-model-and-package-proof-input-package" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("owner guidance only", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/real-model-and-package-proof-input-package.json");
    }

    [Fact]
    public void ReleaseEvidenceBundleDerivesPackageConsumerStatusFromPascalCaseSummary()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "jyppx-release-evidence-" + Guid.NewGuid().ToString("N"));
        string packageConsumerRoot = Path.Combine(tempRoot, "artifacts", "package-consumer");
        Directory.CreateDirectory(packageConsumerRoot);

        try
        {
            string packageConsumerPath = Path.Combine(packageConsumerRoot, "package-consumer-validation-summary.json");
            var packageConsumerSummary = new
            {
                RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
                ValidationState = "pending-local-validation",
                NativeAssetsExpected = 19,
                NativeAssetsFound = 19,
                MissingNativeAssets = Array.Empty<string>(),
                SmokeRequested = true,
                SmokeResult = "blocked-by-cuda-driver",
                EvidenceKind = "full-runtime-package-consumer-smoke-driver-blocked",
                IsRuntimeExecutionEvidence = false,
                IsDependencyProbeOnly = true
            };
            File.WriteAllText(packageConsumerPath, JsonSerializer.Serialize(packageConsumerSummary));

            string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1");
            string output = RunPowerShell(script, "-RepositoryRoot", tempRoot);

            Assert.Contains("Release evidence bundle written", output, StringComparison.Ordinal);

            string bundlePath = Path.Combine(tempRoot, "artifacts", "final-release", "release-evidence-bundle.json");
            using JsonDocument bundle = JsonDocument.Parse(File.ReadAllText(bundlePath));
            JsonElement root = bundle.RootElement;

            string status = root.GetProperty("packageConsumerStatus").GetString()!;
            Assert.Contains("native-assets-ready", status, StringComparison.Ordinal);
            Assert.Contains("validation=pending-local-validation", status, StringComparison.Ordinal);
            Assert.Contains("smoke=blocked-by-cuda-driver", status, StringComparison.Ordinal);
            Assert.Contains("evidence=full-runtime-package-consumer-smoke-driver-blocked", status, StringComparison.Ordinal);
            Assert.Contains("runtimeExecution=False", status, StringComparison.Ordinal);
            Assert.Contains("dependencyProbeOnly=True", status, StringComparison.Ordinal);
            Assert.True(root.GetProperty("packageConsumerNativeAssetsReady").GetBoolean());
            Assert.Equal("blocked-by-cuda-driver", root.GetProperty("packageConsumerSmokeResult").GetString());
            Assert.Equal("full-runtime-package-consumer-smoke-driver-blocked", root.GetProperty("packageConsumerEvidenceKind").GetString());
            Assert.False(root.GetProperty("packageConsumerIsRuntimeExecutionEvidence").GetBoolean());
            Assert.True(root.GetProperty("packageConsumerIsDependencyProbeOnly").GetBoolean());

            Assert.Contains(root.GetProperty("evidenceItems").EnumerateArray(), static item =>
                item.GetProperty("id").GetString() == "package-consumer-validation" &&
                item.GetProperty("state").GetString()!.Contains("native-assets-ready", StringComparison.Ordinal) &&
                item.GetProperty("state").GetString()!.Contains("smoke=blocked-by-cuda-driver", StringComparison.Ordinal) &&
                item.GetProperty("passed").GetBoolean() &&
                item.GetProperty("boundary").GetString()!.Contains("not runtime execution proof", StringComparison.Ordinal));
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void ReleasePackageAndDocsReadinessBundlesStayOwnerReviewOnly()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishVerificationRecord.ps1"));

        string finalPackageOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPackageReviewBundle.ps1"));
        string packageOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePackageProofBundle.ps1"));
        string docsOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DocsPublishReadinessBundle.ps1"));

        Assert.Contains("Final package review bundle written", finalPackageOutput, StringComparison.Ordinal);
        Assert.Contains("Release package proof bundle written", packageOutput, StringComparison.Ordinal);
        Assert.Contains("Docs publish readiness bundle written", docsOutput, StringComparison.Ordinal);

        string finalPackagePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "final-package-review-bundle.json");
        using JsonDocument finalPackage = JsonDocument.Parse(File.ReadAllText(finalPackagePath));
        JsonElement finalPackageRoot = finalPackage.RootElement;

        Assert.Equal("final-package-review-bundle", finalPackageRoot.GetProperty("recordKind").GetString());
        Assert.Equal("owner-review-required", finalPackageRoot.GetProperty("bundleState").GetString());
        Assert.False(finalPackageRoot.GetProperty("performsPublish").GetBoolean());
        Assert.False(finalPackageRoot.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(finalPackageRoot.GetProperty("canUseAsPublicPackageProof").GetBoolean());
        Assert.False(finalPackageRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(finalPackageRoot.GetProperty("postPublishStdoutSummaryReady").GetBoolean());
        Assert.False(finalPackageRoot.GetProperty("postPublishStderrSummaryReady").GetBoolean());
        Assert.False(finalPackageRoot.GetProperty("postPublishStdoutStderrSummaryReady").GetBoolean());
        Assert.False(finalPackageRoot.GetProperty("postPublishAllLogSha256Matches").GetBoolean());
        Assert.True(finalPackageRoot.GetProperty("managedPackageCount").GetInt32() >= 1);
        Assert.True(finalPackageRoot.GetProperty("runtimePackageCount").GetInt32() >= 1);
        Assert.True(finalPackageRoot.GetProperty("splitRuntimePackageCount").GetInt32() >= 1);
        Assert.True(finalPackageRoot.GetProperty("nativeAssetCount").GetInt32() > 0);
        Assert.Contains(finalPackageRoot.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-package-proof-bundle.json");
        Assert.Contains(finalPackageRoot.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-evidence-bundle.json");
        Assert.Contains(finalPackageRoot.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/post-publish-verification-validation.json");
        Assert.Contains(finalPackageRoot.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/external-runtime-proof-validation.json");
        Assert.Contains(finalPackageRoot.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/release/release-candidate-freeze-summary.json");
        Assert.Contains(finalPackageRoot.GetProperty("guardrails").EnumerateArray(), static item => item.GetString()!.Contains("Local package review is not public channel proof", StringComparison.Ordinal));
        Assert.Contains(finalPackageRoot.GetProperty("guardrails").EnumerateArray(), static item => item.GetString()!.Contains("does not publish packages", StringComparison.Ordinal));
        Assert.All(finalPackageRoot.GetProperty("packages").EnumerateArray(), static package =>
        {
            Assert.True(package.GetProperty("sizeBytes").GetInt64() > 0);
            Assert.Matches("^[a-f0-9]{64}$", package.GetProperty("sha256").GetString()!);
            Assert.True(package.GetProperty("sha256Ready").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(package.GetProperty("packageId").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(package.GetProperty("version").GetString()));
        });

        string packagePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-package-proof-bundle.json");
        using JsonDocument package = JsonDocument.Parse(File.ReadAllText(packagePath));
        JsonElement packageRoot = package.RootElement;

        Assert.Equal("release-package-proof-bundle", packageRoot.GetProperty("recordKind").GetString());
        Assert.Equal("package-evidence-owner-review-required", packageRoot.GetProperty("proofState").GetString());
        Assert.False(packageRoot.GetProperty("canUseAsPublicPackageProof").GetBoolean());
        Assert.False(packageRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(packageRoot.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.True(packageRoot.GetProperty("isDependencyProbeOnly").GetBoolean());
        Assert.Equal("blocked-by-cuda-driver", packageRoot.GetProperty("runtimeProofStatus").GetString());
        Assert.Equal("template-only", packageRoot.GetProperty("postPublishVerificationState").GetString());
        Assert.False(packageRoot.GetProperty("postPublishConsumerProjectIdentityReady").GetBoolean());
        Assert.False(packageRoot.GetProperty("postPublishSmokeCommandRuntimeKeyReady").GetBoolean());
        Assert.False(packageRoot.GetProperty("postPublishHostReady").GetBoolean());
        Assert.False(packageRoot.GetProperty("postPublishCommandsReady").GetBoolean());
        Assert.False(packageRoot.GetProperty("postPublishStdoutStderrSummaryReady").GetBoolean());
        Assert.False(packageRoot.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(packageRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        AssertPostPublishRequiredEvidence(packageRoot);
        Assert.True(packageRoot.GetProperty("nativeAssetCopyReady").GetBoolean());
        Assert.True(packageRoot.GetProperty("localFeedDependencyProbeReady").GetBoolean());
        Assert.Contains(packageRoot.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "pack/runtime/runtime-packages.manifest.json");
        Assert.Contains(packageRoot.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "pack/runtime-split/split-runtime-packages.manifest.json");
        Assert.Contains(packageRoot.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/post-publish-verification-validation.json");
        Assert.Contains(packageRoot.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-verification" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("state").GetString()!.Contains("commandsReady=False", StringComparison.Ordinal));

        string docsPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "docs-publish-readiness-bundle.json");
        using JsonDocument docs = JsonDocument.Parse(File.ReadAllText(docsPath));
        JsonElement docsRoot = docs.RootElement;

        Assert.Equal("docs-publish-readiness-bundle", docsRoot.GetProperty("recordKind").GetString());
        Assert.Equal("ready-for-owner-review", docsRoot.GetProperty("readinessState").GetString());
        Assert.False(docsRoot.GetProperty("canPublishDocsExternally").GetBoolean());
        Assert.True(docsRoot.GetProperty("articleCount").GetInt32() >= 30);
        Assert.True(docsRoot.GetProperty("highQualityArticleCount").GetInt32() >= 30);
        Assert.True(docsRoot.GetProperty("sampleBackedArticleCount").GetInt32() > 0);
        Assert.True(docsRoot.GetProperty("coreArticleCount").GetInt32() >= 10);
        Assert.Equal(docsRoot.GetProperty("coreArticleCount").GetInt32(), docsRoot.GetProperty("coreArticleCoveredCount").GetInt32());
        Assert.Equal(0, docsRoot.GetProperty("coreArticleMissingCount").GetInt32());
        Assert.True(docsRoot.GetProperty("ownerActionCount").GetInt32() >= 4);
        Assert.True(docsRoot.GetProperty("hasArticleRoadmap").GetBoolean());
        Assert.Equal("ready", docsRoot.GetProperty("docfxValidationState").GetString());
        Assert.Contains(docsRoot.GetProperty("coreArticleCoverage").EnumerateArray(), static article =>
            article.GetProperty("id").GetString() == "post-publish-verification-record" &&
            article.GetProperty("covered").GetBoolean() &&
            article.GetProperty("includedInToc").GetBoolean() &&
            article.GetProperty("linkedFromIndex").GetBoolean());
        Assert.Contains(docsRoot.GetProperty("ownerActions").EnumerateArray(), static action =>
            action.GetProperty("id").GetString() == "proof-claim-review" &&
            action.GetProperty("state").GetString() == "owner-action-required" &&
            action.GetProperty("boundary").GetString()!.Contains("dependency-probe-only", StringComparison.Ordinal));
    }

    [Fact]
    public void ReleaseOwnerDecisionRecordKeepsApprovalPendingAndPublicationBlocked()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalRuntimeProofRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CompatibleHostRuntimeProofRunbook.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishVerificationRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPackageReviewBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofBackfillPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationBackfillPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseOwnerApprovalInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseOwnerApprovalInput.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseOwnerDecisionTemplate.ps1"));

        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseOwnerDecisionRecord.ps1");
        string output = RunPowerShell(script);

        Assert.Contains("Release owner decision record written", output, StringComparison.Ordinal);

        string recordPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-owner-decision-record.json");
        using JsonDocument record = JsonDocument.Parse(File.ReadAllText(recordPath));
        JsonElement root = record.RootElement;

        Assert.Equal("release-owner-decision-record", root.GetProperty("recordKind").GetString());
        Assert.Equal("pending-release-owner-approval", root.GetProperty("recordState").GetString());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.True(root.GetProperty("requiresHumanOwner").GetBoolean());
        Assert.False(root.GetProperty("realCallbackRuntimeProof").GetBoolean());
        Assert.True(root.GetProperty("allowRuntimeSmokeBlocked").GetBoolean());
        Assert.Equal("blocked-by-cuda-driver", root.GetProperty("runtimeProofStatus").GetString());
        Assert.True(root.GetProperty("runtimeProofRequiredForRelease").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("runtimeProofBlockerOwnerActionStatus").GetString());
        Assert.Equal("cuda-driver-runtime-compatibility", root.GetProperty("runtimeProofBlockerCategory").GetString());
        Assert.False(root.GetProperty("externalRuntimeExecutionEvidence").GetBoolean());
        Assert.Contains(root.GetProperty("externalRuntimeProofState").GetString(), new[] { "template-only", "draft-blocked-by-cuda-driver" });
        Assert.Contains(root.GetProperty("externalRuntimeProofClassification").GetString(), new[] { "template-only", "dependency-probe-only" });
        Assert.True(root.GetProperty("externalRuntimeProofRuntimePackageKeyMatches").GetBoolean());
        Assert.True(root.GetProperty("externalRuntimeProofPackageSourceRuntimePackageKeyMatches").GetBoolean());
        Assert.True(root.GetProperty("externalRuntimeProofFailedProofItemCount").GetInt32() > 0);
        Assert.Equal("owner-action-required", root.GetProperty("externalRuntimeProofOwnerActionStatus").GetString());
        Assert.False(root.GetProperty("externalRuntimeProofCanPromoteRuntimeProof").GetBoolean());
        Assert.Equal("draft-blocked-by-cuda-driver", root.GetProperty("externalRuntimeProofDraftState").GetString());
        Assert.False(root.GetProperty("externalRuntimeProofDraftCanPromoteRuntimeProof").GetBoolean());
        Assert.True(root.GetProperty("draftManagedNupkgSha256Ready").GetBoolean());
        Assert.True(root.GetProperty("draftRuntimeNupkgSha256Ready").GetBoolean());
        Assert.True(root.GetProperty("draftSmokeLogSha256Ready").GetBoolean());
        Assert.True(root.GetProperty("draftNoProjectReference").GetBoolean());
        Assert.Equal("blocked-by-cuda-driver", root.GetProperty("draftSmokeStatus").GetString());
        Assert.True(root.GetProperty("compatibleHostRequired").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("compatibleHostRuntimeProofRunbookState").GetString());
        Assert.True(root.GetProperty("compatibleHostRuntimeProofRunbookCompatibleHostRequired").GetBoolean());
        Assert.False(root.GetProperty("compatibleHostRuntimeProofRunbookPerformsPublish").GetBoolean());
        Assert.False(root.GetProperty("compatibleHostRuntimeProofRunbookApprovesPublicRelease").GetBoolean());
        Assert.False(root.GetProperty("compatibleHostRuntimeProofRunbookCanPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("compatibleHostRuntimeProofRunbookRuntimeExecutionEvidence").GetBoolean());
        Assert.Equal("blocked-by-cuda-driver is not smoke passed", root.GetProperty("compatibleHostRuntimeProofRunbookPromotionBlockedReason").GetString());
        Assert.Contains("Test-PackageConsumer.ps1", root.GetProperty("compatibleHostRuntimeProofRunbookRunPackageConsumerSmokeCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("-FailOnNotProof", root.GetProperty("compatibleHostRuntimeProofRunbookValidateFilledRecordCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("compatible", root.GetProperty("requiredHostAction").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("blocked-by-cuda-driver is not smoke passed", root.GetProperty("promotionBlockedReason").GetString());
        Assert.Equal("template-only", root.GetProperty("postPublishVerificationState").GetString());
        Assert.False(root.GetProperty("postPublishConsumerProjectIdentityReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishSmokeCommandRuntimeKeyReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishHostReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishCommandsReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishStdoutStderrSummaryReady").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal("owner-review-required", root.GetProperty("finalPackageReviewState").GetString());
        Assert.True(root.GetProperty("finalPackageReviewPackageCount").GetInt32() >= 1);
        Assert.True(root.GetProperty("finalPackageReviewNativeAssetCount").GetInt32() > 0);
        Assert.False(root.GetProperty("finalPackageReviewCanUseAsPublicPackageProof").GetBoolean());
        Assert.Equal("blocked-compatible-host-proof-required", root.GetProperty("externalRuntimeProofBackfillPlanState").GetString());
        Assert.True(root.GetProperty("externalRuntimeProofBackfillStepCount").GetInt32() >= 7);
        Assert.False(root.GetProperty("externalRuntimeProofBackfillCanPromoteRuntimeProof").GetBoolean());
        Assert.Equal("blocked-real-post-publish-proof-required", root.GetProperty("postPublishVerificationBackfillPlanState").GetString());
        Assert.True(root.GetProperty("postPublishVerificationBackfillStepCount").GetInt32() >= 9);
        Assert.False(root.GetProperty("postPublishVerificationBackfillCanCloseReleaseIssue").GetBoolean());
        Assert.Contains("Test-PackageConsumer.ps1", root.GetProperty("runtimeProofOwnerCommand").GetString(), StringComparison.Ordinal);
        Assert.Equal("blocked-owner-input-required", root.GetProperty("ownerApprovalInputValidationStatus").GetString());
        Assert.False(root.GetProperty("ownerApprovalCanPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("isRealLinuxRunnerProof").GetBoolean());
        Assert.True(root.GetProperty("requiredDecisionCount").GetInt32() >= 6);
        Assert.Equal(root.GetProperty("requiredDecisionCount").GetInt32(), root.GetProperty("unresolvedDecisionCount").GetInt32());
        Assert.Contains(root.GetProperty("decisions").EnumerateArray(), static decision => decision.GetProperty("id").GetString() == "linux-runner-proof");
        Assert.Contains(root.GetProperty("decisions").EnumerateArray(), static decision => decision.GetProperty("id").GetString() == "owner-approval-input");
        Assert.Contains(root.GetProperty("decisions").EnumerateArray(), static decision =>
            decision.GetProperty("id").GetString() == "runtime-smoke" &&
            decision.GetProperty("currentStatus").GetString()!.Contains("ownerAction=owner-action-required", StringComparison.Ordinal) &&
            decision.GetProperty("currentStatus").GetString()!.Contains("blockerCategory=cuda-driver-runtime-compatibility", StringComparison.Ordinal) &&
            decision.GetProperty("currentStatus").GetString()!.Contains("compatibleHostRunbookState=owner-action-required", StringComparison.Ordinal) &&
            decision.GetProperty("currentStatus").GetString()!.Contains("externalRuntimeProofFailedProofItemCount=", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("decisions").EnumerateArray(), static decision =>
            decision.GetProperty("id").GetString() == "post-publish-verification" &&
            decision.GetProperty("currentStatus").GetString()!.Contains("commandsReady=False", StringComparison.Ordinal) &&
            decision.GetProperty("boundary").GetString()!.Contains("stdout/stderr summary", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("decisions").EnumerateArray(), static decision =>
            decision.GetProperty("id").GetString() == "final-package-review-acknowledgement" &&
            decision.GetProperty("currentStatus").GetString()!.Contains("canUseAsPublicPackageProof=False", StringComparison.Ordinal) &&
            decision.GetProperty("evidence").GetString()!.Contains("final-package-review-bundle.json", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("decisions").EnumerateArray(), static decision =>
            decision.GetProperty("id").GetString() == "backfill-plan-boundary" &&
            decision.GetProperty("currentStatus").GetString()!.Contains("externalRuntimeProofBackfillPlanState=blocked-compatible-host-proof-required", StringComparison.Ordinal) &&
            decision.GetProperty("evidence").GetString()!.Contains("post-publish-verification-backfill-plan.json", StringComparison.Ordinal) &&
            decision.GetProperty("boundary").GetString()!.Contains("guidance only", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("decisions").EnumerateArray(), static decision => decision.GetProperty("boundary").GetString()!.Contains("allowRuntimeSmokeBlocked=true", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("decisions").EnumerateArray(), static decision => decision.GetProperty("boundary").GetString()!.Contains("runtimeProofRequiredForRelease=true", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetProperty("path").GetString() == "artifacts/final-release/compatible-host-runtime-proof-runbook.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetProperty("path").GetString() == "artifacts/final-release/final-package-review-bundle.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetProperty("path").GetString() == "artifacts/final-release/post-publish-verification-validation.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetProperty("path").GetString() == "artifacts/final-release/external-runtime-proof-backfill-plan.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetProperty("path").GetString() == "artifacts/final-release/post-publish-verification-backfill-plan.json");
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note => note.GetString()!.Contains("compatible-host-runtime-proof-runbook is an owner-action runbook, not runtime proof", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note => note.GetString()!.Contains("final-package-review-bundle is local package inventory", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note => note.GetString()!.Contains("post-publish verification remains not closeable", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note => note.GetString()!.Contains("Backfill plans are guidance only", StringComparison.Ordinal));
    }

    [Fact]
    public void ReleasePublishExecutionChecklistDoesNotRunOrApprovePublish()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseOwnerApprovalInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseOwnerApprovalInput.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseOwnerDecisionRecord.ps1"));
        RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Export-LinuxRunnerEvidenceRecordTemplate.ps1"),
            "-RuntimePackageKey", "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22");
        RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Test-LinuxRunnerEvidenceRecord.ps1"),
            "-RuntimePackageKey", "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePromotionIssueRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalRuntimeProofRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishVerificationRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPackageReviewBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePackageProofBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DocsPublishReadinessBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePublishExecutionChecklist.ps1");
        string output = RunPowerShell(script);

        Assert.Contains("Release publish execution checklist written", output, StringComparison.Ordinal);

        string recordPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-publish-execution-checklist.json");
        using JsonDocument record = JsonDocument.Parse(File.ReadAllText(recordPath));
        JsonElement root = record.RootElement;

        Assert.Equal("release-publish-execution-checklist", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-input-required", root.GetProperty("executionState").GetString());
        Assert.False(root.GetProperty("canExecutePublicPublish").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("ownerActionStatus").GetString());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isRealModelRuntimeProof").GetBoolean());
        Assert.True(root.GetProperty("requiresHumanOwner").GetBoolean());
        Assert.Equal("blocked-owner-input-required", root.GetProperty("ownerApprovalInputValidationStatus").GetString());
        Assert.False(root.GetProperty("ownerApprovalCanPublishPublicly").GetBoolean());
        Assert.Equal("blocked-by-cuda-driver", root.GetProperty("runtimeProofStatus").GetString());
        Assert.True(root.GetProperty("runtimeProofRequiredForRelease").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("runtimeProofBlockerOwnerActionStatus").GetString());
        Assert.Equal("cuda-driver-runtime-compatibility", root.GetProperty("runtimeProofBlockerCategory").GetString());
        Assert.Contains("Test-PackageConsumer.ps1", root.GetProperty("runtimeProofOwnerCommand").GetString(), StringComparison.Ordinal);
        Assert.False(root.GetProperty("realCallbackRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isRealLinuxRunnerProof").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeExecutionEvidence").GetBoolean());
        Assert.Contains(root.GetProperty("externalRuntimeProofState").GetString(), new[] { "template-only", "draft-blocked-by-cuda-driver" });
        Assert.Contains(root.GetProperty("externalRuntimeProofClassification").GetString(), new[] { "template-only", "dependency-probe-only" });
        Assert.True(root.GetProperty("externalRuntimeProofRuntimePackageKeyMatches").GetBoolean());
        Assert.True(root.GetProperty("externalRuntimeProofPackageSourceRuntimePackageKeyMatches").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofConsumerProjectIdentityReady").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofSmokeCommandRuntimeKeyReady").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofHostReady").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofCommandsReady").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofManagedNupkgSha256Ready").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofRuntimeNupkgSha256Ready").GetBoolean());
        Assert.True(root.GetProperty("externalRuntimeProofFailedProofItemCount").GetInt32() > 0);
        Assert.False(root.GetProperty("externalRuntimeProofLogSha256FormatReady").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofLogSha256Matches").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("externalRuntimeProofOwnerActionStatus").GetString());
        Assert.False(root.GetProperty("externalRuntimeProofCanPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeExecutionEvidence").GetBoolean());
        Assert.Equal("draft-blocked-by-cuda-driver", root.GetProperty("externalRuntimeProofDraftState").GetString());
        Assert.False(root.GetProperty("externalRuntimeProofDraftCanPromoteRuntimeProof").GetBoolean());
        Assert.True(root.GetProperty("draftManagedNupkgSha256Ready").GetBoolean());
        Assert.True(root.GetProperty("draftRuntimeNupkgSha256Ready").GetBoolean());
        Assert.True(root.GetProperty("draftSmokeLogSha256Ready").GetBoolean());
        Assert.True(root.GetProperty("draftNoProjectReference").GetBoolean());
        Assert.Equal("blocked-by-cuda-driver", root.GetProperty("draftSmokeStatus").GetString());
        Assert.True(root.GetProperty("compatibleHostRequired").GetBoolean());
        Assert.Contains("compatible", root.GetProperty("requiredHostAction").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("blocked-by-cuda-driver is not smoke passed", root.GetProperty("promotionBlockedReason").GetString());
        Assert.Equal("template-only", root.GetProperty("postPublishVerificationState").GetString());
        Assert.False(root.GetProperty("postPublishConsumerProjectIdentityReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishSmokeCommandRuntimeKeyReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishHostReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishCommandsReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishStdoutStderrSummaryReady").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishVerificationProof").GetBoolean());
        AssertPostPublishRequiredEvidence(root);
        Assert.Equal("blocked-evidence-incomplete", root.GetProperty("releaseEvidenceBundleState").GetString());
        Assert.False(root.GetProperty("isReleaseEvidenceComplete").GetBoolean());
        Assert.Equal("package-evidence-owner-review-required", root.GetProperty("releasePackageProofState").GetString());
        Assert.False(root.GetProperty("canUseAsPublicPackageProof").GetBoolean());
        Assert.False(root.GetProperty("packageProofIsRuntimeExecutionProof").GetBoolean());
        Assert.Equal("owner-review-required", root.GetProperty("finalPackageReviewState").GetString());
        Assert.True(root.GetProperty("finalPackageReviewPackageCount").GetInt32() >= 1);
        Assert.True(root.GetProperty("finalPackageReviewNativeAssetCount").GetInt32() > 0);
        Assert.False(root.GetProperty("finalPackageReviewCanUseAsPublicPackageProof").GetBoolean());
        Assert.Equal("ready-for-owner-review", root.GetProperty("docsPublishReadinessState").GetString());
        Assert.False(root.GetProperty("canPublishDocsExternally").GetBoolean());
        Assert.True(root.GetProperty("docsArticleCount").GetInt32() >= 30);
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-evidence-bundle.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/final-package-review-bundle.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-package-proof-bundle.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/docs-publish-readiness-bundle.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/external-runtime-proof-validation.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/post-publish-verification-validation.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/real-model-and-package-proof-input-package.json");
        Assert.Contains(root.GetProperty("preflightItems").EnumerateArray(), static item => item.GetProperty("id").GetString() == "owner-approval-input");
        Assert.Contains(root.GetProperty("preflightItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "release-evidence-bundle" &&
            item.GetProperty("requiredEvidence").GetString()!.Contains("release-evidence-bundle.json", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("blocked-evidence-incomplete", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("preflightItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "final-package-review-bundle" &&
            item.GetProperty("requiredEvidence").GetString()!.Contains("final-package-review-bundle.json", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("owner-review-required", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("canUseAsPublicPackageProof=False", StringComparison.Ordinal) &&
            item.GetProperty("boundary").GetString()!.Contains("not public channel proof", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("preflightItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "release-package-proof-bundle" &&
            item.GetProperty("requiredEvidence").GetString()!.Contains("release-package-proof-bundle.json", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("canUseAsPublicPackageProof=False", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("preflightItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "docs-publish-readiness-bundle" &&
            item.GetProperty("requiredEvidence").GetString()!.Contains("docs-publish-readiness-bundle.json", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("canPublishDocsExternally=False", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("preflightItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "external-runtime-proof-record" &&
            item.GetProperty("requiredEvidence").GetString()!.Contains("external-runtime-proof-validation.json", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("proofState=template-only", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("consumerProjectIdentityReady=False", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("smokeCommandRuntimeKeyReady=False", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("hostReady=False", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("logSha256FormatReady=False", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("ownerAction=owner-action-required", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("preflightItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-verification-record" &&
            item.GetProperty("requiredEvidence").GetString()!.Contains("post-publish-verification-validation.json", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("verificationState=template-only", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("consumerProjectIdentityReady=False", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("smokeCommandRuntimeKeyReady=False", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("hostReady=False", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("commandsReady=False", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("stdoutStderrSummaryReady=False", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("canCloseReleaseIssue=False", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("preflightItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "runtime-proof" &&
            item.GetProperty("boundary").GetString()!.Contains("runtime-deserialization-dependency-diagnostics", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("ownerAction=owner-action-required", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("postPublishVerification").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "runtime-proof-owner-action" &&
            item.GetProperty("boundary").GetString()!.Contains("not package-consumer-runtime proof", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("channelPlans").EnumerateArray(), static channel => channel.GetProperty("id").GetString() == "nuget-org");
        Assert.Contains(root.GetProperty("channelPlans").EnumerateArray(), static channel =>
            channel.GetProperty("id").GetString() == "github-release-assets" &&
            !channel.GetProperty("performsPublish").GetBoolean());
        Assert.Contains(root.GetProperty("postPublishVerification").EnumerateArray(), static item => item.GetProperty("id").GetString() == "clean-consumer");
        Assert.Contains(root.GetProperty("postPublishVerification").EnumerateArray(), static item => item.GetProperty("id").GetString() == "clean-consumer-project-identity");
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("final package review bundle is local package inventory", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(root.GetProperty("postPublishVerification").EnumerateArray(), static item => item.GetProperty("id").GetString() == "host-runtime-metadata");
        Assert.Contains(root.GetProperty("postPublishVerification").EnumerateArray(), static item => item.GetProperty("id").GetString() == "runtime-key-smoke-command");
        Assert.Contains(root.GetProperty("postPublishVerification").EnumerateArray(), static item => item.GetProperty("id").GetString() == "stdout-stderr-summary");
        Assert.Contains(root.GetProperty("postPublishVerification").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "dependency-probe" &&
            item.GetProperty("boundary").GetString()!.Contains("not runtime execution proof", StringComparison.Ordinal));
    }

    [Fact]
    public void ReleasePromotionIssueRecordDoesNotPublishOrApprovePublicPromotion()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseOwnerApprovalInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseOwnerApprovalInput.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseOwnerDecisionRecord.ps1"));
        RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Export-LinuxRunnerEvidenceRecordTemplate.ps1"),
            "-RuntimePackageKey", "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalRuntimeProofRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishVerificationRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofBackfillPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationBackfillPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofCollectionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationCollectionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePackageProofBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DocsPublishReadinessBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePublishExecutionChecklist.ps1"));

        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePromotionIssueRecord.ps1");
        string output = RunPowerShell(script);

        Assert.Contains("Release promotion issue record written", output, StringComparison.Ordinal);

        string recordPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-promotion-issue-record.json");
        using JsonDocument record = JsonDocument.Parse(File.ReadAllText(recordPath));
        JsonElement root = record.RootElement;

        Assert.Equal("release-promotion-issue-record", root.GetProperty("recordKind").GetString());
        Assert.Equal("pending-release-owner-approval", root.GetProperty("promotionState").GetString());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.True(root.GetProperty("requiresHumanOwner").GetBoolean());
        Assert.False(root.GetProperty("realCallbackRuntimeProof").GetBoolean());
        Assert.True(root.GetProperty("allowRuntimeSmokeBlocked").GetBoolean());
        Assert.Equal("blocked-by-cuda-driver", root.GetProperty("runtimeProofStatus").GetString());
        Assert.True(root.GetProperty("runtimeProofRequiredForRelease").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("runtimeProofBlockerOwnerActionStatus").GetString());
        Assert.Equal("cuda-driver-runtime-compatibility", root.GetProperty("runtimeProofBlockerCategory").GetString());
        Assert.Contains("Test-PackageConsumer.ps1", root.GetProperty("runtimeProofOwnerCommand").GetString(), StringComparison.Ordinal);
        Assert.Equal("blocked-owner-input-required", root.GetProperty("ownerApprovalInputValidationStatus").GetString());
        Assert.False(root.GetProperty("ownerApprovalCanPublishPublicly").GetBoolean());
        Assert.Equal("blocked-owner-input-required", root.GetProperty("publishExecutionChecklistState").GetString());
        Assert.False(root.GetProperty("canExecutePublicPublish").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeExecutionEvidence").GetBoolean());
        Assert.Contains(root.GetProperty("externalRuntimeProofState").GetString(), new[] { "template-only", "draft-blocked-by-cuda-driver" });
        Assert.Contains(root.GetProperty("externalRuntimeProofClassification").GetString(), new[] { "template-only", "dependency-probe-only" });
        Assert.True(root.GetProperty("externalRuntimeProofRuntimePackageKeyMatches").GetBoolean());
        Assert.True(root.GetProperty("externalRuntimeProofPackageSourceRuntimePackageKeyMatches").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofConsumerProjectIdentityReady").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofSmokeCommandRuntimeKeyReady").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofHostReady").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofCommandsReady").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofManagedNupkgSha256Ready").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofRuntimeNupkgSha256Ready").GetBoolean());
        Assert.True(root.GetProperty("externalRuntimeProofFailedProofItemCount").GetInt32() > 0);
        Assert.False(root.GetProperty("externalRuntimeProofLogSha256FormatReady").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofLogSha256Matches").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("externalRuntimeProofOwnerActionStatus").GetString());
        Assert.False(root.GetProperty("externalRuntimeProofCanPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeExecutionEvidence").GetBoolean());
        Assert.Equal("draft-blocked-by-cuda-driver", root.GetProperty("externalRuntimeProofDraftState").GetString());
        Assert.False(root.GetProperty("externalRuntimeProofDraftCanPromoteRuntimeProof").GetBoolean());
        Assert.True(root.GetProperty("draftManagedNupkgSha256Ready").GetBoolean());
        Assert.True(root.GetProperty("draftRuntimeNupkgSha256Ready").GetBoolean());
        Assert.True(root.GetProperty("draftSmokeLogSha256Ready").GetBoolean());
        Assert.True(root.GetProperty("draftNoProjectReference").GetBoolean());
        Assert.Equal("blocked-by-cuda-driver", root.GetProperty("draftSmokeStatus").GetString());
        Assert.True(root.GetProperty("compatibleHostRequired").GetBoolean());
        Assert.Contains("compatible", root.GetProperty("requiredHostAction").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Equal("blocked-by-cuda-driver is not smoke passed", root.GetProperty("promotionBlockedReason").GetString());
        Assert.Equal("template-only", root.GetProperty("postPublishVerificationState").GetString());
        Assert.False(root.GetProperty("postPublishConsumerProjectIdentityReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishSmokeCommandRuntimeKeyReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishHostReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishCommandsReady").GetBoolean());
        Assert.False(root.GetProperty("postPublishStdoutStderrSummaryReady").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(root.GetProperty("isRealLinuxRunnerProof").GetBoolean());
        Assert.Equal("blocked-evidence-incomplete", root.GetProperty("releaseEvidenceBundleState").GetString());
        Assert.False(root.GetProperty("isReleaseEvidenceComplete").GetBoolean());
        Assert.Equal("package-evidence-owner-review-required", root.GetProperty("releasePackageProofState").GetString());
        Assert.False(root.GetProperty("canUseAsPublicPackageProof").GetBoolean());
        Assert.False(root.GetProperty("packageProofIsRuntimeExecutionProof").GetBoolean());
        Assert.Equal("owner-review-required", root.GetProperty("finalPackageReviewState").GetString());
        Assert.True(root.GetProperty("finalPackageReviewPackageCount").GetInt32() >= 1);
        Assert.True(root.GetProperty("finalPackageReviewNativeAssetCount").GetInt32() > 0);
        Assert.False(root.GetProperty("finalPackageReviewCanUseAsPublicPackageProof").GetBoolean());
        Assert.Equal("ready-for-owner-review", root.GetProperty("docsPublishReadinessState").GetString());
        Assert.False(root.GetProperty("canPublishDocsExternally").GetBoolean());
        Assert.True(root.GetProperty("docsArticleCount").GetInt32() >= 30);
        Assert.Equal("blocked-compatible-host-proof-required", root.GetProperty("externalRuntimeProofBackfillPlanState").GetString());
        Assert.True(root.GetProperty("externalRuntimeProofBackfillStepCount").GetInt32() >= 7);
        Assert.False(root.GetProperty("externalRuntimeProofBackfillCanPromoteRuntimeProof").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("externalRuntimeProofCollectionPackageState").GetString());
        Assert.True(root.GetProperty("externalRuntimeProofCollectionPackageStepCount").GetInt32() >= 8);
        Assert.True(root.GetProperty("externalRuntimeProofCollectionPackageExecutionOrderCount").GetInt32() >= 5);
        Assert.False(root.GetProperty("externalRuntimeProofCollectionPackageCanPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofCollectionPackageCanCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("externalRuntimeProofCollectionPackageRuntimeExecutionEvidence").GetBoolean());
        Assert.Equal("blocked-real-post-publish-proof-required", root.GetProperty("postPublishVerificationBackfillPlanState").GetString());
        Assert.True(root.GetProperty("postPublishVerificationBackfillStepCount").GetInt32() >= 9);
        Assert.False(root.GetProperty("postPublishVerificationBackfillCanCloseReleaseIssue").GetBoolean());
        Assert.Equal("blocked-real-publication-required", root.GetProperty("postPublishVerificationCollectionPackageState").GetString());
        Assert.True(root.GetProperty("postPublishVerificationCollectionPackageStepCount").GetInt32() >= 8);
        Assert.True(root.GetProperty("postPublishVerificationCollectionPackageExecutionOrderCount").GetInt32() >= 5);
        Assert.False(root.GetProperty("postPublishVerificationCollectionPackageProof").GetBoolean());
        Assert.False(root.GetProperty("postPublishVerificationCollectionPackageCanCloseReleaseIssue").GetBoolean());
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-evidence-bundle.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/final-package-review-bundle.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-package-proof-bundle.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/docs-publish-readiness-bundle.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/external-runtime-proof-validation.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/post-publish-verification-validation.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/external-runtime-proof-backfill-plan.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/post-publish-verification-backfill-plan.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/external-runtime-proof-collection-package.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/post-publish-verification-collection-package.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/real-model-and-package-proof-input-package.json");
        Assert.Contains(root.GetProperty("promotionItems").EnumerateArray(), static item => item.GetProperty("id").GetString() == "nvidia-redistribution");
        Assert.Contains(root.GetProperty("promotionItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "release-evidence-bundle" &&
            item.GetProperty("requiredEvidence").GetString()!.Contains("release-evidence-bundle.json", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("blocked-evidence-incomplete", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("promotionItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "final-package-review-bundle" &&
            item.GetProperty("requiredEvidence").GetString()!.Contains("final-package-review-bundle.json", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("owner-review-required", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("canUseAsPublicPackageProof=False", StringComparison.Ordinal) &&
            item.GetProperty("boundary").GetString()!.Contains("not public channel proof", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("promotionItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "release-package-proof-bundle" &&
            item.GetProperty("requiredEvidence").GetString()!.Contains("release-package-proof-bundle.json", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("canUseAsPublicPackageProof=False", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("promotionItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "docs-publish-readiness-bundle" &&
            item.GetProperty("requiredEvidence").GetString()!.Contains("docs-publish-readiness-bundle.json", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("canPublishDocsExternally=False", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("promotionItems").EnumerateArray(), static item => item.GetProperty("id").GetString() == "owner-approval-input");
        Assert.Contains(root.GetProperty("promotionItems").EnumerateArray(), static item => item.GetProperty("id").GetString() == "publish-execution-checklist");
        Assert.Contains(root.GetProperty("promotionItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "external-runtime-proof-record" &&
            item.GetProperty("requiredEvidence").GetString()!.Contains("external-runtime-proof-validation.json", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("proofState=template-only", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("consumerProjectIdentityReady=False", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("smokeCommandRuntimeKeyReady=False", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("logSha256Matches=False", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("ownerAction=owner-action-required", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("promotionItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "external-runtime-proof-backfill-plan" &&
            item.GetProperty("currentStatus").GetString()!.Contains("planState=blocked-compatible-host-proof-required", StringComparison.Ordinal) &&
            item.GetProperty("boundary").GetString()!.Contains("guidance only", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("promotionItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-verification-record" &&
            item.GetProperty("requiredEvidence").GetString()!.Contains("post-publish-verification-validation.json", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("verificationState=template-only", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("consumerProjectIdentityReady=False", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("smokeCommandRuntimeKeyReady=False", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("hostReady=False", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("commandsReady=False", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("stdoutStderrSummaryReady=False", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("canCloseReleaseIssue=False", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("promotionItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-verification-backfill-plan" &&
            item.GetProperty("currentStatus").GetString()!.Contains("planState=blocked-real-post-publish-proof-required", StringComparison.Ordinal) &&
            item.GetProperty("boundary").GetString()!.Contains("guidance only", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("promotionItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "external-runtime-proof-collection-package" &&
            item.GetProperty("currentStatus").GetString()!.Contains("packageState=owner-action-required", StringComparison.Ordinal) &&
            item.GetProperty("boundary").GetString()!.Contains("guidance only", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("promotionItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-verification-collection-package" &&
            item.GetProperty("currentStatus").GetString()!.Contains("packageState=blocked-real-publication-required", StringComparison.Ordinal) &&
            item.GetProperty("boundary").GetString()!.Contains("guidance only", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("promotionItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-clean-consumer-project-scan" &&
            item.GetProperty("currentStatus").GetString()!.Contains("canCloseReleaseIssue=False", StringComparison.Ordinal) &&
            item.GetProperty("boundary").GetString()!.Contains("helper evidence only", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("promotionItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-verification-input-draft" &&
            item.GetProperty("currentStatus").GetString()!.Contains("isPostPublishVerificationProof=False", StringComparison.Ordinal) &&
            item.GetProperty("boundary").GetString()!.Contains("not post-publish proof", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("promotionItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "release-close-preflight" &&
            item.GetProperty("currentStatus").GetString()!.Contains("canCloseReleaseIssue=False", StringComparison.Ordinal) &&
            item.GetProperty("boundary").GetString()!.Contains("not owner authorization", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("promotionItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "runtime-smoke" &&
            item.GetProperty("currentStatus").GetString()!.Contains("runtimeProofRequiredForRelease=True", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("ownerAction=owner-action-required", StringComparison.Ordinal) &&
            item.GetProperty("boundary").GetString()!.Contains("runtime-deserialization-dependency-diagnostics", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("channelOptions").EnumerateArray(), static channel => channel.GetProperty("id").GetString() == "nuget-org");
        Assert.Contains(root.GetProperty("channelOptions").EnumerateArray(), static channel => channel.GetProperty("boundary").GetString()!.Contains("does not execute dotnet nuget push", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("postPublishVerification").EnumerateArray(), static item => item.GetString()!.Contains("without ProjectReference", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("postPublishVerification").EnumerateArray(), static item => item.GetString()!.Contains("consumerProjectName", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("postPublishVerification").EnumerateArray(), static item => item.GetString()!.Contains("--runtime-package-key", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("postPublishVerification").EnumerateArray(), static item => item.GetString()!.Contains("stdoutSummary", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("final package review bundle is local package inventory", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("Backfill plans are guidance only", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("Collection packages are copyable owner guidance only", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("clean consumer scan and input draft are helper artifacts only", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("Release close preflight aggregates real-proof gaps", StringComparison.Ordinal));
    }

    [Fact]
    public void ReleaseCandidateFreezeSummaryAndChecklistKeepOwnerBoundaryExplicit()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalRuntimeProofRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishVerificationRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePackageProofBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DocsPublishReadinessBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalReleaseDryRun.ps1"),
            "-RuntimePackageKey", "win-x64-trt11.0-cuda13.2-cudnn9.22",
            "-AllowRuntimeSmokeBlocked",
            "-WarnOnly");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseOwnerApprovalInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseOwnerApprovalInput.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseOwnerDecisionRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePublishExecutionChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePromotionIssueRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofBackfillPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationBackfillPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofCollectionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationCollectionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePromotionIssueRecord.ps1"));

        string summaryOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCandidateFreezeSummary.ps1"));
        string checklistOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCandidateFreezeChecklist.ps1"));
        string validationOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseCandidateFreezeSummary.ps1"));
        string ownerPlanOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerAuthorizedPublishCommandPlan.ps1"));
        string ownerPlanValidationOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerAuthorizedPublishCommandPlan.ps1"));

        Assert.Contains("Release candidate freeze summary written", summaryOutput, StringComparison.Ordinal);
        Assert.Contains("Release candidate freeze checklist written", checklistOutput, StringComparison.Ordinal);
        Assert.Contains("ValidationState=blocked-freeze-owner-action-required", validationOutput, StringComparison.Ordinal);
        Assert.Contains("Owner authorized publish command plan written", ownerPlanOutput, StringComparison.Ordinal);
        Assert.Contains("PlanState=blocked-owner-authorization-required", ownerPlanOutput, StringComparison.Ordinal);
        Assert.Contains("Owner authorized publish command plan validation written", ownerPlanValidationOutput, StringComparison.Ordinal);
        Assert.Contains("ValidationState=blocked-owner-authorization-required", ownerPlanValidationOutput, StringComparison.Ordinal);

        string summaryPath = Path.Combine(RepositoryPaths.Root, "artifacts", "release", "release-candidate-freeze-summary.json");
        using JsonDocument summary = JsonDocument.Parse(File.ReadAllText(summaryPath));
        JsonElement summaryRoot = summary.RootElement;

        Assert.Equal("release-candidate-freeze-summary", summaryRoot.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-freeze-owner-action-required", summaryRoot.GetProperty("freezeState").GetString());
        Assert.False(summaryRoot.GetProperty("performsPublish").GetBoolean());
        Assert.True(summaryRoot.GetProperty("requiresHumanOwner").GetBoolean());
        Assert.False(summaryRoot.GetProperty("canPublish").GetBoolean());
        Assert.False(summaryRoot.GetProperty("canPromote").GetBoolean());
        Assert.False(summaryRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.True(summaryRoot.GetProperty("closeReadinessConsistent").GetBoolean());
        Assert.False(summaryRoot.GetProperty("realExternalRuntimeProofReady").GetBoolean());
        Assert.False(summaryRoot.GetProperty("realPostPublishVerificationReady").GetBoolean());
        Assert.Contains(summaryRoot.GetProperty("externalRuntimeProofState").GetString(), new[] { "template-only", "draft-blocked-by-cuda-driver" });
        Assert.Equal("template-only", summaryRoot.GetProperty("postPublishVerificationState").GetString());
        Assert.False(summaryRoot.GetProperty("postPublishStdoutSummaryReady").GetBoolean());
        Assert.False(summaryRoot.GetProperty("postPublishStderrSummaryReady").GetBoolean());
        Assert.False(summaryRoot.GetProperty("postPublishAllLogSha256Matches").GetBoolean());
        AssertPostPublishRequiredEvidence(summaryRoot);
        Assert.Equal("blocked-by-cuda-driver", summaryRoot.GetProperty("runtimeProofStatus").GetString());
        Assert.Equal("owner-review-required", summaryRoot.GetProperty("finalPackageReviewState").GetString());
        Assert.True(summaryRoot.GetProperty("finalPackageReviewPackageCount").GetInt32() >= 1);
        Assert.True(summaryRoot.GetProperty("finalPackageReviewNativeAssetCount").GetInt32() > 0);
        Assert.False(summaryRoot.GetProperty("finalPackageReviewCanUseAsPublicPackageProof").GetBoolean());
        Assert.Equal("blocked-compatible-host-proof-required", summaryRoot.GetProperty("externalRuntimeProofBackfillPlanState").GetString());
        Assert.True(summaryRoot.GetProperty("externalRuntimeProofBackfillStepCount").GetInt32() >= 7);
        Assert.False(summaryRoot.GetProperty("externalRuntimeProofBackfillCanPromoteRuntimeProof").GetBoolean());
        Assert.Equal("owner-action-required", summaryRoot.GetProperty("externalRuntimeProofCollectionPackageState").GetString());
        Assert.True(summaryRoot.GetProperty("externalRuntimeProofCollectionPackageStepCount").GetInt32() >= 8);
        Assert.False(summaryRoot.GetProperty("externalRuntimeProofCollectionPackageCanPromoteRuntimeProof").GetBoolean());
        Assert.False(summaryRoot.GetProperty("externalRuntimeProofCollectionPackageCanCloseReleaseIssue").GetBoolean());
        Assert.False(summaryRoot.GetProperty("externalRuntimeProofCollectionPackageRuntimeExecutionEvidence").GetBoolean());
        Assert.Equal("blocked-real-post-publish-proof-required", summaryRoot.GetProperty("postPublishVerificationBackfillPlanState").GetString());
        Assert.True(summaryRoot.GetProperty("postPublishVerificationBackfillStepCount").GetInt32() >= 9);
        Assert.False(summaryRoot.GetProperty("postPublishVerificationBackfillCanCloseReleaseIssue").GetBoolean());
        Assert.Equal("blocked-real-publication-required", summaryRoot.GetProperty("postPublishVerificationCollectionPackageState").GetString());
        Assert.True(summaryRoot.GetProperty("postPublishVerificationCollectionPackageStepCount").GetInt32() >= 8);
        Assert.False(summaryRoot.GetProperty("postPublishVerificationCollectionPackageProof").GetBoolean());
        Assert.False(summaryRoot.GetProperty("postPublishVerificationCollectionPackageCanCloseReleaseIssue").GetBoolean());
        Assert.False(summaryRoot.GetProperty("postPublishCleanConsumerProjectScanIsProof").GetBoolean());
        Assert.False(summaryRoot.GetProperty("postPublishCleanConsumerProjectScanCanCloseReleaseIssue").GetBoolean());
        Assert.Equal("post-publish-verification-record-input-draft", summaryRoot.GetProperty("postPublishVerificationInputDraftKind").GetString());
        Assert.True(summaryRoot.GetProperty("postPublishVerificationInputDraftOnly").GetBoolean());
        Assert.False(summaryRoot.GetProperty("postPublishVerificationInputDraftIsProof").GetBoolean());
        Assert.False(summaryRoot.GetProperty("postPublishVerificationInputDraftCanCloseReleaseIssue").GetBoolean());
        Assert.Contains(
            summaryRoot.GetProperty("releaseClosePreflightState").GetString(),
            new[] { "missing-release-close-preflight", "blocked-real-proof-required" });
        Assert.False(summaryRoot.GetProperty("releaseClosePreflightCanCloseReleaseIssue").GetBoolean());

        string externalRuntimeProofValidationPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "external-runtime-proof-validation.json");
        using JsonDocument externalRuntimeProofValidation = JsonDocument.Parse(File.ReadAllText(externalRuntimeProofValidationPath));
        JsonElement externalRuntimeProofValidationRoot = externalRuntimeProofValidation.RootElement;

        Assert.False(externalRuntimeProofValidationRoot.GetProperty("stdoutSummaryReady").GetBoolean());
        Assert.False(externalRuntimeProofValidationRoot.GetProperty("stderrSummaryReady").GetBoolean());
        Assert.False(externalRuntimeProofValidationRoot.GetProperty("stdoutStderrSummariesReady").GetBoolean());
        Assert.Contains(externalRuntimeProofValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "stdout-summary" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("detail").GetString()!.Contains("reviewed stdoutSummary", StringComparison.Ordinal));
        Assert.Contains(externalRuntimeProofValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "stderr-summary" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("detail").GetString()!.Contains("no-stderr-emitted", StringComparison.Ordinal));
        Assert.Contains(externalRuntimeProofValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "stdout-stderr-summary" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("detail").GetString()!.Contains("Both stdoutSummary and stderrSummary", StringComparison.Ordinal));
        Assert.True(summaryRoot.GetProperty("blockingItemCount").GetInt32() >= 1);
        Assert.Contains(summaryRoot.GetProperty("blockingItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "real-external-runtime-proof" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("blocked-by-cuda-driver", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("blockingItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-verification" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("cannot close release issue", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("blockingItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "external-runtime-proof-backfill-required" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("currentStatus").GetString()!.Contains("planState=blocked-compatible-host-proof-required", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("blockingItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-verification-backfill-required" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("currentStatus").GetString()!.Contains("planState=blocked-real-post-publish-proof-required", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("blockingItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "external-runtime-proof-collection-package-boundary" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("currentStatus").GetString()!.Contains("packageState=owner-action-required", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("blockingItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-verification-collection-package-boundary" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("currentStatus").GetString()!.Contains("packageState=blocked-real-publication-required", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("blockingItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-clean-consumer-scan-boundary" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("helper evidence only", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("blockingItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-verification-input-draft-boundary" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("helper evidence only", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("blockingItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "release-close-preflight-boundary" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("aggregates real-proof gaps", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("artifactRefs").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "external-runtime-proof" &&
            !item.GetProperty("ready").GetBoolean());
        Assert.Contains(summaryRoot.GetProperty("artifactRefs").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "external-runtime-proof-backfill-plan" &&
            !item.GetProperty("ready").GetBoolean());
        Assert.Contains(summaryRoot.GetProperty("artifactRefs").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-verification-backfill-plan" &&
            !item.GetProperty("ready").GetBoolean());
        Assert.Contains(summaryRoot.GetProperty("artifactRefs").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "external-runtime-proof-collection-package" &&
            !item.GetProperty("ready").GetBoolean());
        Assert.Contains(summaryRoot.GetProperty("artifactRefs").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-verification-collection-package" &&
            !item.GetProperty("ready").GetBoolean());
        Assert.Contains(summaryRoot.GetProperty("artifactRefs").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-clean-consumer-project-scan" &&
            !item.GetProperty("ready").GetBoolean());
        Assert.Contains(summaryRoot.GetProperty("artifactRefs").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-verification-input-draft" &&
            !item.GetProperty("ready").GetBoolean());
        Assert.Contains(summaryRoot.GetProperty("artifactRefs").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "release-close-preflight" &&
            !item.GetProperty("ready").GetBoolean());
        Assert.Contains(summaryRoot.GetProperty("artifactRefs").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "final-package-review-bundle" &&
            item.GetProperty("path").GetString()!.Contains("final-package-review-bundle.json", StringComparison.Ordinal) &&
            !item.GetProperty("ready").GetBoolean());
        Assert.Contains(summaryRoot.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString()!.Contains("final-package-review-bundle.json", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString()!.Contains("external-runtime-proof-backfill-plan.json", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString()!.Contains("post-publish-verification-backfill-plan.json", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString()!.Contains("post-publish-clean-consumer-project-scan.json", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString()!.Contains("post-publish-verification-record.input-draft.json", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString()!.Contains("release-close-preflight.json", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString()!.Contains("external-runtime-proof-collection-package.json", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString()!.Contains("post-publish-verification-collection-package.json", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("does not execute dotnet nuget push", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("final package review bundle is local package inventory", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(summaryRoot.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("Backfill plans and collection packages are guidance only", StringComparison.Ordinal));
        Assert.Contains(summaryRoot.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("clean consumer scan and input draft are helper artifacts only", StringComparison.Ordinal));

        string checklistPath = Path.Combine(RepositoryPaths.Root, "artifacts", "release", "release-candidate-freeze-checklist.json");
        using JsonDocument checklist = JsonDocument.Parse(File.ReadAllText(checklistPath));
        JsonElement checklistRoot = checklist.RootElement;

        Assert.Equal("release-candidate-freeze-checklist", checklistRoot.GetProperty("recordKind").GetString());
        Assert.False(checklistRoot.GetProperty("performsPublish").GetBoolean());
        Assert.True(checklistRoot.GetProperty("requiresHumanOwner").GetBoolean());
        Assert.False(checklistRoot.GetProperty("canPublish").GetBoolean());
        Assert.False(checklistRoot.GetProperty("canPromote").GetBoolean());
        Assert.False(checklistRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains(checklistRoot.GetProperty("ownerDecisionItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "external-runtime-proof" &&
            item.GetProperty("boundary").GetString()!.Contains("blocked-by-cuda-driver", StringComparison.Ordinal));
        Assert.Contains(checklistRoot.GetProperty("ownerDecisionItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-verification" &&
            item.GetProperty("currentStatus").GetString()!.Contains("canCloseReleaseIssue=False", StringComparison.Ordinal));
        Assert.Contains(checklistRoot.GetProperty("publishPlaceholders").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "nuget-org" &&
            item.GetProperty("command").GetString()!.Contains("dotnet nuget push", StringComparison.Ordinal) &&
            !item.GetProperty("performsPublish").GetBoolean());
        Assert.Contains(checklistRoot.GetProperty("postPublishActions").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "stdout-stderr-summary" &&
            item.GetProperty("boundary").GetString()!.Contains("stdout/stderr", StringComparison.Ordinal));
        Assert.Contains(checklistRoot.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString()!.Contains("external-runtime-proof-collection-package.json", StringComparison.Ordinal));
        Assert.Contains(checklistRoot.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString()!.Contains("post-publish-verification-collection-package.json", StringComparison.Ordinal));
        Assert.Contains(checklistRoot.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("Collection packages are copyable owner guidance only", StringComparison.Ordinal));

        string validationPath = Path.Combine(RepositoryPaths.Root, "artifacts", "release", "release-candidate-freeze-validation.json");
        using JsonDocument validation = JsonDocument.Parse(File.ReadAllText(validationPath));
        JsonElement validationRoot = validation.RootElement;

        Assert.Equal("release-candidate-freeze-summary-validation", validationRoot.GetProperty("validationKind").GetString());
        Assert.Equal("blocked-freeze-owner-action-required", validationRoot.GetProperty("validationState").GetString());
        Assert.Equal(0, validationRoot.GetProperty("failedValidationItemCount").GetInt32());
        Assert.False(validationRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validationRoot.GetProperty("realExternalRuntimeProofReady").GetBoolean());
        Assert.False(validationRoot.GetProperty("realPostPublishVerificationReady").GetBoolean());
        Assert.Equal("blocked-compatible-host-proof-required", validationRoot.GetProperty("externalRuntimeProofBackfillPlanState").GetString());
        Assert.True(validationRoot.GetProperty("externalRuntimeProofBackfillStepCount").GetInt32() >= 7);
        Assert.Equal("blocked-real-post-publish-proof-required", validationRoot.GetProperty("postPublishVerificationBackfillPlanState").GetString());
        Assert.True(validationRoot.GetProperty("postPublishVerificationBackfillStepCount").GetInt32() >= 9);
        Assert.Contains(validationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "external-backfill-plan-visible-and-blocked" &&
            item.GetProperty("passed").GetBoolean());
        Assert.Contains(validationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-backfill-plan-visible-and-blocked" &&
            item.GetProperty("passed").GetBoolean());
        Assert.All(validationRoot.GetProperty("validationItems").EnumerateArray(), item => Assert.True(item.GetProperty("passed").GetBoolean()));

        string ownerPlanPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-authorized-publish-command-plan.json");
        using JsonDocument ownerPlan = JsonDocument.Parse(File.ReadAllText(ownerPlanPath));
        JsonElement ownerPlanRoot = ownerPlan.RootElement;

        Assert.Equal("owner-authorized-publish-command-plan", ownerPlanRoot.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-authorization-required", ownerPlanRoot.GetProperty("planState").GetString());
        Assert.False(ownerPlanRoot.GetProperty("performsPublish").GetBoolean());
        Assert.True(ownerPlanRoot.GetProperty("requiresHumanOwner").GetBoolean());
        Assert.True(ownerPlanRoot.GetProperty("requiresExplicitOwnerAuthorization").GetBoolean());
        Assert.False(ownerPlanRoot.GetProperty("canMaterializeExecutableCommands").GetBoolean());
        Assert.False(ownerPlanRoot.GetProperty("realExternalRuntimeProofReady").GetBoolean());
        Assert.False(ownerPlanRoot.GetProperty("realPostPublishVerificationReady").GetBoolean());
        Assert.Equal("owner-review-required", ownerPlanRoot.GetProperty("finalPackageReviewState").GetString());
        Assert.True(ownerPlanRoot.GetProperty("finalPackageReviewPackageCount").GetInt32() >= 1);
        Assert.True(ownerPlanRoot.GetProperty("finalPackageReviewNativeAssetCount").GetInt32() > 0);
        Assert.False(ownerPlanRoot.GetProperty("finalPackageReviewCanUseAsPublicPackageProof").GetBoolean());
        Assert.Equal("blocked-compatible-host-proof-required", ownerPlanRoot.GetProperty("externalRuntimeProofBackfillPlanState").GetString());
        Assert.True(ownerPlanRoot.GetProperty("externalRuntimeProofBackfillStepCount").GetInt32() >= 7);
        Assert.False(ownerPlanRoot.GetProperty("externalRuntimeProofBackfillCanPromoteRuntimeProof").GetBoolean());
        Assert.Equal("owner-action-required", ownerPlanRoot.GetProperty("externalRuntimeProofCollectionPackageState").GetString());
        Assert.True(ownerPlanRoot.GetProperty("externalRuntimeProofCollectionPackageStepCount").GetInt32() >= 8);
        Assert.False(ownerPlanRoot.GetProperty("externalRuntimeProofCollectionPackageCanPromoteRuntimeProof").GetBoolean());
        Assert.False(ownerPlanRoot.GetProperty("externalRuntimeProofCollectionPackageCanCloseReleaseIssue").GetBoolean());
        Assert.False(ownerPlanRoot.GetProperty("externalRuntimeProofCollectionPackageRuntimeExecutionEvidence").GetBoolean());
        Assert.Equal("blocked-real-post-publish-proof-required", ownerPlanRoot.GetProperty("postPublishVerificationBackfillPlanState").GetString());
        Assert.True(ownerPlanRoot.GetProperty("postPublishVerificationBackfillStepCount").GetInt32() >= 9);
        Assert.False(ownerPlanRoot.GetProperty("postPublishVerificationBackfillCanCloseReleaseIssue").GetBoolean());
        Assert.Equal("blocked-real-publication-required", ownerPlanRoot.GetProperty("postPublishVerificationCollectionPackageState").GetString());
        Assert.True(ownerPlanRoot.GetProperty("postPublishVerificationCollectionPackageStepCount").GetInt32() >= 8);
        Assert.False(ownerPlanRoot.GetProperty("postPublishVerificationCollectionPackageProof").GetBoolean());
        Assert.False(ownerPlanRoot.GetProperty("postPublishVerificationCollectionPackageCanCloseReleaseIssue").GetBoolean());
        Assert.Contains(ownerPlanRoot.GetProperty("blockingReasons").EnumerateArray(), static reason =>
            reason.GetString()!.Contains("blocked-by-cuda-driver is not smoke passed", StringComparison.Ordinal));
        Assert.Contains(ownerPlanRoot.GetProperty("publishCommands").EnumerateArray(), static command =>
            command.GetProperty("command").GetString()!.Contains("dotnet nuget push", StringComparison.Ordinal) &&
            !command.GetProperty("authorized").GetBoolean() &&
            !command.GetProperty("executable").GetBoolean() &&
            !command.GetProperty("performsPublish").GetBoolean());
        Assert.Contains(ownerPlanRoot.GetProperty("publishCommands").EnumerateArray(), static command =>
            command.GetProperty("command").GetString()!.Contains("gh release upload", StringComparison.Ordinal) &&
            !command.GetProperty("authorized").GetBoolean() &&
            !command.GetProperty("executable").GetBoolean());
        Assert.Contains(ownerPlanRoot.GetProperty("postPublishCommandPlan").EnumerateArray(), static command =>
            command.GetProperty("id").GetString() == "run-runtime-smoke" &&
            command.GetProperty("command").GetString()!.Contains("--runtime-package-key", StringComparison.Ordinal));
        Assert.Contains(ownerPlanRoot.GetProperty("postPublishCommandPlan").EnumerateArray(), static command =>
            command.GetProperty("id").GetString() == "validate-post-publish-record" &&
            command.GetProperty("command").GetString()!.Contains("Test-PostPublishVerificationRecord.ps1", StringComparison.Ordinal));
        Assert.Contains(ownerPlanRoot.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("does not run dotnet nuget push", StringComparison.Ordinal));
        Assert.Contains(ownerPlanRoot.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString()!.Contains("final-package-review-bundle.json", StringComparison.Ordinal));
        Assert.Contains(ownerPlanRoot.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString()!.Contains("external-runtime-proof-backfill-plan.json", StringComparison.Ordinal));
        Assert.Contains(ownerPlanRoot.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString()!.Contains("post-publish-verification-backfill-plan.json", StringComparison.Ordinal));
        Assert.Contains(ownerPlanRoot.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString()!.Contains("external-runtime-proof-collection-package.json", StringComparison.Ordinal));
        Assert.Contains(ownerPlanRoot.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString()!.Contains("post-publish-verification-collection-package.json", StringComparison.Ordinal));
        Assert.Contains(ownerPlanRoot.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("final package review bundle is local package inventory", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(ownerPlanRoot.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("Backfill plans and collection packages are guidance only", StringComparison.Ordinal));

        string[] ownerAuthorizationRequiredFields = ownerPlanRoot.GetProperty("ownerAuthorizationRequiredFields").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.True(ownerAuthorizationRequiredFields.Length >= 12);
        Assert.Contains("ownerName", ownerAuthorizationRequiredFields);
        Assert.Contains("approvalTimestampUtc", ownerAuthorizationRequiredFields);
        Assert.Contains("targetChannel", ownerAuthorizationRequiredFields);
        Assert.Contains("approvedCommandPlanSha256", ownerAuthorizationRequiredFields);
        Assert.Contains("approvedProofBundleSha256", ownerAuthorizationRequiredFields);
        Assert.Contains("rollbackPlan", ownerAuthorizationRequiredFields);
        Assert.Contains("credentialHandlingAcknowledged", ownerAuthorizationRequiredFields);
        Assert.Contains("nvidiaRedistributionApproval", ownerAuthorizationRequiredFields);

        JsonElement ownerAuthorizationProofGate = ownerPlanRoot.GetProperty("ownerAuthorizationProofGate");
        Assert.Equal("blocked-owner-authorization-required", ownerAuthorizationProofGate.GetProperty("gateState").GetString());
        Assert.Equal(ownerAuthorizationRequiredFields.Length, ownerAuthorizationProofGate.GetProperty("requiredFieldCount").GetInt32());
        Assert.Equal(ownerAuthorizationRequiredFields.Length, ownerAuthorizationProofGate.GetProperty("missingOwnerInputCount").GetInt32());
        Assert.False(ownerAuthorizationProofGate.GetProperty("canMaterializeExecutableCommands").GetBoolean());
        Assert.False(ownerAuthorizationProofGate.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(ownerAuthorizationProofGate.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains(ownerAuthorizationProofGate.GetProperty("requiredValidators").EnumerateArray(), static item => item.GetString() == "Test-ReleaseOwnerApprovalInput.ps1");
        Assert.Contains(ownerAuthorizationProofGate.GetProperty("requiredValidators").EnumerateArray(), static item => item.GetString() == "Test-OwnerAuthorizedPublishCommandPlan.ps1");
        Assert.Contains(ownerAuthorizationProofGate.GetProperty("requiredArtifacts").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-owner-approval-input-record.json");
        Assert.Contains(ownerAuthorizationProofGate.GetProperty("requiredArtifacts").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/owner-authorized-publish-command-plan-validation.json");

        JsonElement[] manualMaterializationPrerequisites = ownerPlanRoot.GetProperty("manualMaterializationPrerequisites").EnumerateArray().ToArray();
        Assert.Contains(manualMaterializationPrerequisites, static item => item.GetProperty("id").GetString() == "owner-authorization" && item.GetProperty("blocksMaterialization").GetBoolean());
        Assert.Contains(manualMaterializationPrerequisites, static item => item.GetProperty("id").GetString() == "owner-decision" && item.GetProperty("blocksMaterialization").GetBoolean());
        Assert.Contains(manualMaterializationPrerequisites, static item => item.GetProperty("id").GetString() == "package-consumer-runtime-proof" && item.GetProperty("blocksMaterialization").GetBoolean());
        Assert.Contains(manualMaterializationPrerequisites, static item => item.GetProperty("id").GetString() == "publish-checklist");
        Assert.Contains(manualMaterializationPrerequisites, static item => item.GetProperty("id").GetString() == "stale-release-claims");

        string[] postPublishRequiredEvidence = ownerPlanRoot.GetProperty("postPublishRequiredEvidence").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("selectedChannel", postPublishRequiredEvidence);
        Assert.Contains("channelSourceUri", postPublishRequiredEvidence);
        Assert.Contains("managedNupkgSha256", postPublishRequiredEvidence);
        Assert.Contains("runtimeNupkgSha256", postPublishRequiredEvidence);
        Assert.Contains("cleanConsumerRootOutsideRepository", postPublishRequiredEvidence);
        Assert.Contains("noProjectReference", postPublishRequiredEvidence);
        Assert.Contains("runtimeSmokeLogSha256", postPublishRequiredEvidence);
        Assert.Contains("stdoutSummary", postPublishRequiredEvidence);
        Assert.Contains("stderrSummary", postPublishRequiredEvidence);
        Assert.Contains("hostMetadata", postPublishRequiredEvidence);

        string ownerPlanValidationPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-authorized-publish-command-plan-validation.json");
        using JsonDocument ownerPlanValidation = JsonDocument.Parse(File.ReadAllText(ownerPlanValidationPath));
        JsonElement ownerPlanValidationRoot = ownerPlanValidation.RootElement;

        Assert.Equal("owner-authorized-publish-command-plan-validation", ownerPlanValidationRoot.GetProperty("validationKind").GetString());
        Assert.Equal("blocked-owner-authorization-required", ownerPlanValidationRoot.GetProperty("validationState").GetString());
        Assert.Equal(0, ownerPlanValidationRoot.GetProperty("failedValidationItemCount").GetInt32());
        Assert.False(ownerPlanValidationRoot.GetProperty("performsPublish").GetBoolean());
        Assert.False(ownerPlanValidationRoot.GetProperty("canMaterializeExecutableCommands").GetBoolean());
        Assert.True(ownerPlanValidationRoot.GetProperty("publishCommandCount").GetInt32() >= 5);
        Assert.Equal("blocked-compatible-host-proof-required", ownerPlanValidationRoot.GetProperty("externalRuntimeProofBackfillPlanState").GetString());
        Assert.True(ownerPlanValidationRoot.GetProperty("externalRuntimeProofBackfillStepCount").GetInt32() >= 7);
        Assert.Equal("owner-action-required", ownerPlanValidationRoot.GetProperty("externalRuntimeProofCollectionPackageState").GetString());
        Assert.True(ownerPlanValidationRoot.GetProperty("externalRuntimeProofCollectionPackageStepCount").GetInt32() >= 8);
        Assert.Equal("blocked-real-post-publish-proof-required", ownerPlanValidationRoot.GetProperty("postPublishVerificationBackfillPlanState").GetString());
        Assert.True(ownerPlanValidationRoot.GetProperty("postPublishVerificationBackfillStepCount").GetInt32() >= 9);
        Assert.Equal("blocked-real-publication-required", ownerPlanValidationRoot.GetProperty("postPublishVerificationCollectionPackageState").GetString());
        Assert.True(ownerPlanValidationRoot.GetProperty("postPublishVerificationCollectionPackageStepCount").GetInt32() >= 8);
        Assert.True(ownerPlanValidationRoot.GetProperty("ownerAuthorizationRequiredFieldCount").GetInt32() >= 12);
        Assert.Equal("blocked-owner-authorization-required", ownerPlanValidationRoot.GetProperty("ownerAuthorizationProofGateState").GetString());
        Assert.True(ownerPlanValidationRoot.GetProperty("ownerAuthorizationProofGateMissingOwnerInputCount").GetInt32() >= 12);
        Assert.True(ownerPlanValidationRoot.GetProperty("manualMaterializationPrerequisiteCount").GetInt32() >= 6);
        Assert.True(ownerPlanValidationRoot.GetProperty("postPublishRequiredEvidenceCount").GetInt32() >= 10);
        Assert.Contains(ownerPlanValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "publish-commands-safe" &&
            item.GetProperty("passed").GetBoolean());
        Assert.Contains(ownerPlanValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "no-materialized-push-command" &&
            item.GetProperty("passed").GetBoolean());
        Assert.True(ownerPlanValidationRoot.GetProperty("publishCommandsPlaceholderOnly").GetBoolean());
        Assert.Contains(ownerPlanValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "backfill-plan-source-evidence-visible" &&
            item.GetProperty("passed").GetBoolean());
        Assert.Contains(ownerPlanValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "collection-package-source-evidence-visible" &&
            item.GetProperty("passed").GetBoolean());
        Assert.Contains(ownerPlanValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "external-collection-package-blocked" &&
            item.GetProperty("passed").GetBoolean());
        Assert.Contains(ownerPlanValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-collection-package-blocked" &&
            item.GetProperty("passed").GetBoolean());
        Assert.Contains(ownerPlanValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "owner-authorization-required-fields-visible" &&
            item.GetProperty("passed").GetBoolean());
        Assert.Contains(ownerPlanValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "owner-authorization-proof-gate-blocked" &&
            item.GetProperty("passed").GetBoolean());
        Assert.Contains(ownerPlanValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "owner-authorization-proof-gate-sources" &&
            item.GetProperty("passed").GetBoolean());
        Assert.Contains(ownerPlanValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "manual-materialization-prerequisites-visible" &&
            item.GetProperty("passed").GetBoolean());
        Assert.Contains(ownerPlanValidationRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-required-evidence-visible" &&
            item.GetProperty("passed").GetBoolean());

        string externalBackfillPlanPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "external-runtime-proof-backfill-plan.json");
        using JsonDocument externalBackfillPlan = JsonDocument.Parse(File.ReadAllText(externalBackfillPlanPath));
        JsonElement externalBackfillRoot = externalBackfillPlan.RootElement;

        Assert.Equal("external-runtime-proof-backfill-plan", externalBackfillRoot.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-compatible-host-proof-required", externalBackfillRoot.GetProperty("planState").GetString());
        Assert.False(externalBackfillRoot.GetProperty("performsPublish").GetBoolean());
        Assert.False(externalBackfillRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(externalBackfillRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(externalBackfillRoot.GetProperty("finalPackageReviewCanUseAsPublicPackageProof").GetBoolean());
        Assert.Contains(externalBackfillRoot.GetProperty("backfillSteps").EnumerateArray(), static step =>
            step.GetProperty("id").GetString() == "validate-real-record" &&
            step.GetProperty("command").GetString()!.Contains("-RequireExistingLog -FailOnNotProof", StringComparison.Ordinal));
        Assert.Contains(externalBackfillRoot.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("blocked-by-cuda-driver is not smoke passed", StringComparison.Ordinal));

        string postPublishBackfillPlanPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "post-publish-verification-backfill-plan.json");
        using JsonDocument postPublishBackfillPlan = JsonDocument.Parse(File.ReadAllText(postPublishBackfillPlanPath));
        JsonElement postPublishBackfillRoot = postPublishBackfillPlan.RootElement;

        Assert.Equal("post-publish-verification-backfill-plan", postPublishBackfillRoot.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-post-publish-proof-required", postPublishBackfillRoot.GetProperty("planState").GetString());
        Assert.False(postPublishBackfillRoot.GetProperty("performsPublish").GetBoolean());
        Assert.False(postPublishBackfillRoot.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(postPublishBackfillRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(postPublishBackfillRoot.GetProperty("finalPackageReviewCanUseAsPublicPackageProof").GetBoolean());
        Assert.Contains(postPublishBackfillRoot.GetProperty("backfillSteps").EnumerateArray(), static step =>
            step.GetProperty("id").GetString() == "validate-post-publish-record" &&
            step.GetProperty("command").GetString()!.Contains("-RequireExistingLog -FailOnNotProof", StringComparison.Ordinal));
        Assert.Contains(postPublishBackfillRoot.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("ProjectReference", StringComparison.Ordinal));
    }

    [Fact]
    public void LinuxRunnerEvidenceTemplateDoesNotClaimRealRunnerProof()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-LinuxRunnerEvidenceTemplate.ps1");
        string output = RunPowerShell(script, "-RuntimePackageKey", "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22");

        Assert.Contains("Linux runner evidence template written", output, StringComparison.Ordinal);
        Assert.Contains("Linux runner issue template written", output, StringComparison.Ordinal);

        string templatePath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "linux-dry-run",
            "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22",
            "linux-runner-evidence-template.json");
        using JsonDocument template = JsonDocument.Parse(File.ReadAllText(templatePath));
        JsonElement root = template.RootElement;

        Assert.Equal("template-only", root.GetProperty("evidenceState").GetString());
        Assert.False(root.GetProperty("isRealLinuxRunnerProof").GetBoolean());
        Assert.Contains(root.GetProperty("commands").EnumerateArray(), static command => command.GetString()!.Contains("Test-PackageConsumer.ps1", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("promotionRules").EnumerateArray(), static rule => rule.GetString()!.Contains("InvocationCount>0", StringComparison.Ordinal));

        string issuePath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "linux-dry-run",
            "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22",
            "linux-runner-issue-template.md");
        string issue = File.ReadAllText(issuePath);

        Assert.Contains("Linux Runner Validation Issue Template", issue, StringComparison.Ordinal);
        Assert.Contains("isRealLinuxRunnerProof=false", issue, StringComparison.Ordinal);
        Assert.Contains("dry-run-only", issue, StringComparison.Ordinal);
    }

    [Fact]
    public void LinuxRunnerEvidenceRecordTemplateKeepsRunnerProofFalseUntilFilled()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-LinuxRunnerEvidenceRecordTemplate.ps1");
        string output = RunPowerShell(script, "-RuntimePackageKey", "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22");

        Assert.Contains("Linux runner evidence record template written", output, StringComparison.Ordinal);

        string templatePath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "linux-dry-run",
            "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22",
            "linux-runner-evidence-record-template.json");
        using JsonDocument template = JsonDocument.Parse(File.ReadAllText(templatePath));
        JsonElement root = template.RootElement;

        Assert.Equal("linux-runner-evidence-record-template", root.GetProperty("recordKind").GetString());
        Assert.Equal("template-only", root.GetProperty("recordState").GetString());
        Assert.Equal("pending-linux-runner-execution", root.GetProperty("executionState").GetString());
        Assert.False(root.GetProperty("isRealLinuxRunnerProof").GetBoolean());
        Assert.False(root.GetProperty("canPromoteLinuxPackage").GetBoolean());
        Assert.Contains(root.GetProperty("commands").EnumerateArray(), static command => command.GetProperty("id").GetString() == "cmake-build");
        Assert.Contains(root.GetProperty("promotionRules").EnumerateArray(), static rule => rule.GetString()!.Contains("blocked-by-cuda-driver is not smoke passed", StringComparison.Ordinal));
    }

    [Fact]
    public void LinuxRunnerEvidenceRecordValidatorKeepsTemplateOnlyEvidenceOutOfProof()
    {
        RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Export-LinuxRunnerEvidenceRecordTemplate.ps1"),
            "-RuntimePackageKey", "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22");

        string script = Path.Combine(RepositoryPaths.Root, "eng", "Test-LinuxRunnerEvidenceRecord.ps1");
        string output = RunPowerShell(script, "-RuntimePackageKey", "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22");

        Assert.Contains("Linux runner evidence validation written", output, StringComparison.Ordinal);
        Assert.Contains("ValidationState=template-only", output, StringComparison.Ordinal);
        Assert.Contains("IsRealLinuxRunnerProof=False", output, StringComparison.Ordinal);

        string validationPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "linux-dry-run",
            "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22",
            "linux-runner-evidence-validation.json");
        using JsonDocument validation = JsonDocument.Parse(File.ReadAllText(validationPath));
        JsonElement root = validation.RootElement;

        Assert.Equal("linux-runner-evidence-validation", root.GetProperty("validationKind").GetString());
        Assert.Equal("template-only", root.GetProperty("validationState").GetString());
        Assert.True(root.GetProperty("isTemplateOnly").GetBoolean());
        Assert.False(root.GetProperty("isRealLinuxRunnerProof").GetBoolean());
        Assert.False(root.GetProperty("canPromoteLinuxPackage").GetBoolean());
        Assert.Contains(root.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "runner-linux-x64" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(root.GetProperty("promotionRules").EnumerateArray(), static rule => rule.GetString()!.Contains("Template-only records", StringComparison.Ordinal));
    }

    [Fact]
    public void StaleReleaseClaimsAuditStaysCleanForReleaseFacingDocs()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Test-StaleReleaseClaims.ps1");
        string output = RunPowerShell(script);

        Assert.Contains("Stale release claims audit written", output, StringComparison.Ordinal);

        string auditPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "stale-release-claims-audit.json");
        using JsonDocument audit = JsonDocument.Parse(File.ReadAllText(auditPath));
        JsonElement root = audit.RootElement;

        Assert.Equal(0, root.GetProperty("findingCount").GetInt32());
        Assert.True(root.GetProperty("scannedFileCount").GetInt32() > 0);
        Assert.Contains(root.GetProperty("rules").EnumerateArray(), static rule => rule.GetProperty("id").GetString() == "allow-runtime-smoke-blocked-ready");
        Assert.Contains(root.GetProperty("rules").EnumerateArray(), static rule => rule.GetProperty("id").GetString() == "dependency-probe-runtime-proof");
        Assert.Contains(root.GetProperty("rules").EnumerateArray(), static rule => rule.GetProperty("id").GetString() == "runtime-proof-complete");
        Assert.Contains(root.GetProperty("rules").EnumerateArray(), static rule => rule.GetProperty("id").GetString() == "post-publish-verified");
        Assert.Contains(root.GetProperty("rules").EnumerateArray(), static rule => rule.GetProperty("id").GetString() == "published-to-nuget");
        Assert.Contains(root.GetProperty("rules").EnumerateArray(), static rule => rule.GetProperty("id").GetString() == "package-consumer-runtime-passed");
        Assert.Contains(root.GetProperty("rules").EnumerateArray(), static rule => rule.GetProperty("id").GetString() == "can-publish-publicly-true");
        Assert.Contains(root.GetProperty("rules").EnumerateArray(), static rule => rule.GetProperty("id").GetString() == "can-close-release-issue-true");
        Assert.Contains(root.GetProperty("rules").EnumerateArray(), static rule => rule.GetProperty("id").GetString() == "sidecar-cn-close-release-issue");
        Assert.Contains(root.GetProperty("rules").EnumerateArray(), static rule => rule.GetProperty("id").GetString() == "sidecar-en-close-release-issue");
        Assert.Contains(root.GetProperty("rules").EnumerateArray(), static rule => rule.GetProperty("id").GetString() == "classification-passed-without-boundary");
        Assert.Contains(root.GetProperty("rules").EnumerateArray(), static rule => rule.GetProperty("id").GetString() == "yolovision-passed-without-boundary");
        Assert.Contains(root.GetProperty("rules").EnumerateArray(), static rule => rule.GetProperty("id").GetString() == "retired-sample-project-name");
        Assert.Contains(root.GetProperty("rules").EnumerateArray(), static rule => rule.GetProperty("id").GetString() == "retired-sample-project-file");
        Assert.Contains(root.GetProperty("rules").EnumerateArray(), static rule => rule.GetProperty("id").GetString() == "retired-sample-path");
        Assert.Contains(root.GetProperty("rules").EnumerateArray(), static rule => rule.GetProperty("id").GetString() == "retired-sample-path-windows");
    }

    [Fact]
    public void SampleAssetManifestAuditKeepsCandidatesSeparateFromSmokePasses()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Test-SampleAssetManifest.ps1");
        string output = RunPowerShell(script);

        Assert.Contains("Sample asset manifest audit written", output, StringComparison.Ordinal);

        string auditPath = Path.Combine(RepositoryPaths.Root, "artifacts", "user-acceptance", "sample-asset-manifest-audit.json");
        using JsonDocument audit = JsonDocument.Parse(File.ReadAllText(auditPath));
        JsonElement root = audit.RootElement;

        Assert.True(root.GetProperty("manifestCount").GetInt32() >= 8);
        Assert.Equal(
            root.GetProperty("manifestCount").GetInt32(),
            root.GetProperty("items").GetArrayLength());
        Assert.Equal(0, root.GetProperty("errorCount").GetInt32());
        Assert.Contains(root.GetProperty("sidecarCrossCheckRules").EnumerateArray(), static rule =>
            rule.GetString()!.Contains("modelSha256 and inputAssetSha256", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("sampleRunEvidenceCrossCheckRules").EnumerateArray(), static rule =>
            rule.GetString()!.Contains("modelSha256, labelsSha256, and inputAssetSha256", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("sampleProjectCrossCheckRules").EnumerateArray(), static rule =>
            rule.GetString()!.Contains("samples/<sampleName>/<sampleName>.csproj", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("allowedProofClassifications").EnumerateArray(), static item =>
            item.GetString() == "real-model-runtime");
        Assert.Contains(root.GetProperty("allowedProofClassifications").EnumerateArray(), static item =>
            item.GetString() == "package-consumer-runtime");

        JsonElement[] items = root.GetProperty("items").EnumerateArray().ToArray();
        Assert.Contains(items, static item =>
            item.GetProperty("sampleName").GetString() == "Classification" &&
            item.GetProperty("status").GetString() == "candidate-not-downloaded" &&
            item.GetProperty("proofClassification").GetString() == "template-only" &&
            item.GetProperty("sampleRunEvidenceRecord").GetString() == "models/classifier-sample-run-evidence.json" &&
            item.GetProperty("sampleRunEvidenceCrossCheckState").GetString() == "owner-action-required" &&
            item.GetProperty("sampleProjectExists").GetBoolean() &&
            item.GetProperty("sampleProjectNameMatches").GetBoolean() &&
            item.GetProperty("sampleProjectRelativePath").GetString() == "samples\\Classification\\Classification.csproj" &&
            !item.GetProperty("isSmokePassed").GetBoolean());
        Assert.Contains(items, static item =>
            item.GetProperty("sampleName").GetString() == "YoloVision" &&
            item.GetProperty("status").GetString() == "candidate-not-downloaded" &&
            item.GetProperty("proofClassification").GetString() == "template-only" &&
            item.GetProperty("sampleRunEvidenceRecord").GetString() == "models/yolovision-sample-run-evidence.json" &&
            item.GetProperty("sampleRunEvidenceCrossCheckState").GetString() == "owner-action-required" &&
            item.GetProperty("sampleProjectExists").GetBoolean() &&
            item.GetProperty("sampleProjectNameMatches").GetBoolean() &&
            item.GetProperty("sampleProjectRelativePath").GetString() == "samples\\YoloVision\\YoloVision.csproj" &&
            !item.GetProperty("isSmokePassed").GetBoolean());
    }

    [Fact]
    public void SampleAssetAcquisitionPlanRequiresOwnerActionAndDoesNotPromoteSamples()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-SampleAssetAcquisitionPlan.ps1");
        string output = RunPowerShell(script);

        Assert.Contains("Sample asset acquisition plan written", output, StringComparison.Ordinal);

        string planPath = Path.Combine(RepositoryPaths.Root, "artifacts", "user-acceptance", "sample-asset-acquisition-plan.json");
        using JsonDocument plan = JsonDocument.Parse(File.ReadAllText(planPath));
        JsonElement root = plan.RootElement;

        Assert.Equal("sample-asset-acquisition-plan", root.GetProperty("planKind").GetString());
        Assert.Equal("owner-action-required", root.GetProperty("planState").GetString());
        Assert.False(root.GetProperty("performsDownload").GetBoolean());
        Assert.False(root.GetProperty("performsSampleRun").GetBoolean());
        Assert.False(root.GetProperty("canPromoteSamples").GetBoolean());
        JsonElement[] items = root.GetProperty("items").EnumerateArray().ToArray();
        Assert.True(items.Length >= 8);
        Assert.Equal(items.Length, root.GetProperty("itemCount").GetInt32());
        Assert.Contains(items, static item =>
            item.GetProperty("sampleName").GetString() == "Classification" &&
            item.GetProperty("planState").GetString() == "owner-action-required" &&
            item.GetProperty("proofClassification").GetString() == "template-only" &&
            item.GetProperty("evidenceSidecar").GetString() == "models/classifier-evidence.sidecar.json" &&
            !item.GetProperty("isSmokePassed").GetBoolean());
        Assert.Contains(items, static item =>
            item.GetProperty("sampleName").GetString() == "YoloVision" &&
            item.GetProperty("planState").GetString() == "owner-action-required" &&
            item.GetProperty("proofClassification").GetString() == "template-only" &&
            item.GetProperty("evidenceSidecar").GetString() == "models/yolo-evidence.sidecar.json" &&
            !item.GetProperty("isSmokePassed").GetBoolean());
        Assert.Contains(items.SelectMany(static item => item.GetProperty("steps").EnumerateArray()), static step =>
            step.GetProperty("id").GetString() == "evidence-sidecar" &&
            step.GetProperty("boundary").GetString()!.Contains("cannot claim package-consumer-runtime", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("Evidence sidecars enrich reports", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("package-consumer-runtime belongs to release proof records", StringComparison.Ordinal));
    }

    [Fact]
    public void OnnxEngineBuildEvidenceSidecarAuditKeepsMissingTemplatesAsOwnerAction()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Test-OnnxEngineBuildEvidenceSidecar.ps1");
        string output = RunPowerShell(script);

        Assert.Contains("ONNX engine build evidence sidecar audit written", output, StringComparison.Ordinal);

        string auditPath = Path.Combine(RepositoryPaths.Root, "artifacts", "user-acceptance", "onnx-engine-build-evidence-sidecar-audit.json");
        using JsonDocument audit = JsonDocument.Parse(File.ReadAllText(auditPath));
        JsonElement root = audit.RootElement;

        Assert.Equal("onnx-engine-build-evidence-sidecar-audit", root.GetProperty("auditKind").GetString());
        Assert.Equal("manifest-scan", root.GetProperty("scanMode").GetString());
        Assert.Equal("owner-action-required", root.GetProperty("auditState").GetString());
        Assert.Equal(0, root.GetProperty("errorCount").GetInt32());
        Assert.True(root.GetProperty("ownerActionRequiredCount").GetInt32() > 0);
        Assert.False(root.GetProperty("items")[0].GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.Contains(root.GetProperty("allowedProofClassifications").EnumerateArray(), static item =>
            item.GetString() == "real-model-runtime");
        Assert.Contains(root.GetProperty("allowedProofClassifications").EnumerateArray(), static item =>
            item.GetString() == "package-consumer-runtime");
        Assert.Contains(root.GetProperty("findings").EnumerateArray(), static finding =>
            finding.GetProperty("ruleId").GetString() == "sidecar-file-missing" &&
            finding.GetProperty("severity").GetString() == "owner-action-required");
        Assert.Contains(root.GetProperty("sidecarRules").EnumerateArray(), static rule =>
            rule.GetString()!.Contains("Missing sidecars in sample asset templates are owner-action-required", StringComparison.Ordinal));
    }

    [Fact]
    public void OnnxEngineBuildEvidenceSidecarTemplateExportCreatesGenericAndSampleTemplates()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-OnnxEngineBuildEvidenceSidecarTemplate.ps1");
        string output = RunPowerShell(script);

        Assert.Contains("ONNX engine build evidence sidecar template written", output, StringComparison.Ordinal);

        string exportPath = Path.Combine(RepositoryPaths.Root, "artifacts", "user-acceptance", "onnx-engine-build-evidence-sidecar-template-export.json");
        using JsonDocument export = JsonDocument.Parse(File.ReadAllText(exportPath));
        JsonElement root = export.RootElement;

        Assert.Equal("onnx-engine-build-evidence-sidecar-template-export", root.GetProperty("exportKind").GetString());
        Assert.Equal("template-only", root.GetProperty("proofClassification").GetString());
        Assert.Equal(3, root.GetProperty("sampleSpecificTemplateCount").GetInt32());
        Assert.Contains(root.GetProperty("evidenceClassifications").EnumerateArray(), static item =>
            item.GetString() == "real-model-runtime");
        Assert.Contains(root.GetProperty("promotionRules").EnumerateArray(), static rule =>
            rule.GetString()!.Contains("Do not use a sidecar to promote", StringComparison.Ordinal));

        string genericPath = Path.Combine(RepositoryPaths.Root, "artifacts", "user-acceptance", "onnx-engine-build-evidence-sidecar.template.json");
        using JsonDocument generic = JsonDocument.Parse(File.ReadAllText(genericPath));
        Assert.Equal("onnx-engine-build-evidence-sidecar-template", generic.RootElement.GetProperty("recordKind").GetString());
        Assert.Equal("template-only", generic.RootElement.GetProperty("proofClassification").GetString());
        Assert.Equal("", generic.RootElement.GetProperty("modelEvidence").GetProperty("modelSha256").GetString());
        Assert.True(generic.RootElement.GetProperty("modelEvidence").TryGetProperty("preprocessedInputTensorName", out _));
        Assert.True(generic.RootElement.GetProperty("modelEvidence").TryGetProperty("preprocessedInputTensorSha256", out _));
        Assert.Contains(generic.RootElement.GetProperty("classificationRules").EnumerateArray(), static rule =>
            rule.GetString()!.Contains("package-consumer-runtime belongs to release proof records", StringComparison.Ordinal));
    }

    [Fact]
    public void SampleRunEvidenceRecordTemplateExportCreatesGenericAndSampleTemplates()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-SampleRunEvidenceRecordTemplate.ps1");
        string output = RunPowerShell(script);

        Assert.Contains("Sample run evidence record template written", output, StringComparison.Ordinal);

        string exportPath = Path.Combine(RepositoryPaths.Root, "artifacts", "user-acceptance", "sample-run-evidence-record-template-export.json");
        using JsonDocument export = JsonDocument.Parse(File.ReadAllText(exportPath));
        JsonElement root = export.RootElement;

        Assert.Equal("sample-run-evidence-record-template-export", root.GetProperty("exportKind").GetString());
        Assert.Equal("template-only", root.GetProperty("proofClassification").GetString());
        Assert.Equal(3, root.GetProperty("sampleSpecificTemplateCount").GetInt32());
        Assert.Contains(root.GetProperty("evidenceClassifications").EnumerateArray(), static item =>
            item.GetString() == "real-model-runtime");
        Assert.DoesNotContain(root.GetProperty("evidenceClassifications").EnumerateArray(), static item =>
            item.GetString() == "package-consumer-runtime");
        Assert.Contains(root.GetProperty("promotionRules").EnumerateArray(), static rule =>
            rule.GetString()!.Contains("package-consumer-runtime cannot be claimed", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("requiredEvidenceSummary").EnumerateArray(), static item =>
            item.GetString()!.Contains("modelLicense", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("requiredEvidenceSummary").EnumerateArray(), static item =>
            item.GetString()!.Contains("preprocessedInputTensorElementCount", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("promotionRules").EnumerateArray(), static rule =>
            rule.GetString()!.Contains("Do not fabricate modelLicense", StringComparison.Ordinal));

        string yoloPath = Path.Combine(RepositoryPaths.Root, "artifacts", "user-acceptance", "sample-run-evidence-record.yolox-s.template.json");
        using JsonDocument yolo = JsonDocument.Parse(File.ReadAllText(yoloPath));
        Assert.Equal("sample-run-evidence-record-template", yolo.RootElement.GetProperty("recordKind").GetString());
        Assert.Equal("YoloVision", yolo.RootElement.GetProperty("sampleName").GetString());
        Assert.Equal("template-only", yolo.RootElement.GetProperty("proofClassification").GetString());
        Assert.Equal(".\\models\\yolox_s-preprocessed-fp32.bin", yolo.RootElement.GetProperty("preprocessedInputTensorPath").GetString());
        Assert.True(yolo.RootElement.TryGetProperty("preprocessedInputTensorSha256", out _));
        Assert.Equal("owner-required", yolo.RootElement.GetProperty("modelLicense").GetString());
        Assert.Equal("owner-required", yolo.RootElement.GetProperty("labelsLicense").GetString());
        Assert.Equal("owner-required", yolo.RootElement.GetProperty("inputAssetLicense").GetString());
        Assert.Equal(string.Empty, yolo.RootElement.GetProperty("preprocessedInputTensorElementCount").GetString());
        Assert.Equal("owner-action-required", yolo.RootElement.GetProperty("validatorState").GetString());
        Assert.Contains(yolo.RootElement.GetProperty("failureReasons").EnumerateArray(), static reason =>
            reason.GetString()!.Contains("validator must pass", StringComparison.Ordinal));
        Assert.Contains("--input-data .\\models\\yolox_s-preprocessed-fp32.bin", yolo.RootElement.GetProperty("sampleRunCommand").GetString(), StringComparison.Ordinal);
        Assert.False(yolo.RootElement.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.Contains(yolo.RootElement.GetProperty("expectedEvidenceLines").EnumerateArray(), static line =>
            line.GetString()!.Contains("YoloVision Passed=True", StringComparison.Ordinal));
        Assert.Contains(yolo.RootElement.GetProperty("expectedEvidenceLines").EnumerateArray(), static line =>
            line.GetString()!.Contains("InputSource=external", StringComparison.Ordinal));
    }

    [Fact]
    public void SampleRunEvidenceRecordValidationKeepsTemplatesAsOwnerAction()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-SampleRunEvidenceRecordTemplate.ps1"));

        string script = Path.Combine(RepositoryPaths.Root, "eng", "Test-SampleRunEvidenceRecord.ps1");
        string output = RunPowerShell(script);

        Assert.Contains("Sample run evidence record validation written", output, StringComparison.Ordinal);

        string validationPath = Path.Combine(RepositoryPaths.Root, "artifacts", "user-acceptance", "sample-run-evidence-record-validation.json");
        using JsonDocument validation = JsonDocument.Parse(File.ReadAllText(validationPath));
        JsonElement root = validation.RootElement;

        Assert.Equal("sample-run-evidence-record-validation", root.GetProperty("validationKind").GetString());
        Assert.Equal("owner-action-required", root.GetProperty("validationState").GetString());
        Assert.True(root.GetProperty("templateOnly").GetBoolean());
        Assert.Equal("template-only", root.GetProperty("proofClassification").GetString());
        Assert.False(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.True(root.GetProperty("packageConsumerRuntimeForbidden").GetBoolean());
        Assert.Equal(0, root.GetProperty("errorCount").GetInt32());
        Assert.Contains(root.GetProperty("disallowedProofClassifications").EnumerateArray(), static item =>
            item.GetString() == "package-consumer-runtime");
        Assert.Contains(root.GetProperty("promotionRules").EnumerateArray(), static rule =>
            rule.GetString()!.Contains("package-consumer-runtime is forbidden", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "preprocessed-input-tensor-sha256");
        Assert.Contains(root.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "preprocessed-input-tensor-element-count");
        Assert.Contains(root.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "model-license");
        Assert.Contains(root.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "labels-license");
        Assert.Contains(root.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "input-asset-license");
        Assert.False(root.GetProperty("modelLicenseReady").GetBoolean());
        Assert.False(root.GetProperty("labelsLicenseReady").GetBoolean());
        Assert.False(root.GetProperty("inputAssetLicenseReady").GetBoolean());
        Assert.False(root.GetProperty("preprocessedInputTensorElementCountReady").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("validatorStateDeclared").GetString());
        Assert.True(root.GetProperty("validatorStateMatches").GetBoolean());
        Assert.True(root.TryGetProperty("failureReasons", out _));
        Assert.Contains(root.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "yolovision-external-input-evidence-line");
    }

    [Fact]
    public void OnnxEngineBuildEvidenceSidecarAuditRecordsPackageConsumerWithoutPromotingIt()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "jyppx-sidecar-boundary-" + Guid.NewGuid().ToString("N"));
        string sidecarPath = Path.Combine(tempRoot, "model-evidence.sidecar.json");
        string outputRoot = Path.Combine(tempRoot, "audit");
        Directory.CreateDirectory(tempRoot);
        File.WriteAllText(sidecarPath, """
{
  "proofClassification": "package-consumer-runtime",
  "stdoutSummary": "owner reviewed stdout",
  "stderrSummary": "owner reviewed stderr",
  "modelEvidence": {
    "modelSource": ".\\models\\model.onnx",
    "modelSha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "modelLicense": "Apache-2.0",
    "inputAssetName": ".\\models\\image.jpg",
    "inputAssetSha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  }
}
""");

        try
        {
            string script = Path.Combine(RepositoryPaths.Root, "eng", "Test-OnnxEngineBuildEvidenceSidecar.ps1");
            string output = RunPowerShell(script, "-SidecarPath", sidecarPath, "-OutputRoot", outputRoot);

            Assert.Contains("ONNX engine build evidence sidecar audit written", output, StringComparison.Ordinal);

            string auditPath = Path.Combine(outputRoot, "onnx-engine-build-evidence-sidecar-audit.json");
            using JsonDocument audit = JsonDocument.Parse(File.ReadAllText(auditPath));
            JsonElement root = audit.RootElement;
            JsonElement item = root.GetProperty("items")[0];

            Assert.Equal("single-sidecar", root.GetProperty("scanMode").GetString());
            Assert.Equal("package-consumer-runtime", item.GetProperty("proofClassification").GetString());
            Assert.False(item.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
            Assert.True(item.GetProperty("proofClassificationIsPackageConsumerRuntime").GetBoolean());
            Assert.Contains(root.GetProperty("findings").EnumerateArray(), static finding =>
                finding.GetProperty("ruleId").GetString() == "sidecar-package-consumer-record-only" &&
                finding.GetProperty("severity").GetString() == "boundary");
            Assert.Contains(root.GetProperty("sidecarRules").EnumerateArray(), static rule =>
                rule.GetString()!.Contains("cannot promote TensorRtExec/OnnxToEngine build reports", StringComparison.Ordinal));
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void SampleRunEvidenceRecordRejectsPackageConsumerRuntimeClassification()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "jyppx-sample-evidence-boundary-" + Guid.NewGuid().ToString("N"));
        string recordPath = Path.Combine(tempRoot, "sample-run-evidence.json");
        string outputRoot = Path.Combine(tempRoot, "validation");
        Directory.CreateDirectory(tempRoot);
        File.WriteAllText(recordPath, """
{
  "schemaVersion": 1,
  "recordKind": "sample-run-evidence-record",
  "templateOnly": false,
  "sampleName": "YoloVision",
  "manifestPath": "samples/assets/yolovision-assets.template.json",
  "proofClassification": "package-consumer-runtime",
  "modelPath": ".\\models\\yolo.onnx",
  "modelSha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "labelsPath": ".\\models\\coco.names",
  "labelsSha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "inputAssetPath": ".\\models\\image.jpg",
  "inputAssetSha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "preprocessedInputTensorPath": ".\\models\\yolo-preprocessed-fp32.bin",
  "preprocessedInputTensorSha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
  "evidenceSidecarPath": ".\\models\\yolo-evidence.sidecar.json",
  "buildReportPath": ".\\models\\yolo-build-report.json",
  "sampleRunCommand": "dotnet run --project .\\samples\\YoloVision -- --model .\\models\\yolo.onnx --labels .\\models\\coco.names --input-data .\\models\\yolo-preprocessed-fp32.bin",
  "sampleRunLogPath": ".\\models\\yolo-sample-run.log",
  "sampleRunLogSha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  "stdoutSummary": "YoloVision Passed=True",
  "stderrSummary": "no-stderr-emitted",
  "expectedEvidenceLines": [
    "YoloVision Passed=True",
    "InputSource=external"
  ],
  "isSmokePassed": true,
  "canPromoteRealModelRuntime": true
}
""");

        try
        {
            string script = Path.Combine(RepositoryPaths.Root, "eng", "Test-SampleRunEvidenceRecord.ps1");
            string output = RunPowerShellAllowFailure(script, "-InputPath", recordPath, "-OutputRoot", outputRoot);

            Assert.Contains("Sample run evidence record validation written", output, StringComparison.Ordinal);

            string validationPath = Path.Combine(outputRoot, "sample-run-evidence-record-validation.json");
            using JsonDocument validation = JsonDocument.Parse(File.ReadAllText(validationPath));
            JsonElement root = validation.RootElement;

            Assert.Equal("invalid", root.GetProperty("validationState").GetString());
            Assert.Equal("package-consumer-runtime", root.GetProperty("proofClassification").GetString());
            Assert.False(root.GetProperty("packageConsumerRuntimeForbidden").GetBoolean());
            Assert.False(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.True(root.GetProperty("errorCount").GetInt32() > 0);
            Assert.Contains(root.GetProperty("validationItems").EnumerateArray(), static item =>
                item.GetProperty("id").GetString() == "package-consumer-runtime-forbidden" &&
                !item.GetProperty("passed").GetBoolean());
            Assert.Contains(root.GetProperty("disallowedProofClassifications").EnumerateArray(), static item =>
                item.GetString() == "package-consumer-runtime");
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void RealModelOwnerHandoffKeepsModelBackfillAsOwnerAction()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-SampleAssetManifest.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-SampleAssetAcquisitionPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-SampleRunEvidenceRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-SampleRunEvidenceRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OnnxEngineBuildEvidenceSidecar.ps1"));

        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-RealModelOwnerHandoff.ps1");
        string output = RunPowerShell(script);

        Assert.Contains("Real model owner handoff written", output, StringComparison.Ordinal);
        Assert.Contains("HandoffState=owner-action-required", output, StringComparison.Ordinal);
        Assert.Contains("CanPromoteRealModelRuntime=False", output, StringComparison.Ordinal);

        string handoffPath = Path.Combine(RepositoryPaths.Root, "artifacts", "user-acceptance", "real-model-owner-handoff.json");
        using JsonDocument handoff = JsonDocument.Parse(File.ReadAllText(handoffPath));
        JsonElement root = handoff.RootElement;

        Assert.Equal("real-model-owner-handoff", root.GetProperty("recordKind").GetString());
        Assert.Equal("owner-action-required", root.GetProperty("handoffState").GetString());
        Assert.Equal("owner-action-required", root.GetProperty("ownerActionStatus").GetString());
        Assert.False(root.GetProperty("performsDownload").GetBoolean());
        Assert.False(root.GetProperty("performsSampleRun").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.True(root.GetProperty("packageConsumerRuntimeForbidden").GetBoolean());
        Assert.Equal("template-only", root.GetProperty("proofClassification").GetString());

        JsonElement[] items = root.GetProperty("items").EnumerateArray().ToArray();
        Assert.True(root.GetProperty("itemCount").GetInt32() >= 9);
        Assert.Equal(root.GetProperty("itemCount").GetInt32(), items.Length);
        Assert.Contains(items, static item =>
            item.GetProperty("sampleName").GetString() == "Classification" &&
            item.GetProperty("sampleRunEvidenceRecord").GetString() == "models/classifier-sample-run-evidence.json" &&
            item.GetProperty("sampleRunEvidenceTemplate").GetString()!.Contains("classification", StringComparison.Ordinal) &&
            !item.GetProperty("isSmokePassed").GetBoolean());
        Assert.Contains(items, static item =>
            item.GetProperty("sampleName").GetString() == "YoloVision" &&
            item.GetProperty("sampleRunEvidenceRecord").GetString() == "models/yolovision-sample-run-evidence.json" &&
            item.GetProperty("sampleRunEvidenceTemplate").GetString()!.Contains("yolovision", StringComparison.Ordinal) &&
            !item.GetProperty("isSmokePassed").GetBoolean());
        Assert.Contains(items, static item =>
            item.GetProperty("modelName").GetString() == "YOLOX-S candidate" &&
            item.GetProperty("sampleRunEvidenceRecord").GetString() == "models/yolox_s-sample-run-evidence.json" &&
            item.GetProperty("sampleRunEvidenceTemplate").GetString()!.Contains("yolox-s", StringComparison.Ordinal) &&
            item.GetProperty("proofClassification").GetString() == "build-only");
        Assert.Contains(items.SelectMany(static item => item.GetProperty("steps").EnumerateArray()), static step =>
            step.GetProperty("id").GetString() == "hash-backfill" &&
            step.GetProperty("command").GetString()!.Contains("Get-FileHash", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("package-consumer-runtime is forbidden", StringComparison.Ordinal));
    }

    [Fact]
    public void DeferredReadOnlyApiCandidatePlanKeepsPromotionAsPlanningInputOnly()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredReadOnlyApiCandidatePlan.ps1");
        string output = RunPowerShell(script, "-MaxItems", "15");

        Assert.Contains("Deferred readonly API candidate plan written", output, StringComparison.Ordinal);

        string planPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-api-candidate-plan.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-api-candidate-plan.md");
        using JsonDocument plan = JsonDocument.Parse(File.ReadAllText(planPath));
        JsonElement root = plan.RootElement;
        string markdown = File.ReadAllText(markdownPath);

        Assert.Equal("deferred-readonly-api-candidate-plan", root.GetProperty("planKind").GetString());
        Assert.True(root.GetProperty("totalDeferredRowCount").GetInt32() >= 0);
        Assert.True(root.GetProperty("lowRiskDeferredRowCount").GetInt32() >= root.GetProperty("selectedCandidateCount").GetInt32());
        Assert.True(root.GetProperty("highRiskDeferredRowCount").GetInt32() > 0);
        Assert.True(root.GetProperty("manualReviewDesignGroupCount").GetInt32() >= 0);
        Assert.False(root.GetProperty("includeMediumRisk").GetBoolean());
        Assert.True(root.GetProperty("selectedCandidateCount").GetInt32() <= 15);
        Assert.Contains(root.GetProperty("selectionPolicy").EnumerateArray(), static policy => policy.GetString()!.Contains("Exclude callback", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("selectionPolicy").EnumerateArray(), static policy => policy.GetString()!.Contains("IAlgorithm result snapshots", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("selectionPolicy").EnumerateArray(), static policy => policy.GetString()!.Contains("add/remove/enable/disable", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("selectionPolicy").EnumerateArray(), static policy => policy.GetString()!.Contains("shape-values/profile-values/shape-binding array rows", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("selectionPolicy").EnumerateArray(), static policy => policy.GetString()!.Contains("IAlgorithm/IAlgorithmContext/IAlgorithmIOInfo/IAlgorithmVariant", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("selectionPolicy").EnumerateArray(), static policy => policy.GetString()!.Contains("IDimensionExpr, IExprBuilder, IOnnxConfig, IVersionedInterface", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("selectionPolicy").EnumerateArray(), static policy => policy.GetString()!.Contains("IPluginResource, and IPluginResourceContext", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("selectionPolicy").EnumerateArray(), static policy => policy.GetString()!.Contains("Deduplicate by interface", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("selectionPolicy").EnumerateArray(), static policy => policy.GetString()!.Contains("cross-version guards", StringComparison.Ordinal));

        JsonElement[] candidates = root.GetProperty("candidates").EnumerateArray().ToArray();
        Assert.All(candidates, static candidate =>
        {
            Assert.Equal("low", candidate.GetProperty("ownershipRisk").GetString());
            Assert.Equal("candidate-for-readonly-diagnostic-promotion", candidate.GetProperty("recommendedAction").GetString());
            Assert.True(candidate.GetProperty("canPromoteWithoutDesignGate").GetBoolean());
            Assert.Equal("manual-review-other", candidate.GetProperty("designGroup").GetString());

            string interfaceName = candidate.GetProperty("interface").GetString()!;
            string method = candidate.GetProperty("method").GetString()!;
            string text = $"{interfaceName} {method}";
            Assert.DoesNotContain("callback", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("listener", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("allocator", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("buffer", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("weights", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("state", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("plugin", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("resource", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("IAlgorithm", text, StringComparison.Ordinal);
            Assert.DoesNotContain("register", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("deregister", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("enqueue", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("clone", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("IVersionedInterface", text, StringComparison.Ordinal);
            Assert.DoesNotContain("IPluginResource", text, StringComparison.Ordinal);
            Assert.DoesNotContain("IPluginResourceContext", text, StringComparison.Ordinal);
            Assert.DoesNotContain("IRNNv2Layer", text, StringComparison.Ordinal);
            Assert.DoesNotContain("IDimensionExpr", text, StringComparison.Ordinal);
            Assert.DoesNotContain("IExprBuilder", text, StringComparison.Ordinal);
            Assert.DoesNotContain("IOnnxConfig", text, StringComparison.Ordinal);
            Assert.DoesNotContain("addVerbosity", text, StringComparison.Ordinal);
            Assert.DoesNotContain("getGpuAllocator", text, StringComparison.Ordinal);
            Assert.DoesNotContain("getErrorBuffer", text, StringComparison.Ordinal);
            Assert.DoesNotContain("getLayerOutputTensor", text, StringComparison.Ordinal);
            Assert.DoesNotContain("getProfileShapeValues", text, StringComparison.Ordinal);
            Assert.DoesNotContain("getProfileTensorValues", text, StringComparison.Ordinal);
            Assert.DoesNotContain("getShapeBinding", text, StringComparison.Ordinal);
            Assert.DoesNotContain("getAPILanguage", text, StringComparison.Ordinal);
            Assert.DoesNotContain("getInterfaceInfo", text, StringComparison.Ordinal);
            Assert.DoesNotContain("getBiasForGate", text, StringComparison.Ordinal);
            Assert.DoesNotContain("getCellState", text, StringComparison.Ordinal);
            Assert.DoesNotContain("getDataLength", text, StringComparison.Ordinal);
            Assert.DoesNotContain("getHiddenState", text, StringComparison.Ordinal);
            Assert.False(method.StartsWith("add", StringComparison.OrdinalIgnoreCase), $"Unsafe add method selected: {interfaceName}");
            Assert.False(method.StartsWith("remove", StringComparison.OrdinalIgnoreCase), $"Unsafe remove method selected: {interfaceName}");
            Assert.False(method.StartsWith("enable", StringComparison.OrdinalIgnoreCase), $"Unsafe enable method selected: {interfaceName}");
            Assert.False(method.StartsWith("disable", StringComparison.OrdinalIgnoreCase), $"Unsafe disable method selected: {interfaceName}");
            Assert.False(method.StartsWith("reduce", StringComparison.OrdinalIgnoreCase), $"Unsafe reduce method selected: {interfaceName}");
            Assert.False(method.StartsWith("create", StringComparison.OrdinalIgnoreCase), $"Unsafe create method selected: {interfaceName}");
            Assert.False(method.StartsWith("set", StringComparison.OrdinalIgnoreCase), $"Unsafe setter selected: {interfaceName}");
        });

        Assert.Equal(candidates.Length, candidates.Select(static candidate => $"{candidate.GetProperty("interface").GetString()}|{candidate.GetProperty("tensorRtLine").GetString()}").Distinct(StringComparer.Ordinal).Count());
    }

    [Fact]
    public void DeferredReadOnlyApiCandidatePlanClassifiesMediumRiskAsManualDesignWork()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredReadOnlyApiCandidatePlan.ps1");
        string output = RunPowerShell(script, "-IncludeMediumRisk", "-MaxItems", "60");

        Assert.Contains("Deferred readonly API candidate plan written", output, StringComparison.Ordinal);

        string planPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-api-candidate-plan.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-api-candidate-plan.md");
        using JsonDocument plan = JsonDocument.Parse(File.ReadAllText(planPath));
        JsonElement root = plan.RootElement;
        string markdown = File.ReadAllText(markdownPath);

        Assert.True(root.GetProperty("includeMediumRisk").GetBoolean());
        Assert.True(root.GetProperty("mediumRiskDeferredRowCount").GetInt32() > 0);
        Assert.True(root.GetProperty("manualReviewDesignGroupCount").GetInt32() > 0);
        Assert.Contains("| Design group | Count | Risk levels | Promote without design gate |", markdown, StringComparison.Ordinal);
        Assert.Contains("High-risk design groups are boundary planning input only", markdown, StringComparison.Ordinal);
        Assert.Contains("`algorithm-selector-ownership-boundary`", markdown, StringComparison.Ordinal);
        Assert.Contains("`plugin-ownership-boundary`", markdown, StringComparison.Ordinal);

        JsonElement[] designGroups = root.GetProperty("manualReviewDesignGroups").EnumerateArray().ToArray();
        Assert.Contains(designGroups, static group =>
            group.GetProperty("designGroup").GetString() == "runtime-execution-boundary" &&
            !group.GetProperty("canPromoteWithoutDesignGate").GetBoolean() &&
            group.GetProperty("recommendedDesignAction").GetString()!.Contains("Do not promote", StringComparison.Ordinal));
        Assert.Contains(designGroups, static group =>
            group.GetProperty("designGroup").GetString() == "algorithm-selector-ownership-boundary" &&
            group.GetProperty("ownershipRiskLevels").EnumerateArray().Any(risk => risk.GetString() == "high") &&
            !group.GetProperty("canPromoteWithoutDesignGate").GetBoolean() &&
            group.GetProperty("recommendedDesignAction").GetString()!.Contains("Do not expose borrowed algorithm pointers", StringComparison.Ordinal));
        Assert.Contains(designGroups, static group =>
            group.GetProperty("designGroup").GetString() == "runtime-deserialization-boundary" &&
            group.GetProperty("recommendedDesignAction").GetString()!.Contains("bridge-owned engine lifetime", StringComparison.Ordinal));
        Assert.Contains(designGroups, static group =>
            group.GetProperty("designGroup").GetString() == "dimension-expression-snapshot-design" &&
            group.GetProperty("recommendedDesignAction").GetString()!.Contains("pointer-free snapshot", StringComparison.Ordinal));
        Assert.Contains(designGroups, static group =>
            group.GetProperty("designGroup").GetString() == "expression-builder-design-gate" &&
            group.GetProperty("recommendedDesignAction").GetString()!.Contains("Do not create expression nodes", StringComparison.Ordinal));
        Assert.Contains(designGroups, static group =>
            group.GetProperty("designGroup").GetString() == "error-recorder-diagnostics-design" &&
            group.GetProperty("recommendedDesignAction").GetString()!.Contains("Do not expose recorder pointers", StringComparison.Ordinal));
        Assert.Contains(designGroups, static group =>
            group.GetProperty("designGroup").GetString() == "calibrator-callback-metadata-design" &&
            group.GetProperty("recommendedDesignAction").GetString()!.Contains("Do not invoke calibration callbacks", StringComparison.Ordinal));
        Assert.Contains(designGroups, static group =>
            group.GetProperty("designGroup").GetString() == "plugin-ownership-boundary" &&
            group.GetProperty("ownershipRiskLevels").EnumerateArray().Any(risk => risk.GetString() == "high") &&
            group.GetProperty("recommendedDesignAction").GetString()!.Contains("copied metadata", StringComparison.Ordinal));

        JsonElement[] candidates = root.GetProperty("candidates").EnumerateArray().ToArray();
        Assert.True(candidates.Length > 0);
        Assert.All(candidates, static candidate =>
        {
            Assert.False(candidate.GetProperty("canPromoteWithoutDesignGate").GetBoolean());
            Assert.Equal("manual-review-before-promotion", candidate.GetProperty("recommendedAction").GetString());
            Assert.NotEqual("candidate-for-readonly-diagnostic-promotion", candidate.GetProperty("recommendedAction").GetString());
            Assert.False(string.IsNullOrWhiteSpace(candidate.GetProperty("designGroup").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(candidate.GetProperty("promotionBoundary").GetString()));
        });
        Assert.Contains(candidates, static candidate =>
            candidate.GetProperty("interface").GetString()!.Contains("IExecutionContext::execute", StringComparison.Ordinal) &&
            candidate.GetProperty("designGroup").GetString() == "runtime-execution-boundary");
        Assert.Contains(candidates, static candidate =>
            candidate.GetProperty("interface").GetString()!.Contains("IRuntime::deserializeCudaEngine", StringComparison.Ordinal) &&
            candidate.GetProperty("designGroup").GetString() == "runtime-deserialization-boundary");
        Assert.Contains(candidates, static candidate =>
            candidate.GetProperty("interface").GetString()!.Contains("IInt8", StringComparison.Ordinal) &&
            candidate.GetProperty("designGroup").GetString() == "calibrator-callback-metadata-design");
    }

    [Fact]
    public void DeferredCandidateSafetyTriageSeparatesPromotionProofDesignAndKeepDeferredRows()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredReadOnlyApiCandidatePlan.ps1");
        string output = RunPowerShell(script, "-IncludeMediumRisk", "-MaxItems", "60");

        Assert.Contains("Deferred candidate safety triage written", output, StringComparison.Ordinal);

        string triagePath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-candidate-safety-triage.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-candidate-safety-triage.md");
        using JsonDocument triage = JsonDocument.Parse(File.ReadAllText(triagePath));
        JsonElement root = triage.RootElement;
        string markdown = File.ReadAllText(markdownPath);

        Assert.Equal("deferred-candidate-safety-triage", root.GetProperty("triageKind").GetString());
        Assert.Equal("artifacts/interface-coverage/tensorrt-interface-comparison.csv", root.GetProperty("sourceMatrix").GetString());
        Assert.True(root.GetProperty("totalTriageRowCount").GetInt32() > 0);
        Assert.Contains(root.GetProperty("policy").EnumerateArray(), static policy => policy.GetString()!.Contains("Do not treat this triage as permission to delete deferred records", StringComparison.Ordinal));

        JsonElement[] tierSummaries = root.GetProperty("tierSummaries").EnumerateArray().ToArray();
        Assert.Contains(tierSummaries, static tier =>
            tier.GetProperty("safetyTier").GetString() == "A - immediate-safe" &&
            tier.GetProperty("recommendedAction").GetString() == "promote-real-api");
        Assert.Contains(tierSummaries, static tier =>
            tier.GetProperty("safetyTier").GetString() == "B - safe-alternative-or-alias" &&
            tier.GetProperty("candidateCount").GetInt32() > 0 &&
            tier.GetProperty("recommendedAction").GetString() == "alias-or-proof-safe-alternative" &&
            tier.GetProperty("designGroups").EnumerateArray().Any(group => group.GetString() == "error-recorder-diagnostics-design"));
        Assert.Contains(tierSummaries, static tier =>
            tier.GetProperty("safetyTier").GetString() == "C - design-gate-required" &&
            tier.GetProperty("candidateCount").GetInt32() > 0 &&
            tier.GetProperty("recommendedAction").GetString() == "design-gate-required" &&
            tier.GetProperty("designGroups").EnumerateArray().Any(group => group.GetString() == "algorithm-selector-ownership-boundary") &&
            tier.GetProperty("designGroups").EnumerateArray().Any(group => group.GetString() == "plugin-ownership-boundary"));
        Assert.Contains(tierSummaries, static tier =>
            tier.GetProperty("safetyTier").GetString() == "D - keep-deferred" &&
            tier.GetProperty("candidateCount").GetInt32() > 0 &&
            tier.GetProperty("recommendedAction").GetString() == "keep-deferred");

        Assert.Contains("# Deferred Candidate Safety Triage", markdown, StringComparison.Ordinal);
        Assert.Contains("## Safety Tier Summary", markdown, StringComparison.Ordinal);
        Assert.Contains("A - immediate-safe", markdown, StringComparison.Ordinal);
        Assert.Contains("B - safe-alternative-or-alias", markdown, StringComparison.Ordinal);
        Assert.Contains("C - design-gate-required", markdown, StringComparison.Ordinal);
        Assert.Contains("D - keep-deferred", markdown, StringComparison.Ordinal);
        Assert.Contains("registry load/unload, plugin resource acquire/release, plugin instance create/clone/enqueue, callback trampolines, and borrowed pointer boundaries", markdown, StringComparison.Ordinal);
        Assert.Contains("Do not treat this triage as permission to delete deferred records", markdown, StringComparison.Ordinal);

        JsonElement[] rows = root.GetProperty("rows").EnumerateArray().ToArray();
        Assert.Contains(rows, static row =>
            row.GetProperty("interface").GetString() == "IPluginRegistry::getErrorRecorder" &&
            row.GetProperty("safetyTier").GetString() == "B - safe-alternative-or-alias" &&
            row.GetProperty("recommendedAction").GetString() == "alias-or-proof-safe-alternative");
        Assert.Contains(rows, static row =>
            row.GetProperty("interface").GetString() == "IErrorRecorder::getInterfaceInfo" &&
            row.GetProperty("safetyTier").GetString() == "B - safe-alternative-or-alias" &&
            row.GetProperty("designGroup").GetString() == "error-recorder-diagnostics-design" &&
            row.GetProperty("matchedManifestIds").GetString()!.Contains("error-recorder-get-interface-info-deferred", StringComparison.Ordinal) &&
            row.GetProperty("matchedManifestIds").GetString()!.Contains("error-recorder-snapshot-info", StringComparison.Ordinal));
        Assert.Contains(rows, static row =>
            row.GetProperty("interface").GetString() == "IParser::getLayerOutputTensor" &&
            row.GetProperty("safetyTier").GetString() == "B - safe-alternative-or-alias" &&
            row.GetProperty("matchedManifestIds").GetString()!.Contains("onnx-parser-layer-output-tensor-exists", StringComparison.Ordinal) &&
            row.GetProperty("matchedManifestIds").GetString()!.Contains("parser-get-layer-output-tensor-deferred", StringComparison.Ordinal));
        Assert.Contains(rows, static row =>
            row.GetProperty("safetyTier").GetString() == "D - keep-deferred" &&
            row.GetProperty("designGroup").GetString() == "callback-allocator-boundary" &&
            row.GetProperty("reason").GetString()!.Contains("callback trampoline", StringComparison.Ordinal));
        Assert.Contains(rows, static row =>
            row.GetProperty("safetyTier").GetString() == "D - keep-deferred" &&
            row.GetProperty("interface").GetString()!.Contains("IPluginResource", StringComparison.Ordinal));

        Assert.DoesNotContain(rows, static row =>
            row.GetProperty("interface").GetString() == "IErrorRecorder::getInterfaceInfo" &&
            row.GetProperty("safetyTier").GetString() == "A - immediate-safe");
    }

    [Fact]
    public void ReleaseProofArtifactsConsumeDeferredSafetyTriageWithoutPromotingProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredReadOnlyApiCandidatePlan.ps1"), "-IncludeMediumRisk", "-MaxItems", "60");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePackageProofBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseGapDashboard.ps1"));

        string triagePath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-candidate-safety-triage.json");
        using JsonDocument triage = JsonDocument.Parse(File.ReadAllText(triagePath));
        JsonElement triageRoot = triage.RootElement;
        int triageRows = triageRoot.GetProperty("totalTriageRowCount").GetInt32();
        JsonElement[] tierSummaries = triageRoot.GetProperty("tierSummaries").EnumerateArray().ToArray();
        int tierA = Assert.Single(tierSummaries, static item => item.GetProperty("safetyTier").GetString() == "A - immediate-safe").GetProperty("candidateCount").GetInt32();
        int tierB = Assert.Single(tierSummaries, static item => item.GetProperty("safetyTier").GetString() == "B - safe-alternative-or-alias").GetProperty("candidateCount").GetInt32();
        int tierC = Assert.Single(tierSummaries, static item => item.GetProperty("safetyTier").GetString() == "C - design-gate-required").GetProperty("candidateCount").GetInt32();
        int tierD = Assert.Single(tierSummaries, static item => item.GetProperty("safetyTier").GetString() == "D - keep-deferred").GetProperty("candidateCount").GetInt32();

        string[] jsonArtifacts =
        [
            Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-package-proof-bundle.json"),
            Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.json"),
            Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-close-gap-dashboard.json"),
        ];

        foreach (string jsonPath in jsonArtifacts)
        {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(jsonPath));
            JsonElement root = document.RootElement;
            Assert.Equal("triage-ready-planning-input-only", root.GetProperty("deferredSafetyTriageState").GetString());
            Assert.Equal("deferred-candidate-safety-triage", root.GetProperty("deferredSafetyTriageKind").GetString());
            Assert.Equal(triageRows, root.GetProperty("deferredSafetyTriageTotalRows").GetInt32());
            Assert.Equal(tierA, root.GetProperty("deferredSafetyTierAImmediateSafeCount").GetInt32());
            Assert.Equal(tierB, root.GetProperty("deferredSafetyTierBSafeAlternativeCount").GetInt32());
            Assert.Equal(tierC, root.GetProperty("deferredSafetyTierCDesignGateCount").GetInt32());
            Assert.Equal(tierD, root.GetProperty("deferredSafetyTierDKeepDeferredCount").GetInt32());
            Assert.Contains("permission to delete deferred records", root.GetProperty("deferredSafetyTriageProofBoundary").GetString()!, StringComparison.Ordinal);
            Assert.Contains("artifacts/interface-coverage/deferred-candidate-safety-triage.json", root.GetProperty("sourceEvidence").EnumerateArray().Select(static item => item.GetString()!));
            Assert.Contains("artifacts/interface-coverage/deferred-candidate-safety-triage.md", root.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!));
            Assert.Contains("deferred safety triage", root.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!));
        }

        using JsonDocument packageBundle = JsonDocument.Parse(File.ReadAllText(jsonArtifacts[0]));
        Assert.False(packageBundle.RootElement.GetProperty("deferredSafetyTriageIsPackageProof").GetBoolean());
        Assert.False(packageBundle.RootElement.GetProperty("deferredSafetyTriageIsRuntimeExecutionProof").GetBoolean());
        Assert.False(packageBundle.RootElement.GetProperty("deferredSafetyTriageCanUseAsPublicPackageProof").GetBoolean());
        Assert.Contains(packageBundle.RootElement.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "deferred-safety-triage" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("state").GetString()!.Contains("proof=false", StringComparison.Ordinal));

        using JsonDocument evidenceBundle = JsonDocument.Parse(File.ReadAllText(jsonArtifacts[1]));
        Assert.False(evidenceBundle.RootElement.GetProperty("deferredSafetyTriageIsReleaseProof").GetBoolean());
        Assert.False(evidenceBundle.RootElement.GetProperty("deferredSafetyTriageCanPublishPublicly").GetBoolean());
        Assert.False(evidenceBundle.RootElement.GetProperty("deferredSafetyTriageCanCloseReleaseIssue").GetBoolean());
        Assert.Contains(evidenceBundle.RootElement.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "deferred-safety-triage" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("no tier is package-consumer-runtime proof", StringComparison.Ordinal));

        using JsonDocument closeDashboard = JsonDocument.Parse(File.ReadAllText(jsonArtifacts[2]));
        Assert.False(closeDashboard.RootElement.GetProperty("deferredSafetyTriageIsReleaseProof").GetBoolean());
        Assert.False(closeDashboard.RootElement.GetProperty("deferredSafetyTriageCanPublishPublicly").GetBoolean());
        Assert.False(closeDashboard.RootElement.GetProperty("deferredSafetyTriageCanCloseReleaseIssue").GetBoolean());
        Assert.False(closeDashboard.RootElement.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(closeDashboard.RootElement.GetProperty("canCloseReleaseIssue").GetBoolean());

        foreach (string markdownPath in new[]
        {
            Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-package-proof-bundle.md"),
            Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"),
            Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-close-gap-dashboard.md"),
        })
        {
            string markdown = File.ReadAllText(markdownPath);
            Assert.Contains("deferred safety triage", markdown, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("safe-alternative", markdown, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("design-gate", markdown, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("keep-deferred", markdown, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("False", markdown, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ReleaseFreezeFinalVerificationAggregatesProofBlockersAndTriageWithoutPromotingRelease()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredReadOnlyApiCandidatePlan.ps1"), "-IncludeMediumRisk", "-MaxItems", "60");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePackageProofBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseGapDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseClosePreflight.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCandidateFinalEvidenceFreeze.ps1"));

        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseFreezeFinalVerification.ps1"));
        Assert.Contains("Release freeze final verification written", output, StringComparison.Ordinal);

        string triagePath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-candidate-safety-triage.json");
        using JsonDocument triage = JsonDocument.Parse(File.ReadAllText(triagePath));
        JsonElement triageRoot = triage.RootElement;
        int triageRows = triageRoot.GetProperty("totalTriageRowCount").GetInt32();
        JsonElement[] tierSummaries = triageRoot.GetProperty("tierSummaries").EnumerateArray().ToArray();
        int tierA = Assert.Single(tierSummaries, static item => item.GetProperty("safetyTier").GetString() == "A - immediate-safe").GetProperty("candidateCount").GetInt32();
        int tierB = Assert.Single(tierSummaries, static item => item.GetProperty("safetyTier").GetString() == "B - safe-alternative-or-alias").GetProperty("candidateCount").GetInt32();
        int tierC = Assert.Single(tierSummaries, static item => item.GetProperty("safetyTier").GetString() == "C - design-gate-required").GetProperty("candidateCount").GetInt32();
        int tierD = Assert.Single(tierSummaries, static item => item.GetProperty("safetyTier").GetString() == "D - keep-deferred").GetProperty("candidateCount").GetInt32();

        string jsonPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-freeze-final-verification.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-freeze-final-verification.md");
        Assert.True(File.Exists(jsonPath), $"Expected {jsonPath} to exist.");
        Assert.True(File.Exists(markdownPath), $"Expected {markdownPath} to exist.");

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(jsonPath));
        JsonElement root = document.RootElement;

        Assert.Equal("release-freeze-final-verification", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("verificationState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("ownerActionStatus").GetString());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isRealModelRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishVerificationProof").GetBoolean());

        string[] expectedBlockers =
        {
            "owner-authorization",
            "package-consumer-runtime",
            "linux-runner-proof",
            "real-model-runtime",
            "post-publish-verification",
        };
        string[] releaseBlockers = root
            .GetProperty("releaseBlockers")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .Order(StringComparer.Ordinal)
            .ToArray();
        Assert.Equal(expectedBlockers.Order(StringComparer.Ordinal).ToArray(), releaseBlockers);
        Assert.Equal(expectedBlockers.Length, root.GetProperty("releaseBlockerCount").GetInt32());

        string jsonText = File.ReadAllText(jsonPath);
        foreach (string marker in new[]
        {
            "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof",
            "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof",
            "Test-LinuxRunnerEvidenceRecord.ps1",
            "Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog",
            "Test-ReleaseOwnerApprovalInput.ps1",
            "Test-OwnerAuthorizedPublishCommandPlan.ps1",
            "Test-StaleReleaseClaims.ps1",
            "deferred safety triage",
        })
        {
            Assert.Contains(marker, jsonText, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Equal("triage-ready-planning-input-only", root.GetProperty("deferredSafetyTriageState").GetString());
        Assert.Equal("deferred-candidate-safety-triage", root.GetProperty("deferredSafetyTriageKind").GetString());
        Assert.Equal(triageRows, root.GetProperty("deferredSafetyTriageTotalRows").GetInt32());
        Assert.Equal(tierA, root.GetProperty("deferredSafetyTierAImmediateSafeCount").GetInt32());
        Assert.Equal(tierB, root.GetProperty("deferredSafetyTierBSafeAlternativeCount").GetInt32());
        Assert.Equal(tierC, root.GetProperty("deferredSafetyTierCDesignGateCount").GetInt32());
        Assert.Equal(tierD, root.GetProperty("deferredSafetyTierDKeepDeferredCount").GetInt32());
        Assert.Contains("permission to delete deferred records", root.GetProperty("deferredSafetyTriageProofBoundary").GetString()!, StringComparison.Ordinal);
        Assert.Contains("artifacts/interface-coverage/deferred-candidate-safety-triage.json", root.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!));
        Assert.Contains("deferred safety triage", root.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!));

        JsonElement triageSummary = root.GetProperty("deferredSafetyTriageSummary");
        Assert.False(triageSummary.GetProperty("isReleaseProof").GetBoolean());
        Assert.False(triageSummary.GetProperty("isPackageProof").GetBoolean());
        Assert.False(triageSummary.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(triageSummary.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(triageSummary.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(triageSummary.GetProperty("canDeleteDeferredRecords").GetBoolean());

        string markdown = File.ReadAllText(markdownPath);
        foreach (string marker in new[]
        {
            "release-freeze-final-verification",
            "blocked-real-proof-required",
            "deferred safety triage",
            "safe-alternative",
            "design-gate",
            "keep-deferred",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom",
            "det、cls、seg、obb、pose、sem",
        })
        {
            Assert.Contains(marker, markdown, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void PublicApiBilingualDocumentationBacklogGroupsExistingAuditIntoActionableBatches()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-PublicApiBilingualDocumentationBacklog.ps1");
        string output = RunPowerShell(script);

        Assert.Contains("Public API bilingual documentation backlog written", output, StringComparison.Ordinal);

        string auditPath = Path.Combine(RepositoryPaths.Root, "artifacts", "api-doc-audit", "public-api-bilingual-documentation-audit.json");
        string backlogPath = Path.Combine(RepositoryPaths.Root, "artifacts", "api-doc-audit", "public-api-bilingual-documentation-backlog.json");
        using JsonDocument audit = JsonDocument.Parse(File.ReadAllText(auditPath));
        using JsonDocument backlog = JsonDocument.Parse(File.ReadAllText(backlogPath));

        JsonElement root = backlog.RootElement;
        Assert.Equal(audit.RootElement.GetProperty("findingCount").GetInt32(), root.GetProperty("backlogFindingCount").GetInt32());
        if (audit.RootElement.GetProperty("findingCount").GetInt32() > 0)
        {
            Assert.Contains(root.GetProperty("batchSummaries").EnumerateArray(), static batch => batch.GetProperty("recommendedBatch").GetString() == "P2-callback-allocator-boundary-docs");
            Assert.Contains(root.GetProperty("topTypeSummaries").EnumerateArray(), static type => type.TryGetProperty("sourceHint", out JsonElement sourceHint) && sourceHint.GetString()!.EndsWith(".cs", StringComparison.Ordinal));
        }
        else
        {
            Assert.Equal(0, root.GetProperty("batchSummaries").GetArrayLength());
            Assert.Equal(0, root.GetProperty("topTypeSummaries").GetArrayLength());
        }
    }

    [Fact]
    public void UserAcceptanceCatalogKeepsAssetRequiredSamplesSeparateFromSmokePasses()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-SampleAssetManifest.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-SampleAssetAcquisitionPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-SampleRunEvidenceRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-SampleRunEvidenceRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealModelOwnerHandoff.ps1"));

        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-UserAcceptanceSampleCatalog.ps1");
        string output = RunPowerShell(script);

        Assert.Contains("User acceptance catalog written", output, StringComparison.Ordinal);

        string catalogPath = Path.Combine(RepositoryPaths.Root, "artifacts", "user-acceptance", "sample-smoke-catalog.json");
        using JsonDocument catalog = JsonDocument.Parse(File.ReadAllText(catalogPath));
        JsonElement root = catalog.RootElement;

        Assert.True(root.GetProperty("itemCount").GetInt32() >= 10);
        Assert.Equal(0, root.GetProperty("missingItemCount").GetInt32());
        Assert.Equal("ready", root.GetProperty("sampleAssetManifestAuditStatus").GetString());
        Assert.Equal(0, root.GetProperty("sampleAssetManifestErrorCount").GetInt32());
        Assert.Equal("owner-action-required", root.GetProperty("sampleAssetAcquisitionPlanState").GetString());
        Assert.True(root.GetProperty("sampleAssetAcquisitionPlanItemCount").GetInt32() >= 8);
        Assert.Equal("owner-action-required", root.GetProperty("sampleRunEvidenceValidationState").GetString());
        Assert.False(root.GetProperty("sampleRunEvidenceCanPromoteRealModelRuntime").GetBoolean());
        Assert.Equal("template-only", root.GetProperty("sampleRunEvidenceProofClassification").GetString());
        Assert.Equal("owner-action-required", root.GetProperty("realModelOwnerHandoffState").GetString());
        Assert.False(root.GetProperty("realModelOwnerHandoffCanPromoteRealModelRuntime").GetBoolean());
        Assert.Equal("template-only", root.GetProperty("realModelOwnerHandoffProofClassification").GetString());

        JsonElement[] items = root.GetProperty("items").EnumerateArray().ToArray();
        Assert.Contains(items, static item => item.GetProperty("name").GetString() == "DynamicShape" && item.GetProperty("status").GetString() == "ready-to-run");
        Assert.Contains(items, static item =>
            item.GetProperty("name").GetString() == "Classification" &&
            item.GetProperty("status").GetString() == "asset-required" &&
            item.GetProperty("assetRequirement").GetString()!.Contains("Manifest status: candidate-not-downloaded", StringComparison.Ordinal) &&
            item.GetProperty("assetRequirement").GetString()!.Contains("proof classification: template-only", StringComparison.Ordinal) &&
            item.GetProperty("assetRequirement").GetString()!.Contains("acquisition plan: owner-action-required", StringComparison.Ordinal) &&
            item.GetProperty("assetRequirement").GetString()!.Contains("runner evidence state: owner-action-required", StringComparison.Ordinal) &&
            item.GetProperty("assetRequirement").GetString()!.Contains("runner can promote real model runtime: False", StringComparison.Ordinal) &&
            item.GetProperty("assetRequirement").GetString()!.Contains("owner handoff state: owner-action-required", StringComparison.Ordinal) &&
            item.GetProperty("assetRequirement").GetString()!.Contains("owner handoff can promote real model runtime: False", StringComparison.Ordinal));
        Assert.Contains(items, static item =>
            item.GetProperty("name").GetString() == "YoloVision" &&
            item.GetProperty("status").GetString() == "asset-required" &&
            item.GetProperty("command").GetString()!.Contains("--input-data .\\models\\yolo-preprocessed-fp32.bin", StringComparison.Ordinal) &&
            !item.GetProperty("command").GetString()!.Contains("--input .\\models\\yolo", StringComparison.OrdinalIgnoreCase) &&
            item.GetProperty("assetRequirement").GetString()!.Contains("Manifest status: candidate-not-downloaded", StringComparison.Ordinal) &&
            item.GetProperty("assetRequirement").GetString()!.Contains("proof classification: template-only", StringComparison.Ordinal) &&
            item.GetProperty("assetRequirement").GetString()!.Contains("acquisition plan: owner-action-required", StringComparison.Ordinal) &&
            item.GetProperty("assetRequirement").GetString()!.Contains("runner evidence state: owner-action-required", StringComparison.Ordinal) &&
            item.GetProperty("assetRequirement").GetString()!.Contains("runner can promote real model runtime: False", StringComparison.Ordinal) &&
            item.GetProperty("assetRequirement").GetString()!.Contains("owner handoff state: owner-action-required", StringComparison.Ordinal) &&
            item.GetProperty("assetRequirement").GetString()!.Contains("owner handoff can promote real model runtime: False", StringComparison.Ordinal));
        Assert.Contains(items, static item => item.GetProperty("name").GetString() == "PluginRegistryInventorySmokeRunner" && item.GetProperty("status").GetString() == "cataloged-not-run");
    }

    [Fact]
    public void ReleaseCandidateFullAcceptanceSummaryAggregatesNonPublishReadiness()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPackageReviewBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePackageProofBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DocsPublishReadinessBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofBackfillPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationBackfillPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofCollectionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationCollectionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePromotionIssueRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCandidateFreezeSummary.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCandidateFreezeChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseCandidateFreezeSummary.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerAuthorizedPublishCommandPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerAuthorizedPublishCommandPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-StaleReleaseClaims.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-UserAcceptanceSampleCatalog.ps1"));

        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCandidateFullAcceptanceSummary.ps1");
        string output = RunPowerShell(script);

        Assert.Contains("Release candidate full acceptance summary written", output, StringComparison.Ordinal);
        Assert.Contains("PerformsPublish=False", output, StringComparison.Ordinal);
        Assert.Contains("CanCloseReleaseIssue=False", output, StringComparison.Ordinal);

        string summaryPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-candidate-full-acceptance-summary.json");
        using JsonDocument summary = JsonDocument.Parse(File.ReadAllText(summaryPath));
        JsonElement root = summary.RootElement;

        Assert.Equal("release-candidate-full-acceptance-summary", root.GetProperty("recordKind").GetString());
        Assert.Equal("ready-for-owner-proof-collection", root.GetProperty("acceptanceState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.True(root.GetProperty("canPromoteToOwnerProofCollection").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canUseAsPublicPackageProof").GetBoolean());
        Assert.False(root.GetProperty("canPublishDocsExternally").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal(0, root.GetProperty("failedRequiredItemCount").GetInt32());
        Assert.True(root.GetProperty("requiredItemCount").GetInt32() >= 10);
        Assert.Equal("owner-review-required", root.GetProperty("finalPackageReviewState").GetString());
        Assert.Equal("package-evidence-owner-review-required", root.GetProperty("releasePackageProofState").GetString());
        Assert.Equal("ready-for-owner-review", root.GetProperty("docsPublishReadinessState").GetString());
        Assert.Equal("blocked-evidence-incomplete", root.GetProperty("releaseEvidenceBundleState").GetString());
        Assert.Equal("blocked-freeze-owner-action-required", root.GetProperty("freezeState").GetString());
        Assert.Equal("blocked-freeze-owner-action-required", root.GetProperty("freezeValidationState").GetString());
        Assert.Equal("blocked-owner-authorization-required", root.GetProperty("ownerCommandPlanState").GetString());
        Assert.Equal("blocked-owner-authorization-required", root.GetProperty("ownerCommandPlanValidationState").GetString());
        Assert.Equal("owner-action-required", root.GetProperty("externalRuntimeProofCollectionPackageState").GetString());
        Assert.Equal("blocked-real-publication-required", root.GetProperty("postPublishVerificationCollectionPackageState").GetString());
        Assert.False(root.GetProperty("postPublishCleanConsumerProjectScanState").GetString() is null);
        Assert.Equal("post-publish-verification-record-input-draft", root.GetProperty("postPublishVerificationInputDraftKind").GetString());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("releaseClosePreflightState").GetString());
        Assert.True(root.GetProperty("releaseClosePreflightFailedItemCount").GetInt32() >= 4);
        Assert.Equal(0, root.GetProperty("staleReleaseClaimsFindingCount").GetInt32());

        JsonElement[] items = root.GetProperty("acceptanceItems").EnumerateArray().ToArray();
        Assert.Contains(items, static item =>
            item.GetProperty("id").GetString() == "external-runtime-proof-collection-package" &&
            item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("not compatible-host runtime proof", StringComparison.Ordinal));
        Assert.Contains(items, static item =>
            item.GetProperty("id").GetString() == "post-publish-verification-collection-package" &&
            item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("cannot be proof", StringComparison.Ordinal));
        Assert.Contains(items, static item =>
            item.GetProperty("id").GetString() == "post-publish-clean-consumer-project-scan" &&
            item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("helper evidence only", StringComparison.Ordinal));
        Assert.Contains(items, static item =>
            item.GetProperty("id").GetString() == "post-publish-verification-input-draft" &&
            item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("helper artifact only", StringComparison.Ordinal));
        Assert.Contains(items, static item =>
            item.GetProperty("id").GetString() == "release-close-preflight" &&
            item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("cannot substitute proof", StringComparison.Ordinal));
        Assert.Contains(items, static item =>
            item.GetProperty("id").GetString() == "real-proof-boundary" &&
            item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("state").GetString()!.Contains("isRuntimeProof=False", StringComparison.Ordinal) &&
            item.GetProperty("state").GetString()!.Contains("isProof=False", StringComparison.Ordinal));
        Assert.Contains(items, static item =>
            item.GetProperty("id").GetString() == "sample-surface-yolovision" &&
            item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("YoloVision", StringComparison.Ordinal));

        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/external-runtime-proof-collection-package.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/post-publish-verification-collection-package.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/post-publish-clean-consumer-project-scan.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/post-publish-verification-record.input-draft.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/release-close-preflight.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/stale-release-claims-audit.json");
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("canCloseReleaseIssue remains false", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("blocked-by-cuda-driver is not smoke passed", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("safetyNotes").EnumerateArray(), static note =>
            note.GetString()!.Contains("clean consumer scan and input draft are helper artifacts only", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("nextOwnerActions").EnumerateArray(), static action =>
            action.GetString()!.Contains("ReleaseClosePreflight", StringComparison.Ordinal));
        Assert.Contains(root.GetProperty("nextOwnerActions").EnumerateArray(), static action =>
            action.GetString()!.Contains("External Runtime Proof Collection Package", StringComparison.Ordinal));
    }

    [Fact]
    public void OwnerProofValidatorsExposeActionableBoundaries()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordDraft.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalRuntimeProofRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishVerificationRecord.ps1"));

        string externalValidationPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "external-runtime-proof-validation.json");
        using JsonDocument externalValidation = JsonDocument.Parse(File.ReadAllText(externalValidationPath));
        JsonElement externalRoot = externalValidation.RootElement;

        Assert.Equal("template-only", externalRoot.GetProperty("validationState").GetString());
        Assert.False(externalRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(externalRoot.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.True(externalRoot.GetProperty("ownerActionSummary").GetArrayLength() >= 4);
        Assert.Contains(externalRoot.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static item => item.GetString() == "dependency-probe-only");
        Assert.Contains(externalRoot.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static item => item.GetString() == "blocked-by-cuda-driver");
        Assert.Contains(externalRoot.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static item => item.GetString() == "bridge-only package consumer log");
        Assert.Contains(externalRoot.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static item => item.GetString() == "Skipped=True");
        Assert.All(externalRoot.GetProperty("validationItems").EnumerateArray(), static item =>
        {
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("ownerAction").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("boundary").GetString()));
        });

        string postPublishValidationPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "post-publish-verification-validation.json");
        using JsonDocument postPublishValidation = JsonDocument.Parse(File.ReadAllText(postPublishValidationPath));
        JsonElement postRoot = postPublishValidation.RootElement;

        Assert.Equal("template-only", postRoot.GetProperty("validationState").GetString());
        Assert.False(postRoot.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(postRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.True(postRoot.GetProperty("ownerActionSummary").GetArrayLength() >= 5);
        Assert.Contains(postRoot.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static item => item.GetString() == "local feed");
        Assert.Contains(postRoot.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static item => item.GetString() == "ProjectReference");
        Assert.Contains(postRoot.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static item => item.GetString() == "bridge-only package consumer log");
        Assert.Contains(postRoot.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static item => item.GetString() == "Skipped=True");
        Assert.Contains(postRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "published-package-url" &&
            !item.GetProperty("passed").GetBoolean() &&
            !string.IsNullOrWhiteSpace(item.GetProperty("ownerAction").GetString()) &&
            !string.IsNullOrWhiteSpace(item.GetProperty("boundary").GetString()));
        Assert.Contains(postRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "published-package-sha256-source" &&
            !item.GetProperty("passed").GetBoolean() &&
            !string.IsNullOrWhiteSpace(item.GetProperty("ownerAction").GetString()) &&
            !string.IsNullOrWhiteSpace(item.GetProperty("boundary").GetString()));
        Assert.Contains(postRoot.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "clean-consumer-outside-repository" &&
            !item.GetProperty("passed").GetBoolean() &&
            !string.IsNullOrWhiteSpace(item.GetProperty("ownerAction").GetString()) &&
            !string.IsNullOrWhiteSpace(item.GetProperty("boundary").GetString()));
        Assert.All(postRoot.GetProperty("validationItems").EnumerateArray(), static item =>
        {
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("ownerAction").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("boundary").GetString()));
        });

        string externalValidationMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "external-runtime-proof-validation.md"));
        string postPublishValidationMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "post-publish-verification-validation.md"));
        string externalTemplateMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "external-runtime-proof-record-template.md"));
        string postPublishTemplateMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "post-publish-verification-record-template.md"));

        Assert.Contains("## Owner Action Summary", externalValidationMarkdown, StringComparison.Ordinal);
        Assert.Contains("## Non-Substitute Proof Kinds", externalValidationMarkdown, StringComparison.Ordinal);
        Assert.Contains("## Owner Action Summary", postPublishValidationMarkdown, StringComparison.Ordinal);
        Assert.Contains("## Non-Substitute Proof Kinds", postPublishValidationMarkdown, StringComparison.Ordinal);
        Assert.Contains("## Owner Action Summary", externalTemplateMarkdown, StringComparison.Ordinal);
        Assert.Contains("## Owner Action Summary", postPublishTemplateMarkdown, StringComparison.Ordinal);
        Assert.Contains("managedPackageSha256Source", postPublishTemplateMarkdown, StringComparison.Ordinal);

        string externalCollectionDoc = ReadSource("docs", "articles", "zh-cn", "external-runtime-proof-collection-package.md");
        string postPublishCollectionDoc = ReadSource("docs", "articles", "zh-cn", "post-publish-verification-collection-package.md");
        string ownerApprovalDoc = ReadSource("docs", "articles", "zh-cn", "release-owner-approval-guide.md");
        string ownerCommandPlanDoc = ReadSource("docs", "articles", "zh-cn", "owner-authorized-publish-command-plan.md");
        string postPublishRecordDoc = ReadSource("docs", "articles", "zh-cn", "post-publish-verification-record.md");
        string cleanConsumerScanDoc = ReadSource("docs", "articles", "zh-cn", "post-publish-clean-consumer-project-scan.md");
        string postPublishInputDraftDoc = ReadSource("docs", "articles", "zh-cn", "post-publish-verification-record-input-draft.md");
        string releaseClosePreflightDoc = ReadSource("docs", "articles", "zh-cn", "release-close-preflight.md");
        string docsIndex = ReadSource("docs", "index.md");
        string docsToc = ReadSource("docs", "toc.yml");

        Assert.Contains("## 不可替代 Proof 清单", externalCollectionDoc, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record.json", externalCollectionDoc, StringComparison.Ordinal);
        Assert.Contains("-FailOnNotProof", externalCollectionDoc, StringComparison.Ordinal);
        Assert.Contains("## 不可替代 Proof 清单", postPublishCollectionDoc, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-record.json", postPublishCollectionDoc, StringComparison.Ordinal);
        Assert.Contains("-FailOnNotProof", postPublishCollectionDoc, StringComparison.Ordinal);
        Assert.Contains("## Owner 冻结前最小复核", ownerApprovalDoc, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue", ownerApprovalDoc, StringComparison.Ordinal);
        Assert.Contains("## 大模型执行边界", ownerCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("不得执行真实发布", ownerCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("materializedExecutableCommand", ownerCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("Owner Authorization Proof Gate", ownerCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("manualMaterializationPrerequisites", ownerCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("credentialHandlingAcknowledged", ownerCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("nvidiaRedistributionApproval", ownerCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("postPublishRequiredEvidence", ownerCommandPlanDoc, StringComparison.Ordinal);
        Assert.Contains("managedPackageSha256Source", postPublishRecordDoc, StringComparison.Ordinal);
        Assert.Contains("download timestamp", postPublishRecordDoc, StringComparison.Ordinal);
        Assert.Contains("postPublishRequiredEvidence", postPublishRecordDoc, StringComparison.Ordinal);
        Assert.Contains("publishedPackageUrl", postPublishRecordDoc, StringComparison.Ordinal);
        Assert.Contains("cleanConsumerRootOutsideRepository", postPublishRecordDoc, StringComparison.Ordinal);
        Assert.Contains("runtimeSmokeLogPath", postPublishRecordDoc, StringComparison.Ordinal);
        Assert.Contains("runtimeSmokeLogSha256", postPublishRecordDoc, StringComparison.Ordinal);
        Assert.Contains("host metadata", postPublishRecordDoc, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Owner Authorization Proof Gate", postPublishRecordDoc, StringComparison.Ordinal);
        Assert.Contains("Test-PostPublishCleanConsumerProject.ps1", postPublishRecordDoc, StringComparison.Ordinal);
        Assert.Contains("Export-PostPublishVerificationRecordInputDraft.ps1", postPublishRecordDoc, StringComparison.Ordinal);
        Assert.Contains("Export-ReleaseClosePreflight.ps1", postPublishRecordDoc, StringComparison.Ordinal);

        Assert.Contains("Test-PostPublishCleanConsumerProject.ps1", cleanConsumerScanDoc, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue=false", cleanConsumerScanDoc, StringComparison.Ordinal);
        Assert.Contains("isPostPublishVerificationProof=false", cleanConsumerScanDoc, StringComparison.Ordinal);
        Assert.Contains("blocked-project-reference-present", cleanConsumerScanDoc, StringComparison.Ordinal);
        Assert.Contains("blocked-project-inside-repository", cleanConsumerScanDoc, StringComparison.Ordinal);

        Assert.Contains("Export-PostPublishVerificationRecordInputDraft.ps1", postPublishInputDraftDoc, StringComparison.Ordinal);
        Assert.Contains("inputDraftOnly=true", postPublishInputDraftDoc, StringComparison.Ordinal);
        Assert.Contains("isPostPublishVerificationProof=false", postPublishInputDraftDoc, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue=false", postPublishInputDraftDoc, StringComparison.Ordinal);
        Assert.Contains("64 位 SHA256", postPublishInputDraftDoc, StringComparison.Ordinal);

        Assert.Contains("Export-ReleaseClosePreflight.ps1", releaseClosePreflightDoc, StringComparison.Ordinal);
        Assert.Contains("blocked-real-proof-required", releaseClosePreflightDoc, StringComparison.Ordinal);
        Assert.Contains("post-publish-clean-consumer-scan", releaseClosePreflightDoc, StringComparison.Ordinal);
        Assert.Contains("dependency-probe-only", releaseClosePreflightDoc, StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", releaseClosePreflightDoc, StringComparison.Ordinal);

        Assert.Contains("articles/zh-cn/post-publish-clean-consumer-project-scan.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/post-publish-verification-record-input-draft.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-close-preflight.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/post-publish-clean-consumer-project-scan.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/post-publish-verification-record-input-draft.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-close-preflight.md", docsToc, StringComparison.Ordinal);
    }

    [Fact]
    public void NegativeProofValidatorFixturesDoNotPolluteDefaultFinalReleaseArtifacts()
    {
        string externalDefaultPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "external-runtime-proof-validation.json");
        string postPublishDefaultPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "post-publish-verification-validation.json");
        string preflightPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-close-preflight.json");

        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalRuntimeProofRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishVerificationRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseClosePreflight.ps1"));

        using JsonDocument externalBefore = JsonDocument.Parse(File.ReadAllText(externalDefaultPath));
        using JsonDocument postPublishBefore = JsonDocument.Parse(File.ReadAllText(postPublishDefaultPath));
        using JsonDocument preflightBefore = JsonDocument.Parse(File.ReadAllText(preflightPath));

        AssertDefaultFinalReleaseProofState(externalBefore.RootElement, postPublishBefore.RootElement, preflightBefore.RootElement);

        ValidateExternalRuntimeNegativeRecordUsesTemporaryOutputRoot();
        ValidatePostPublishNegativeRecordUsesTemporaryOutputRoot();

        using JsonDocument externalAfter = JsonDocument.Parse(File.ReadAllText(externalDefaultPath));
        using JsonDocument postPublishAfter = JsonDocument.Parse(File.ReadAllText(postPublishDefaultPath));
        using JsonDocument preflightAfter = JsonDocument.Parse(File.ReadAllText(preflightPath));

        AssertDefaultFinalReleaseProofState(externalAfter.RootElement, postPublishAfter.RootElement, preflightAfter.RootElement);
    }

    [Fact]
    public void ExternalRuntimeProofValidatorRejectsBridgeOnlySkippedAndHashMismatchRecords()
    {
        ValidateExternalRuntimeNegativeRecordUsesTemporaryOutputRoot();
    }

    [Fact]
    public void PostPublishValidatorRejectsBridgeOnlyLocalFeedAndMissingCleanConsumerProof()
    {
        ValidatePostPublishNegativeRecordUsesTemporaryOutputRoot();
    }

    private static void ValidateExternalRuntimeNegativeRecordUsesTemporaryOutputRoot()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "jyppx-external-runtime-negative-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        try
        {
            string logPath = Path.Combine(tempRoot, "bridge-only-smoke.log");
            File.WriteAllText(
                logPath,
                """
                HighLevelWrapperSurface=compiled:onnx-parser-diagnostic-snapshot
                WrapperSurfaceEvidenceKind=compile-surface-proof
                IsRuntimeExecutionProof=False
                Skipped=True Reason=blocked-by-cuda-driver
                bridge-only package consumer log
                DependencyProbeStatus=dependency-probe-only
                """);

            string recordPath = Path.Combine(tempRoot, "external-runtime-proof-record.json");
            File.WriteAllText(
                recordPath,
                $$"""
                {
                  "recordKind": "external-runtime-proof-record",
                  "templateOnly": false,
                  "runtimePackageKey": "win-x64-trt11.0-cuda13.2-cudnn9.22",
                  "proofState": "owner-filled-negative-test",
                  "proofClassification": "package-consumer-runtime",
                  "publicationState": "not-for-publication-negative-test",
                  "isRuntimeExecutionEvidence": false,
                  "isDependencyProbeOnly": true,
                  "canPromoteRuntimeProof": false,
                  "host": {
                    "ownerName": "negative-test-owner",
                    "machineName": "negative-test-host",
                    "osDescription": "Windows negative test",
                    "gpuName": "NVIDIA negative test GPU",
                    "driverVersion": "999.0",
                    "cudaDriverSupportedRuntime": "13.2",
                    "cudaRuntimeVersion": "13.2",
                    "tensorRtRuntimeVersion": "11.0",
                    "cudnnVersion": "9.22",
                    "tensorRtLine": "TRT11"
                  },
                  "packageSource": {
                    "managedPackageSource": "https://packages.example.invalid/JYPPX.TensorRtSharp.4.0.0.nupkg",
                    "runtimePackageSource": "https://packages.example.invalid/JYPPX.TensorRtSharp.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22.4.0.0.nupkg",
                    "runtimePackageKey": "win-x64-trt11.0-cuda13.2-cudnn9.22",
                    "managedNupkgSha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "runtimeNupkgSha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "noProjectReference": false,
                    "consumerProjectName": "NegativeConsumer",
                    "consumerProjectPath": "C:\\negative\\NegativeConsumer\\NegativeConsumer.csproj"
                  },
                  "command": {
                    "restoreCommand": "dotnet restore",
                    "buildCommand": "dotnet build",
                    "smokeCommand": "dotnet run -- --runtime-package-key win-x64-trt11.0-cuda13.2-cudnn9.22",
                    "exitCode": 0,
                    "logPath": "{{logPath.Replace("\\", "\\\\")}}",
                    "logSha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
                  },
                  "results": {
                    "dependencyProbeStatus": "dependency-probe-only",
                    "smokeStatus": "Skipped=True",
                    "nativeAssetsCopied": true,
                    "stdoutSummary": "Bridge-only wrapper surface compiled; no TensorRT runtime execution occurred.",
                    "stderrSummary": "no-stderr-emitted",
                    "failureDiagnostic": "blocked-by-cuda-driver"
                  },
                  "modelEvidence": {
                    "modelName": "",
                    "modelSha256": "",
                    "modelLicense": "",
                    "inputAssetName": "",
                    "inputAssetSha256": ""
                  }
                }
                """);

            string output = RunPowerShellAllowFailure(
                Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalRuntimeProofRecord.ps1"),
                "-InputPath",
                recordPath,
                "-OutputRoot",
                Path.Combine(tempRoot, "external-validation"),
                "-RuntimePackageKey",
                "win-x64-trt11.0-cuda13.2-cudnn9.22",
                "-RequireExistingLog",
                "-FailOnNotProof");

            Assert.Contains("External runtime proof is not real runtime proof", output, StringComparison.Ordinal);

            string validationPath = Path.Combine(tempRoot, "external-validation", "external-runtime-proof-validation.json");
            using JsonDocument validation = JsonDocument.Parse(File.ReadAllText(validationPath));
            JsonElement root = validation.RootElement;

            Assert.False(root.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
            Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.Equal("blocked-by-cuda-driver", root.GetProperty("validationState").GetString());
            Assert.False(root.GetProperty("logSha256Matches").GetBoolean());
            Assert.True(root.GetProperty("failedProofItemCount").GetInt32() > 0);

            string[] nonSubstitutes = root.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
            Assert.Contains("bridge-only package consumer log", nonSubstitutes);
            Assert.Contains("Skipped=True", nonSubstitutes);
            Assert.Contains("WrapperSurfaceEvidenceKind=compile-surface-proof", nonSubstitutes);
            Assert.Contains("IsRuntimeExecutionProof=False", nonSubstitutes);

            JsonElement[] items = root.GetProperty("validationItems").EnumerateArray().ToArray();
            Assert.Contains(items, static item => item.GetProperty("id").GetString() == "package-source" && !item.GetProperty("passed").GetBoolean());
            Assert.Contains(items, static item => item.GetProperty("id").GetString() == "log-sha256-match" && !item.GetProperty("passed").GetBoolean());
            Assert.Contains(items, static item => item.GetProperty("id").GetString() == "not-dependency-probe-only" && !item.GetProperty("passed").GetBoolean());
            Assert.Contains(items, static item => item.GetProperty("id").GetString() == "not-driver-blocked" && !item.GetProperty("passed").GetBoolean());
            Assert.Contains(items, static item => item.GetProperty("id").GetString() == "declared-runtime-proof" && !item.GetProperty("passed").GetBoolean());
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    private static void ValidatePostPublishNegativeRecordUsesTemporaryOutputRoot()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "jyppx-post-publish-negative-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        try
        {
            string restoreLogPath = Path.Combine(tempRoot, "restore.log");
            string nativeAssetsLogPath = Path.Combine(tempRoot, "native-assets.log");
            string dependencyProbeLogPath = Path.Combine(tempRoot, "dependency-probe.log");
            string smokeLogPath = Path.Combine(tempRoot, "smoke.log");

            File.WriteAllText(restoreLogPath, "restore from local feed");
            File.WriteAllText(nativeAssetsLogPath, "native assets copied by bridge-only smoke");
            File.WriteAllText(dependencyProbeLogPath, "dependency-probe-only");
            File.WriteAllText(
                smokeLogPath,
                """
                Skipped=True
                bridge-only package consumer log
                WrapperSurfaceEvidenceKind=compile-surface-proof
                IsRuntimeExecutionProof=False
                DependencyProbeStatus=dependency-probe-only
                """);

            string recordPath = Path.Combine(tempRoot, "post-publish-verification-record.json");
            File.WriteAllText(
                recordPath,
                $$"""
                {
                  "recordKind": "post-publish-verification-record",
                  "templateOnly": false,
                  "verificationState": "owner-filled-negative-test",
                  "postPublishProofClassification": "post-publish-package-consumer-runtime",
                  "performsPublish": false,
                  "selectedChannel": "local-feed",
                  "channelSourceUri": "",
                  "ownerName": "negative-test-owner",
                  "reviewerName": "negative-test-reviewer",
                  "publishedVersion": "4.0.0-negative",
                  "managedPackageSource": "local-feed",
                  "runtimePackageSource": "local-feed",
                  "expectedRuntimePackageKey": "win-x64-trt11.0-cuda13.2-cudnn9.22",
                  "cleanConsumerRoot": "",
                  "consumerProjectName": "NegativeConsumer",
                  "consumerProjectPath": "C:\\negative\\NegativeConsumer\\NegativeConsumer.csproj",
                  "noProjectReference": false,
                  "restoreCommand": "dotnet restore --source local-feed",
                  "buildCommand": "dotnet build",
                  "smokeCommand": "dotnet run -- --runtime-package-key win-x64-trt11.0-cuda13.2-cudnn9.22",
                  "stdoutSummary": "Bridge-only wrapper surface compiled; no published-channel runtime execution occurred.",
                  "stderrSummary": "no-stderr-emitted",
                  "packageIdentity": {
                    "managedPackageId": "JYPPX.TensorRtSharp",
                    "managedPackageVersion": "4.0.0-negative",
                    "managedPackageUrl": "",
                    "managedNupkgSha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "managedPackageSha256Source": "",
                    "managedPackageDownloadTimestampUtc": "",
                    "runtimePackageId": "JYPPX.TensorRtSharp.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22",
                    "runtimePackageVersion": "4.0.0-negative",
                    "runtimePackageUrl": "",
                    "runtimeNupkgSha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "runtimePackageSha256Source": "",
                    "runtimePackageDownloadTimestampUtc": ""
                  },
                  "host": {
                    "ownerName": "negative-test-owner",
                    "machineName": "negative-test-host",
                    "osDescription": "Windows negative test",
                    "gpuName": "NVIDIA negative test GPU",
                    "driverVersion": "999.0",
                    "cudaDriverSupportedRuntime": "13.2",
                    "cudaRuntimeVersion": "13.2",
                    "tensorRtRuntimeVersion": "11.0",
                    "tensorRtLine": "TRT11",
                    "cudnnVersion": "9.22"
                  },
                  "nativeAssetsCopied": true,
                  "dependencyProbePassed": true,
                  "runtimeSmokePassed": false,
                  "runtimeSmokeExitCode": 0,
                  "smokeStatus": "Skipped=True",
                  "restoreLogPath": "{{restoreLogPath.Replace("\\", "\\\\")}}",
                  "restoreLogSha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                  "nativeAssetListingPath": "{{nativeAssetsLogPath.Replace("\\", "\\\\")}}",
                  "nativeAssetListingSha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                  "dependencyProbeLogPath": "{{dependencyProbeLogPath.Replace("\\", "\\\\")}}",
                  "dependencyProbeLogSha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                  "smokeLogPath": "{{smokeLogPath.Replace("\\", "\\\\")}}",
                  "smokeLogSha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                  "isPostPublishVerificationProof": false,
                  "canCloseReleaseIssue": false,
                  "verificationItems": [
                    { "id": "published-package-version", "status": "failed", "evidence": "local feed package identity is not a published-channel proof" },
                    { "id": "clean-directory", "status": "failed", "evidence": "cleanConsumerRoot missing" },
                    { "id": "clean-consumer-project-identity", "status": "passed", "evidence": "project name/path present" },
                    { "id": "no-project-reference", "status": "failed", "evidence": "ProjectReference present" },
                    { "id": "managed-package-source", "status": "failed", "evidence": "local-feed" },
                    { "id": "runtime-package-source", "status": "failed", "evidence": "local-feed" },
                    { "id": "host-runtime-metadata", "status": "passed", "evidence": "host metadata present" },
                    { "id": "native-assets-copied", "status": "passed", "evidence": "bridge-only native copy log" },
                    { "id": "dependency-probe", "status": "passed", "evidence": "dependency-probe-only" },
                    { "id": "runtime-key-smoke-command", "status": "passed", "evidence": "--runtime-package-key win-x64-trt11.0-cuda13.2-cudnn9.22" },
                    { "id": "stdout-summary", "status": "passed", "evidence": "stdout reviewed" },
                    { "id": "stderr-summary", "status": "passed", "evidence": "no-stderr-emitted" },
                    { "id": "stdout-stderr-summary", "status": "passed", "evidence": "stdout/stderr reviewed" },
                    { "id": "compatible-host-smoke", "status": "failed", "evidence": "Skipped=True" }
                  ],
                  "executionSteps": []
                }
                """);

            string output = RunPowerShellAllowFailure(
                Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishVerificationRecord.ps1"),
                "-InputPath",
                recordPath,
                "-OutputRoot",
                Path.Combine(tempRoot, "post-publish-validation"),
                "-RequireExistingLog",
                "-FailOnNotProof");

            Assert.Contains("Post-publish verification", output, StringComparison.Ordinal);
            Assert.Contains("not proof", output, StringComparison.Ordinal);

            string validationPath = Path.Combine(tempRoot, "post-publish-validation", "post-publish-verification-validation.json");
            using JsonDocument validation = JsonDocument.Parse(File.ReadAllText(validationPath));
            JsonElement root = validation.RootElement;

            Assert.False(root.GetProperty("isPostPublishVerificationProof").GetBoolean());
            Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.True(root.GetProperty("failedProofItemCount").GetInt32() > 0);
            Assert.False(root.GetProperty("smokeLogSha256Matches").GetBoolean());

            string[] nonSubstitutes = root.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
            Assert.Contains("local feed", nonSubstitutes);
            Assert.Contains("ProjectReference", nonSubstitutes);
            Assert.Contains("bridge-only package consumer log", nonSubstitutes);
            Assert.Contains("Skipped=True", nonSubstitutes);
            Assert.Contains("WrapperSurfaceEvidenceKind=compile-surface-proof", nonSubstitutes);
            Assert.Contains("IsRuntimeExecutionProof=False", nonSubstitutes);

            JsonElement[] items = root.GetProperty("validationItems").EnumerateArray().ToArray();
            Assert.Contains(items, static item => item.GetProperty("id").GetString() == "channel-source" && !item.GetProperty("passed").GetBoolean());
            Assert.Contains(items, static item => item.GetProperty("id").GetString() == "published-package-url" && !item.GetProperty("passed").GetBoolean());
            Assert.Contains(items, static item => item.GetProperty("id").GetString() == "published-package-sha256-source" && !item.GetProperty("passed").GetBoolean());
            Assert.Contains(items, static item => item.GetProperty("id").GetString() == "clean-consumer-root" && !item.GetProperty("passed").GetBoolean());
            Assert.Contains(items, static item => item.GetProperty("id").GetString() == "clean-consumer-outside-repository" && !item.GetProperty("passed").GetBoolean());
            Assert.Contains(items, static item => item.GetProperty("id").GetString() == "no-project-reference" && !item.GetProperty("passed").GetBoolean());
            Assert.Contains(items, static item => item.GetProperty("id").GetString() == "runtime-smoke-log" && !item.GetProperty("passed").GetBoolean());
            Assert.Contains(items, static item => item.GetProperty("id").GetString() == "smoke-log-sha256-match" && !item.GetProperty("passed").GetBoolean());
            Assert.Contains(items, static item => item.GetProperty("id").GetString() == "execution-steps-present" && !item.GetProperty("passed").GetBoolean());
            Assert.Contains(items, static item => item.GetProperty("id").GetString() == "declared-proof" && !item.GetProperty("passed").GetBoolean());
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void PostPublishCleanConsumerScanClassifiesProjectBoundaries()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "jyppx-post-publish-clean-consumer-" + Guid.NewGuid().ToString("N"));
        string outsideRoot = Path.Combine(tempRoot, "outside");
        string goodRoot = Path.Combine(outsideRoot, "GoodConsumer");
        string projectReferenceRoot = Path.Combine(outsideRoot, "ProjectReferenceConsumer");
        string missingTargetRoot = Path.Combine(outsideRoot, "MissingTargetConsumer");
        string localPackageSourceRoot = Path.Combine(outsideRoot, "LocalPackageSourceConsumer");
        string localNupkgRoot = Path.Combine(outsideRoot, "LocalNupkgConsumer");
        Directory.CreateDirectory(goodRoot);
        Directory.CreateDirectory(projectReferenceRoot);
        Directory.CreateDirectory(missingTargetRoot);
        Directory.CreateDirectory(localPackageSourceRoot);
        Directory.CreateDirectory(localNupkgRoot);

        try
        {
            string goodProject = Path.Combine(goodRoot, "GoodConsumer.csproj");
            string projectReferenceProject = Path.Combine(projectReferenceRoot, "ProjectReferenceConsumer.csproj");
            string missingTargetProject = Path.Combine(missingTargetRoot, "MissingTargetConsumer.csproj");
            string localPackageSourceProject = Path.Combine(localPackageSourceRoot, "LocalPackageSourceConsumer.csproj");
            string localNupkgProject = Path.Combine(localNupkgRoot, "LocalNupkgConsumer.csproj");
            string insideProject = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "post-publish-clean-consumer-inside-repo-fixture.csproj");

            File.WriteAllText(goodProject, """
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="JYPPX.TensorRtSharp" Version="4.0.0-rc.1" />
    <PackageReference Include="JYPPX.TensorRtSharp.win-x64-trt11.0-cuda13.2-cudnn9.22" Version="4.0.0-rc.1" />
  </ItemGroup>
</Project>
""");

            File.WriteAllText(projectReferenceProject, """
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="JYPPX.TensorRtSharp" Version="4.0.0-rc.1" />
    <ProjectReference Include="..\..\TensorRtSharp4.0\src\JYPPX.TensorRtSharp\JYPPX.TensorRtSharp.csproj" />
  </ItemGroup>
</Project>
""");

            File.WriteAllText(missingTargetProject, """
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />
  </ItemGroup>
</Project>
""");

            File.WriteAllText(localPackageSourceProject, """
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <RestoreSources>$(MSBuildProjectDirectory)\local-feed;https://api.nuget.org/v3/index.json</RestoreSources>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="JYPPX.TensorRtSharp" Version="4.0.0-rc.1" />
  </ItemGroup>
</Project>
""");

            File.WriteAllText(localNupkgProject, """
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="JYPPX.TensorRtSharp" Version="..\packages\JYPPX.TensorRtSharp.4.0.0-rc.1.nupkg" />
  </ItemGroup>
</Project>
""");

            Directory.CreateDirectory(Path.GetDirectoryName(insideProject)!);
            File.WriteAllText(insideProject, """
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="JYPPX.TensorRtSharp" Version="4.0.0-rc.1" />
  </ItemGroup>
</Project>
""");

            string script = Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishCleanConsumerProject.ps1");
            string outputRoot = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "post-publish-clean-consumer-test-fixtures");

            RunPowerShell(script, "-ProjectPath", goodProject, "-OutputRoot", Path.Combine(outputRoot, "good"));
            RunPowerShell(script, "-ProjectPath", projectReferenceProject, "-OutputRoot", Path.Combine(outputRoot, "project-reference"));
            RunPowerShell(script, "-ProjectPath", missingTargetProject, "-OutputRoot", Path.Combine(outputRoot, "missing-target"));
            RunPowerShell(script, "-ProjectPath", localPackageSourceProject, "-OutputRoot", Path.Combine(outputRoot, "local-package-source"));
            RunPowerShell(script, "-ProjectPath", localNupkgProject, "-OutputRoot", Path.Combine(outputRoot, "local-nupkg"));
            RunPowerShell(script, "-ProjectPath", insideProject, "-OutputRoot", Path.Combine(outputRoot, "inside-repo"));

            using JsonDocument good = JsonDocument.Parse(File.ReadAllText(Path.Combine(outputRoot, "good", "post-publish-clean-consumer-project-scan.json")));
            using JsonDocument projectReference = JsonDocument.Parse(File.ReadAllText(Path.Combine(outputRoot, "project-reference", "post-publish-clean-consumer-project-scan.json")));
            using JsonDocument missingTarget = JsonDocument.Parse(File.ReadAllText(Path.Combine(outputRoot, "missing-target", "post-publish-clean-consumer-project-scan.json")));
            using JsonDocument localPackageSource = JsonDocument.Parse(File.ReadAllText(Path.Combine(outputRoot, "local-package-source", "post-publish-clean-consumer-project-scan.json")));
            using JsonDocument localNupkg = JsonDocument.Parse(File.ReadAllText(Path.Combine(outputRoot, "local-nupkg", "post-publish-clean-consumer-project-scan.json")));
            using JsonDocument insideRepo = JsonDocument.Parse(File.ReadAllText(Path.Combine(outputRoot, "inside-repo", "post-publish-clean-consumer-project-scan.json")));

            JsonElement goodRootElement = good.RootElement;
            Assert.Equal("post-publish-clean-consumer-project-scan", goodRootElement.GetProperty("recordKind").GetString());
            Assert.Equal("clean-consumer-project-scan-passed", goodRootElement.GetProperty("scanState").GetString());
            Assert.True(goodRootElement.GetProperty("scanPassed").GetBoolean());
            Assert.False(goodRootElement.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(goodRootElement.GetProperty("isPostPublishVerificationProof").GetBoolean());
            Assert.True(goodRootElement.GetProperty("projectOutsideRepository").GetBoolean());
            Assert.Equal(0, goodRootElement.GetProperty("projectReferenceCount").GetInt32());
            Assert.True(goodRootElement.GetProperty("targetTensorRtReferenceFound").GetBoolean());
            Assert.Contains(goodRootElement.GetProperty("packageReferences").EnumerateArray(), static item =>
                item.GetProperty("include").GetString() == "JYPPX.TensorRtSharp");

            JsonElement projectReferenceRootElement = projectReference.RootElement;
            Assert.Equal("blocked-project-reference-present", projectReferenceRootElement.GetProperty("scanState").GetString());
            Assert.False(projectReferenceRootElement.GetProperty("scanPassed").GetBoolean());
            Assert.False(projectReferenceRootElement.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(projectReferenceRootElement.GetProperty("isPostPublishVerificationProof").GetBoolean());
            Assert.True(projectReferenceRootElement.GetProperty("projectReferenceCount").GetInt32() > 0);

            JsonElement missingTargetRootElement = missingTarget.RootElement;
            Assert.Equal("blocked-target-package-reference-missing", missingTargetRootElement.GetProperty("scanState").GetString());
            Assert.False(missingTargetRootElement.GetProperty("scanPassed").GetBoolean());
            Assert.False(missingTargetRootElement.GetProperty("targetTensorRtReferenceFound").GetBoolean());

            JsonElement localPackageSourceRootElement = localPackageSource.RootElement;
            Assert.Equal("blocked-local-package-source-present", localPackageSourceRootElement.GetProperty("scanState").GetString());
            Assert.False(localPackageSourceRootElement.GetProperty("scanPassed").GetBoolean());
            Assert.True(localPackageSourceRootElement.GetProperty("hasLocalPackageSource").GetBoolean());
            Assert.True(localPackageSourceRootElement.GetProperty("localPackageSourceCount").GetInt32() > 0);
            Assert.False(localPackageSourceRootElement.GetProperty("isPostPublishVerificationProof").GetBoolean());
            Assert.Contains(localPackageSourceRootElement.GetProperty("checks").EnumerateArray(), static item =>
                item.GetProperty("id").GetString() == "no-local-package-source" &&
                item.GetProperty("passed").GetBoolean() == false);

            JsonElement localNupkgRootElement = localNupkg.RootElement;
            Assert.Equal("blocked-local-nupkg-reference-present", localNupkgRootElement.GetProperty("scanState").GetString());
            Assert.False(localNupkgRootElement.GetProperty("scanPassed").GetBoolean());
            Assert.True(localNupkgRootElement.GetProperty("hasLocalNupkgPackageReference").GetBoolean());
            Assert.True(localNupkgRootElement.GetProperty("localNupkgPackageReferenceCount").GetInt32() > 0);
            Assert.False(localNupkgRootElement.GetProperty("isPostPublishVerificationProof").GetBoolean());
            Assert.Contains(localNupkgRootElement.GetProperty("checks").EnumerateArray(), static item =>
                item.GetProperty("id").GetString() == "no-local-nupkg-reference" &&
                item.GetProperty("passed").GetBoolean() == false);

            JsonElement insideRepoRootElement = insideRepo.RootElement;
            Assert.Equal("blocked-project-inside-repository", insideRepoRootElement.GetProperty("scanState").GetString());
            Assert.False(insideRepoRootElement.GetProperty("scanPassed").GetBoolean());
            Assert.False(insideRepoRootElement.GetProperty("projectOutsideRepository").GetBoolean());
        }
        finally
        {
            string insideProject = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "post-publish-clean-consumer-inside-repo-fixture.csproj");
            if (File.Exists(insideProject))
            {
                File.Delete(insideProject);
            }

            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void PostPublishVerificationInputDraftComputesLogHashesButStaysNonProof()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "jyppx-post-publish-input-draft-" + Guid.NewGuid().ToString("N"));
        string consumerRoot = Path.Combine(tempRoot, "CleanConsumer");
        Directory.CreateDirectory(consumerRoot);

        try
        {
            string consumerProject = Path.Combine(consumerRoot, "CleanConsumer.csproj");
            string restoreLog = Path.Combine(tempRoot, "restore.log");
            string nativeAssetsLog = Path.Combine(tempRoot, "native-assets.log");
            string dependencyProbeLog = Path.Combine(tempRoot, "dependency-probe.log");
            string smokeLog = Path.Combine(tempRoot, "smoke.log");

            File.WriteAllText(consumerProject, """
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="JYPPX.TensorRtSharp" Version="4.0.0-rc.1" />
  </ItemGroup>
</Project>
""");
            File.WriteAllText(restoreLog, "restore succeeded from public channel");
            File.WriteAllText(nativeAssetsLog, "native runtime assets copied");
            File.WriteAllText(dependencyProbeLog, "DependencyProbe BridgeInitialized=True");
            File.WriteAllText(smokeLog, "smoke passed with --runtime-package-key win-x64-trt11.0-cuda13.2-cudnn9.22");

            string outputRoot = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "post-publish-input-draft-test");
            string scanOutputRoot = Path.Combine(outputRoot, "scan");
            string draftPath = Path.Combine(outputRoot, "post-publish-verification-record.input-draft.json");

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishCleanConsumerProject.ps1"),
                "-ProjectPath",
                consumerProject,
                "-OutputRoot",
                scanOutputRoot);

            RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationRecordTemplate.ps1"));
            string output = RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationRecordInputDraft.ps1"),
                "-CleanConsumerProjectScanPath",
                Path.Combine(scanOutputRoot, "post-publish-clean-consumer-project-scan.json"),
                "-OutputPath",
                draftPath,
                "-SelectedChannel",
                "nuget.org",
                "-ChannelSourceUri",
                "https://api.nuget.org/v3/index.json",
                "-ManagedPackageId",
                "JYPPX.TensorRtSharp",
                "-ManagedPackageVersion",
                "4.0.0-rc.1",
                "-RuntimePackageId",
                "JYPPX.TensorRtSharp.win-x64-trt11.0-cuda13.2-cudnn9.22",
                "-RuntimePackageVersion",
                "4.0.0-rc.1",
                "-ManagedPackageUrl",
                "https://www.nuget.org/packages/JYPPX.TensorRtSharp/4.0.0-rc.1",
                "-RuntimePackageUrl",
                "https://www.nuget.org/packages/JYPPX.TensorRtSharp.win-x64-trt11.0-cuda13.2-cudnn9.22/4.0.0-rc.1",
                "-ManagedPackageSha256Source",
                "owner-downloaded-nupkg",
                "-RuntimePackageSha256Source",
                "owner-downloaded-nupkg",
                "-RestoreLogPath",
                restoreLog,
                "-NativeAssetListingPath",
                nativeAssetsLog,
                "-DependencyProbeLogPath",
                dependencyProbeLog,
                "-SmokeLogPath",
                smokeLog,
                "-RestoreCommand",
                "dotnet restore",
                "-BuildCommand",
                "dotnet build",
                "-SmokeCommand",
                "dotnet run -- --runtime-package-key win-x64-trt11.0-cuda13.2-cudnn9.22");

            Assert.Contains("InputDraftOnly=True", output, StringComparison.Ordinal);
            Assert.Contains("IsPostPublishVerificationProof=False", output, StringComparison.Ordinal);
            Assert.Contains("CanCloseReleaseIssue=False", output, StringComparison.Ordinal);
            Assert.Contains("CleanConsumerProjectScanPassed=True", output, StringComparison.Ordinal);

            using JsonDocument draft = JsonDocument.Parse(File.ReadAllText(draftPath));
            JsonElement root = draft.RootElement;

            Assert.Equal("post-publish-verification-record-input-draft", root.GetProperty("recordKind").GetString());
            Assert.False(root.GetProperty("templateOnly").GetBoolean());
            Assert.True(root.GetProperty("inputDraftOnly").GetBoolean());
            Assert.True(root.GetProperty("cleanConsumerProjectScanPassed").GetBoolean());
            Assert.False(root.GetProperty("isPostPublishVerificationProof").GetBoolean());
            Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.Equal("owner-action-required", root.GetProperty("smokeStatus").GetString());
            Assert.True(root.GetProperty("noProjectReference").GetBoolean());
            Assert.Equal(64, root.GetProperty("restoreLogSha256").GetString()!.Length);
            Assert.Equal(64, root.GetProperty("nativeAssetListingSha256").GetString()!.Length);
            Assert.Equal(64, root.GetProperty("dependencyProbeLogSha256").GetString()!.Length);
            Assert.Equal(64, root.GetProperty("smokeLogSha256").GetString()!.Length);
            Assert.Equal("owner-downloaded-nupkg", root.GetProperty("packageIdentity").GetProperty("managedPackageSha256Source").GetString());
            Assert.Equal("owner-downloaded-nupkg", root.GetProperty("packageIdentity").GetProperty("runtimePackageSha256Source").GetString());
            Assert.Equal(consumerProject, root.GetProperty("consumerProjectPath").GetString());
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void ReleaseClosePreflightAggregatesRealProofGaps()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalRuntimeProofRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishVerificationRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPackageReviewBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePackageProofBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DocsPublishReadinessBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofBackfillPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationBackfillPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofCollectionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationCollectionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePromotionIssueRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCandidateFreezeSummary.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCandidateFreezeChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseCandidateFreezeSummary.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerAuthorizedPublishCommandPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerAuthorizedPublishCommandPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-StaleReleaseClaims.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-UserAcceptanceSampleCatalog.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCandidateFullAcceptanceSummary.ps1"));

        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseClosePreflight.ps1"));

        Assert.Contains("PreflightState=blocked-real-proof-required", output, StringComparison.Ordinal);
        Assert.Contains("CanCloseReleaseIssue=False", output, StringComparison.Ordinal);
        Assert.Contains("PerformsPublish=False", output, StringComparison.Ordinal);

        string preflightPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-close-preflight.json");
        using JsonDocument preflight = JsonDocument.Parse(File.ReadAllText(preflightPath));
        JsonElement root = preflight.RootElement;

        Assert.Equal("release-close-preflight", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("preflightState").GetString());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.True(root.GetProperty("failedItemCount").GetInt32() >= 7);
        Assert.Equal("linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22", root.GetProperty("linuxRuntimePackageKey").GetString());
        Assert.False(root.GetProperty("isRealLinuxRunnerProof").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.Equal("blocked-owner-authorization-required", root.GetProperty("ownerAuthorizationProofGateState").GetString());
        Assert.True(root.GetProperty("ownerAuthorizationRequiredFieldCount").GetInt32() >= 12);
        Assert.Equal(root.GetProperty("ownerAuthorizationRequiredFieldCount").GetInt32(), root.GetProperty("ownerAuthorizationMissingOwnerInputCount").GetInt32());
        Assert.True(root.GetProperty("manualMaterializationPrerequisiteCount").GetInt32() >= 6);
        Assert.True(root.GetProperty("postPublishRequiredEvidenceCount").GetInt32() >= 10);

        JsonElement[] items = root.GetProperty("preflightItems").EnumerateArray().ToArray();
        Assert.Contains(items, static item => item.GetProperty("id").GetString() == "external-runtime-proof-record");
        Assert.Contains(items, static item => item.GetProperty("id").GetString() == "owner-authorized-command-plan");
        Assert.Contains(items, static item => item.GetProperty("id").GetString() == "linux-runner-proof");
        Assert.Contains(items, static item => item.GetProperty("id").GetString() == "real-model-runtime-proof");
        Assert.Contains(items, static item => item.GetProperty("id").GetString() == "post-publish-clean-consumer-scan");
        Assert.Contains(items, static item => item.GetProperty("id").GetString() == "post-publish-verification-record");
        Assert.Contains(items, static item => item.GetProperty("id").GetString() == "stale-release-claims");
        Assert.Contains(items, static item => item.GetProperty("id").GetString() == "full-acceptance-close-readiness");
        Assert.Contains(items, static item =>
            item.GetProperty("id").GetString() == "linux-runner-proof" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("Windows handoff", StringComparison.Ordinal));
        Assert.Contains(items, static item =>
            item.GetProperty("id").GetString() == "real-model-runtime-proof" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("ownerAction").GetString()!.Contains("Classification/YoloVision", StringComparison.Ordinal));
        Assert.Contains(items, static item =>
            item.GetProperty("id").GetString() == "external-runtime-proof-record" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("dependency-probe-only", StringComparison.Ordinal));
        Assert.Contains(items, static item =>
            item.GetProperty("id").GetString() == "post-publish-verification-record" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("ProjectReference", StringComparison.Ordinal));
        Assert.Contains(items, static item =>
            item.GetProperty("id").GetString() == "owner-authorized-command-plan" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("currentStatus").GetString()!.Contains("ownerAuthorizationProofGateState=blocked-owner-authorization-required", StringComparison.Ordinal) &&
            item.GetProperty("currentStatus").GetString()!.Contains("manualMaterializationPrerequisiteCount=", StringComparison.Ordinal) &&
            item.GetProperty("ownerAction").GetString()!.Contains("fill every owner authorization field", StringComparison.Ordinal));

        string[] preflightOwnerAuthorizationRequiredFields = root.GetProperty("ownerAuthorizationRequiredFields").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("approvedCommandPlanSha256", preflightOwnerAuthorizationRequiredFields);
        Assert.Contains("approvedProofBundleSha256", preflightOwnerAuthorizationRequiredFields);
        Assert.Contains("nvidiaRedistributionApproval", preflightOwnerAuthorizationRequiredFields);
        Assert.Contains(root.GetProperty("postPublishRequiredEvidence").EnumerateArray(), static item => item.GetString() == "runtimeSmokeLogSha256");
        Assert.Contains(root.GetProperty("postPublishRequiredEvidence").EnumerateArray(), static item => item.GetString() == "hostMetadata");

        string[] nonSubstitutes = root.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("template", nonSubstitutes);
        Assert.Contains("draft", nonSubstitutes);
        Assert.Contains("runbook", nonSubstitutes);
        Assert.Contains("collection package", nonSubstitutes);
        Assert.Contains("local inventory", nonSubstitutes);
        Assert.Contains("local feed", nonSubstitutes);
        Assert.Contains("ProjectReference", nonSubstitutes);
        Assert.Contains("helper", nonSubstitutes);
        Assert.Contains("build-only", nonSubstitutes);
        Assert.Contains("parse-only", nonSubstitutes);
        Assert.Contains("sidecar-only", nonSubstitutes);
        Assert.Contains("input package", nonSubstitutes);
        Assert.Contains("dependency-probe-only", nonSubstitutes);
        Assert.Contains("Skipped=True", nonSubstitutes);
        Assert.Contains("blocked-by-cuda-driver", nonSubstitutes);
        Assert.Contains("bridge-only package consumer log", nonSubstitutes);
        Assert.Contains("bridge-only wrapper surface", nonSubstitutes);
        Assert.Contains("WrapperSurfaceEvidenceKind=compile-surface-proof", nonSubstitutes);
        Assert.Contains("IsRuntimeExecutionProof=False", nonSubstitutes);
        Assert.Contains("Parser/ParserRefitter diagnostic snapshots", nonSubstitutes);
        Assert.Contains("copied managed diagnostic snapshot", nonSubstitutes);
        Assert.Contains("Windows handoff for Linux proof", nonSubstitutes);
        Assert.Contains(items, static item =>
            item.GetProperty("id").GetString() == "external-runtime-proof-record" &&
            item.GetProperty("boundary").GetString()!.Contains("sidecar-only", StringComparison.Ordinal));

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-close-preflight.md"));
        Assert.Contains("## Non-Substitute Proof Kinds", markdown, StringComparison.Ordinal);
        Assert.Contains("post-publish-clean-consumer-scan", markdown, StringComparison.Ordinal);
        Assert.Contains("linux-runner-proof", markdown, StringComparison.Ordinal);
        Assert.Contains("real-model-runtime-proof", markdown, StringComparison.Ordinal);
        Assert.Contains("Windows handoff for Linux proof", markdown, StringComparison.Ordinal);
        Assert.Contains("Owner Authorization Required Fields", markdown, StringComparison.Ordinal);
        Assert.Contains("approvedCommandPlanSha256", markdown, StringComparison.Ordinal);
        Assert.Contains("Post-Publish Required Evidence", markdown, StringComparison.Ordinal);
        Assert.Contains("runtimeSmokeLogSha256", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseCandidateFinalGapReviewSummarizesRemainingRealProofBlockers()
    {
        string jsonPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-candidate-final-gap-review.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-candidate-final-gap-review.md");

        Assert.True(File.Exists(jsonPath));
        Assert.True(File.Exists(markdownPath));

        using JsonDocument review = JsonDocument.Parse(File.ReadAllText(jsonPath));
        JsonElement root = review.RootElement;

        Assert.Equal("release-candidate-final-gap-review", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("reviewState").GetString());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("preflightState").GetString());
        Assert.True(root.GetProperty("preflightFailedItemCount").GetInt32() >= 4);

        JsonElement articleMatrix = root.GetProperty("articleMatrix");
        Assert.True(articleMatrix.GetProperty("articleCount").GetInt32() >= 81);
        Assert.True(articleMatrix.GetProperty("maxArticleId").GetDouble() >= 81);
        Assert.Empty(articleMatrix.GetProperty("duplicateArticleIds").EnumerateArray());
        Assert.Equal("81-article-matrix-linked-no-duplicate-id", articleMatrix.GetProperty("status").GetString());

        string[] blockers = root.GetProperty("openBlockers").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("owner authorization proof missing", blockers);
        Assert.Contains("package-consumer-runtime proof missing", blockers);
        Assert.Contains("post-publish verification proof missing", blockers);
        Assert.Contains("real-model-runtime proof for Classification/YoloVision missing", blockers);
        Assert.Contains("Linux compatible host runner proof missing", blockers);

        string[] nonSubstitutes = root.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("build-only", nonSubstitutes);
        Assert.Contains("parse-only", nonSubstitutes);
        Assert.Contains("sidecar-only", nonSubstitutes);
        Assert.Contains("ProjectReference", nonSubstitutes);
        Assert.Contains("blocked-by-cuda-driver", nonSubstitutes);

        string markdown = File.ReadAllText(markdownPath);
        foreach (string marker in new[]
        {
            "manifest/source coverage is not treated as 100% usable",
            "TrtexecAlignmentStatus=parse-only",
            "det/cls/seg/obb/pose/sem",
            "package-consumer-runtime",
            "real-model-runtime",
            "post-publish verification",
            "blocked-by-cuda-driver",
            "owner-action-required",
            "Non-Substitute Proof Kinds",
            "canCloseReleaseIssue=false",
            "performsPublish=false"
        })
        {
            Assert.Contains(marker, markdown, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void OwnerReleaseExecutionPackageKeepsPublishActionsManualAndProofBoundariesExplicit()
    {
        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerReleaseExecutionPackage.ps1"));

        Assert.Contains("Owner release execution package written", output, StringComparison.Ordinal);
        Assert.Contains("PerformsPublish=False", output, StringComparison.Ordinal);
        Assert.Contains("CanCloseReleaseIssue=False", output, StringComparison.Ordinal);

        string jsonPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-release-execution-package.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-release-execution-package.md");

        Assert.True(File.Exists(jsonPath));
        Assert.True(File.Exists(markdownPath));

        using JsonDocument package = JsonDocument.Parse(File.ReadAllText(jsonPath));
        JsonElement root = package.RootElement;

        Assert.Equal("owner-release-execution-package", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("packageState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("ownerActionStatus").GetString());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isRealModelRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.True(root.GetProperty("requiresHumanOwner").GetBoolean());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("finalGapReviewState").GetString());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("preflightState").GetString());
        AssertPostPublishRequiredEvidence(root);

        Assert.Equal(5, root.GetProperty("oneScreenReleaseHoldChecklistCount").GetInt32());
        JsonElement[] oneScreenItems = root.GetProperty("oneScreenReleaseHoldChecklist").EnumerateArray().ToArray();
        Assert.Contains(oneScreenItems, static item =>
            item.GetProperty("id").GetString() == "package-consumer-runtime" &&
            item.GetProperty("ownerVisibleBlocker").GetString()!.Contains("Package-consumer-runtime proof is missing", StringComparison.Ordinal) &&
            item.GetProperty("validatorCommand").GetString()!.Contains("Test-ExternalRuntimeProofRecord.ps1", StringComparison.Ordinal) &&
            !item.GetProperty("performsPublish").GetBoolean() &&
            !item.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains(oneScreenItems, static item =>
            item.GetProperty("id").GetString() == "post-publish-verification" &&
            item.GetProperty("ownerNextAction").GetString()!.Contains("clean external consumer", StringComparison.Ordinal) &&
            item.GetProperty("validatorCommand").GetString()!.Contains("Test-PostPublishVerificationRecord.ps1", StringComparison.Ordinal));
        Assert.All(oneScreenItems, static item =>
        {
            Assert.True(item.GetProperty("requiredRealInputs").GetArrayLength() > 0);
            Assert.True(item.GetProperty("cannotUse").GetArrayLength() > 0);
        });

        JsonElement[] steps = root.GetProperty("executionSteps").EnumerateArray().ToArray();
        Assert.Contains(steps, static item => item.GetProperty("id").GetString() == "collect-external-runtime-proof");
        Assert.Contains(steps, static item => item.GetProperty("id").GetString() == "collect-real-model-runtime-proof");
        Assert.Contains(steps, static item => item.GetProperty("id").GetString() == "collect-linux-runner-proof");
        Assert.Contains(steps, static item => item.GetProperty("id").GetString() == "validate-post-publish-verification");
        Assert.All(steps, static item =>
        {
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
        });

        JsonElement[] placeholders = root.GetProperty("manualPublishPlaceholders").EnumerateArray().ToArray();
        Assert.Contains(placeholders, static item =>
            item.GetProperty("id").GetString() == "nuget-org-managed-package" &&
            item.GetProperty("commandTemplate").GetString()!.Contains("dotnet nuget push", StringComparison.Ordinal) &&
            !item.GetProperty("performsPublish").GetBoolean() &&
            !item.GetProperty("executedByThisScript").GetBoolean());
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/external-runtime-proof-validation.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/real-model-and-package-proof-input-package.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/post-publish-verification-validation.json");
        Assert.Contains(placeholders, static item =>
            item.GetProperty("id").GetString() == "github-release-assets" &&
            item.GetProperty("commandTemplate").GetString()!.Contains("gh release upload", StringComparison.Ordinal) &&
            !item.GetProperty("performsPublish").GetBoolean());

        string[] nonSubstitutes = root.GetProperty("mustNotSubstitute").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("ProjectReference", nonSubstitutes);
        Assert.Contains("build-only", nonSubstitutes);
        Assert.Contains("parse-only", nonSubstitutes);
        Assert.Contains("sidecar-only", nonSubstitutes);
        Assert.Contains("blocked-by-cuda-driver", nonSubstitutes);

        string[] requiredOwnerInputs = root.GetProperty("requiredOwnerInputs").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains(requiredOwnerInputs, static item => item.Contains("runtimePackageKey=win-x64-trt11.0-cuda13.2-cudnn9.22", StringComparison.Ordinal));
        Assert.Contains(requiredOwnerInputs, static item => item.Contains("linuxRuntimePackageKey=linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22", StringComparison.Ordinal));
        Assert.Contains(requiredOwnerInputs, static item => item.Contains("YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom", StringComparison.Ordinal));
        Assert.Contains(requiredOwnerInputs, static item => item.Contains("post-publish logs", StringComparison.Ordinal));
        Assert.Contains("publishedPackageUrl", requiredOwnerInputs);
        Assert.Contains("runtimeSmokeLogSha256", requiredOwnerInputs);
        Assert.Contains("hostMetadata", requiredOwnerInputs);

        string[] validatorCommands = root.GetProperty("validatorCommands").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains(validatorCommands, static item => item.Contains("Test-ExternalRuntimeProofRecord.ps1", StringComparison.Ordinal) && item.Contains("-RequireExistingLog -FailOnNotProof", StringComparison.Ordinal));
        Assert.Contains(validatorCommands, static item => item.Contains("Test-PostPublishVerificationRecord.ps1", StringComparison.Ordinal) && item.Contains("-RequireExistingLog -FailOnNotProof", StringComparison.Ordinal));
        Assert.Contains(validatorCommands, static item => item.Contains("Test-LinuxRunnerEvidenceRecord.ps1", StringComparison.Ordinal));
        Assert.Contains(validatorCommands, static item => item.Contains("Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog", StringComparison.Ordinal));

        string[] expectedArtifacts = root.GetProperty("expectedArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/external-runtime-proof-record.json", expectedArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-verification-record.json", expectedArtifacts);
        Assert.Contains("artifacts/user-acceptance/sample-run-evidence-record.json", expectedArtifacts);
        Assert.Contains("artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runner-evidence-record.json", expectedArtifacts);

        string[] promotionBlockers = root.GetProperty("promotionBlockers").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("bridge-only package consumer log", promotionBlockers);
        Assert.Contains("mismatched log SHA256", promotionBlockers);
        Assert.Contains("Windows handoff for Linux proof", promotionBlockers);
        Assert.Contains("missing execution steps", promotionBlockers);

        string ownerPackageMarkdown = File.ReadAllText(markdownPath);
        Assert.Contains("一屏 Release Hold 清单", ownerPackageMarkdown, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("oneScreenReleaseHoldChecklistCount", ownerPackageMarkdown, StringComparison.Ordinal);
        Assert.Contains("Package-consumer-runtime proof is missing", ownerPackageMarkdown, StringComparison.Ordinal);
        Assert.Contains("Post-publish verification proof is missing", ownerPackageMarkdown, StringComparison.Ordinal);

        Assert.All(steps, static item =>
        {
            Assert.True(item.GetProperty("requiredOwnerInputs").GetArrayLength() > 0);
            Assert.True(item.GetProperty("validatorCommands").GetArrayLength() > 0);
            Assert.True(item.GetProperty("expectedArtifacts").GetArrayLength() > 0);
            Assert.True(item.GetProperty("promotionBlockers").GetArrayLength() > 0);
        });

        string markdown = File.ReadAllText(markdownPath);
        foreach (string marker in new[]
        {
            "performsPublish=false",
            "canCloseReleaseIssue=false",
            "Required Owner Inputs",
            "Validator Commands",
            "Expected Artifacts",
            "Promotion Blockers",
            "dotnet nuget push",
            "owner manually executes publish commands outside this script",
            "package-consumer-runtime",
            "real-model-runtime",
            "post-publish verification",
            "Post-Publish Required Evidence",
            "runtimeSmokeLogSha256",
            "hostMetadata",
            "Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json",
            "Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json",
            "Test-LinuxRunnerEvidenceRecord.ps1",
            "Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog",
            "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom",
            "blocked-by-cuda-driver",
            "mismatched log SHA256",
            "TrtexecAlignmentStatus=parse-only"
        })
        {
            Assert.Contains(marker, markdown, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void CompatibleHostProofBackfillPackageKeepsOwnerActionsExplicitAndNonPromotable()
    {
        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CompatibleHostProofBackfillPackage.ps1"));

        Assert.Contains("Compatible host proof backfill package written", output, StringComparison.Ordinal);
        Assert.Contains("PerformsPublish=False", output, StringComparison.Ordinal);
        Assert.Contains("CanCloseReleaseIssue=False", output, StringComparison.Ordinal);

        string jsonPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "compatible-host-proof-backfill-package.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "compatible-host-proof-backfill-package.md");

        Assert.True(File.Exists(jsonPath));
        Assert.True(File.Exists(markdownPath));

        using JsonDocument package = JsonDocument.Parse(File.ReadAllText(jsonPath));
        JsonElement root = package.RootElement;

        Assert.Equal("compatible-host-proof-backfill-package", root.GetProperty("recordKind").GetString());
        Assert.Equal("owner-action-required", root.GetProperty("packageState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.True(root.GetProperty("requiresCompatibleHost").GetBoolean());
        Assert.True(root.GetProperty("requiresHumanOwner").GetBoolean());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("ownerReleaseExecutionPackageState").GetString());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("preflightState").GetString());

        JsonElement[] steps = root.GetProperty("backfillSteps").EnumerateArray().ToArray();
        Assert.Contains(steps, static item => item.GetProperty("id").GetString() == "prepare-external-runtime-proof-record");
        Assert.Contains(steps, static item => item.GetProperty("id").GetString() == "collect-linux-runner-evidence");
        Assert.Contains(steps, static item => item.GetProperty("id").GetString() == "collect-real-model-sample-evidence");
        Assert.Contains(steps, static item => item.GetProperty("id").GetString() == "prepare-post-publish-verification");
        Assert.All(steps, static item =>
        {
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
        });

        JsonElement externalProof = root.GetProperty("externalRuntimeProof");
        Assert.Equal("template-only", externalProof.GetProperty("validationState").GetString());
        Assert.False(externalProof.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.Contains("proofClassification=package-consumer-runtime", externalProof.GetProperty("requiredFields").EnumerateArray().Select(static item => item.GetString()!));

        JsonElement sampleEvidence = root.GetProperty("sampleEvidence");
        Assert.False(sampleEvidence.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.Contains(sampleEvidence.GetProperty("requirements").EnumerateArray(), static item =>
            item.GetProperty("sampleName").GetString() == "YoloVision" &&
            item.GetProperty("taskScope").GetString()!.Contains("det/cls/seg/obb/pose/sem", StringComparison.Ordinal));

        JsonElement[] tracks = root.GetProperty("ownerProofFinalBackfillTracks").EnumerateArray().ToArray();
        Assert.Equal(4, tracks.Length);
        Assert.Contains(tracks, static item =>
            item.GetProperty("trackId").GetString() == "package-consumer-runtime" &&
            item.GetProperty("proofClass").GetString() == "package-consumer-runtime" &&
            item.GetProperty("inputJsonPath").GetString() == "artifacts/final-release/external-runtime-proof-record.json" &&
            item.GetProperty("validatorCommand").GetString()!.Contains("Test-ExternalRuntimeProofRecord.ps1", StringComparison.Ordinal) &&
            item.GetProperty("validatorCommand").GetString()!.Contains("-RequireExistingLog -FailOnNotProof", StringComparison.Ordinal));
        Assert.Contains(tracks, static item =>
            item.GetProperty("trackId").GetString() == "linux-runner-proof" &&
            item.GetProperty("proofClass").GetString() == "linux-runner-proof" &&
            item.GetProperty("inputJsonPath").GetString()!.Contains("linux-runner-evidence-record.json", StringComparison.Ordinal) &&
            item.GetProperty("validatorCommand").GetString()!.Contains("Test-LinuxRunnerEvidenceRecord.ps1", StringComparison.Ordinal));
        Assert.Contains(tracks, static item =>
            item.GetProperty("trackId").GetString() == "real-model-runtime" &&
            item.GetProperty("proofClass").GetString() == "real-model-runtime" &&
            item.GetProperty("inputJsonPath").GetString() == "artifacts/user-acceptance/sample-run-evidence-record.json" &&
            item.GetProperty("validatorCommand").GetString()!.Contains("Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog", StringComparison.Ordinal));
        Assert.Contains(tracks, static item =>
            item.GetProperty("trackId").GetString() == "post-publish-verification" &&
            item.GetProperty("proofClass").GetString() == "post-publish-package-consumer-runtime" &&
            item.GetProperty("inputJsonPath").GetString() == "artifacts/final-release/post-publish-verification-record.json" &&
            item.GetProperty("validatorCommand").GetString()!.Contains("Test-PostPublishVerificationRecord.ps1", StringComparison.Ordinal) &&
            item.GetProperty("validatorCommand").GetString()!.Contains("-RequireExistingLog -FailOnNotProof", StringComparison.Ordinal));
        Assert.All(tracks, static item =>
        {
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(item.GetProperty("canPromoteProof").GetBoolean());
            Assert.True(item.GetProperty("requiredOwnerInputs").GetArrayLength() > 0);
            Assert.True(item.GetProperty("logFields").GetArrayLength() > 0);
            Assert.True(item.GetProperty("sha256Fields").GetArrayLength() > 0);
            Assert.True(item.GetProperty("expectedArtifacts").GetArrayLength() > 0);
            Assert.True(item.GetProperty("promotionBlockers").GetArrayLength() > 0);
        });

        string[] requiredOwnerInputs = root.GetProperty("requiredOwnerInputs").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains(requiredOwnerInputs, static item => item.Contains("runtimePackageKey=win-x64-trt11.0-cuda13.2-cudnn9.22", StringComparison.Ordinal));
        Assert.Contains(requiredOwnerInputs, static item => item.Contains("linuxRuntimePackageKey=linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22", StringComparison.Ordinal));
        Assert.Contains(requiredOwnerInputs, static item => item.Contains("YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom", StringComparison.Ordinal));
        Assert.Contains(requiredOwnerInputs, static item => item.Contains("post-publish-verification-record", StringComparison.Ordinal) || item.Contains("postPublishProofClassification", StringComparison.Ordinal));
        Assert.Contains(requiredOwnerInputs, static item => item.Contains("validator-promoted runtime proof state", StringComparison.Ordinal));
        Assert.Contains(requiredOwnerInputs, static item => item.Contains("validator-promoted close readiness", StringComparison.Ordinal));
        Assert.DoesNotContain("canPromoteRuntimeProof=true", requiredOwnerInputs);
        Assert.DoesNotContain("canCloseReleaseIssue=true", requiredOwnerInputs);

        string[] validatorCommands = root.GetProperty("validatorCommands").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains(validatorCommands, static item => item.Contains("Test-ExternalRuntimeProofRecord.ps1", StringComparison.Ordinal) && item.Contains("-InputPath artifacts/final-release/external-runtime-proof-record.json", StringComparison.Ordinal));
        Assert.Contains(validatorCommands, static item => item.Contains("Test-LinuxRunnerEvidenceRecord.ps1", StringComparison.Ordinal));
        Assert.Contains(validatorCommands, static item => item.Contains("Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog", StringComparison.Ordinal));
        Assert.Contains(validatorCommands, static item => item.Contains("Test-PostPublishVerificationRecord.ps1", StringComparison.Ordinal) && item.Contains("-InputPath artifacts/final-release/post-publish-verification-record.json", StringComparison.Ordinal));

        string[] expectedArtifacts = root.GetProperty("expectedArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/external-runtime-proof-record.json", expectedArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-verification-record.json", expectedArtifacts);
        Assert.Contains("artifacts/user-acceptance/sample-run-evidence-record.json", expectedArtifacts);
        Assert.Contains("artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runner-evidence-record.json", expectedArtifacts);

        string[] promotionBlockers = root.GetProperty("promotionBlockers").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("bridge-only package consumer log", promotionBlockers);
        Assert.Contains("mismatched log SHA256", promotionBlockers);
        Assert.Contains("Windows handoff for Linux proof", promotionBlockers);
        Assert.Contains("missing execution steps", promotionBlockers);

        string[] nonSubstitutes = root.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("ProjectReference", nonSubstitutes);
        Assert.Contains("build-only", nonSubstitutes);
        Assert.Contains("parse-only", nonSubstitutes);
        Assert.Contains("sidecar-only", nonSubstitutes);
        Assert.Contains("blocked-by-cuda-driver", nonSubstitutes);
        Assert.Contains("local feed", nonSubstitutes);
        Assert.Contains("bridge-only package consumer log", nonSubstitutes);
        Assert.Contains("mismatched log SHA256", nonSubstitutes);

        string markdown = File.ReadAllText(markdownPath);
        foreach (string marker in new[]
        {
            "performsPublish=false",
            "canCloseReleaseIssue=false",
            "Owner Proof Final Backfill Tracks",
            "Required Owner Inputs",
            "Validator Commands",
            "Expected Artifacts",
            "Promotion Blockers",
            "package-consumer-runtime",
            "real-model-runtime",
            "post-publish verification",
            "Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json",
            "Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json",
            "Test-LinuxRunnerEvidenceRecord.ps1",
            "Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog",
            "blocked-by-cuda-driver",
            "ProjectReference",
            "mismatched log SHA256",
            "Windows handoff for Linux proof",
            "sidecar-only",
            "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom",
            "det/cls/seg/obb/pose/sem"
        })
        {
            Assert.Contains(marker, markdown, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void RealModelAndPackageProofInputPackageKeepsInputsExplicitAndNonPromotable()
    {
        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealModelAndPackageProofInputPackage.ps1"));
        Assert.Contains("Real model and package proof input package written", output, StringComparison.Ordinal);
        Assert.Contains("PerformsPublish=False", output, StringComparison.Ordinal);
        Assert.Contains("CanPublishPublicly=False", output, StringComparison.Ordinal);
        Assert.Contains("CanCloseReleaseIssue=False", output, StringComparison.Ordinal);

        string jsonPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "real-model-and-package-proof-input-package.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "real-model-and-package-proof-input-package.md");
        Assert.True(File.Exists(jsonPath));
        Assert.True(File.Exists(markdownPath));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(jsonPath));
        JsonElement root = document.RootElement;
        Assert.Equal("real-model-and-package-proof-input-package", root.GetProperty("recordKind").GetString());
        Assert.Equal("owner-action-required", root.GetProperty("packageState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());

        JsonElement[] checklists = root.GetProperty("inputChecklists").EnumerateArray().ToArray();
        Assert.Contains(checklists, static item => item.GetProperty("id").GetString() == "package-consumer-runtime-input");
        Assert.Contains(checklists, static item => item.GetProperty("id").GetString() == "real-model-runtime-input");
        Assert.Contains(checklists, static item => item.GetProperty("id").GetString() == "post-publish-verification-input");
        Assert.All(checklists, static item =>
        {
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPromoteProof").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
        });

        JsonElement packageConsumer = root.GetProperty("packageConsumerRuntimeInputChecklist");
        Assert.Equal("package-consumer-runtime", packageConsumer.GetProperty("proofClass").GetString());
        Assert.Equal("Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof", packageConsumer.GetProperty("validator").GetString());
        Assert.Contains(packageConsumer.GetProperty("requiredInputs").EnumerateArray(), static item =>
            item.GetString()!.Contains("managed nupkg SHA256", StringComparison.Ordinal));
        Assert.Contains(packageConsumer.GetProperty("requiredInputs").EnumerateArray(), static item =>
            item.GetString()!.Contains("no ProjectReference", StringComparison.Ordinal));

        JsonElement realModel = root.GetProperty("realModelRuntimeInputChecklist");
        Assert.Equal("real-model-runtime", realModel.GetProperty("proofClass").GetString());
        Assert.Contains(realModel.GetProperty("requiredInputs").EnumerateArray(), static item =>
            item.GetString()!.Contains("YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom", StringComparison.Ordinal));
        Assert.Contains(realModel.GetProperty("requiredInputs").EnumerateArray(), static item =>
            item.GetString()!.Contains("det/cls/seg/obb/pose/sem", StringComparison.Ordinal));
        Assert.Contains(realModel.GetProperty("requiredInputs").EnumerateArray(), static item =>
            item.GetString()!.Contains("never package-consumer-runtime", StringComparison.Ordinal));

        JsonElement postPublish = root.GetProperty("postPublishVerificationInputChecklist");
        Assert.Equal("post-publish-package-consumer-runtime", postPublish.GetProperty("proofClass").GetString());
        Assert.Equal("Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof", postPublish.GetProperty("validator").GetString());

        string[] nonSubstitutes = root.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("template", nonSubstitutes);
        Assert.Contains("draft", nonSubstitutes);
        Assert.Contains("runbook", nonSubstitutes);
        Assert.Contains("collection package", nonSubstitutes);
        Assert.Contains("local feed", nonSubstitutes);
        Assert.Contains("ProjectReference", nonSubstitutes);
        Assert.Contains("build-only", nonSubstitutes);
        Assert.Contains("parse-only", nonSubstitutes);
        Assert.Contains("sidecar-only", nonSubstitutes);
        Assert.Contains("blocked-by-cuda-driver", nonSubstitutes);

        string markdown = File.ReadAllText(markdownPath);
        foreach (string marker in new[]
        {
            "recordKind=real-model-and-package-proof-input-package",
            "packageState=owner-action-required",
            "performsPublish=false",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "package-consumer-runtime",
            "real-model-runtime",
            "post-publish verification",
            "blocked-by-cuda-driver",
            "ProjectReference",
            "sidecar-only",
            "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof",
            "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof",
            "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom",
            "det、cls、seg、obb、pose、sem"
        })
        {
            Assert.Contains(marker, markdown, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void ReleaseCloseGapDashboardSummarizesRemainingProofBlockersWithoutPromotingInputs()
    {
        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseGapDashboard.ps1"));
        Assert.Contains("Release close gap dashboard written", output, StringComparison.Ordinal);
        Assert.Contains("DashboardState=blocked-real-proof-required", output, StringComparison.Ordinal);
        Assert.Contains("PerformsPublish=False", output, StringComparison.Ordinal);
        Assert.Contains("CanCloseReleaseIssue=False", output, StringComparison.Ordinal);

        string jsonPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-close-gap-dashboard.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-close-gap-dashboard.md");
        Assert.True(File.Exists(jsonPath));
        Assert.True(File.Exists(markdownPath));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(jsonPath));
        JsonElement root = document.RootElement;
        Assert.Equal("release-close-gap-dashboard", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("dashboardState").GetString());
        Assert.Equal("owner-action-required", root.GetProperty("packageState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("ownerActionStatus").GetString());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isRealModelRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishVerificationProof").GetBoolean());
        string[] expectedPostPublishRequiredEvidence =
        {
            "selectedChannel",
            "channelSourceUri",
            "publishedPackageUrl",
            "managedPackageUrl",
            "runtimePackageUrl",
            "managedNupkgSha256",
            "runtimeNupkgSha256",
            "cleanConsumerRootOutsideRepository",
            "consumerProjectPath",
            "noProjectReference",
            "noLocalPackageSource",
            "noLocalNupkgPackageReference",
            "restoreLogPath",
            "nativeAssetListingSha256",
            "dependencyProbeLogPath",
            "dependencyProbeLogSha256",
            "runtimeSmokeLogPath",
            "runtimeSmokeLogSha256",
            "runtimeSmokePassed",
            "runtimeSmokeExitCode",
            "stdoutSummary",
            "stderrSummary",
            "hostMetadata",
        };
        string[] postPublishRequiredEvidence = root.GetProperty("postPublishRequiredEvidence").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Equal(expectedPostPublishRequiredEvidence.Length, root.GetProperty("postPublishRequiredEvidenceCount").GetInt32());
        Assert.Equal(expectedPostPublishRequiredEvidence.Length, root.GetProperty("releaseRuntimeProofExecutionMatrixPostPublishRequiredEvidenceCount").GetInt32());
        Assert.Equal(expectedPostPublishRequiredEvidence.Order(StringComparer.Ordinal).ToArray(), postPublishRequiredEvidence.Order(StringComparer.Ordinal).ToArray());
        Assert.Equal(5, root.GetProperty("gapCount").GetInt32());
        Assert.True(root.GetProperty("preflightFailedItemCount").GetInt32() >= 5);
        Assert.Equal("blocked-evidence-incomplete", root.GetProperty("releaseEvidenceBundleState").GetString());
        Assert.Equal("owner-action-required", root.GetProperty("realModelAndPackageProofInputPackageState").GetString());
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/release-evidence-bundle.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/release-promotion-issue-record.json");
        Assert.Contains(root.GetProperty("sourceEvidence").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/real-model-and-package-proof-input-package.json");

        JsonElement[] gaps = root.GetProperty("gapItems").EnumerateArray().ToArray();
        Assert.Equal(5, gaps.Length);
        Assert.All(gaps, static item =>
        {
            Assert.False(item.GetProperty("passed").GetBoolean());
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.True(item.GetProperty("requiredRealInputs").GetArrayLength() > 0);
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("validator").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("ownerAction").GetString()));
            Assert.Contains("blocked-by-cuda-driver", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
            Assert.Contains("Skipped=True", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
            Assert.Contains("bridge-only wrapper surface", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
            Assert.Contains("WrapperSurfaceEvidenceKind=compile-surface-proof", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
            Assert.Contains("IsRuntimeExecutionProof=False", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
            Assert.Contains("Parser/ParserRefitter diagnostic snapshots", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
            Assert.Contains("copied managed diagnostic snapshot", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
            Assert.Contains("ProjectReference", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
            Assert.Contains("sidecar-only", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
        });

        Assert.Contains(gaps, static item =>
            item.GetProperty("gapId").GetString() == "owner-authorization" &&
            item.GetProperty("proofClass").GetString() == "owner-authorization");
        Assert.Contains(gaps, static item =>
            item.GetProperty("gapId").GetString() == "package-consumer-runtime" &&
            item.GetProperty("proofClass").GetString() == "package-consumer-runtime" &&
            item.GetProperty("validator").GetString() == "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof");
        Assert.Contains(gaps, static item =>
            item.GetProperty("gapId").GetString() == "linux-runner-proof" &&
            item.GetProperty("proofClass").GetString() == "linux-runner-proof");
        Assert.Contains(gaps, static item =>
            item.GetProperty("gapId").GetString() == "real-model-runtime" &&
            item.GetProperty("proofClass").GetString() == "real-model-runtime" &&
            item.GetProperty("currentState").GetString()!.Contains("inputPackageState=owner-action-required", StringComparison.Ordinal));
        Assert.Contains(gaps, static item =>
            item.GetProperty("gapId").GetString() == "post-publish-verification" &&
            item.GetProperty("proofClass").GetString() == "post-publish verification" &&
            item.GetProperty("validator").GetString() == "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof" &&
            item.GetProperty("requiredRealInputs").EnumerateArray().Select(static value => value.GetString()!).Contains("publishedPackageUrl", StringComparer.Ordinal) &&
            item.GetProperty("requiredRealInputs").EnumerateArray().Select(static value => value.GetString()!).Contains("runtimeSmokeLogSha256", StringComparer.Ordinal) &&
            item.GetProperty("requiredRealInputs").EnumerateArray().Select(static value => value.GetString()!).Contains("hostMetadata", StringComparer.Ordinal));

        string[] nonSubstitutes = root.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("input package", nonSubstitutes);
        Assert.Contains("ProjectReference", nonSubstitutes);
        Assert.Contains("build-only", nonSubstitutes);
        Assert.Contains("parse-only", nonSubstitutes);
        Assert.Contains("sidecar-only", nonSubstitutes);
        Assert.Contains("Skipped=True", nonSubstitutes);
        Assert.Contains("blocked-by-cuda-driver", nonSubstitutes);
        Assert.Contains("bridge-only wrapper surface", nonSubstitutes);
        Assert.Contains("WrapperSurfaceEvidenceKind=compile-surface-proof", nonSubstitutes);
        Assert.Contains("IsRuntimeExecutionProof=False", nonSubstitutes);
        Assert.Contains("Parser/ParserRefitter diagnostic snapshots", nonSubstitutes);
        Assert.Contains("copied managed diagnostic snapshot", nonSubstitutes);
        Assert.Contains("mismatched log SHA256", nonSubstitutes);

        Assert.Contains(root.GetProperty("sourceArtifacts").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/release-close-preflight.json");
        Assert.Contains(root.GetProperty("sourceArtifacts").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/release-evidence-bundle.json");
        Assert.Contains(root.GetProperty("sourceArtifacts").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/release-promotion-issue-record.json");
        Assert.Contains(root.GetProperty("sourceArtifacts").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/real-model-and-package-proof-input-package.json");

        string markdown = File.ReadAllText(markdownPath);
        foreach (string marker in new[]
        {
            "recordKind=release-close-gap-dashboard",
            "blocked-real-proof-required",
            "performsPublish=false",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "postPublishRequiredEvidenceCount",
            "runtimeSmokeLogSha256",
            "hostMetadata",
            "package-consumer-runtime",
            "real-model-runtime",
            "post-publish verification",
            "blocked-by-cuda-driver",
            "Skipped=True",
            "bridge-only wrapper surface",
            "WrapperSurfaceEvidenceKind=compile-surface-proof",
            "IsRuntimeExecutionProof=False",
            "ONNX Parser diagnostic snapshot",
            "ONNX ParserRefitter diagnostic snapshot",
            "copied managed diagnostic snapshot",
            "ProjectReference",
            "sidecar-only",
            "real-model-and-package-proof-input-package",
            "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom",
            "det、cls、seg、obb、pose、sem"
        })
        {
            Assert.Contains(marker, markdown, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void CompatibleHostProofExecutionPackAggregatesReleaseCloseGapsWithoutPromotingProof()
    {
        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CompatibleHostProofExecutionPack.ps1"));
        Assert.Contains("Compatible host proof execution pack written", output, StringComparison.Ordinal);
        Assert.Contains("PackageState=owner-action-required", output, StringComparison.Ordinal);
        Assert.Contains("BlockerCount=5", output, StringComparison.Ordinal);
        Assert.Contains("PerformsPublish=False", output, StringComparison.Ordinal);
        Assert.Contains("CanPublishPublicly=False", output, StringComparison.Ordinal);
        Assert.Contains("CanCloseReleaseIssue=False", output, StringComparison.Ordinal);

        string jsonPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "compatible-host-proof-execution-pack.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "compatible-host-proof-execution-pack.md");
        Assert.True(File.Exists(jsonPath));
        Assert.True(File.Exists(markdownPath));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(jsonPath));
        JsonElement root = document.RootElement;
        Assert.Equal("compatible-host-proof-execution-pack", root.GetProperty("recordKind").GetString());
        Assert.Equal("owner-action-required", root.GetProperty("packageState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal(5, root.GetProperty("blockerCount").GetInt32());

        JsonElement[] items = root.GetProperty("executionItems").EnumerateArray().ToArray();
        Assert.Equal(5, items.Length);
        Assert.Contains(items, static item => item.GetProperty("gapId").GetString() == "owner-authorization");
        Assert.Contains(items, static item => item.GetProperty("gapId").GetString() == "package-consumer-runtime" && item.GetProperty("proofClass").GetString() == "package-consumer-runtime");
        Assert.Contains(items, static item => item.GetProperty("gapId").GetString() == "linux-runner-proof" && item.GetProperty("proofClass").GetString() == "linux-runner-proof");
        Assert.Contains(items, static item => item.GetProperty("gapId").GetString() == "real-model-runtime" && item.GetProperty("proofClass").GetString() == "real-model-runtime");
        Assert.Contains(items, static item => item.GetProperty("gapId").GetString() == "post-publish-verification" && item.GetProperty("proofClass").GetString() == "post-publish verification");

        Assert.All(items, static item =>
        {
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.Equal("owner-action-required", item.GetProperty("ownerAction").GetString());
            Assert.True(item.GetProperty("requiredInputs").GetArrayLength() > 0);
            Assert.True(item.GetProperty("validateCommands").GetArrayLength() > 0);
            Assert.Contains("ProjectReference", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
            Assert.Contains("blocked-by-cuda-driver", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
            Assert.Contains("Skipped=True", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
            Assert.Contains("bridge-only package consumer log", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
            Assert.Contains("bridge-only wrapper surface", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
            Assert.Contains("WrapperSurfaceEvidenceKind=compile-surface-proof", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
            Assert.Contains("IsRuntimeExecutionProof=False", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
            Assert.Contains("Parser/ParserRefitter diagnostic snapshots", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
            Assert.Contains("copied managed diagnostic snapshot", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
            Assert.Contains("sidecar-only", item.GetProperty("nonSubstitutes").EnumerateArray().Select(static value => value.GetString()!));
        });

        string[] nonSubstitutes = root.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("input package", nonSubstitutes);
        Assert.Contains("collection package", nonSubstitutes);
        Assert.Contains("local feed", nonSubstitutes);
        Assert.Contains("ProjectReference", nonSubstitutes);
        Assert.Contains("build-only", nonSubstitutes);
        Assert.Contains("parse-only", nonSubstitutes);
        Assert.Contains("sidecar-only", nonSubstitutes);
        Assert.Contains("Skipped=True", nonSubstitutes);
        Assert.Contains("blocked-by-cuda-driver", nonSubstitutes);
        Assert.Contains("bridge-only package consumer log", nonSubstitutes);
        Assert.Contains("bridge-only wrapper surface", nonSubstitutes);
        Assert.Contains("WrapperSurfaceEvidenceKind=compile-surface-proof", nonSubstitutes);
        Assert.Contains("IsRuntimeExecutionProof=False", nonSubstitutes);
        Assert.Contains("Parser/ParserRefitter diagnostic snapshots", nonSubstitutes);
        Assert.Contains("copied managed diagnostic snapshot", nonSubstitutes);

        Assert.Contains(root.GetProperty("sourceArtifacts").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/release-close-gap-dashboard.json");
        Assert.Contains(root.GetProperty("sourceArtifacts").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/release-evidence-bundle.json");
        Assert.Contains(root.GetProperty("sourceArtifacts").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/release-promotion-issue-record.json");

        string markdown = File.ReadAllText(markdownPath);
        foreach (string marker in new[]
        {
            "recordKind=compatible-host-proof-execution-pack",
            "packageState=owner-action-required",
            "performsPublish=false",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "owner-authorization",
            "package-consumer-runtime",
            "linux-runner-proof",
            "real-model-runtime",
            "post-publish verification",
            "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof",
            "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof",
            "Test-LinuxRunnerEvidenceRecord.ps1",
            "Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog",
            "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom",
            "det、cls、seg、obb、pose、sem",
            "Skipped=True",
            "bridge-only package consumer log",
            "bridge-only wrapper surface",
            "WrapperSurfaceEvidenceKind=compile-surface-proof",
            "IsRuntimeExecutionProof=False",
            "ONNX Parser diagnostic snapshot",
            "ONNX ParserRefitter diagnostic snapshot",
            "copied managed diagnostic snapshot",
            "blocked-by-cuda-driver",
            "ProjectReference",
            "sidecar-only"
        })
        {
            Assert.Contains(marker, markdown, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void OwnerProofExecutionChainKeepsFiveBlockersActionableAndNonPromotable()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerReleaseExecutionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CompatibleHostProofBackfillPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CompatibleHostProofExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseGapDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseClosePreflight.ps1"));

        string[] expectedBlockers =
        {
            "owner-authorization",
            "package-consumer-runtime",
            "linux-runner-proof",
            "real-model-runtime",
            "post-publish-verification",
        };

        Dictionary<string, string[]> expectedValidators = new(StringComparer.Ordinal)
        {
            ["owner-authorization"] =
            [
                "Test-ReleaseOwnerApprovalInput.ps1",
                "Test-OwnerAuthorizedPublishCommandPlan.ps1",
            ],
            ["package-consumer-runtime"] =
            [
                "Test-ExternalRuntimeProofRecord.ps1",
                "-RequireExistingLog",
                "-FailOnNotProof",
            ],
            ["linux-runner-proof"] =
            [
                "Test-LinuxRunnerEvidenceRecord.ps1",
            ],
            ["real-model-runtime"] =
            [
                "Test-SampleAssetManifest.ps1",
                "Test-SampleRunEvidenceRecord.ps1",
                "-RequireExistingLog",
            ],
            ["post-publish-verification"] =
            [
                "Test-PostPublishVerificationRecord.ps1",
                "-RequireExistingLog",
                "-FailOnNotProof",
            ],
        };

        using JsonDocument executionPack = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "compatible-host-proof-execution-pack.json")));
        JsonElement executionPackRoot = executionPack.RootElement;
        Assert.Equal("owner-action-required", executionPackRoot.GetProperty("packageState").GetString());
        Assert.False(executionPackRoot.GetProperty("performsPublish").GetBoolean());
        Assert.False(executionPackRoot.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(executionPackRoot.GetProperty("canCloseReleaseIssue").GetBoolean());

        JsonElement[] executionItems = executionPackRoot.GetProperty("executionItems").EnumerateArray().ToArray();
        Assert.Equal(expectedBlockers.Length, executionItems.Length);
        foreach (string blocker in expectedBlockers)
        {
            JsonElement item = Assert.Single(executionItems, candidate => candidate.GetProperty("gapId").GetString() == blocker);
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.True(item.GetProperty("requiredInputs").GetArrayLength() >= 5);
            Assert.True(item.GetProperty("expectedArtifacts").GetArrayLength() >= 1);
            Assert.True(item.GetProperty("nonSubstitutes").GetArrayLength() >= 10);

            string validators = string.Join(" ", item.GetProperty("validateCommands").EnumerateArray().Select(static value => value.GetString()));
            foreach (string expectedValidator in expectedValidators[blocker])
            {
                Assert.Contains(expectedValidator, validators, StringComparison.Ordinal);
            }
        }

        using JsonDocument gapDashboard = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-close-gap-dashboard.json")));
        JsonElement[] gapItems = gapDashboard.RootElement.GetProperty("gapItems").EnumerateArray().ToArray();
        Assert.Equal(expectedBlockers.Length, gapItems.Length);
        foreach (string blocker in expectedBlockers)
        {
            JsonElement item = Assert.Single(gapItems, candidate => candidate.GetProperty("gapId").GetString() == blocker);
            Assert.False(item.GetProperty("passed").GetBoolean());
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.True(item.GetProperty("requiredRealInputs").GetArrayLength() >= 5);
            Assert.True(item.GetProperty("relatedArtifacts").GetArrayLength() >= 1);

            string validator = item.GetProperty("validator").GetString()!;
            foreach (string expectedValidator in expectedValidators[blocker])
            {
                Assert.Contains(expectedValidator, validator, StringComparison.Ordinal);
            }
        }

        using JsonDocument backfillPackage = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "compatible-host-proof-backfill-package.json")));
        JsonElement[] backfillTracks = backfillPackage.RootElement.GetProperty("ownerProofFinalBackfillTracks").EnumerateArray().ToArray();
        Assert.Equal(4, backfillTracks.Length);
        foreach (string trackId in expectedBlockers.Where(static id => id != "owner-authorization"))
        {
            JsonElement track = Assert.Single(backfillTracks, candidate => candidate.GetProperty("trackId").GetString() == trackId);
            Assert.False(track.GetProperty("performsPublish").GetBoolean());
            Assert.False(track.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(track.GetProperty("canPromoteProof").GetBoolean());
            Assert.True(track.GetProperty("requiredOwnerInputs").GetArrayLength() >= 7);
            Assert.True(track.GetProperty("expectedArtifacts").GetArrayLength() >= 2);

            string validator = track.GetProperty("validatorCommand").GetString()!;
            foreach (string expectedValidator in expectedValidators[trackId])
            {
                Assert.Contains(expectedValidator, validator, StringComparison.Ordinal);
            }
        }
    }

    [Fact]
    public void ReleaseCloseArtifactsShareOwnerProofBlockerIdsAndValidators()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerReleaseExecutionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CompatibleHostProofExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseRuntimeProofExecutionMatrix.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseProofReadinessSnapshot.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofInputReadiness.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerProofInputReadiness.ps1"), "-FailOnInvalid");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseGapDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseClosePreflight.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePublishExecutionChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleasePromotionIssueRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCandidateFinalEvidenceFreeze.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCandidateFreezeSummary.ps1"));

        string[] expectedBlockers =
        {
            "owner-authorization",
            "package-consumer-runtime",
            "linux-runner-proof",
            "real-model-runtime",
            "post-publish-verification",
        };

        using JsonDocument executionPack = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "compatible-host-proof-execution-pack.json")));
        using JsonDocument releaseEvidence = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.json")));
        using JsonDocument gapDashboard = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-close-gap-dashboard.json")));
        using JsonDocument ownerPackage = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-release-execution-package.json")));
        using JsonDocument proofSnapshot = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-proof-readiness-snapshot.json")));
        using JsonDocument ownerProofInputReadiness = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-proof-input-readiness.json")));
        using JsonDocument ownerProofInputReadinessValidation = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-proof-input-readiness-validation.json")));
        using JsonDocument preflight = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-close-preflight.json")));
        using JsonDocument publishChecklist = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-publish-execution-checklist.json")));
        using JsonDocument promotionIssue = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-promotion-issue-record.json")));
        using JsonDocument finalFreeze = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-candidate-final-evidence-freeze.json")));
        using JsonDocument freezeSummary = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "release", "release-candidate-freeze-summary.json")));

        string[] executionIds = executionPack.RootElement
            .GetProperty("executionItems")
            .EnumerateArray()
            .Select(static item => item.GetProperty("gapId").GetString()!)
            .Order(StringComparer.Ordinal)
            .ToArray();
        string[] gapIds = gapDashboard.RootElement
            .GetProperty("gapItems")
            .EnumerateArray()
            .Select(static item => item.GetProperty("gapId").GetString()!)
            .Order(StringComparer.Ordinal)
            .ToArray();
        Assert.Equal(expectedBlockers.Order(StringComparer.Ordinal).ToArray(), executionIds);
        Assert.Equal(expectedBlockers.Order(StringComparer.Ordinal).ToArray(), gapIds);
        AssertReleaseHoldChecklistMirrorsOwnerPackage(ownerPackage.RootElement, executionPack.RootElement);
        AssertReleaseHoldChecklistMirrorsOwnerPackage(ownerPackage.RootElement, releaseEvidence.RootElement);
        AssertReleaseHoldChecklistMirrorsOwnerPackage(ownerPackage.RootElement, gapDashboard.RootElement);
        AssertReleaseHoldChecklistMirrorsOwnerPackage(ownerPackage.RootElement, publishChecklist.RootElement);
        AssertReleaseHoldChecklistMirrorsOwnerPackage(ownerPackage.RootElement, promotionIssue.RootElement);
        AssertReleaseHoldChecklistMirrorsOwnerPackage(ownerPackage.RootElement, finalFreeze.RootElement);
        AssertReleaseHoldChecklistMirrorsOwnerPackage(ownerPackage.RootElement, freezeSummary.RootElement);

        foreach (string artifactPath in new[]
        {
            Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-publish-execution-checklist.md"),
            Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-promotion-issue-record.md"),
            Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-candidate-final-evidence-freeze.md"),
            Path.Combine(RepositoryPaths.Root, "artifacts", "release", "release-candidate-freeze-summary.md"),
        })
        {
            string markdown = File.ReadAllText(artifactPath);
            Assert.Contains("One-Screen Release Hold Checklist", markdown, StringComparison.Ordinal);
            Assert.Contains("package-consumer-runtime", markdown, StringComparison.Ordinal);
            Assert.Contains("post-publish-verification", markdown, StringComparison.Ordinal);
            Assert.Contains("ProjectReference", markdown, StringComparison.Ordinal);
            Assert.Contains("blocked-by-cuda-driver", markdown, StringComparison.Ordinal);
        }

        Assert.Contains(publishChecklist.RootElement.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/owner-release-execution-package.json");
        Assert.Contains(promotionIssue.RootElement.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/owner-release-execution-package.json");
        Assert.Contains(finalFreeze.RootElement.GetProperty("sourceArtifacts").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/owner-release-execution-package.json");
        Assert.Contains(freezeSummary.RootElement.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/owner-release-execution-package.json");
        AssertReleaseProofReadinessSnapshotIsBlocked(proofSnapshot.RootElement);
        AssertOwnerProofInputReadinessIsBlocked(ownerProofInputReadiness.RootElement);
        AssertOwnerProofInputReadinessValidationIsValid(ownerProofInputReadinessValidation.RootElement);
        AssertReleaseProofReadinessSnapshotIsTransitive(releaseEvidence.RootElement, "sourceEvidence");
        AssertReleaseProofReadinessSnapshotIsTransitive(releaseEvidence.RootElement, "sourceArtifacts");
        AssertReleaseProofReadinessSnapshotIsTransitive(publishChecklist.RootElement, "sourceEvidence");
        AssertReleaseProofReadinessSnapshotIsTransitive(promotionIssue.RootElement, "sourceEvidence");
        AssertReleaseProofReadinessSnapshotIsTransitive(finalFreeze.RootElement, "sourceArtifacts");
        AssertReleaseProofReadinessSnapshotIsTransitive(freezeSummary.RootElement, "sourceEvidence");
        AssertOwnerProofInputReadinessIsTransitive(releaseEvidence.RootElement, "sourceEvidence");
        AssertOwnerProofInputReadinessIsTransitive(releaseEvidence.RootElement, "sourceArtifacts");
        AssertOwnerProofInputReadinessIsTransitive(publishChecklist.RootElement, "sourceEvidence");
        AssertOwnerProofInputReadinessIsTransitive(promotionIssue.RootElement, "sourceEvidence");
        AssertOwnerProofInputReadinessIsTransitive(finalFreeze.RootElement, "sourceArtifacts");
        AssertOwnerProofInputReadinessIsTransitive(freezeSummary.RootElement, "sourceEvidence");
        AssertOwnerProofInputReadinessValidationIsTransitive(releaseEvidence.RootElement, "sourceEvidence");
        AssertOwnerProofInputReadinessValidationIsTransitive(releaseEvidence.RootElement, "sourceArtifacts");
        AssertOwnerProofInputReadinessValidationIsTransitive(publishChecklist.RootElement, "sourceEvidence");
        AssertOwnerProofInputReadinessValidationIsTransitive(promotionIssue.RootElement, "sourceEvidence");
        AssertOwnerProofInputReadinessValidationIsTransitive(finalFreeze.RootElement, "sourceArtifacts");
        AssertOwnerProofInputReadinessValidationIsTransitive(freezeSummary.RootElement, "sourceEvidence");
        AssertReleaseProofReadinessSnapshotState(releaseEvidence.RootElement);
        AssertReleaseProofReadinessSnapshotState(publishChecklist.RootElement);
        AssertReleaseProofReadinessSnapshotState(promotionIssue.RootElement);
        AssertReleaseProofReadinessSnapshotState(finalFreeze.RootElement);
        AssertReleaseProofReadinessSnapshotState(freezeSummary.RootElement);
        AssertOwnerProofInputReadinessState(releaseEvidence.RootElement);
        AssertOwnerProofInputReadinessState(publishChecklist.RootElement);
        AssertOwnerProofInputReadinessState(promotionIssue.RootElement);
        AssertOwnerProofInputReadinessState(finalFreeze.RootElement);
        AssertOwnerProofInputReadinessState(freezeSummary.RootElement);
        AssertOwnerProofInputReadinessValidationState(releaseEvidence.RootElement);
        AssertOwnerProofInputReadinessValidationState(publishChecklist.RootElement);
        AssertOwnerProofInputReadinessValidationState(promotionIssue.RootElement);
        AssertOwnerProofInputReadinessValidationState(finalFreeze.RootElement);
        AssertOwnerProofInputReadinessValidationState(freezeSummary.RootElement);

        Assert.Equal(23, releaseEvidence.RootElement.GetProperty("postPublishRequiredEvidenceCount").GetInt32());
        Assert.Equal(23, gapDashboard.RootElement.GetProperty("postPublishRequiredEvidenceCount").GetInt32());
        Assert.Equal(23, preflight.RootElement.GetProperty("postPublishRequiredEvidenceCount").GetInt32());
        Assert.Equal(
            releaseEvidence.RootElement.GetProperty("postPublishRequiredEvidence").EnumerateArray().Select(static item => item.GetString()!).Order(StringComparer.Ordinal).ToArray(),
            gapDashboard.RootElement.GetProperty("postPublishRequiredEvidence").EnumerateArray().Select(static item => item.GetString()!).Order(StringComparer.Ordinal).ToArray());
        Assert.Equal(
            preflight.RootElement.GetProperty("postPublishRequiredEvidence").EnumerateArray().Select(static item => item.GetString()!).Order(StringComparer.Ordinal).ToArray(),
            gapDashboard.RootElement.GetProperty("postPublishRequiredEvidence").EnumerateArray().Select(static item => item.GetString()!).Order(StringComparer.Ordinal).ToArray());

        string preflightText = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-close-preflight.json"));
        foreach (string marker in new[]
        {
            "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof",
            "Test-LinuxRunnerEvidenceRecord.ps1",
            "Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog",
            "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof",
            "owner authorization",
            "blocked-real-proof-required",
        })
        {
            Assert.Contains(marker, preflightText, StringComparison.OrdinalIgnoreCase);
        }

        Assert.False(preflight.RootElement.GetProperty("performsPublish").GetBoolean());
        Assert.False(preflight.RootElement.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(preflight.RootElement.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal("blocked-real-proof-required", preflight.RootElement.GetProperty("preflightState").GetString());
    }

    [Fact]
    public void OwnerProofFinalHoldArtifactsFrontDoorsAndTemplatesStayBlockedUntilRealRecordsPass()
    {
        using JsonDocument freeze = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-candidate-final-evidence-freeze.json")));
        using JsonDocument externalTemplate = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "external-runtime-proof-record-template.json")));
        using JsonDocument externalDraft = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "external-runtime-proof-record.draft.json")));
        using JsonDocument externalValidation = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "external-runtime-proof-validation.json")));
        using JsonDocument postPublishTemplate = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "post-publish-verification-record-template.json")));
        using JsonDocument postPublishValidation = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "post-publish-verification-validation.json")));

        JsonElement freezeRoot = freeze.RootElement;
        Assert.Equal("blocked-real-proof-required", freezeRoot.GetProperty("freezeState").GetString());
        Assert.False(freezeRoot.GetProperty("performsPublish").GetBoolean());
        Assert.False(freezeRoot.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(freezeRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal(5, freezeRoot.GetProperty("blockerCount").GetInt32());

        string[] expectedBlockers =
        [
            "owner-authorization",
            "package-consumer-runtime",
            "linux-runner-proof",
            "real-model-runtime",
            "post-publish-verification",
        ];
        JsonElement[] blockers = freezeRoot.GetProperty("remainingBlockers").EnumerateArray().ToArray();
        Assert.Equal(expectedBlockers.Length, blockers.Length);

        foreach (string blockerId in expectedBlockers)
        {
            JsonElement blocker = Assert.Single(blockers, item => item.GetProperty("gapId").GetString() == blockerId);
            Assert.False(blocker.GetProperty("passed").GetBoolean());
            Assert.False(blocker.GetProperty("performsPublish").GetBoolean());
            Assert.False(blocker.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(blocker.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(blocker.GetProperty("validator").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(blocker.GetProperty("requiredRealEvidence").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(blocker.GetProperty("ownerAction").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(blocker.GetProperty("freezeBoundary").GetString()));

            string[] nonSubstitutes = blocker
                .GetProperty("nonSubstitutes")
                .EnumerateArray()
                .Select(static item => item.GetString() ?? string.Empty)
                .ToArray();

            foreach (string marker in new[]
            {
                "template",
                "draft",
                "runbook",
                "collection package",
                "input package",
                "local feed",
                "ProjectReference",
                "blocked-by-cuda-driver",
                "build-only",
                "parse-only",
                "sidecar-only",
                "Windows handoff for Linux proof",
                "owner-action-required without validator pass",
            })
            {
                Assert.Contains(marker, nonSubstitutes);
            }
        }

        JsonElement externalTemplateRoot = externalTemplate.RootElement;
        Assert.True(externalTemplateRoot.GetProperty("templateOnly").GetBoolean());
        Assert.Equal("template-only", externalTemplateRoot.GetProperty("proofClassification").GetString());
        Assert.False(externalTemplateRoot.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.False(externalTemplateRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.Equal("blocked-by-cuda-driver", externalTemplateRoot.GetProperty("currentRuntimeProofStatus").GetString());
        Assert.Contains(externalTemplateRoot.GetProperty("promotionRules").EnumerateArray(), static item =>
            (item.GetString() ?? string.Empty).Contains("proofClassification must be package-consumer-runtime", StringComparison.Ordinal));
        Assert.Contains(externalTemplateRoot.GetProperty("promotionRules").EnumerateArray(), static item =>
            (item.GetString() ?? string.Empty).Contains("blocked-by-cuda-driver is not smoke passed", StringComparison.Ordinal));

        JsonElement externalDraftRoot = externalDraft.RootElement;
        Assert.True(externalDraftRoot.GetProperty("templateOnly").GetBoolean());
        Assert.Equal("dependency-probe-only", externalDraftRoot.GetProperty("proofClassification").GetString());
        Assert.False(externalDraftRoot.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.True(externalDraftRoot.GetProperty("isDependencyProbeOnly").GetBoolean());
        Assert.False(externalDraftRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.Equal("blocked-by-cuda-driver", externalDraftRoot.GetProperty("currentRuntimeProofStatus").GetString());

        JsonElement externalValidationRoot = externalValidation.RootElement;
        Assert.Equal("template-only", externalValidationRoot.GetProperty("validationState").GetString());
        Assert.Equal("template-only", externalValidationRoot.GetProperty("proofClassification").GetString());
        Assert.False(externalValidationRoot.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.False(externalValidationRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.True(externalValidationRoot.GetProperty("failedProofItemCount").GetInt32() > 0);

        JsonElement postPublishTemplateRoot = postPublishTemplate.RootElement;
        Assert.True(postPublishTemplateRoot.GetProperty("templateOnly").GetBoolean());
        Assert.Equal("template-only", postPublishTemplateRoot.GetProperty("postPublishProofClassification").GetString());
        Assert.False(postPublishTemplateRoot.GetProperty("performsPublish").GetBoolean());
        Assert.False(postPublishTemplateRoot.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(postPublishTemplateRoot.GetProperty("canCloseReleaseIssue").GetBoolean());

        JsonElement postPublishValidationRoot = postPublishValidation.RootElement;
        Assert.Equal("template-only", postPublishValidationRoot.GetProperty("validationState").GetString());
        Assert.Equal("template-only", postPublishValidationRoot.GetProperty("postPublishProofClassification").GetString());
        Assert.False(postPublishValidationRoot.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(postPublishValidationRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.True(postPublishValidationRoot.GetProperty("failedProofItemCount").GetInt32() > 0);

        string readmeFrontDoors = string.Join(
            Environment.NewLine,
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md")),
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md")));
        string frontDoorAndOwnerDocs = string.Join(
            Environment.NewLine,
            readmeFrontDoors,
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-owner-action-checklist-final-hold.md")),
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-hold-final-inspection.md")),
            File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-proof-non-substitutes.md")));

        foreach (string marker in new[]
        {
            "blocked-real-proof-required",
            "performsPublish=false",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "owner authorization",
            "package-consumer-runtime",
            "Linux runner proof",
            "real-model-runtime",
            "post-publish verification",
            "Owner Authorization Proof Gate",
            "requiredFieldCount=14",
            "missingOwnerInputCount=14",
            "approvedCommandPlanSha256",
            "approvedProofBundleSha256",
            "credentialHandlingAcknowledged",
            "nvidiaRedistributionApproval",
            "manualMaterializationPrerequisites",
            "postPublishRequiredEvidence",
            "Test-ReleaseOwnerApprovalInput.ps1",
            "Test-OwnerAuthorizedPublishCommandPlan.ps1",
            "Test-ExternalRuntimeProofRecord.ps1",
            "Test-SampleAssetManifest.ps1",
            "Test-SampleRunEvidenceRecord.ps1",
            "Test-LinuxRunnerEvidenceRecord.ps1",
            "Test-PostPublishVerificationRecord.ps1",
            "local feed",
            "ProjectReference",
            "build-only",
            "parse-only",
            "sidecar-only",
            "blocked-by-cuda-driver",
            "真实记录字段置位要求",
            "templateOnly=false",
            "recordKind=external-runtime-proof-record",
            "runtimePackageKey",
            "packageSource.runtimePackageKey",
            "proofClassification=package-consumer-runtime",
            "isRuntimeExecutionEvidence=true",
            "canPromoteRuntimeProof=true",
            "currentRuntimeProofStatus=smoke-passed",
            "packageSource.noProjectReference=true",
            "packageSource.managedNupkgSha256",
            "packageSource.runtimeNupkgSha256",
            "command.exitCode=0",
            "command.logSha256",
            "results.stdoutSummary",
            "results.stderrSummary",
            "recordKind=post-publish-verification-record",
            "postPublishProofClassification=post-publish-package-consumer-runtime",
            "isPostPublishVerificationProof=true",
            "packageIdentity.managedPackageUrl",
            "packageIdentity.runtimePackageUrl",
            "packageIdentity.managedNupkgSha256",
            "packageIdentity.runtimeNupkgSha256",
            "noProjectReference=true",
            "nativeAssetListingSha256",
            "dependencyProbeLogSha256",
            "smokeLogSha256",
            "dependencyProbePassed=true",
            "runtimeSmokePassed=true",
            "runtimeSmokeLogSha256",
            "runtimeSmokeExitCode",
            "hostMetadata",
            "canCloseReleaseIssue` 必须保持 false",
        })
        {
            Assert.Contains(marker, frontDoorAndOwnerDocs, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string forbiddenClaim in new[]
        {
            "canCloseReleaseIssue=true",
            "canPublishPublicly=true",
            "performsPublish=true",
            "published to NuGet",
            "post-publish verification completed",
            "release issue can be closed",
        })
        {
            Assert.DoesNotContain(forbiddenClaim, readmeFrontDoors, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void FinalReleaseFrontDoorDoesNotPromoteOwnerGuidanceAsProof()
    {
        string[] frontDoorFiles =
        {
            Path.Combine("README.md"),
            Path.Combine("README.zh-CN.md"),
            Path.Combine("docs", "index.md"),
            Path.Combine("docs", "articles", "zh-cn", "release-owner-handoff.md"),
            Path.Combine("docs", "articles", "zh-cn", "release-candidate-freeze.md"),
            Path.Combine("docs", "articles", "zh-cn", "release-owner-action-checklist-final-hold.md"),
            Path.Combine("docs", "articles", "zh-cn", "release-hold-final-inspection.md"),
        };

        foreach (string relativePath in frontDoorFiles)
        {
            string source = File.ReadAllText(Path.Combine(RepositoryPaths.Root, relativePath));
            foreach (string requiredBoundary in new[]
            {
                "owner authorization",
                "package-consumer-runtime",
                "post-publish",
                "Linux",
                "real-model-runtime",
                "owner-release-execution-package",
                "oneScreenReleaseHoldChecklist",
            })
            {
                Assert.Contains(requiredBoundary, source, StringComparison.OrdinalIgnoreCase);
            }

            foreach (string forbiddenClaim in new[]
            {
                "canCloseReleaseIssue=true",
                "canPublishPublicly=true",
                "performsPublish=true",
                "release issue can be closed",
                "public release is approved",
                "post-publish verification passed",
            })
            {
                Assert.DoesNotContain(forbiddenClaim, source, StringComparison.OrdinalIgnoreCase);
            }
        }

        string[] releaseFacingFiles =
        {
            Path.Combine("docs", "articles", "zh-cn", "release-candidate-gate.md"),
            Path.Combine("docs", "articles", "zh-cn", "release-evidence-bundle.md"),
            Path.Combine("docs", "articles", "zh-cn", "release-close-gap-dashboard.md"),
            Path.Combine("docs", "articles", "zh-cn", "release-publish-execution-checklist.md"),
            Path.Combine("docs", "articles", "zh-cn", "release-candidate-final-evidence-freeze.md"),
            Path.Combine("docs", "articles", "zh-cn", "project-release-story-and-boundaries.md"),
        };

        foreach (string relativePath in releaseFacingFiles)
        {
            string source = File.ReadAllText(Path.Combine(RepositoryPaths.Root, relativePath));
            Assert.Contains("owner-release-execution-package", source, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("oneScreenReleaseHoldChecklist", source, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("blocked", source, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("guidance", source, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("proof", source, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("canCloseReleaseIssue=false", source, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void OwnerProofRealRecordFieldContractsStayAlignedAcrossTemplatesValidatorsAndDocs()
    {
        string ownerChecklist = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "release-owner-action-checklist-final-hold.md"));
        string externalValidator = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Test-ExternalRuntimeProofRecord.ps1"));
        string postPublishValidator = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Test-PostPublishVerificationRecord.ps1"));

        using JsonDocument externalTemplate = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "external-runtime-proof-record-template.json")));
        using JsonDocument externalValidation = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "external-runtime-proof-validation.json")));
        using JsonDocument postPublishTemplate = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "post-publish-verification-record-template.json")));
        using JsonDocument postPublishValidation = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "post-publish-verification-validation.json")));

        JsonElement externalTemplateRoot = externalTemplate.RootElement;
        JsonElement externalPackageSource = externalTemplateRoot.GetProperty("packageSource");
        foreach (string propertyName in new[]
        {
            "recordKind",
            "runtimePackageKey",
            "templateOnly",
            "proofClassification",
            "isRuntimeExecutionEvidence",
            "canPromoteRuntimeProof",
            "currentRuntimeProofStatus",
        })
        {
            AssertHasProperty(externalTemplateRoot, propertyName);
        }

        foreach (string propertyName in new[]
        {
            "runtimePackageKey",
            "managedNupkgSha256",
            "runtimeNupkgSha256",
            "noProjectReference",
        })
        {
            AssertHasProperty(externalPackageSource, propertyName);
        }

        JsonElement externalValidationRoot = externalValidation.RootElement;
        foreach (string propertyName in new[]
        {
            "inputRecordKind",
            "runtimePackageKey",
            "expectedRuntimePackageKey",
            "runtimePackageKeyMatches",
            "packageSourceRuntimePackageKeyMatches",
            "managedNupkgSha256Ready",
            "runtimeNupkgSha256Ready",
            "proofClassification",
            "proofClassificationPromotable",
            "isRuntimeExecutionEvidence",
            "canPromoteRuntimeProof",
            "stdoutSummaryReady",
            "stderrSummaryReady",
            "stdoutStderrSummariesReady",
            "logSha256FormatReady",
            "logSha256Matches",
            "noProjectReference",
        })
        {
            AssertHasProperty(externalValidationRoot, propertyName);
        }

        string externalContractSources = string.Join(Environment.NewLine, ownerChecklist, externalValidator);
        foreach (string marker in new[]
        {
            "recordKind=external-runtime-proof-record",
            "templateOnly=false",
            "runtimePackageKey",
            "packageSource.runtimePackageKey",
            "proofClassification=package-consumer-runtime",
            "isRuntimeExecutionEvidence=true",
            "canPromoteRuntimeProof=true",
            "currentRuntimeProofStatus=smoke-passed",
            "packageSource.noProjectReference=true",
            "managedNupkgSha256",
            "runtimeNupkgSha256",
            "command.exitCode=0",
            "command.logSha256",
            "stdoutSummary",
            "stderrSummary",
            "logSha256 must match",
        })
        {
            Assert.Contains(marker, externalContractSources, StringComparison.OrdinalIgnoreCase);
        }

        JsonElement postPublishTemplateRoot = postPublishTemplate.RootElement;
        JsonElement postPublishPackageIdentity = postPublishTemplateRoot.GetProperty("packageIdentity");
        foreach (string propertyName in new[]
        {
            "recordKind",
            "runtimePackageKey",
            "templateOnly",
            "postPublishProofClassification",
            "noProjectReference",
            "nativeAssetListingSha256",
            "dependencyProbeLogSha256",
            "smokeLogSha256",
            "dependencyProbePassed",
            "runtimeSmokePassed",
            "performsPublish",
            "isPostPublishVerificationProof",
            "canCloseReleaseIssue",
        })
        {
            AssertHasProperty(postPublishTemplateRoot, propertyName);
        }

        foreach (string propertyName in new[]
        {
            "managedPackageUrl",
            "runtimePackageUrl",
            "managedNupkgSha256",
            "runtimeNupkgSha256",
        })
        {
            AssertHasProperty(postPublishPackageIdentity, propertyName);
        }

        Assert.DoesNotContain("packageIdentity.noProjectReference", ownerChecklist, StringComparison.OrdinalIgnoreCase);

        JsonElement postPublishValidationRoot = postPublishValidation.RootElement;
        foreach (string propertyName in new[]
        {
            "inputRecordKind",
            "postPublishProofClassification",
            "postPublishProofClassificationPromotable",
            "managedPackageUrlReady",
            "runtimePackageUrlReady",
            "managedNupkgSha256Ready",
            "runtimeNupkgSha256Ready",
            "noProjectReference",
            "nativeAssetListingSha256Ready",
            "dependencyProbeLogSha256Ready",
            "smokeLogSha256Ready",
            "dependencyProbePassed",
            "runtimeSmokePassed",
            "runtimeSmokeExitCodeIsZero",
            "isPostPublishVerificationProof",
            "canCloseReleaseIssue",
        })
        {
            AssertHasProperty(postPublishValidationRoot, propertyName);
        }

        string postPublishContractSources = string.Join(Environment.NewLine, ownerChecklist, postPublishValidator);
        foreach (string marker in new[]
        {
            "recordKind=post-publish-verification-record",
            "templateOnly=false",
            "postPublishProofClassification=post-publish-package-consumer-runtime",
            "isPostPublishVerificationProof=true",
            "performsPublish=false",
            "packageIdentity.managedPackageUrl",
            "packageIdentity.runtimePackageUrl",
            "packageIdentity.managedNupkgSha256",
            "packageIdentity.runtimeNupkgSha256",
            "noProjectReference=true",
            "nativeAssetListingSha256",
            "dependencyProbeLogSha256",
            "smokeLogSha256",
            "dependencyProbePassed=true",
            "runtimeSmokePassed=true",
            "canCloseReleaseIssue` 必须保持 false",
        })
        {
            Assert.Contains(marker, postPublishContractSources, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void ReleaseProofBusUsesConsistentNonSubstituteProofKinds()
    {
        string[] requiredMarkers =
        {
            "template",
            "draft",
            "runbook",
            "collection package",
            "input package",
            "local feed",
            "ProjectReference",
            "build-only",
            "parse-only",
            "sidecar-only",
            "dependency-probe-only",
            "Skipped=True",
            "blocked-by-cuda-driver",
            "bridge-only package consumer log",
            "bridge-only wrapper surface",
            "WrapperSurfaceEvidenceKind=compile-surface-proof",
            "IsRuntimeExecutionProof=False",
            "mismatched log SHA256",
            "Parser/ParserRefitter diagnostic snapshots",
            "copied managed diagnostic snapshot",
            "Windows handoff for Linux proof",
        };

        string[] releaseProofBusSources =
        {
            Path.Combine("eng", "Test-ExternalRuntimeProofRecord.ps1"),
            Path.Combine("eng", "Test-PostPublishVerificationRecord.ps1"),
            Path.Combine("eng", "Export-ReleaseClosePreflight.ps1"),
            Path.Combine("eng", "Export-CompatibleHostProofBackfillPackage.ps1"),
            Path.Combine("eng", "Export-CompatibleHostProofExecutionPack.ps1"),
            Path.Combine("eng", "Export-ReleaseCloseGapDashboard.ps1"),
            Path.Combine("eng", "Export-ReleaseEvidenceBundle.ps1"),
            Path.Combine("eng", "Export-OwnerReleaseExecutionPackage.ps1"),
            Path.Combine("eng", "Export-RealModelAndPackageProofInputPackage.ps1"),
            Path.Combine("docs", "articles", "zh-cn", "release-proof-non-substitutes.md"),
            Path.Combine("docs", "articles", "zh-cn", "release-close-gap-dashboard.md"),
            Path.Combine("docs", "articles", "zh-cn", "release-evidence-bundle.md"),
            Path.Combine("docs", "articles", "zh-cn", "compatible-host-proof-execution-pack.md"),
            Path.Combine("docs", "articles", "zh-cn", "compatible-host-proof-backfill-package.md"),
            Path.Combine("docs", "articles", "zh-cn", "post-publish-verification-record.md"),
        };

        foreach (string relativePath in releaseProofBusSources)
        {
            string source = File.ReadAllText(Path.Combine(RepositoryPaths.Root, relativePath));
            foreach (string marker in requiredMarkers)
            {
                Assert.Contains(marker, source, StringComparison.OrdinalIgnoreCase);
            }
        }

        string[] releaseProofBusExports =
        {
            Path.Combine("eng", "Export-CompatibleHostProofBackfillPackage.ps1"),
            Path.Combine("eng", "Export-CompatibleHostProofExecutionPack.ps1"),
            Path.Combine("eng", "Export-ReleaseClosePreflight.ps1"),
            Path.Combine("eng", "Export-ReleaseCloseGapDashboard.ps1"),
            Path.Combine("eng", "Export-ReleaseEvidenceBundle.ps1"),
            Path.Combine("eng", "Export-ReleaseRuntimeProofExecutionMatrix.ps1"),
            Path.Combine("eng", "Export-OwnerReleaseExecutionPackage.ps1"),
            Path.Combine("eng", "Export-RealModelAndPackageProofInputPackage.ps1"),
        };

        foreach (string scriptPath in releaseProofBusExports)
        {
            RunPowerShell(Path.Combine(RepositoryPaths.Root, scriptPath));
        }

        string[] releaseProofBusArtifacts =
        {
            Path.Combine("artifacts", "final-release", "compatible-host-proof-backfill-package.json"),
            Path.Combine("artifacts", "final-release", "compatible-host-proof-backfill-package.md"),
            Path.Combine("artifacts", "final-release", "compatible-host-proof-execution-pack.json"),
            Path.Combine("artifacts", "final-release", "compatible-host-proof-execution-pack.md"),
            Path.Combine("artifacts", "final-release", "release-close-preflight.json"),
            Path.Combine("artifacts", "final-release", "release-close-preflight.md"),
            Path.Combine("artifacts", "final-release", "release-close-gap-dashboard.json"),
            Path.Combine("artifacts", "final-release", "release-close-gap-dashboard.md"),
            Path.Combine("artifacts", "final-release", "release-evidence-bundle.json"),
            Path.Combine("artifacts", "final-release", "release-evidence-bundle.md"),
            Path.Combine("artifacts", "final-release", "owner-release-execution-package.json"),
            Path.Combine("artifacts", "final-release", "owner-release-execution-package.md"),
            Path.Combine("artifacts", "final-release", "real-model-and-package-proof-input-package.json"),
            Path.Combine("artifacts", "final-release", "real-model-and-package-proof-input-package.md"),
        };

        foreach (string relativePath in releaseProofBusArtifacts)
        {
            string source = File.ReadAllText(Path.Combine(RepositoryPaths.Root, relativePath));
            foreach (string marker in requiredMarkers)
            {
                Assert.Contains(marker, source, StringComparison.OrdinalIgnoreCase);
            }
        }
    }

    private static void AssertReleaseHoldChecklistMirrorsOwnerPackage(JsonElement ownerRoot, JsonElement targetRoot)
    {
        string[] expectedIds =
        [
            "owner-authorization",
            "package-consumer-runtime",
            "linux-runner-proof",
            "real-model-runtime",
            "post-publish-verification",
        ];

        Assert.Equal(5, ownerRoot.GetProperty("oneScreenReleaseHoldChecklistCount").GetInt32());
        Assert.Equal(5, targetRoot.GetProperty("oneScreenReleaseHoldChecklistCount").GetInt32());
        JsonElement[] ownerItems = ownerRoot.GetProperty("oneScreenReleaseHoldChecklist").EnumerateArray().ToArray();
        JsonElement[] targetItems = targetRoot.GetProperty("oneScreenReleaseHoldChecklist").EnumerateArray().ToArray();
        Assert.Equal(expectedIds.Order(StringComparer.Ordinal).ToArray(), ownerItems.Select(static item => item.GetProperty("id").GetString()!).Order(StringComparer.Ordinal).ToArray());
        Assert.Equal(expectedIds.Order(StringComparer.Ordinal).ToArray(), targetItems.Select(static item => item.GetProperty("id").GetString()!).Order(StringComparer.Ordinal).ToArray());

        foreach (string id in expectedIds)
        {
            JsonElement ownerItem = Assert.Single(ownerItems, item => item.GetProperty("id").GetString() == id);
            JsonElement targetItem = Assert.Single(targetItems, item => item.GetProperty("id").GetString() == id);
            Assert.Equal(ownerItem.GetProperty("ownerVisibleBlocker").GetString(), targetItem.GetProperty("ownerVisibleBlocker").GetString());
            Assert.Equal(ownerItem.GetProperty("validatorCommand").GetString(), targetItem.GetProperty("validatorCommand").GetString());
            Assert.False(targetItem.GetProperty("performsPublish").GetBoolean());
            Assert.False(targetItem.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.True(targetItem.GetProperty("requiredRealInputs").GetArrayLength() > 0);
            Assert.True(targetItem.GetProperty("cannotUse").GetArrayLength() > 0);
        }

        string cannotUseCorpus = string.Join(
            Environment.NewLine,
            targetItems.SelectMany(static item => item.GetProperty("cannotUse").EnumerateArray().Select(static value => value.GetString()!)));

        foreach (string marker in new[]
        {
            "ProjectReference",
            "blocked-by-cuda-driver",
            "local feed",
            "sidecar-only",
        })
        {
            Assert.Contains(marker, cannotUseCorpus, StringComparison.Ordinal);
        }
    }

    private static void AssertReleaseProofReadinessSnapshotIsBlocked(JsonElement root)
    {
        Assert.Equal("release-proof-readiness-snapshot", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("readinessState").GetString());
        Assert.Equal(5, root.GetProperty("readinessItemCount").GetInt32());
        Assert.Equal(0, root.GetProperty("readyProofItemCount").GetInt32());
        Assert.Equal(5, root.GetProperty("blockedProofItemCount").GetInt32());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    private static void AssertReleaseProofReadinessSnapshotIsTransitive(JsonElement root, string sourcePropertyName)
    {
        string[] sourcePaths = root.GetProperty(sourcePropertyName)
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        Assert.Contains("artifacts/final-release/release-proof-readiness-snapshot.json", sourcePaths);
        Assert.Contains("artifacts/final-release/release-proof-readiness-snapshot.md", sourcePaths);
    }

    private static void AssertReleaseProofReadinessSnapshotState(JsonElement root)
    {
        Assert.Equal("blocked-real-proof-required", root.GetProperty("releaseProofReadinessSnapshotState").GetString());
        Assert.Equal(5, root.GetProperty("releaseProofReadinessSnapshotItemCount").GetInt32());
        Assert.Equal(0, root.GetProperty("releaseProofReadinessSnapshotReadyProofItemCount").GetInt32());
        Assert.Equal(5, root.GetProperty("releaseProofReadinessSnapshotBlockedProofItemCount").GetInt32());
        Assert.False(root.GetProperty("releaseProofReadinessSnapshotPerformsPublish").GetBoolean());
        Assert.False(root.GetProperty("releaseProofReadinessSnapshotCanPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("releaseProofReadinessSnapshotCanCloseReleaseIssue").GetBoolean());
    }

    private static void AssertOwnerProofInputReadinessIsBlocked(JsonElement root)
    {
        Assert.Equal("owner-proof-input-readiness", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-input-required", root.GetProperty("readinessState").GetString());
        Assert.Equal(5, root.GetProperty("contractCount").GetInt32());
        Assert.Equal(0, root.GetProperty("readyContractCount").GetInt32());
        Assert.Equal(5, root.GetProperty("blockedContractCount").GetInt32());
        Assert.True(root.GetProperty("requiresHumanOwner").GetBoolean());
        Assert.True(root.GetProperty("requiresCompatibleHost").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());

        string[] expectedBlockers =
        [
            "owner-authorization",
            "package-consumer-runtime",
            "linux-runner-proof",
            "real-model-runtime",
            "post-publish-verification",
        ];
        JsonElement[] contracts = root.GetProperty("ownerInputContracts").EnumerateArray().ToArray();
        Assert.Equal(expectedBlockers.Length, contracts.Length);
        foreach (string blocker in expectedBlockers)
        {
            JsonElement contract = Assert.Single(contracts, candidate => candidate.GetProperty("blockerId").GetString() == blocker);
            Assert.Equal(blocker, contract.GetProperty("proofClass").GetString());
            Assert.Equal("blocked-real-owner-input-required", contract.GetProperty("contractState").GetString());
            Assert.False(contract.GetProperty("ready").GetBoolean());
            Assert.False(contract.GetProperty("canPromoteOnPass").GetBoolean());
            Assert.False(contract.GetProperty("performsPublish").GetBoolean());
            Assert.False(contract.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(contract.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.True(contract.GetProperty("requiredInputFiles").GetArrayLength() >= 2);
            Assert.True(contract.GetProperty("templateFiles").GetArrayLength() >= 2);
            Assert.True(contract.GetProperty("replaceOrFillInstructions").GetArrayLength() >= 3);
            Assert.True(contract.GetProperty("successCriteria").GetArrayLength() >= 3);
            Assert.True(contract.GetProperty("expectedOutputArtifacts").GetArrayLength() >= 2);
            Assert.False(string.IsNullOrWhiteSpace(contract.GetProperty("validatorCommand").GetString()));

            string nonSubstitutes = string.Join(" ", contract.GetProperty("nonSubstitutes").EnumerateArray().Select(static item => item.GetString()));
            foreach (string marker in new[] { "template", "draft", "collection package", "input package", "local feed", "ProjectReference", "DependencyProbe", "blocked-by-cuda-driver" })
            {
                Assert.Contains(marker, nonSubstitutes, StringComparison.Ordinal);
            }
        }
    }

    private static void AssertOwnerProofInputReadinessIsTransitive(JsonElement root, string sourcePropertyName)
    {
        string[] sourcePaths = root.GetProperty(sourcePropertyName)
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        Assert.Contains("artifacts/final-release/owner-proof-input-readiness.json", sourcePaths);
        Assert.Contains("artifacts/final-release/owner-proof-input-readiness.md", sourcePaths);
    }

    private static void AssertOwnerProofInputReadinessState(JsonElement root)
    {
        Assert.Equal("blocked-real-proof-input-required", root.GetProperty("ownerProofInputReadinessState").GetString());
        Assert.Equal(5, root.GetProperty("ownerProofInputReadinessContractCount").GetInt32());
        Assert.Equal(0, root.GetProperty("ownerProofInputReadinessReadyContractCount").GetInt32());
        Assert.Equal(5, root.GetProperty("ownerProofInputReadinessBlockedContractCount").GetInt32());
        Assert.False(root.GetProperty("ownerProofInputReadinessPerformsPublish").GetBoolean());
        Assert.False(root.GetProperty("ownerProofInputReadinessCanPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("ownerProofInputReadinessCanCloseReleaseIssue").GetBoolean());
    }

    private static void AssertOwnerProofInputReadinessValidationIsValid(JsonElement root)
    {
        Assert.Equal("owner-proof-input-readiness-validation", root.GetProperty("recordKind").GetString());
        Assert.Equal("valid-owner-proof-input-readiness", root.GetProperty("validationState").GetString());
        Assert.True(root.GetProperty("isValidOwnerProofInputReadiness").GetBoolean());
        Assert.Equal(5, root.GetProperty("contractCount").GetInt32());
        Assert.Equal(0, root.GetProperty("readyContractCount").GetInt32());
        Assert.Equal(5, root.GetProperty("blockedContractCount").GetInt32());
        Assert.Equal(0, root.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.True(root.GetProperty("validationItems").GetArrayLength() >= 100);
    }

    private static void AssertOwnerProofInputReadinessValidationIsTransitive(JsonElement root, string sourcePropertyName)
    {
        string[] sourcePaths = root.GetProperty(sourcePropertyName)
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        Assert.Contains("artifacts/final-release/owner-proof-input-readiness-validation.json", sourcePaths);
        Assert.Contains("artifacts/final-release/owner-proof-input-readiness-validation.md", sourcePaths);
    }

    private static void AssertOwnerProofInputReadinessValidationState(JsonElement root)
    {
        Assert.Equal("valid-owner-proof-input-readiness", root.GetProperty("ownerProofInputReadinessValidationState").GetString());
        Assert.True(root.GetProperty("ownerProofInputReadinessIsValid").GetBoolean());
        Assert.Equal(0, root.GetProperty("ownerProofInputReadinessValidationFailedBlockerCount").GetInt32());
        Assert.False(root.GetProperty("ownerProofInputReadinessValidationPerformsPublish").GetBoolean());
        Assert.False(root.GetProperty("ownerProofInputReadinessValidationCanPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("ownerProofInputReadinessValidationCanCloseReleaseIssue").GetBoolean());
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }

    private static void AssertHasProperty(JsonElement element, string propertyName)
    {
        Assert.True(element.TryGetProperty(propertyName, out _), $"Expected JSON property '{propertyName}'.");
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = "pwsh",
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        startInfo.ArgumentList.Add("-NoProfile");
        startInfo.ArgumentList.Add("-ExecutionPolicy");
        startInfo.ArgumentList.Add("Bypass");
        startInfo.ArgumentList.Add("-File");
        startInfo.ArgumentList.Add(scriptPath);

        foreach (string argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using Process process = Process.Start(startInfo)!;
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"PowerShell command failed with exit code {process.ExitCode}:{Environment.NewLine}{output}{error}");
        return output + error;
    }

    private static string RunPowerShellAllowFailure(string scriptPath, params string[] arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = "pwsh",
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        startInfo.ArgumentList.Add("-NoProfile");
        startInfo.ArgumentList.Add("-ExecutionPolicy");
        startInfo.ArgumentList.Add("Bypass");
        startInfo.ArgumentList.Add("-File");
        startInfo.ArgumentList.Add(scriptPath);

        foreach (string argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using Process process = Process.Start(startInfo)!;
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.NotEqual(0, process.ExitCode);
        return output + error;
    }

    private static void AssertDefaultFinalReleaseProofState(
        JsonElement externalValidation,
        JsonElement postPublishValidation,
        JsonElement releaseClosePreflight)
    {
        Assert.Equal("template-only", externalValidation.GetProperty("validationState").GetString());
        Assert.Equal("template-only", externalValidation.GetProperty("proofClassification").GetString());
        Assert.False(externalValidation.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
        Assert.False(externalValidation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.Contains(externalValidation.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static item => item.GetString() == "bridge-only package consumer log");
        Assert.Contains(externalValidation.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static item => item.GetString() == "Skipped=True");

        Assert.Equal("template-only", postPublishValidation.GetProperty("validationState").GetString());
        Assert.Equal("template-only", postPublishValidation.GetProperty("postPublishProofClassification").GetString());
        Assert.False(postPublishValidation.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(postPublishValidation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains(postPublishValidation.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static item => item.GetString() == "local feed");
        Assert.Contains(postPublishValidation.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static item => item.GetString() == "ProjectReference");
        Assert.Contains(postPublishValidation.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static item => item.GetString() == "bridge-only package consumer log");
        Assert.Contains(postPublishValidation.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static item => item.GetString() == "Skipped=True");

        Assert.Equal("blocked-real-proof-required", releaseClosePreflight.GetProperty("preflightState").GetString());
        Assert.False(releaseClosePreflight.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(releaseClosePreflight.GetProperty("performsPublish").GetBoolean());
        Assert.False(releaseClosePreflight.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(releaseClosePreflight.GetProperty("isRealLinuxRunnerProof").GetBoolean());
        Assert.False(releaseClosePreflight.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.True(releaseClosePreflight.GetProperty("failedItemCount").GetInt32() >= 7);

        string[] nonSubstitutes = releaseClosePreflight
            .GetProperty("nonSubstituteProofKinds")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("local feed", nonSubstitutes);
        Assert.Contains("ProjectReference", nonSubstitutes);
        Assert.Contains("bridge-only package consumer log", nonSubstitutes);
        Assert.Contains("Skipped=True", nonSubstitutes);
        Assert.Contains("mismatched log SHA256", nonSubstitutes);
    }

    private static void AssertPostPublishRequiredEvidence(JsonElement root)
    {
        string[] expected =
        {
            "selectedChannel",
            "channelSourceUri",
            "publishedPackageUrl",
            "managedPackageUrl",
            "runtimePackageUrl",
            "managedNupkgSha256",
            "runtimeNupkgSha256",
            "cleanConsumerRootOutsideRepository",
            "consumerProjectPath",
            "noProjectReference",
            "noLocalPackageSource",
            "noLocalNupkgPackageReference",
            "restoreLogPath",
            "nativeAssetListingSha256",
            "dependencyProbeLogPath",
            "dependencyProbeLogSha256",
            "runtimeSmokeLogPath",
            "runtimeSmokeLogSha256",
            "runtimeSmokePassed",
            "runtimeSmokeExitCode",
            "stdoutSummary",
            "stderrSummary",
            "hostMetadata",
        };

        string[] actual = root.GetProperty("postPublishRequiredEvidence")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        Assert.Equal(expected.Length, root.GetProperty("postPublishRequiredEvidenceCount").GetInt32());
        Assert.Equal(expected.Order(StringComparer.Ordinal).ToArray(), actual.Order(StringComparer.Ordinal).ToArray());
    }
}




