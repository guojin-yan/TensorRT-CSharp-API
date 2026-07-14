[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot

function Get-ValidationState {
  param([string]$RelativePath, [string]$MissingState)
  $record = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath $RelativePath
  return [string](Get-PropertyOrDefault -Object $record -Name "validationState" -DefaultValue $MissingState)
}

function Get-AuditState {
  param([string]$RelativePath, [string]$MissingState)
  $record = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath $RelativePath
  return [string](Get-PropertyOrDefault -Object $record -Name "auditState" -DefaultValue $MissingState)
}

function New-IntakeField {
  param([string]$Name, [string]$Category, [string]$EvidenceSource)
  [pscustomobject]@{
    name = $Name
    category = $Category
    evidenceSource = $EvidenceSource
    state = "blocked-owner-real-input-required"
    ownerMustProvide = $true
    acceptsTemplate = $false
    acceptsDraft = $false
    acceptsLocalFeed = $false
    acceptsProjectReference = $false
    acceptsDirectNupkg = $false
    acceptsDryRun = $false
    acceptsDashboard = $false
    acceptsQueuedWorkflow = $false
    performsPublish = $false
    usesPublishToken = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
  }
}

function New-IntakePhase {
  param(
    [string]$Id,
    [string]$Title,
    [string]$SourceState,
    [string[]]$RequiredFields,
    [string[]]$ValidatorScripts,
    [string[]]$OwnerActions,
    [string[]]$ForbiddenSubstitutes
  )

  $fields = @(
    foreach ($field in $RequiredFields) {
      New-IntakeField -Name $field -Category $Id -EvidenceSource "owner-real-public-release-evidence"
    }
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    phaseState = "blocked-owner-real-evidence-required"
    sourceState = $SourceState
    requiredFields = @($fields)
    requiredFieldCount = @($fields).Count
    validatorScripts = @($ValidatorScripts)
    validatorScriptCount = @($ValidatorScripts).Count
    ownerActions = @($OwnerActions)
    ownerActionCount = @($OwnerActions).Count
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
    forbiddenSubstituteCount = @($ForbiddenSubstitutes).Count
    proofReady = $false
    blocked = $true
    ownerActionRequired = $true
    notExecutedByAutomation = $true
    ownerExecutionOnly = $true
    performsPublish = $false
    usesPublishToken = $false
    performsRuntimeExecution = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Owner public publish execution final intake phase only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$sourceStates = [ordered]@{
  publicPublishFinalOwnerExecutionPack = Get-ValidationState "artifacts/final-release/public-publish-final-owner-execution-pack-validation.json" "missing-public-publish-final-owner-execution-pack-validation"
  publicPackageDownloadProofOwnerExecutionPack = Get-ValidationState "artifacts/final-release/public-package-download-proof-owner-execution-pack-validation.json" "missing-public-package-download-proof-owner-execution-pack-validation"
  githubPublishAndCiStatusSnapshot = Get-ValidationState "artifacts/final-release/github-publish-and-ci-status-snapshot-validation.json" "missing-github-publish-and-ci-status-snapshot-validation"
  ownerRealPublishEvidenceImportReadinessDashboard = Get-ValidationState "artifacts/final-release/owner-real-publish-evidence-import-readiness-dashboard-validation.json" "missing-owner-real-publish-evidence-import-readiness-dashboard-validation"
  releaseCloseFinalCandidateAuditPack = Get-ValidationState "artifacts/final-release/release-close-final-candidate-audit-pack-validation.json" "missing-release-close-final-candidate-audit-pack-validation"
  publicPublishResultImport = Get-ValidationState "artifacts/final-release/public-publish-result-import-validation.json" "missing-public-publish-result-import-validation"
  publicPackageDownloadProofInput = Get-ValidationState "artifacts/final-release/public-package-download-proof-input-validation.json" "missing-public-package-download-proof-input-validation"
  publicPackageDownloadProofCandidate = Get-ValidationState "artifacts/final-release/public-package-download-proof-candidate-validation.json" "missing-public-package-download-proof-candidate-validation"
  externalCleanConsumerExecutionResult = Get-ValidationState "artifacts/final-release/external-clean-consumer-execution-result-validation.json" "missing-external-clean-consumer-execution-result-validation"
  postPublishCleanConsumerProofResult = Get-ValidationState "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json" "missing-post-publish-clean-consumer-proof-result-validation"
  postPublishUserVerificationPack = Get-ValidationState "artifacts/final-release/post-publish-user-verification-pack-validation.json" "missing-post-publish-user-verification-pack-validation"
  strictCloseReadyConvergenceDashboard = Get-ValidationState "artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json" "missing-strict-close-ready-convergence-dashboard-validation"
  finalPublicReleaseClosureBridge = Get-ValidationState "artifacts/final-release/final-public-release-closure-bridge-validation.json" "missing-final-public-release-closure-bridge-validation"
  releaseIssueCloseOwnerDecisionInput = Get-ValidationState "artifacts/final-release/release-issue-close-owner-decision-input-validation.json" "missing-release-issue-close-owner-decision-input-validation"
  releaseIssueCloseRecord = Get-ValidationState "artifacts/final-release/release-issue-close-record-validation.json" "missing-release-issue-close-record-validation"
  releaseEvidenceClassificationAudit = Get-AuditState "artifacts/final-release/release-evidence-classification-audit.json" "missing-release-evidence-classification-audit"
}

$forbidden = @(
  "template",
  "draft",
  "runbook",
  "dashboard",
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "queued workflow",
  "dry-run",
  "candidate-only",
  "missing runner",
  "pre-publish smoke reused as post-publish proof"
)

$phases = @(
  New-IntakePhase -Id "claim-boundary-preflight" -Title "Claim boundary and documentation preflight" -SourceState $sourceStates.publicPublishFinalOwnerExecutionPack -RequiredFields @("releaseVersion", "packageIds", "runtimePackageKeys", "knownLimitationsUrl", "nonProofBoundaryAcknowledgement", "docsHash", "readmeHash", "nugetMetadataReview", "ownerReviewer") -ValidatorScripts @("Test-PublicDocsAndPackageMetadataGate.ps1 -Strict", "Test-ReleaseDocsAndNuGetMetadataAudit.ps1 -Strict") -OwnerActions @("Review public docs and NuGet metadata claims before real publish.", "Confirm no local-only evidence is presented as public proof.") -ForbiddenSubstitutes $forbidden
  New-IntakePhase -Id "owner-public-publish-execution" -Title "Owner public publish execution result" -SourceState $sourceStates.publicPublishResultImport -RequiredFields @("selectedChannel", "managedPackageId", "managedPackageVersion", "runtimePackageId", "runtimePackageVersion", "runtimePackageKey", "publicPackagePageUrl", "publicPackageDownloadUrl", "publishStartedAtUtc", "publishCompletedAtUtc", "publishTranscriptSha256", "ownerReviewer") -ValidatorScripts @("Test-PublicPublishResultOwnerInput.ps1 -Strict", "Test-PublicPublishResultImport.ps1 -Strict", "Test-OwnerPublicPublishExecutionResultCandidate.ps1 -Strict") -OwnerActions @("Owner executes approved public publish outside automation.", "Owner imports the real publish result with package URLs and transcript hashes.") -ForbiddenSubstitutes $forbidden
  New-IntakePhase -Id "github-actions-run-proof" -Title "GitHub Actions and CI proof import" -SourceState $sourceStates.githubPublishAndCiStatusSnapshot -RequiredFields @("workflowName", "runId", "runUrl", "headSha", "branch", "conclusion", "createdAtUtc", "completedAtUtc", "logSha256", "artifactManifestSha256", "releaseQualityGateSummarySha256") -ValidatorScripts @("Test-GitHubPublishAndCiStatusSnapshot.ps1 -Strict", "Test-RemoteCiAndPublicPublishProofBackfillGate.ps1 -Strict") -OwnerActions @("Record the completed CI run without triggering a publish workflow.", "Cross-check run SHA and release quality gate summary.") -ForbiddenSubstitutes $forbidden
  New-IntakePhase -Id "public-managed-package-download-proof" -Title "Public managed package download proof" -SourceState $sourceStates.publicPackageDownloadProofInput -RequiredFields @("managedPublicPackageUrl", "managedPublicDownloadUrl", "managedPackageId", "managedPackageVersion", "managedNupkgPath", "managedNupkgSha256", "managedNupkgSizeBytes", "downloadedAtUtc", "downloadTranscriptSha256") -ValidatorScripts @("Test-PublicPackageDownloadProofInput.ps1 -Strict", "Test-PublicPackageDownloadProofCandidate.ps1 -Strict") -OwnerActions @("Download managed package from public channel.", "Record URL, path, size, SHA256 and timestamp.") -ForbiddenSubstitutes $forbidden
  New-IntakePhase -Id "public-runtime-package-download-proof" -Title "Public runtime package download proof" -SourceState $sourceStates.publicPackageDownloadProofCandidate -RequiredFields @("runtimePublicPackageUrl", "runtimePublicDownloadUrl", "runtimePackageId", "runtimePackageVersion", "runtimePackageKey", "runtimeNupkgPath", "runtimeNupkgSha256", "runtimeNupkgSizeBytes", "downloadedAtUtc", "downloadTranscriptSha256") -ValidatorScripts @("Test-PublicPackageDownloadProofInput.ps1 -Strict", "Test-PublicPackageDownloadProofCandidate.ps1 -Strict") -OwnerActions @("Download runtime package or GitHub release asset from public channel.", "Record runtime route, package key, SHA256 and source URL.") -ForbiddenSubstitutes $forbidden
  New-IntakePhase -Id "repository-external-clean-consumer" -Title "Repository-external clean consumer proof" -SourceState $sourceStates.externalCleanConsumerExecutionResult -RequiredFields @("externalWorkspaceRoot", "consumerProjectPath", "restoreSource", "restoreCommand", "buildCommand", "smokeCommand", "restoreLogSha256", "buildLogSha256", "smokeLogSha256", "hostMetadataSha256", "noLocalSubstituteConfirmation") -ValidatorScripts @("Test-ExternalCleanConsumerExecutionResult.ps1 -Strict", "Test-CleanConsumerRuntimeProofCrossCheckGate.ps1 -Strict") -OwnerActions @("Run clean consumer outside this repository.", "Use public package source only and import logs plus hashes.") -ForbiddenSubstitutes $forbidden
  New-IntakePhase -Id "post-publish-clean-consumer-proof" -Title "Post-publish clean consumer proof" -SourceState $sourceStates.postPublishCleanConsumerProofResult -RequiredFields @("postPublishPublicPackageUrl", "postPublishPackageSha256", "postPublishConsumerRoot", "postPublishRestoreLogSha256", "postPublishBuildLogSha256", "postPublishSmokeLogSha256", "stdoutSummary", "stderrSummary", "hostMetadata", "ownerReviewedAtUtc") -ValidatorScripts @("Test-PostPublishCleanConsumerProofResult.ps1 -Strict", "Test-PostPublishCleanConsumerRealProofFromOwnerResult.ps1 -Strict") -OwnerActions @("Run post-publication clean consumer validation.", "Link public package download hashes to clean consumer logs.") -ForbiddenSubstitutes $forbidden
  New-IntakePhase -Id "post-publish-user-verification" -Title "Post-publish user verification" -SourceState $sourceStates.postPublishUserVerificationPack -RequiredFields @("managedPackageId", "managedPackageVersion", "runtimePackageId", "runtimePackageVersion", "consumerProjectIdentity", "smokeCommand", "stdoutSummary", "stderrSummary", "allLogSha256Matches", "ownerReviewer") -ValidatorScripts @("Test-PostPublishUserVerificationPack.ps1 -Strict", "Test-PostPublishVerificationRecord.ps1 -Strict") -OwnerActions @("Record end-user style verification after public publication.", "Confirm stdout/stderr summaries and log SHA256 values match.") -ForbiddenSubstitutes $forbidden
  New-IntakePhase -Id "strict-close-convergence" -Title "Strict close convergence" -SourceState $sourceStates.strictCloseReadyConvergenceDashboard -RequiredFields @("strictCloseReadyState", "finalClosureBridgeState", "acceptedProofLaneCount", "blockedProofLaneCount", "classificationAuditState", "releaseQualityGateState", "evidenceBundleSha256", "classificationAuditSha256") -ValidatorScripts @("Test-StrictCloseReadyConvergenceDashboard.ps1 -Strict", "Test-FinalPublicReleaseClosureBridge.ps1 -Strict", "Test-ReleaseCloseFinalCandidateAuditPack.ps1 -Strict") -OwnerActions @("Refresh strict close dashboards after all real proof inputs are accepted.", "Keep close candidate blocked until every real proof lane is accepted.") -ForbiddenSubstitutes $forbidden
  New-IntakePhase -Id "release-issue-close-owner-decision" -Title "Release Issue close Owner decision" -SourceState $sourceStates.releaseIssueCloseOwnerDecisionInput -RequiredFields @("releaseIssueUrl", "releaseIssueNumber", "ownerCloseDecision", "ownerName", "ownerEmail", "ownerDecisionTimestampUtc", "approvedPublicPackageProofHash", "approvedPostPublishProofHash", "rollbackPlan", "knownLimitationsAcknowledgement") -ValidatorScripts @("Test-ReleaseIssueCloseOwnerDecisionInput.ps1 -Strict", "Test-FinalReleaseCloseApprovalRealInputFromOwnerResult.ps1 -Strict") -OwnerActions @("Owner records close decision only after real post-publish evidence is accepted.", "Map close decision to bundle hash, proof hashes, rollback plan and known limitations.") -ForbiddenSubstitutes $forbidden
  New-IntakePhase -Id "release-issue-close-record" -Title "Release Issue close record" -SourceState $sourceStates.releaseIssueCloseRecord -RequiredFields @("closeRecordId", "releaseIssueUrl", "ownerCloseDecision", "releaseEvidenceBundleSha256", "classificationAuditSha256", "postPublishProofSha256", "rollbackDecision", "staleClaimAuditState") -ValidatorScripts @("Test-ReleaseIssueCloseRecord.ps1 -Strict", "Test-ReleaseIssueCloseRecordRealInputMap.ps1 -Strict") -OwnerActions @("Import final close record after Owner decision.", "Verify close record hashes against current evidence bundle and classification audit.") -ForbiddenSubstitutes $forbidden
  New-IntakePhase -Id "final-bundle-classification-lock" -Title "Final bundle and classification audit lock" -SourceState $sourceStates.releaseEvidenceClassificationAudit -RequiredFields @("releaseEvidenceBundleSha256", "classificationAuditSha256", "requiredNonProofItemCount", "nonSubstituteProofKindCount", "auditedNonProofItemCount", "findingCount", "bundleState", "ownerActionStatus", "canCloseReleaseIssue") -ValidatorScripts @("Export-ReleaseEvidenceBundle.ps1", "Test-ReleaseEvidenceClassificationAudit.ps1 -Strict", "Test-ReleaseQualityGate.ps1 -Strict") -OwnerActions @("Refresh release evidence bundle after every real proof import.", "Confirm classification audit still blocks templates, dashboards and local substitutes.") -ForbiddenSubstitutes $forbidden
)

$blockedPhases = @($phases | Where-Object { [bool]$_.blocked })
$proofReadyPhases = @($phases | Where-Object { [bool]$_.proofReady })
$requiredEvidenceFieldCount = 0
$validatorScriptCount = 0
$ownerActionCount = 0
$forbiddenSubstituteCount = 0
foreach ($phase in $phases) {
  $requiredEvidenceFieldCount += [int]$phase.requiredFieldCount
  $validatorScriptCount += [int]$phase.validatorScriptCount
  $ownerActionCount += [int]$phase.ownerActionCount
  $forbiddenSubstituteCount += [int]$phase.forbiddenSubstituteCount
}

$record = [pscustomobject]@{
  recordKind = "owner-public-publish-execution-final-intake-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  intakeState = "blocked-owner-public-publish-execution-final-intake-real-evidence-required"
  phaseCount = @($phases).Count
  blockedPhaseCount = @($blockedPhases).Count
  proofReadyPhaseCount = @($proofReadyPhases).Count
  requiredEvidenceFieldCount = $requiredEvidenceFieldCount
  validatorScriptCount = $validatorScriptCount
  ownerActionCount = $ownerActionCount
  forbiddenSubstituteCount = $forbiddenSubstituteCount
  sourceStates = $sourceStates
  phases = @($phases)
  finalCloseCandidateReady = $false
  ownerActionRequired = $true
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  usesPublishToken = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner public publish execution final intake pack is an owner-action convergence map only; it does not run publish commands, does not download packages, does not run clean consumer smoke, is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "owner-public-publish-execution-final-intake-pack.json"
$mdPath = Join-Path $OutputRoot "owner-public-publish-execution-final-intake-pack.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 20)

$phaseRows = foreach ($phase in $phases) {
  "| ``$($phase.id)`` | ``$($phase.phaseState)`` | ``$($phase.requiredFieldCount)`` | ``$($phase.validatorScriptCount)`` | ``$($phase.proofReady)`` |"
}
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Owner Public Publish Execution Final Intake Pack",
  "",
  "- intakeState: ``$($record.intakeState)``",
  "- phaseCount: ``$($record.phaseCount)``",
  "- blockedPhaseCount: ``$($record.blockedPhaseCount)``",
  "- proofReadyPhaseCount: ``$($record.proofReadyPhaseCount)``",
  "- requiredEvidenceFieldCount: ``$($record.requiredEvidenceFieldCount)``",
  "- validatorScriptCount: ``$($record.validatorScriptCount)``",
  "- canPublishPublicly: ``False``",
  "- canCloseReleaseIssue: ``False``",
  "",
  "| Phase | State | Fields | Validators | Proof Ready |",
  "| --- | --- | ---: | ---: | ---: |",
  @($phaseRows),
  "",
  "## Boundary",
  "",
  $record.boundary
)

Write-Host "OwnerPublicPublishExecutionFinalIntakePackState=$($record.intakeState) Phases=$($record.phaseCount) Blocked=$($record.blockedPhaseCount) ProofReady=$($record.proofReadyPhaseCount)"
