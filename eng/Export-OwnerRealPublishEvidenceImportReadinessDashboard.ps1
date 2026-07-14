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

function Test-OwnerFileExists {
  param([string]$RelativePath)
  return Test-Path -LiteralPath (Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $RelativePath) -PathType Leaf
}

function Get-OwnerSlotById {
  param([AllowNull()][object]$Ledger, [string]$Id)
  foreach ($slot in @(Convert-ToArray (Get-PropertyOrDefault -Object $Ledger -Name "slots" -DefaultValue @()))) {
    if ([string](Get-PropertyOrDefault -Object $slot -Name "id" -DefaultValue "") -eq $Id) {
      return $slot
    }
  }

  return $null
}

function New-ReadinessSlot {
  param(
    [AllowNull()][object]$Ledger,
    [string]$Id,
    [string]$Title,
    [string[]]$OwnerInputPaths,
    [string[]]$ValidatorScripts,
    [string[]]$ExpectedEvidenceFields,
    [string[]]$ForbiddenSubstitutes,
    [string]$NextOwnerAction
  )

  $ledgerSlot = Get-OwnerSlotById -Ledger $Ledger -Id $Id
  $ownerInputFileCount = @($OwnerInputPaths | Where-Object { Test-OwnerFileExists -RelativePath $_ }).Count
  $slotValidationState = [string](Get-PropertyOrDefault -Object $ledgerSlot -Name "validationState" -DefaultValue "missing-$Id-ledger-slot")
  $slotBlockedValidationItems = [int](Get-PropertyOrDefault -Object $ledgerSlot -Name "blockedValidationItemCount" -DefaultValue 0)
  $slotProofReady = [bool](Get-PropertyOrDefault -Object $ledgerSlot -Name "proofReady" -DefaultValue $false)
  $missingFields = @($ExpectedEvidenceFields | ForEach-Object { [string]$_ })
  $missingReasons = New-Object System.Collections.Generic.List[string]
  if ($ownerInputFileCount -eq 0) { $missingReasons.Add("missing-real-owner-input-file") | Out-Null }
  if (-not $slotProofReady) { $missingReasons.Add("ledger-slot-not-proof-ready") | Out-Null }
  if ($slotBlockedValidationItems -gt 0) { $missingReasons.Add("validator-still-has-blocked-items") | Out-Null }

  [pscustomobject]@{
    id = $Id
    title = $Title
    readinessState = if ($slotProofReady -and $ownerInputFileCount -gt 0) { "owner-real-evidence-import-ready" } else { "blocked-owner-real-evidence-import-required" }
    ownerInputPaths = @($OwnerInputPaths)
    ownerInputFileCount = $ownerInputFileCount
    validatorScripts = @($ValidatorScripts)
    validatorScriptCount = @($ValidatorScripts).Count
    sourceLedgerSlotValidationState = $slotValidationState
    sourceLedgerSlotBlockedValidationItemCount = $slotBlockedValidationItems
    expectedEvidenceFields = @($ExpectedEvidenceFields)
    expectedEvidenceFieldCount = @($ExpectedEvidenceFields).Count
    missingEvidenceFields = @($missingFields)
    missingEvidenceFieldCount = @($missingFields).Count
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
    forbiddenSubstituteCount = @($ForbiddenSubstitutes).Count
    missingReasonCount = $missingReasons.Count
    missingReasons = @($missingReasons.ToArray())
    proofReady = $false
    blocked = $true
    nextOwnerAction = $NextOwnerAction
    notExecutedByAutomation = $true
    performsPublish = $false
    usesPublishToken = $false
    performsRuntimeExecution = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Owner evidence import readiness slot only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

if (-not (Test-OwnerFileExists -RelativePath "artifacts/final-release/owner-real-publish-evidence-availability-ledger.json")) {
  & (Join-Path $RepositoryRoot "eng\Export-OwnerRealPublishEvidenceAvailabilityLedger.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$ledger = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/owner-real-publish-evidence-availability-ledger.json"
$ledgerValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/owner-real-publish-evidence-availability-ledger-validation.json"
$landingPackValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/post-publish-docs-and-samples-final-landing-pack-validation.json"
$releaseEvidenceBundle = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/release-evidence-bundle.json"

$commonForbiddenSubstitutes = @(
  "template",
  "draft",
  "runbook",
  "dashboard",
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "candidate",
  "dry-run",
  "sidecar-only",
  "TensorRtExec report only",
  "pre-publish smoke reused as post-publish proof"
)

$slots = @(
  New-ReadinessSlot -Ledger $ledger -Id "owner-public-publish-result" -Title "Owner public publish result import" -OwnerInputPaths @("artifacts/final-release/public-publish-result-owner-input.json", "artifacts/final-release/owner-public-publish-execution-result-input.json") -ValidatorScripts @("Test-PublicPublishResultOwnerInput.ps1 -Strict", "Test-PublicPublishResultImport.ps1 -Strict", "Test-OwnerPublicPublishExecutionResultCandidate.ps1 -Strict") -ExpectedEvidenceFields @("packageId", "packageVersion", "publicPackageUrl", "publicDownloadUrl", "nupkgSha256", "publishTranscriptSha256", "ownerReviewer", "ownerReviewedAtUtc") -ForbiddenSubstitutes $commonForbiddenSubstitutes -NextOwnerAction "Owner fills the public publish result input after the real manual/remote publish has completed."
  New-ReadinessSlot -Ledger $ledger -Id "github-actions-run-proof" -Title "GitHub Actions run proof import" -OwnerInputPaths @("artifacts/final-release/github-actions-run-evidence.owner.json", "artifacts/final-release/github-actions-run-evidence-input.json") -ValidatorScripts @("Test-GitHubActionsRunEvidenceImport.ps1 -Strict", "Test-RemoteCiAndPublicPublishProofBackfillGate.ps1 -Strict") -ExpectedEvidenceFields @("runId", "runUrl", "workflowName", "headSha", "conclusion", "logSha256", "artifactManifestSha256", "ownerReviewer") -ForbiddenSubstitutes $commonForbiddenSubstitutes -NextOwnerAction "Owner records the completed workflow run proof after publish or quality workflow completes."
  New-ReadinessSlot -Ledger $ledger -Id "public-package-download-proof" -Title "Public package download proof import" -OwnerInputPaths @("artifacts/final-release/public-package-download-proof-input.json") -ValidatorScripts @("Test-PublicPackageDownloadProofInput.ps1 -Strict", "Test-PublicPackageDownloadProofCandidate.ps1 -Strict") -ExpectedEvidenceFields @("managedPackageUrl", "managedPublicDownloadUrl", "managedNupkgSha256", "runtimePackageUrl", "runtimePublicDownloadUrl", "runtimeNupkgSha256", "downloadTimestampUtc", "downloadSource") -ForbiddenSubstitutes $commonForbiddenSubstitutes -NextOwnerAction "Owner downloads from public sources and records URL/hash/timestamp evidence."
  New-ReadinessSlot -Ledger $ledger -Id "repository-external-clean-consumer-proof" -Title "Repository-external clean consumer proof import" -OwnerInputPaths @("artifacts/final-release/external-clean-consumer-execution-result.owner.json", "artifacts/final-release/external-clean-consumer-execution-result.json") -ValidatorScripts @("Test-ExternalCleanConsumerExecutionResult.ps1 -Strict", "Test-CleanConsumerRuntimeProofCrossCheckGate.ps1 -Strict") -ExpectedEvidenceFields @("externalWorkspaceRoot", "projectPath", "restoreCommand", "buildCommand", "smokeCommand", "restoreLogSha256", "buildLogSha256", "smokeLogSha256", "hostMetadata", "packageSource") -ForbiddenSubstitutes $commonForbiddenSubstitutes -NextOwnerAction "Owner runs restore/build/smoke in a repository-external workspace using only public package sources."
  New-ReadinessSlot -Ledger $ledger -Id "post-publish-clean-consumer-proof" -Title "Post-publish clean consumer proof import" -OwnerInputPaths @("artifacts/final-release/post-publish-clean-consumer-proof-result.owner.json", "artifacts/final-release/post-publish-clean-consumer-proof-result.json") -ValidatorScripts @("Test-PostPublishCleanConsumerProofResult.ps1 -Strict", "Test-PostPublishCleanConsumerRealProofFromOwnerResult.ps1 -Strict") -ExpectedEvidenceFields @("publicPackageUrl", "publicDownloadSha256", "cleanConsumerRestoreLogSha256", "cleanConsumerBuildLogSha256", "cleanConsumerSmokeLogSha256", "stdoutSummary", "stderrSummary", "ownerReviewer") -ForbiddenSubstitutes $commonForbiddenSubstitutes -NextOwnerAction "Owner records post-publication clean consumer proof tied to public package download hashes."
  New-ReadinessSlot -Ledger $ledger -Id "post-publish-user-verification" -Title "Post-publish user verification import" -OwnerInputPaths @("artifacts/final-release/post-publish-verification-owner-input.json", "artifacts/final-release/post-publish-verification-record.json") -ValidatorScripts @("Test-PostPublishVerificationOwnerInput.ps1 -Strict", "Test-PostPublishVerificationRecord.ps1 -Strict", "Test-PostPublishUserVerificationPack.ps1 -Strict") -ExpectedEvidenceFields @("managedPackageId", "managedPackageVersion", "runtimePackageId", "runtimePackageVersion", "consumerProjectIdentity", "smokeCommand", "stdoutSummary", "stderrSummary", "allLogSha256Matches") -ForbiddenSubstitutes $commonForbiddenSubstitutes -NextOwnerAction "Owner imports post-publish user verification after clean consumer and public package evidence are real."
  New-ReadinessSlot -Ledger $ledger -Id "release-issue-close-owner-decision" -Title "Release Issue close owner decision import" -OwnerInputPaths @("artifacts/final-release/release-issue-close-owner-decision-input.owner.json", "artifacts/final-release/release-issue-close-owner-decision-input.json") -ValidatorScripts @("Test-ReleaseIssueCloseOwnerDecisionInput.ps1 -Strict", "Test-FinalReleaseCloseApprovalRealInputFromOwnerResult.ps1 -Strict") -ExpectedEvidenceFields @("releaseIssueUrl", "releaseIssueNumber", "ownerCloseDecision", "releaseEvidenceBundleSha256", "postPublishProofSha256", "rollbackPlanSha256", "ownerReviewer", "ownerReviewedAtUtc") -ForbiddenSubstitutes $commonForbiddenSubstitutes -NextOwnerAction "Owner records the final close decision only after public publish and post-publish proof are accepted."
  New-ReadinessSlot -Ledger $ledger -Id "release-issue-close-record" -Title "Release Issue close record import" -OwnerInputPaths @("artifacts/final-release/release-issue-close-record.json") -ValidatorScripts @("Test-ReleaseIssueCloseRecord.ps1 -Strict", "Test-ReleaseIssueCloseRecordRealInputMap.ps1 -Strict") -ExpectedEvidenceFields @("closeRecordId", "releaseIssueUrl", "ownerCloseDecision", "evidenceBundleSha256", "classificationAuditSha256", "postPublishProofSha256", "rollbackDecision", "staleClaimAuditState") -ForbiddenSubstitutes $commonForbiddenSubstitutes -NextOwnerAction "Owner imports final close record and validates it against real post-publish proof and evidence bundle hash."
  New-ReadinessSlot -Ledger $ledger -Id "strict-close-final-convergence" -Title "Strict close final convergence import" -OwnerInputPaths @("artifacts/final-release/final-owner-close-decision.owner.json", "artifacts/final-release/final-owner-rollback-review.owner.json") -ValidatorScripts @("Test-StrictCloseReadyConvergenceDashboard.ps1 -Strict", "Test-FinalPublicReleaseClosureBridge.ps1 -Strict", "Test-ReleaseQualityGate.ps1 -Strict") -ExpectedEvidenceFields @("strictCloseReadyState", "finalPublicReleaseClosureBridgeState", "ownerRollbackReview", "ownerCloseDecision", "allProofLanesAccepted", "classificationAuditPassed", "releaseQualityGatePassed") -ForbiddenSubstitutes $commonForbiddenSubstitutes -NextOwnerAction "Owner reruns strict close convergence after all real proof input validators are green."
)

$blockedSlots = @($slots | Where-Object { [bool]$_.blocked })
$proofReadySlots = @($slots | Where-Object { [bool]$_.proofReady })
$expectedFieldCount = 0
$missingFieldCount = 0
$validatorScriptCount = 0
foreach ($slot in $slots) {
  $expectedFieldCount += [int]$slot.expectedEvidenceFieldCount
  $missingFieldCount += [int]$slot.missingEvidenceFieldCount
  $validatorScriptCount += [int]$slot.validatorScriptCount
}

$record = [pscustomobject]@{
  recordKind = "owner-real-publish-evidence-import-readiness-dashboard"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  dashboardState = "blocked-owner-real-publish-evidence-import-required"
  sourceLedgerState = [string](Get-PropertyOrDefault -Object $ledger -Name "ledgerState" -DefaultValue "missing-owner-real-publish-evidence-availability-ledger")
  sourceLedgerValidationState = [string](Get-PropertyOrDefault -Object $ledgerValidation -Name "validationState" -DefaultValue "missing-owner-real-publish-evidence-availability-ledger-validation")
  sourceFinalLandingPackValidationState = [string](Get-PropertyOrDefault -Object $landingPackValidation -Name "validationState" -DefaultValue "missing-post-publish-docs-and-samples-final-landing-pack-validation")
  sourceReleaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
  slotCount = @($slots).Count
  blockedSlotCount = @($blockedSlots).Count
  proofReadySlotCount = @($proofReadySlots).Count
  expectedEvidenceFieldCount = $expectedFieldCount
  missingEvidenceFieldCount = $missingFieldCount
  validatorScriptCount = $validatorScriptCount
  slots = @($slots)
  ownerActionRequired = $true
  passed = $false
  notExecutedByAutomation = $true
  performsPublish = $false
  usesPublishToken = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner real publish evidence import readiness dashboard is a blocked owner-action map only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "owner-real-publish-evidence-import-readiness-dashboard.json"
$mdPath = Join-Path $OutputRoot "owner-real-publish-evidence-import-readiness-dashboard.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 18)

$rows = foreach ($slot in $slots) {
  "| ``$($slot.id)`` | ``$($slot.readinessState)`` | ``$($slot.ownerInputFileCount)`` | ``$($slot.expectedEvidenceFieldCount)`` | ``$($slot.validatorScriptCount)`` | ``$($slot.proofReady)`` |"
}
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Owner Real Publish Evidence Import Readiness Dashboard",
  "",
  "- dashboardState: ``$($record.dashboardState)``",
  "- slotCount: ``$($record.slotCount)``",
  "- blockedSlotCount: ``$($record.blockedSlotCount)``",
  "- proofReadySlotCount: ``$($record.proofReadySlotCount)``",
  "- expectedEvidenceFieldCount: ``$($record.expectedEvidenceFieldCount)``",
  "- missingEvidenceFieldCount: ``$($record.missingEvidenceFieldCount)``",
  "- canPublishPublicly: ``False``",
  "- canCloseReleaseIssue: ``False``",
  "",
  "| Slot | State | Owner Input Files | Expected Fields | Validators | Proof Ready |",
  "| --- | --- | ---: | ---: | ---: | ---: |",
  @($rows),
  "",
  "## Boundary",
  "",
  $record.boundary
)
Write-Host "OwnerRealPublishEvidenceImportReadinessDashboardState=$($record.dashboardState) Slots=$($record.slotCount) Blocked=$($record.blockedSlotCount) ProofReady=$($record.proofReadySlotCount)"
