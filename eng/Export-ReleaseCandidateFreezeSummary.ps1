[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$LinuxRuntimePackageKey = "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function Normalize-PostPublishRequiredEvidence {
  param([AllowNull()][object]$Evidence)

  $items = @($Evidence | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
  foreach ($requiredField in @("noLocalPackageSource", "noLocalNupkgPackageReference")) {
    if ($items -notcontains $requiredField) {
      $items += $requiredField
    }
  }

  return @($items)
}

function New-FreezeArtifactRef {
  param(
    [string]$Id,
    [string]$Path,
    [string]$State,
    [bool]$Ready,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    path = $Path
    state = $State
    ready = $Ready
    boundary = $Boundary
  }
}

function New-FreezeBlockingItem {
  param(
    [string]$Id,
    [string]$Title,
    [bool]$Passed,
    [string]$CurrentStatus,
    [string]$RequiredEvidence,
    [string]$OwnerAction,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    passed = $Passed
    state = if ($Passed) { "satisfied" } else { "blocking" }
    currentStatus = $CurrentStatus
    requiredEvidence = $RequiredEvidence
    ownerAction = $OwnerAction
    boundary = $Boundary
  }
}

$finalRelease = Read-JsonOrNull "artifacts\final-release\final-release-dry-run-summary.json"
$releaseEvidence = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$finalPackageReview = Read-JsonOrNull "artifacts\final-release\final-package-review-bundle.json"
$packageProof = Read-JsonOrNull "artifacts\final-release\release-package-proof-bundle.json"
$ownerApproval = Read-JsonOrNull "artifacts\final-release\release-owner-approval-input-validation.json"
$ownerDecision = Read-JsonOrNull "artifacts\final-release\release-owner-decision-record.json"
$ownerReleaseExecutionPackage = Read-JsonOrNull "artifacts\final-release\owner-release-execution-package.json"
$releaseProofReadinessSnapshot = Read-JsonOrNull "artifacts\final-release\release-proof-readiness-snapshot.json"
$ownerProofInputReadiness = Read-JsonOrNull "artifacts\final-release\owner-proof-input-readiness.json"
$ownerProofInputReadinessValidation = Read-JsonOrNull "artifacts\final-release\owner-proof-input-readiness-validation.json"
$publishChecklist = Read-JsonOrNull "artifacts\final-release\release-publish-execution-checklist.json"
$promotionIssue = Read-JsonOrNull "artifacts\final-release\release-promotion-issue-record.json"
$externalRuntimeProof = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-record.json"
$externalRuntimeProofTemplate = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-record-template.json"
$externalRuntimeProofValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$compatibleHostCollectionBundle = Read-JsonOrNull "artifacts\final-release\compatible-host-runtime-proof-collection-bundle.json"
$externalRuntimeProofBackfillPlan = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-backfill-plan.json"
$postPublishVerificationBackfillPlan = Read-JsonOrNull "artifacts\final-release\post-publish-verification-backfill-plan.json"
$externalRuntimeProofCollectionPackage = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-collection-package.json"
$postPublishVerificationCollectionPackage = Read-JsonOrNull "artifacts\final-release\post-publish-verification-collection-package.json"
$postPublishCleanConsumerProjectScan = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-project-scan.json"
$postPublishVerificationInputDraft = Read-JsonOrNull "artifacts\final-release\post-publish-verification-record.input-draft.json"
$releaseClosePreflight = Read-JsonOrNull "artifacts\final-release\release-close-preflight.json"
$postPublishRecord = Read-JsonOrNull "artifacts\final-release\post-publish-verification-record.json"
$postPublishTemplate = Read-JsonOrNull "artifacts\final-release\post-publish-verification-record-template.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$staleAudit = Read-JsonOrNull "artifacts\final-release\stale-release-claims-audit.json"
$packageConsumer = Read-JsonOrNull "artifacts\package-consumer\package-consumer-validation-summary.json"
$linuxValidation = Read-JsonOrNull "artifacts\linux-dry-run\$LinuxRuntimePackageKey\linux-runner-evidence-validation.json"

$commit = "unknown"
try {
  $gitCommit = & git -C $RepositoryRoot rev-parse --short HEAD 2>$null
  if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($gitCommit)) {
    $commit = [string]$gitCommit.Trim()
  }
}
catch {
  $commit = "unknown"
}

$releaseEvidenceState = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$releaseEvidenceComplete = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "isReleaseEvidenceComplete" -DefaultValue $false)
$releaseEvidenceCanPublish = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "canPublishPublicly" -DefaultValue $false)
$releaseEvidenceCanClose = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "canCloseReleaseIssue" -DefaultValue $false)

$finalReleaseStatus = [string](Get-PropertyOrDefault -Object $finalRelease -Name "overallStatus" -DefaultValue "missing-final-release-dry-run")
$finalReleaseBlockingIssueCount = [int](Get-PropertyOrDefault -Object $finalRelease -Name "blockingIssueCount" -DefaultValue -1)
$finalReleaseManualApprovalCount = [int](Get-PropertyOrDefault -Object $finalRelease -Name "manualApprovalCount" -DefaultValue -1)
$finalReleaseCanClose = [bool](Get-PropertyOrDefault -Object $finalRelease -Name "canCloseReleaseIssue" -DefaultValue $false)

$packageProofState = [string](Get-PropertyOrDefault -Object $packageProof -Name "proofState" -DefaultValue "missing-release-package-proof-bundle")
$canUseAsPublicPackageProof = [bool](Get-PropertyOrDefault -Object $packageProof -Name "canUseAsPublicPackageProof" -DefaultValue $false)
$packageProofIsRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $packageProof -Name "isRuntimeExecutionProof" -DefaultValue $false)
$finalPackageReviewState = [string](Get-PropertyOrDefault -Object $finalPackageReview -Name "bundleState" -DefaultValue "missing-final-package-review-bundle")
$finalPackageReviewPackageCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "packageCount" -DefaultValue 0)
$finalPackageReviewNativeAssetCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "nativeAssetCount" -DefaultValue 0)
$finalPackageReviewCanUseAsPublicPackageProof = [bool](Get-PropertyOrDefault -Object $finalPackageReview -Name "canUseAsPublicPackageProof" -DefaultValue $false)

$ownerApprovalStatus = [string](Get-PropertyOrDefault -Object $ownerApproval -Name "overallStatus" -DefaultValue "missing-owner-approval-input-validation")
$ownerApprovalCanPublish = [bool](Get-PropertyOrDefault -Object $ownerApproval -Name "canPublishPublicly" -DefaultValue $false)
$ownerDecisionState = [string](Get-PropertyOrDefault -Object $ownerDecision -Name "recordState" -DefaultValue "missing-release-owner-decision-record")
$ownerDecisionCanPublish = [bool](Get-PropertyOrDefault -Object $ownerDecision -Name "canPublishPublicly" -DefaultValue $false)

$publishChecklistState = [string](Get-PropertyOrDefault -Object $publishChecklist -Name "executionState" -DefaultValue "missing-release-publish-execution-checklist")
$canExecutePublicPublish = [bool](Get-PropertyOrDefault -Object $publishChecklist -Name "canExecutePublicPublish" -DefaultValue $false)
$publishChecklistCanClose = [bool](Get-PropertyOrDefault -Object $publishChecklist -Name "canCloseReleaseIssue" -DefaultValue $false)

$promotionIssueState = [string](Get-PropertyOrDefault -Object $promotionIssue -Name "promotionState" -DefaultValue "missing-release-promotion-issue-record")
$promotionCanPublish = [bool](Get-PropertyOrDefault -Object $promotionIssue -Name "canPublishPublicly" -DefaultValue $false)
$promotionCanClose = [bool](Get-PropertyOrDefault -Object $promotionIssue -Name "canCloseReleaseIssue" -DefaultValue $false)

$externalRuntimeProofStateFallback = if ($externalRuntimeProofValidation) { [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation") } elseif ($externalRuntimeProofTemplate) { [string](Get-PropertyOrDefault -Object $externalRuntimeProofTemplate -Name "proofState" -DefaultValue "template-only") } else { "missing" }
$externalRuntimeProofClassificationFallback = if ($externalRuntimeProofValidation) { [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "proofClassification" -DefaultValue "missing-proof-classification") } elseif ($externalRuntimeProofTemplate) { [string](Get-PropertyOrDefault -Object $externalRuntimeProofTemplate -Name "proofClassification" -DefaultValue "template-only") } else { "missing-proof-classification" }
$externalRuntimeProofState = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofState" -DefaultValue $externalRuntimeProofStateFallback)
$externalRuntimeProofClassification = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofClassification" -DefaultValue $externalRuntimeProofClassificationFallback)
$externalRuntimeProofClassificationPromotable = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofClassificationPromotable" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "proofClassificationPromotable" -DefaultValue $false)))
$externalRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeExecutionEvidence" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "isRuntimeExecutionEvidence" -DefaultValue $false)))
$externalRuntimeProofRuntimePackageKeyMatches = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofRuntimePackageKeyMatches" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "runtimePackageKeyMatches" -DefaultValue $false)))
$externalRuntimeProofPackageSourceRuntimePackageKeyMatches = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofPackageSourceRuntimePackageKeyMatches" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "packageSourceRuntimePackageKeyMatches" -DefaultValue $false)))
$externalRuntimeProofConsumerProjectIdentityReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofConsumerProjectIdentityReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "consumerProjectIdentityReady" -DefaultValue $false)))
$externalRuntimeProofSmokeCommandRuntimeKeyReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofSmokeCommandRuntimeKeyReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "smokeCommandRuntimeKeyReady" -DefaultValue $false)))
$externalRuntimeProofHostReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofHostReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "hostReady" -DefaultValue $false)))
$externalRuntimeProofCommandsReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofCommandsReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "commandsReady" -DefaultValue $false)))
$externalRuntimeProofManagedNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofManagedNupkgSha256Ready" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "managedNupkgSha256Ready" -DefaultValue $false)))
$externalRuntimeProofRuntimeNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofRuntimeNupkgSha256Ready" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "runtimeNupkgSha256Ready" -DefaultValue $false)))
$externalRuntimeProofLogSha256FormatReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofLogSha256FormatReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "logSha256FormatReady" -DefaultValue $false)))
$externalRuntimeProofLogSha256Matches = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofLogSha256Matches" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "logSha256Matches" -DefaultValue $false)))
$externalRuntimeProofFailedProofItemCount = [int](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofFailedProofItemCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "failedProofItemCount" -DefaultValue -1)))
$externalRuntimeProofCanPromote = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "canPromoteRuntimeProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)))
$realExternalRuntimeProofFilePresent = $null -ne $externalRuntimeProof
$realExternalRuntimeProofReady = $realExternalRuntimeProofFilePresent -and $externalRuntimeProofCanPromote -and $externalRuntimeExecutionEvidence -and [string]::Equals($externalRuntimeProofClassification, "package-consumer-runtime", [System.StringComparison]::OrdinalIgnoreCase)

$packageConsumerSmokeResultFallback = [string](Get-PropertyOrDefault -Object $packageConsumer -Name "SmokeResult" -DefaultValue "missing-runtime-proof")
$runtimeProofStatusFallback = [string](Get-PropertyOrDefault -Object $finalRelease -Name "runtimeProofStatus" -DefaultValue $packageConsumerSmokeResultFallback)
$runtimeProofStatus = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "runtimeProofStatus" -DefaultValue $runtimeProofStatusFallback)
$runtimeProofRequiredForRelease = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "runtimeProofRequiredForRelease" -DefaultValue ([bool](Get-PropertyOrDefault -Object $finalRelease -Name "runtimeProofRequiredForRelease" -DefaultValue $true)))
$allowRuntimeSmokeBlocked = [bool](Get-PropertyOrDefault -Object $finalRelease -Name "allowRuntimeSmokeBlocked" -DefaultValue $false)

$postPublishStateFallback = if ($postPublishValidation) { [string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation") } elseif ($postPublishTemplate) { [string](Get-PropertyOrDefault -Object $postPublishTemplate -Name "verificationState" -DefaultValue "template-only") } else { "not-applicable-before-publish" }
$postPublishState = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishVerificationState" -DefaultValue $postPublishStateFallback)
$postPublishProofClassification = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishProofClassification" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublishValidation -Name "postPublishProofClassification" -DefaultValue "missing-post-publish-proof-classification")))
$postPublishProofClassificationPromotable = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishProofClassificationPromotable" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "postPublishProofClassificationPromotable" -DefaultValue $false)))
$postPublishConsumerProjectIdentityReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishConsumerProjectIdentityReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "consumerProjectIdentityReady" -DefaultValue $false)))
$postPublishSmokeCommandRuntimeKeyReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishSmokeCommandRuntimeKeyReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "smokeCommandRuntimeKeyReady" -DefaultValue $false)))
$postPublishHostReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishHostReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "hostReady" -DefaultValue $false)))
$postPublishCommandsReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishCommandsReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "commandsReady" -DefaultValue $false)))
$postPublishStdoutSummaryReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishStdoutSummaryReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "stdoutSummaryReady" -DefaultValue $false)))
$postPublishStderrSummaryReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishStderrSummaryReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "stderrSummaryReady" -DefaultValue $false)))
$postPublishStdoutStderrSummaryReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishStdoutStderrSummaryReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "stdoutStderrSummaryReady" -DefaultValue $false)))
$postPublishAllLogSha256Matches = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishAllLogSha256Matches" -DefaultValue $false)
$isPostPublishVerificationProof = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "isPostPublishVerificationProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "isPostPublishVerificationProof" -DefaultValue $false)))
$postPublishCanClose = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "canCloseReleaseIssue" -DefaultValue $false)
$postPublishRequiredEvidence = @(
  Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishRequiredEvidence" -DefaultValue @(
    Get-PropertyOrDefault -Object $postPublishValidation -Name "postPublishRequiredEvidence" -DefaultValue @()
  )
)
$postPublishRequiredEvidence = Normalize-PostPublishRequiredEvidence -Evidence $postPublishRequiredEvidence
$postPublishRequiredEvidenceCount = if ($postPublishRequiredEvidence.Count -gt 0) {
  $postPublishRequiredEvidence.Count
}
else {
  [int](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishRequiredEvidenceCount" -DefaultValue (
    [int](Get-PropertyOrDefault -Object $postPublishValidation -Name "postPublishRequiredEvidenceCount" -DefaultValue 0)
  ))
}
$realPostPublishRecordPresent = $null -ne $postPublishRecord
$realPostPublishVerificationReady = $realPostPublishRecordPresent -and $postPublishCanClose -and $isPostPublishVerificationProof -and $postPublishProofClassificationPromotable

$staleFindingCount = [int](Get-PropertyOrDefault -Object $staleAudit -Name "findingCount" -DefaultValue -1)
$linuxValidationState = [string](Get-PropertyOrDefault -Object $linuxValidation -Name "validationState" -DefaultValue "missing-linux-runner-validation")
$isRealLinuxRunnerProof = [bool](Get-PropertyOrDefault -Object $linuxValidation -Name "isRealLinuxRunnerProof" -DefaultValue $false)
$compatibleHostCollectionState = [string](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "collectionState" -DefaultValue "missing-compatible-host-runtime-proof-collection-bundle")
$compatibleHostCollectionCanPromote = [bool](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "canPromoteRuntimeProof" -DefaultValue $false)
$externalRuntimeProofBackfillPlanState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofBackfillPlan -Name "planState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofBackfillPlanState" -DefaultValue "missing-external-runtime-proof-backfill-plan")))
$externalRuntimeProofBackfillStepCount = [int](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofBackfillStepCount" -DefaultValue (@((Get-PropertyOrDefault -Object $externalRuntimeProofBackfillPlan -Name "backfillSteps" -DefaultValue @())).Count))
$externalRuntimeProofBackfillCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofBackfillPlan -Name "canPromoteRuntimeProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofBackfillCanPromoteRuntimeProof" -DefaultValue $false)))
$externalRuntimeProofCollectionPackageState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofCollectionPackage -Name "packageState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofCollectionPackageState" -DefaultValue "missing-external-runtime-proof-collection-package")))
$externalRuntimeProofCollectionPackageStepCount = [int](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofCollectionPackageStepCount" -DefaultValue (@((Get-PropertyOrDefault -Object $externalRuntimeProofCollectionPackage -Name "collectionSteps" -DefaultValue @())).Count))
$externalRuntimeProofCollectionPackageCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofCollectionPackage -Name "canPromoteRuntimeProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofCollectionPackageCanPromoteRuntimeProof" -DefaultValue $false)))
$externalRuntimeProofCollectionPackageCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofCollectionPackage -Name "canCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofCollectionPackageCanCloseReleaseIssue" -DefaultValue $false)))
$externalRuntimeProofCollectionPackageRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofCollectionPackage -Name "isRuntimeExecutionEvidence" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofCollectionPackageRuntimeExecutionEvidence" -DefaultValue $false)))
$postPublishVerificationBackfillPlanState = [string](Get-PropertyOrDefault -Object $postPublishVerificationBackfillPlan -Name "planState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishVerificationBackfillPlanState" -DefaultValue "missing-post-publish-verification-backfill-plan")))
$postPublishVerificationBackfillStepCount = [int](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishVerificationBackfillStepCount" -DefaultValue (@((Get-PropertyOrDefault -Object $postPublishVerificationBackfillPlan -Name "backfillSteps" -DefaultValue @())).Count))
$postPublishVerificationBackfillCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $postPublishVerificationBackfillPlan -Name "canCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishVerificationBackfillCanCloseReleaseIssue" -DefaultValue $false)))
$postPublishVerificationCollectionPackageState = [string](Get-PropertyOrDefault -Object $postPublishVerificationCollectionPackage -Name "packageState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishVerificationCollectionPackageState" -DefaultValue "missing-post-publish-verification-collection-package")))
$postPublishVerificationCollectionPackageStepCount = [int](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishVerificationCollectionPackageStepCount" -DefaultValue (@((Get-PropertyOrDefault -Object $postPublishVerificationCollectionPackage -Name "collectionSteps" -DefaultValue @())).Count))
$postPublishVerificationCollectionPackageProof = [bool](Get-PropertyOrDefault -Object $postPublishVerificationCollectionPackage -Name "isPostPublishVerificationProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishVerificationCollectionPackageProof" -DefaultValue $false)))
$postPublishVerificationCollectionPackageCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $postPublishVerificationCollectionPackage -Name "canCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishVerificationCollectionPackageCanCloseReleaseIssue" -DefaultValue $false)))
$postPublishCleanConsumerProjectScanState = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishCleanConsumerProjectScanState" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublishCleanConsumerProjectScan -Name "scanState" -DefaultValue "missing-post-publish-clean-consumer-project-scan")))
$postPublishCleanConsumerProjectScanPassed = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishCleanConsumerProjectScanPassed" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishCleanConsumerProjectScan -Name "scanPassed" -DefaultValue $false)))
$postPublishCleanConsumerProjectScanIsProof = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishCleanConsumerProjectScanIsProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishCleanConsumerProjectScan -Name "isPostPublishVerificationProof" -DefaultValue $false)))
$postPublishCleanConsumerProjectScanCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishCleanConsumerProjectScanCanCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishCleanConsumerProjectScan -Name "canCloseReleaseIssue" -DefaultValue $false)))
$postPublishVerificationInputDraftKind = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishVerificationInputDraftKind" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublishVerificationInputDraft -Name "recordKind" -DefaultValue "missing-post-publish-verification-record-input-draft")))
$postPublishVerificationInputDraftOnly = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishVerificationInputDraftOnly" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationInputDraft -Name "inputDraftOnly" -DefaultValue $false)))
$postPublishVerificationInputDraftIsProof = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishVerificationInputDraftIsProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationInputDraft -Name "isPostPublishVerificationProof" -DefaultValue $false)))
$postPublishVerificationInputDraftCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishVerificationInputDraftCanCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationInputDraft -Name "canCloseReleaseIssue" -DefaultValue $false)))
$ownerReleaseExecutionPackageState = [string](Get-PropertyOrDefault -Object $ownerReleaseExecutionPackage -Name "packageState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerReleaseExecutionPackageState" -DefaultValue "missing-owner-release-execution-package")))
$oneScreenReleaseHoldChecklist = @(Get-PropertyOrDefault -Object $ownerReleaseExecutionPackage -Name "oneScreenReleaseHoldChecklist" -DefaultValue @(Get-PropertyOrDefault -Object $releaseEvidence -Name "oneScreenReleaseHoldChecklist" -DefaultValue @()))
$oneScreenReleaseHoldChecklistCount = if ($oneScreenReleaseHoldChecklist.Count -gt 0) { $oneScreenReleaseHoldChecklist.Count } else { [int](Get-PropertyOrDefault -Object $releaseEvidence -Name "oneScreenReleaseHoldChecklistCount" -DefaultValue 0) }
$releaseProofReadinessSnapshotState = [string](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "readinessState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidence -Name "releaseProofReadinessSnapshotState" -DefaultValue "missing-release-proof-readiness-snapshot")))
$releaseProofReadinessSnapshotItemCount = [int](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "readinessItemCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseEvidence -Name "releaseProofReadinessSnapshotItemCount" -DefaultValue 0)))
$releaseProofReadinessSnapshotReadyProofItemCount = [int](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "readyProofItemCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseEvidence -Name "releaseProofReadinessSnapshotReadyProofItemCount" -DefaultValue 0)))
$releaseProofReadinessSnapshotBlockedProofItemCount = [int](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "blockedProofItemCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseEvidence -Name "releaseProofReadinessSnapshotBlockedProofItemCount" -DefaultValue 0)))
$releaseProofReadinessSnapshotPerformsPublish = [bool](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "performsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "releaseProofReadinessSnapshotPerformsPublish" -DefaultValue $false)))
$releaseProofReadinessSnapshotCanPublishPublicly = [bool](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "canPublishPublicly" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "releaseProofReadinessSnapshotCanPublishPublicly" -DefaultValue $false)))
$releaseProofReadinessSnapshotCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "canCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "releaseProofReadinessSnapshotCanCloseReleaseIssue" -DefaultValue $false)))
$ownerProofInputReadinessState = [string](Get-PropertyOrDefault -Object $ownerProofInputReadiness -Name "readinessState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerProofInputReadinessState" -DefaultValue "missing-owner-proof-input-readiness")))
$ownerProofInputReadinessContractCount = [int](Get-PropertyOrDefault -Object $ownerProofInputReadiness -Name "contractCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerProofInputReadinessContractCount" -DefaultValue 0)))
$ownerProofInputReadinessReadyContractCount = [int](Get-PropertyOrDefault -Object $ownerProofInputReadiness -Name "readyContractCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerProofInputReadinessReadyContractCount" -DefaultValue 0)))
$ownerProofInputReadinessBlockedContractCount = [int](Get-PropertyOrDefault -Object $ownerProofInputReadiness -Name "blockedContractCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerProofInputReadinessBlockedContractCount" -DefaultValue 0)))
$ownerProofInputReadinessPerformsPublish = [bool](Get-PropertyOrDefault -Object $ownerProofInputReadiness -Name "performsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerProofInputReadinessPerformsPublish" -DefaultValue $false)))
$ownerProofInputReadinessCanPublishPublicly = [bool](Get-PropertyOrDefault -Object $ownerProofInputReadiness -Name "canPublishPublicly" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerProofInputReadinessCanPublishPublicly" -DefaultValue $false)))
$ownerProofInputReadinessCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $ownerProofInputReadiness -Name "canCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerProofInputReadinessCanCloseReleaseIssue" -DefaultValue $false)))
$ownerProofInputReadinessValidationState = [string](Get-PropertyOrDefault -Object $ownerProofInputReadinessValidation -Name "validationState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerProofInputReadinessValidationState" -DefaultValue "missing-owner-proof-input-readiness-validation")))
$ownerProofInputReadinessIsValid = [bool](Get-PropertyOrDefault -Object $ownerProofInputReadinessValidation -Name "isValidOwnerProofInputReadiness" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerProofInputReadinessIsValid" -DefaultValue $false)))
$ownerProofInputReadinessValidationFailedBlockerCount = [int](Get-PropertyOrDefault -Object $ownerProofInputReadinessValidation -Name "failedBlockerCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerProofInputReadinessValidationFailedBlockerCount" -DefaultValue -1)))
$ownerProofInputReadinessValidationPerformsPublish = [bool](Get-PropertyOrDefault -Object $ownerProofInputReadinessValidation -Name "performsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerProofInputReadinessValidationPerformsPublish" -DefaultValue $false)))
$ownerProofInputReadinessValidationCanPublishPublicly = [bool](Get-PropertyOrDefault -Object $ownerProofInputReadinessValidation -Name "canPublishPublicly" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerProofInputReadinessValidationCanPublishPublicly" -DefaultValue $false)))
$ownerProofInputReadinessValidationCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $ownerProofInputReadinessValidation -Name "canCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerProofInputReadinessValidationCanCloseReleaseIssue" -DefaultValue $false)))
$releaseClosePreflightState = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "releaseClosePreflightState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "preflightState" -DefaultValue "missing-release-close-preflight")))
$releaseClosePreflightFailedItemCount = [int](Get-PropertyOrDefault -Object $releaseEvidence -Name "releaseClosePreflightFailedItemCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "failedItemCount" -DefaultValue -1)))
$releaseClosePreflightCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "releaseClosePreflightCanCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "canCloseReleaseIssue" -DefaultValue $false)))
$releaseClosePreflightPerformsPublish = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "releaseClosePreflightPerformsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "performsPublish" -DefaultValue $true)))

$canPublish = $ownerApprovalCanPublish -and $ownerDecisionCanPublish -and $canExecutePublicPublish -and $releaseEvidenceCanPublish -and $releaseEvidenceComplete -and $realExternalRuntimeProofReady
$canPromote = $canPublish -and $canUseAsPublicPackageProof -and $externalRuntimeProofCanPromote
$closeReadinessConsistent = ($releaseEvidenceCanClose -eq $finalReleaseCanClose) -and ($releaseEvidenceCanClose -eq $publishChecklistCanClose) -and ($releaseEvidenceCanClose -eq $promotionCanClose) -and ($releaseEvidenceCanClose -eq $postPublishCanClose)
$canCloseReleaseIssue = $closeReadinessConsistent -and $releaseEvidenceCanClose -and $promotionCanClose -and $realPostPublishVerificationReady

$artifactRefs = @(
  New-FreezeArtifactRef -Id "release-evidence-bundle" -Path "artifacts/final-release/release-evidence-bundle.json" -State $releaseEvidenceState -Ready $releaseEvidenceComplete -Boundary "Evidence aggregation is not owner approval or publication."
  New-FreezeArtifactRef -Id "final-release-dry-run" -Path "artifacts/final-release/final-release-dry-run-summary.json" -State $finalReleaseStatus -Ready ($finalReleaseBlockingIssueCount -eq 0 -and $null -ne $finalRelease) -Boundary "Dry run readiness still requires manual owner decisions."
  New-FreezeArtifactRef -Id "final-package-review-bundle" -Path "artifacts/final-release/final-package-review-bundle.json" -State $finalPackageReviewState -Ready $finalPackageReviewCanUseAsPublicPackageProof -Boundary "Local package inventory is not public package proof."
  New-FreezeArtifactRef -Id "release-package-proof-bundle" -Path "artifacts/final-release/release-package-proof-bundle.json" -State $packageProofState -Ready $canUseAsPublicPackageProof -Boundary "Local package proof is not public package proof."
  New-FreezeArtifactRef -Id "owner-approval-input" -Path "artifacts/final-release/release-owner-approval-input-validation.json" -State $ownerApprovalStatus -Ready $ownerApprovalCanPublish -Boundary "Template owner input is not approval."
  New-FreezeArtifactRef -Id "owner-decision-record" -Path "artifacts/final-release/release-owner-decision-record.json" -State $ownerDecisionState -Ready $ownerDecisionCanPublish -Boundary "Generated default decision is pending until owner fills it."
  New-FreezeArtifactRef -Id "publish-execution-checklist" -Path "artifacts/final-release/release-publish-execution-checklist.json" -State $publishChecklistState -Ready $canExecutePublicPublish -Boundary "Checklist contains placeholders and does not push packages."
  New-FreezeArtifactRef -Id "promotion-issue-record" -Path "artifacts/final-release/release-promotion-issue-record.json" -State $promotionIssueState -Ready $promotionCanPublish -Boundary "Issue body is review material, not approval."
  New-FreezeArtifactRef -Id "external-runtime-proof" -Path "artifacts/final-release/external-runtime-proof-record.json" -State $externalRuntimeProofState -Ready $realExternalRuntimeProofReady -Boundary "Template, draft, runbook, collection bundle, and blocked-by-cuda-driver are not runtime proof."
  New-FreezeArtifactRef -Id "external-runtime-proof-backfill-plan" -Path "artifacts/final-release/external-runtime-proof-backfill-plan.json" -State $externalRuntimeProofBackfillPlanState -Ready $false -Boundary "Backfill plan is guidance only and is not compatible-host runtime proof."
  New-FreezeArtifactRef -Id "external-runtime-proof-collection-package" -Path "artifacts/final-release/external-runtime-proof-collection-package.json" -State $externalRuntimeProofCollectionPackageState -Ready $false -Boundary "Collection package is copyable owner guidance only and is not compatible-host runtime proof."
  New-FreezeArtifactRef -Id "post-publish-verification" -Path "artifacts/final-release/post-publish-verification-record.json" -State $postPublishState -Ready $realPostPublishVerificationReady -Boundary "Post-publish proof is only possible after real channel publication and clean consumer smoke."
  New-FreezeArtifactRef -Id "post-publish-verification-backfill-plan" -Path "artifacts/final-release/post-publish-verification-backfill-plan.json" -State $postPublishVerificationBackfillPlanState -Ready $false -Boundary "Backfill plan is guidance only and is not real post-publish proof."
  New-FreezeArtifactRef -Id "post-publish-verification-collection-package" -Path "artifacts/final-release/post-publish-verification-collection-package.json" -State $postPublishVerificationCollectionPackageState -Ready $false -Boundary "Collection package is copyable owner guidance only and is not real post-publish proof."
  New-FreezeArtifactRef -Id "post-publish-clean-consumer-project-scan" -Path "artifacts/final-release/post-publish-clean-consumer-project-scan.json" -State $postPublishCleanConsumerProjectScanState -Ready $false -Boundary "Clean consumer scan is helper evidence only and is not real post-publish proof."
  New-FreezeArtifactRef -Id "post-publish-verification-input-draft" -Path "artifacts/final-release/post-publish-verification-record.input-draft.json" -State $postPublishVerificationInputDraftKind -Ready $false -Boundary "Input draft is helper evidence only and is not real post-publish proof."
  New-FreezeArtifactRef -Id "release-close-preflight" -Path "artifacts/final-release/release-close-preflight.json" -State $releaseClosePreflightState -Ready $false -Boundary "Release close preflight is a gap aggregator, not proof or close approval."
  New-FreezeArtifactRef -Id "stale-release-claims" -Path "artifacts/final-release/stale-release-claims-audit.json" -State ("findingCount=" + $staleFindingCount) -Ready ($staleFindingCount -eq 0) -Boundary "Zero findings is a stale-claim guardrail, not publication approval."
)

$blockingItems = @(
  New-FreezeBlockingItem -Id "real-external-runtime-proof" -Title "Real compatible-host external runtime proof" -Passed $realExternalRuntimeProofReady -CurrentStatus ("state=$externalRuntimeProofState; classification=$externalRuntimeProofClassification; canPromote=$externalRuntimeProofCanPromote; realFilePresent=$realExternalRuntimeProofFilePresent") -RequiredEvidence "artifacts/final-release/external-runtime-proof-record.json validated as package-consumer-runtime with matching runtime package key, host metadata, --runtime-package-key smoke command, package SHA256, and smoke log SHA256." -OwnerAction "Run package-consumer smoke on a compatible CUDA host and validate with Test-ExternalRuntimeProofRecord.ps1 -FailOnNotProof." -Boundary "blocked-by-cuda-driver, dependency-probe-only, template-only, draft-only, runbook, and collection bundle are not smoke passed."
  New-FreezeBlockingItem -Id "external-runtime-proof-backfill-required" -Title "External runtime proof backfill plan boundary" -Passed $false -CurrentStatus ("planState=$externalRuntimeProofBackfillPlanState; stepCount=$externalRuntimeProofBackfillStepCount; canPromoteRuntimeProof=$externalRuntimeProofBackfillCanPromoteRuntimeProof") -RequiredEvidence "A real artifacts/final-release/external-runtime-proof-record.json and -FailOnNotProof validation output, not the backfill plan." -OwnerAction "Use the backfill plan only as execution guidance and collect real compatible-host proof before promotion." -Boundary "Backfill plan is guidance only; it is not runtime proof, publication approval, or release close approval."
  New-FreezeBlockingItem -Id "external-runtime-proof-collection-package-boundary" -Title "External runtime proof collection package boundary" -Passed $false -CurrentStatus ("packageState=$externalRuntimeProofCollectionPackageState; stepCount=$externalRuntimeProofCollectionPackageStepCount; canPromoteRuntimeProof=$externalRuntimeProofCollectionPackageCanPromoteRuntimeProof; canCloseReleaseIssue=$externalRuntimeProofCollectionPackageCanCloseReleaseIssue; isRuntimeExecutionEvidence=$externalRuntimeProofCollectionPackageRuntimeExecutionEvidence") -RequiredEvidence "A real artifacts/final-release/external-runtime-proof-record.json and -FailOnNotProof validation output, not the collection package." -OwnerAction "Use the collection package to reduce owner execution friction, but collect real compatible-host proof before promotion." -Boundary "Collection package is copyable guidance only; it is not runtime proof, publication approval, release close approval, or package push."
  New-FreezeBlockingItem -Id "owner-approval" -Title "Release owner approval input" -Passed $ownerApprovalCanPublish -CurrentStatus ("status=$ownerApprovalStatus; canPublishPublicly=$ownerApprovalCanPublish") -RequiredEvidence "A non-template owner approval input record validated cleanly." -OwnerAction "Fill the owner approval input record after reviewing release evidence." -Boundary "Generated templates and examples cannot approve public publication."
  New-FreezeBlockingItem -Id "owner-decision" -Title "Release owner decision record" -Passed $ownerDecisionCanPublish -CurrentStatus ("recordState=$ownerDecisionState; canPublishPublicly=$ownerDecisionCanPublish") -RequiredEvidence "Owner decision record with explicit approval and required dispositions." -OwnerAction "Record final owner decision after external runtime proof and legal/channel checks." -Boundary "The default generated record remains pending release-owner approval."
  New-FreezeBlockingItem -Id "release-evidence-complete" -Title "Release evidence bundle complete" -Passed $releaseEvidenceComplete -CurrentStatus ("bundleState=$releaseEvidenceState; complete=$releaseEvidenceComplete") -RequiredEvidence "All release evidence bundle proof items complete." -OwnerAction "Fix source evidence items instead of overriding the bundle." -Boundary "Aggregation success is not evidence completion."
  New-FreezeBlockingItem -Id "publish-checklist-authorized" -Title "Publish checklist authorized" -Passed $canExecutePublicPublish -CurrentStatus ("executionState=$publishChecklistState; canExecutePublicPublish=$canExecutePublicPublish") -RequiredEvidence "Owner-approved publish checklist generated after real proof is present." -OwnerAction "Keep publish command placeholders gated until owner approval." -Boundary "The script must not execute dotnet nuget push, GitHub Packages upload, GitHub Release upload, delete, delist, or withdraw."
  New-FreezeBlockingItem -Id "post-publish-verification" -Title "Real post-publish verification proof" -Passed $realPostPublishVerificationReady -CurrentStatus ("state=$postPublishState; classification=$postPublishProofClassification; proof=$isPostPublishVerificationProof; canClose=$postPublishCanClose; realFilePresent=$realPostPublishRecordPresent; stdoutSummaryReady=$postPublishStdoutSummaryReady; stderrSummaryReady=$postPublishStderrSummaryReady; allLogSha256Matches=$postPublishAllLogSha256Matches") -RequiredEvidence "artifacts/final-release/post-publish-verification-record.json from real channel publish plus clean consumer restore/build/smoke, package hashes, host metadata, --runtime-package-key command, stdout/stderr summary, and matching SHA256-backed logs." -OwnerAction "Only fill after an authorized package publish and clean consumer verification." -Boundary "Post-publish template-only, owner-action-required, missing summaries, missing SHA256 match, or synthetic shapes cannot close release issue."
  New-FreezeBlockingItem -Id "post-publish-verification-backfill-required" -Title "Post-publish verification backfill plan boundary" -Passed $false -CurrentStatus ("planState=$postPublishVerificationBackfillPlanState; stepCount=$postPublishVerificationBackfillStepCount; canCloseReleaseIssue=$postPublishVerificationBackfillCanCloseReleaseIssue") -RequiredEvidence "A real artifacts/final-release/post-publish-verification-record.json from authorized channel publication and clean consumer verification, not the backfill plan." -OwnerAction "Use the backfill plan only after authorized publication to collect real post-publish proof." -Boundary "Backfill plan is guidance only; it is not post-publish proof, publication approval, or release close approval."
  New-FreezeBlockingItem -Id "post-publish-verification-collection-package-boundary" -Title "Post-publish verification collection package boundary" -Passed $false -CurrentStatus ("packageState=$postPublishVerificationCollectionPackageState; stepCount=$postPublishVerificationCollectionPackageStepCount; isPostPublishVerificationProof=$postPublishVerificationCollectionPackageProof; canCloseReleaseIssue=$postPublishVerificationCollectionPackageCanCloseReleaseIssue") -RequiredEvidence "A real artifacts/final-release/post-publish-verification-record.json from authorized channel publication and clean consumer verification, not the collection package." -OwnerAction "Use the collection package only after authorized publication to collect real post-publish proof." -Boundary "Collection package is copyable guidance only; it is not post-publish proof, publication approval, release close approval, or package push."
  New-FreezeBlockingItem -Id "post-publish-clean-consumer-scan-boundary" -Title "Post-publish clean consumer scan boundary" -Passed $false -CurrentStatus ("scanState=$postPublishCleanConsumerProjectScanState; scanPassed=$postPublishCleanConsumerProjectScanPassed; isPostPublishVerificationProof=$postPublishCleanConsumerProjectScanIsProof; canCloseReleaseIssue=$postPublishCleanConsumerProjectScanCanCloseReleaseIssue") -RequiredEvidence "A real post-publish verification record from authorized channel publication and clean consumer smoke, not the helper scan." -OwnerAction "Use the clean consumer scan as boundary input only after authorized publication." -Boundary "Clean consumer scan is helper evidence only; it is not post-publish proof, release close approval, or package push."
  New-FreezeBlockingItem -Id "post-publish-verification-input-draft-boundary" -Title "Post-publish verification input draft boundary" -Passed $false -CurrentStatus ("recordKind=$postPublishVerificationInputDraftKind; inputDraftOnly=$postPublishVerificationInputDraftOnly; isPostPublishVerificationProof=$postPublishVerificationInputDraftIsProof; canCloseReleaseIssue=$postPublishVerificationInputDraftCanCloseReleaseIssue") -RequiredEvidence "A real validated post-publish verification record, not the input draft." -OwnerAction "Use the input draft only as fill guidance after authorized publication." -Boundary "Input draft is helper evidence only; it is not post-publish proof, release close approval, or package push."
  New-FreezeBlockingItem -Id "release-close-preflight-boundary" -Title "Release close preflight boundary" -Passed $false -CurrentStatus ("preflightState=$releaseClosePreflightState; failedItemCount=$releaseClosePreflightFailedItemCount; canCloseReleaseIssue=$releaseClosePreflightCanCloseReleaseIssue; performsPublish=$releaseClosePreflightPerformsPublish") -RequiredEvidence "All real proof gates satisfied: owner authorization, external runtime proof, and post-publish verification proof." -OwnerAction "Rerun preflight after attaching real proof artifacts; do not override its blocked state." -Boundary "Release close preflight aggregates real-proof gaps; it is not proof, owner authorization, or package push."
  New-FreezeBlockingItem -Id "close-readiness-consistency" -Title "Close readiness consistency" -Passed $closeReadinessConsistent -CurrentStatus ("releaseEvidence=$releaseEvidenceCanClose; finalDryRun=$finalReleaseCanClose; publishChecklist=$publishChecklistCanClose; promotionIssue=$promotionCanClose; postPublish=$postPublishCanClose") -RequiredEvidence "All close-readiness fields agree and remain false until real post-publish proof exists." -OwnerAction "Regenerate release evidence, final dry run, publish checklist, and promotion issue after proof changes." -Boundary "No single artifact can unilaterally close the release issue."
  New-FreezeBlockingItem -Id "stale-release-claims" -Title "Stale release claims audit" -Passed ($staleFindingCount -eq 0) -CurrentStatus ("findingCount=$staleFindingCount") -RequiredEvidence "stale-release-claims-audit.json with findingCount=0." -OwnerAction "Fix release-facing docs or generated issue text before owner review." -Boundary "The stale-claim audit is a text guardrail, not publication approval."
  New-FreezeBlockingItem -Id "blocked-cuda-driver-visible" -Title "CUDA driver blocker remains visible" -Passed (-not [string]::Equals($runtimeProofStatus, "blocked-by-cuda-driver", [System.StringComparison]::OrdinalIgnoreCase)) -CurrentStatus ("runtimeProofStatus=$runtimeProofStatus; allowRuntimeSmokeBlocked=$allowRuntimeSmokeBlocked; runtimeProofRequiredForRelease=$runtimeProofRequiredForRelease") -RequiredEvidence "A compatible CUDA host smoke pass or explicit owner disposition for RC-only release." -OwnerAction "Do not treat blocked-by-cuda-driver as smoke passed; collect compatible-host proof." -Boundary "AllowRuntimeSmokeBlocked records dry-run intent only."
)

$blockingCount = @($blockingItems | Where-Object { -not $_.passed }).Count
$freezeState = if ($blockingCount -eq 0 -and $canPublish -and $canPromote) { "freeze-ready-for-owner-publish-authorization" } else { "blocked-freeze-owner-action-required" }

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "release-candidate-freeze-summary"
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  repositoryRoot = $RepositoryRoot
  gitCommit = $commit
  freezeState = $freezeState
  blockingItemCount = $blockingCount
  canPublish = $canPublish
  canPromote = $canPromote
  canCloseReleaseIssue = $canCloseReleaseIssue
  performsPublish = $false
  requiresHumanOwner = $true
  releaseEvidenceBundleState = $releaseEvidenceState
  isReleaseEvidenceComplete = $releaseEvidenceComplete
  releaseEvidenceCanCloseReleaseIssue = $releaseEvidenceCanClose
  finalReleaseDryRunStatus = $finalReleaseStatus
  finalReleaseBlockingIssueCount = $finalReleaseBlockingIssueCount
  finalReleaseManualApprovalCount = $finalReleaseManualApprovalCount
  finalReleaseCanCloseReleaseIssue = $finalReleaseCanClose
  finalPackageReviewState = $finalPackageReviewState
  finalPackageReviewPackageCount = $finalPackageReviewPackageCount
  finalPackageReviewNativeAssetCount = $finalPackageReviewNativeAssetCount
  finalPackageReviewCanUseAsPublicPackageProof = $finalPackageReviewCanUseAsPublicPackageProof
  releasePackageProofState = $packageProofState
  canUseAsPublicPackageProof = $canUseAsPublicPackageProof
  packageProofIsRuntimeExecutionProof = $packageProofIsRuntimeExecutionProof
  ownerApprovalInputValidationStatus = $ownerApprovalStatus
  ownerApprovalCanPublishPublicly = $ownerApprovalCanPublish
  ownerDecisionState = $ownerDecisionState
  ownerDecisionCanPublishPublicly = $ownerDecisionCanPublish
  publishExecutionChecklistState = $publishChecklistState
  canExecutePublicPublish = $canExecutePublicPublish
  publishChecklistCanCloseReleaseIssue = $publishChecklistCanClose
  promotionIssueState = $promotionIssueState
  promotionIssueCanPublishPublicly = $promotionCanPublish
  promotionIssueCanCloseReleaseIssue = $promotionCanClose
  runtimeProofStatus = $runtimeProofStatus
  runtimeProofRequiredForRelease = $runtimeProofRequiredForRelease
  allowRuntimeSmokeBlocked = $allowRuntimeSmokeBlocked
  externalRuntimeProofState = $externalRuntimeProofState
  externalRuntimeProofClassification = $externalRuntimeProofClassification
  externalRuntimeProofClassificationPromotable = $externalRuntimeProofClassificationPromotable
  externalRuntimeProofRuntimePackageKeyMatches = $externalRuntimeProofRuntimePackageKeyMatches
  externalRuntimeProofPackageSourceRuntimePackageKeyMatches = $externalRuntimeProofPackageSourceRuntimePackageKeyMatches
  externalRuntimeProofConsumerProjectIdentityReady = $externalRuntimeProofConsumerProjectIdentityReady
  externalRuntimeProofSmokeCommandRuntimeKeyReady = $externalRuntimeProofSmokeCommandRuntimeKeyReady
  externalRuntimeProofHostReady = $externalRuntimeProofHostReady
  externalRuntimeProofCommandsReady = $externalRuntimeProofCommandsReady
  externalRuntimeProofManagedNupkgSha256Ready = $externalRuntimeProofManagedNupkgSha256Ready
  externalRuntimeProofRuntimeNupkgSha256Ready = $externalRuntimeProofRuntimeNupkgSha256Ready
  externalRuntimeProofLogSha256FormatReady = $externalRuntimeProofLogSha256FormatReady
  externalRuntimeProofLogSha256Matches = $externalRuntimeProofLogSha256Matches
  externalRuntimeProofFailedProofItemCount = $externalRuntimeProofFailedProofItemCount
  externalRuntimeExecutionEvidence = $externalRuntimeExecutionEvidence
  externalRuntimeProofCanPromoteRuntimeProof = $externalRuntimeProofCanPromote
  realExternalRuntimeProofFilePresent = $realExternalRuntimeProofFilePresent
  realExternalRuntimeProofReady = $realExternalRuntimeProofReady
  compatibleHostRuntimeProofCollectionBundleState = $compatibleHostCollectionState
  compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof = $compatibleHostCollectionCanPromote
  externalRuntimeProofBackfillPlanState = $externalRuntimeProofBackfillPlanState
  externalRuntimeProofBackfillStepCount = $externalRuntimeProofBackfillStepCount
  externalRuntimeProofBackfillCanPromoteRuntimeProof = $externalRuntimeProofBackfillCanPromoteRuntimeProof
  externalRuntimeProofCollectionPackageState = $externalRuntimeProofCollectionPackageState
  externalRuntimeProofCollectionPackageStepCount = $externalRuntimeProofCollectionPackageStepCount
  externalRuntimeProofCollectionPackageCanPromoteRuntimeProof = $externalRuntimeProofCollectionPackageCanPromoteRuntimeProof
  externalRuntimeProofCollectionPackageCanCloseReleaseIssue = $externalRuntimeProofCollectionPackageCanCloseReleaseIssue
  externalRuntimeProofCollectionPackageRuntimeExecutionEvidence = $externalRuntimeProofCollectionPackageRuntimeExecutionEvidence
  postPublishVerificationState = $postPublishState
  postPublishProofClassification = $postPublishProofClassification
  postPublishProofClassificationPromotable = $postPublishProofClassificationPromotable
  postPublishConsumerProjectIdentityReady = $postPublishConsumerProjectIdentityReady
  postPublishSmokeCommandRuntimeKeyReady = $postPublishSmokeCommandRuntimeKeyReady
  postPublishHostReady = $postPublishHostReady
  postPublishCommandsReady = $postPublishCommandsReady
  postPublishStdoutSummaryReady = $postPublishStdoutSummaryReady
  postPublishStderrSummaryReady = $postPublishStderrSummaryReady
  postPublishStdoutStderrSummaryReady = $postPublishStdoutStderrSummaryReady
  postPublishAllLogSha256Matches = $postPublishAllLogSha256Matches
  postPublishRequiredEvidence = $postPublishRequiredEvidence
  postPublishRequiredEvidenceCount = $postPublishRequiredEvidenceCount
  isPostPublishVerificationProof = $isPostPublishVerificationProof
  postPublishVerificationBackfillPlanState = $postPublishVerificationBackfillPlanState
  postPublishVerificationBackfillStepCount = $postPublishVerificationBackfillStepCount
  postPublishVerificationBackfillCanCloseReleaseIssue = $postPublishVerificationBackfillCanCloseReleaseIssue
  postPublishVerificationCollectionPackageState = $postPublishVerificationCollectionPackageState
  postPublishVerificationCollectionPackageStepCount = $postPublishVerificationCollectionPackageStepCount
  postPublishVerificationCollectionPackageProof = $postPublishVerificationCollectionPackageProof
  postPublishVerificationCollectionPackageCanCloseReleaseIssue = $postPublishVerificationCollectionPackageCanCloseReleaseIssue
  postPublishCleanConsumerProjectScanState = $postPublishCleanConsumerProjectScanState
  postPublishCleanConsumerProjectScanPassed = $postPublishCleanConsumerProjectScanPassed
  postPublishCleanConsumerProjectScanIsProof = $postPublishCleanConsumerProjectScanIsProof
  postPublishCleanConsumerProjectScanCanCloseReleaseIssue = $postPublishCleanConsumerProjectScanCanCloseReleaseIssue
  postPublishVerificationInputDraftKind = $postPublishVerificationInputDraftKind
  postPublishVerificationInputDraftOnly = $postPublishVerificationInputDraftOnly
  postPublishVerificationInputDraftIsProof = $postPublishVerificationInputDraftIsProof
  postPublishVerificationInputDraftCanCloseReleaseIssue = $postPublishVerificationInputDraftCanCloseReleaseIssue
  ownerReleaseExecutionPackageState = $ownerReleaseExecutionPackageState
  oneScreenReleaseHoldChecklist = @($oneScreenReleaseHoldChecklist)
  oneScreenReleaseHoldChecklistCount = $oneScreenReleaseHoldChecklistCount
  releaseProofReadinessSnapshotState = $releaseProofReadinessSnapshotState
  releaseProofReadinessSnapshotItemCount = $releaseProofReadinessSnapshotItemCount
  releaseProofReadinessSnapshotReadyProofItemCount = $releaseProofReadinessSnapshotReadyProofItemCount
  releaseProofReadinessSnapshotBlockedProofItemCount = $releaseProofReadinessSnapshotBlockedProofItemCount
  releaseProofReadinessSnapshotPerformsPublish = $releaseProofReadinessSnapshotPerformsPublish
  releaseProofReadinessSnapshotCanPublishPublicly = $releaseProofReadinessSnapshotCanPublishPublicly
  releaseProofReadinessSnapshotCanCloseReleaseIssue = $releaseProofReadinessSnapshotCanCloseReleaseIssue
  ownerProofInputReadinessState = $ownerProofInputReadinessState
  ownerProofInputReadinessContractCount = $ownerProofInputReadinessContractCount
  ownerProofInputReadinessReadyContractCount = $ownerProofInputReadinessReadyContractCount
  ownerProofInputReadinessBlockedContractCount = $ownerProofInputReadinessBlockedContractCount
  ownerProofInputReadinessPerformsPublish = $ownerProofInputReadinessPerformsPublish
  ownerProofInputReadinessCanPublishPublicly = $ownerProofInputReadinessCanPublishPublicly
  ownerProofInputReadinessCanCloseReleaseIssue = $ownerProofInputReadinessCanCloseReleaseIssue
  ownerProofInputReadinessValidationState = $ownerProofInputReadinessValidationState
  ownerProofInputReadinessIsValid = $ownerProofInputReadinessIsValid
  ownerProofInputReadinessValidationFailedBlockerCount = $ownerProofInputReadinessValidationFailedBlockerCount
  ownerProofInputReadinessValidationPerformsPublish = $ownerProofInputReadinessValidationPerformsPublish
  ownerProofInputReadinessValidationCanPublishPublicly = $ownerProofInputReadinessValidationCanPublishPublicly
  ownerProofInputReadinessValidationCanCloseReleaseIssue = $ownerProofInputReadinessValidationCanCloseReleaseIssue
  releaseClosePreflightState = $releaseClosePreflightState
  releaseClosePreflightFailedItemCount = $releaseClosePreflightFailedItemCount
  releaseClosePreflightCanCloseReleaseIssue = $releaseClosePreflightCanCloseReleaseIssue
  releaseClosePreflightPerformsPublish = $releaseClosePreflightPerformsPublish
  postPublishCanCloseReleaseIssue = $postPublishCanClose
  realPostPublishRecordPresent = $realPostPublishRecordPresent
  realPostPublishVerificationReady = $realPostPublishVerificationReady
  closeReadinessConsistent = $closeReadinessConsistent
  staleReleaseClaimsFindingCount = $staleFindingCount
  linuxRunnerValidationState = $linuxValidationState
  isRealLinuxRunnerProof = $isRealLinuxRunnerProof
  artifactRefs = $artifactRefs
  blockingItems = $blockingItems
  sourceEvidence = @(
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/final-release-dry-run-summary.json",
    "artifacts/final-release/final-package-review-bundle.json",
    "artifacts/final-release/release-package-proof-bundle.json",
    "artifacts/final-release/release-owner-approval-input-validation.json",
    "artifacts/final-release/release-owner-decision-record.json",
    "artifacts/final-release/owner-release-execution-package.json",
    "artifacts/final-release/owner-release-execution-package.md",
    "artifacts/final-release/release-proof-readiness-snapshot.json",
    "artifacts/final-release/release-proof-readiness-snapshot.md",
    "artifacts/final-release/owner-proof-input-readiness.json",
    "artifacts/final-release/owner-proof-input-readiness.md",
    "artifacts/final-release/owner-proof-input-readiness-validation.json",
    "artifacts/final-release/owner-proof-input-readiness-validation.md",
    "artifacts/final-release/release-publish-execution-checklist.json",
    "artifacts/final-release/release-promotion-issue-record.json",
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/final-release/external-runtime-proof-record.json",
    "artifacts/final-release/external-runtime-proof-backfill-plan.json",
    "artifacts/final-release/external-runtime-proof-backfill-plan.md",
    "artifacts/final-release/external-runtime-proof-collection-package.json",
    "artifacts/final-release/external-runtime-proof-collection-package.md",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/post-publish-verification-record.json",
    "artifacts/final-release/post-publish-verification-backfill-plan.json",
    "artifacts/final-release/post-publish-verification-backfill-plan.md",
    "artifacts/final-release/post-publish-verification-collection-package.json",
    "artifacts/final-release/post-publish-verification-collection-package.md",
    "artifacts/final-release/post-publish-clean-consumer-project-scan.json",
    "artifacts/final-release/post-publish-verification-record.input-draft.json",
    "artifacts/final-release/release-close-preflight.json",
    "artifacts/final-release/stale-release-claims-audit.json",
    "artifacts/package-consumer/package-consumer-validation-summary.json",
    "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json"
  )
  safetyNotes = @(
    "Release candidate freeze is not publish.",
    "This script does not execute dotnet nuget push, GitHub Packages upload, GitHub Release upload, delete, delist, or withdraw.",
    "Owner approval and owner decision are required but are not proof by themselves.",
    "The final package review bundle is local package inventory, not public package proof.",
    "Template, draft, example, runbook, collection bundle, collection package, dependency-probe-only, and blocked-by-cuda-driver are not real runtime proof.",
    "Backfill plans and collection packages are guidance only; they are not runtime proof, post-publish proof, publication approval, release close approval, or package push.",
    "Post-publish clean consumer scan and input draft are helper artifacts only; they cannot close the release issue.",
    "Release close preflight aggregates real-proof gaps but cannot substitute external runtime proof, owner authorization, or post-publish verification proof.",
    "Release proof readiness snapshot is a compact status view only; it cannot publish, close the release issue, or substitute real proof records.",
    "blocked-by-cuda-driver is not smoke passed.",
    "Post-publish verification can close the release issue only after a real channel publish and clean consumer package-consumer smoke.",
    "canCloseReleaseIssue remains false until real post-publish proof and consistency across release artifacts are present."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "release-candidate-freeze-summary.json"
$markdownPath = Join-Path $outputRoot "release-candidate-freeze-summary.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Candidate Freeze Summary")
$lines.Add("")
$lines.Add("Runtime key: ``$RuntimePackageKey``")
$lines.Add("")
$lines.Add("Freeze state: ``$freezeState``")
$lines.Add("")
$lines.Add("This summary freezes the current release-candidate evidence state for owner review. It does not publish packages and does not turn templates, drafts, runbooks, or collection bundles into proof.")
$lines.Add("")
$lines.Add("## Readiness")
$lines.Add("")
$lines.Add("- can publish: ``$canPublish``")
$lines.Add("- can promote: ``$canPromote``")
$lines.Add("- can close release issue: ``$canCloseReleaseIssue``")
$lines.Add("- blocking item count: $blockingCount")
$lines.Add("- close readiness consistent: ``$closeReadinessConsistent``")
$lines.Add("- post-publish required evidence count: ``$postPublishRequiredEvidenceCount``")
$lines.Add("")
$lines.Add("## Proof Snapshot")
$lines.Add("")
$lines.Add("- external runtime proof state: ``$externalRuntimeProofState``")
$lines.Add("- external runtime proof classification: ``$externalRuntimeProofClassification``")
$lines.Add("- real external runtime proof file present: ``$realExternalRuntimeProofFilePresent``")
$lines.Add("- real external runtime proof ready: ``$realExternalRuntimeProofReady``")
$lines.Add("- external runtime proof backfill plan: ``$externalRuntimeProofBackfillPlanState``")
$lines.Add("- external runtime proof backfill step count: ``$externalRuntimeProofBackfillStepCount``")
$lines.Add("- external runtime proof backfill can promote proof: ``$externalRuntimeProofBackfillCanPromoteRuntimeProof``")
$lines.Add("- external runtime proof collection package: ``$externalRuntimeProofCollectionPackageState``")
$lines.Add("- external runtime proof collection package step count: ``$externalRuntimeProofCollectionPackageStepCount``")
$lines.Add("- external runtime proof collection package can promote proof: ``$externalRuntimeProofCollectionPackageCanPromoteRuntimeProof``")
$lines.Add("- external runtime proof collection package can close release issue: ``$externalRuntimeProofCollectionPackageCanCloseReleaseIssue``")
$lines.Add("- runtime proof status: ``$runtimeProofStatus``")
$lines.Add("- blocked-by-cuda-driver is smoke passed: ``false``")
$lines.Add("- post-publish verification state: ``$postPublishState``")
$lines.Add("- real post-publish record present: ``$realPostPublishRecordPresent``")
$lines.Add("- real post-publish verification ready: ``$realPostPublishVerificationReady``")
$lines.Add("- post-publish verification backfill plan: ``$postPublishVerificationBackfillPlanState``")
$lines.Add("- post-publish verification backfill step count: ``$postPublishVerificationBackfillStepCount``")
$lines.Add("- post-publish verification backfill can close release issue: ``$postPublishVerificationBackfillCanCloseReleaseIssue``")
$lines.Add("- post-publish verification collection package: ``$postPublishVerificationCollectionPackageState``")
$lines.Add("- post-publish verification collection package step count: ``$postPublishVerificationCollectionPackageStepCount``")
$lines.Add("- post-publish verification collection package is proof: ``$postPublishVerificationCollectionPackageProof``")
$lines.Add("- post-publish verification collection package can close release issue: ``$postPublishVerificationCollectionPackageCanCloseReleaseIssue``")
$lines.Add("- post-publish clean consumer project scan: ``$postPublishCleanConsumerProjectScanState``")
$lines.Add("- post-publish verification input draft: ``$postPublishVerificationInputDraftKind``")
$lines.Add("- owner release execution package: ``$ownerReleaseExecutionPackageState``")
$lines.Add("- one-screen release hold checklist count: ``$oneScreenReleaseHoldChecklistCount``")
$lines.Add("- release proof readiness snapshot: ``$releaseProofReadinessSnapshotState``")
$lines.Add("- release proof readiness item count: ``$releaseProofReadinessSnapshotItemCount``")
$lines.Add("- release proof readiness ready proof item count: ``$releaseProofReadinessSnapshotReadyProofItemCount``")
$lines.Add("- release proof readiness blocked proof item count: ``$releaseProofReadinessSnapshotBlockedProofItemCount``")
$lines.Add("- release proof readiness can publish publicly: ``$releaseProofReadinessSnapshotCanPublishPublicly``")
$lines.Add("- release proof readiness can close release issue: ``$releaseProofReadinessSnapshotCanCloseReleaseIssue``")
$lines.Add("- owner proof input readiness: ``$ownerProofInputReadinessState``")
$lines.Add("- owner proof input contract count: ``$ownerProofInputReadinessContractCount``")
$lines.Add("- owner proof input ready contract count: ``$ownerProofInputReadinessReadyContractCount``")
$lines.Add("- owner proof input blocked contract count: ``$ownerProofInputReadinessBlockedContractCount``")
$lines.Add("- owner proof input can publish publicly: ``$ownerProofInputReadinessCanPublishPublicly``")
$lines.Add("- owner proof input can close release issue: ``$ownerProofInputReadinessCanCloseReleaseIssue``")
$lines.Add("- owner proof input validation: ``$ownerProofInputReadinessValidationState``")
$lines.Add("- owner proof input validation is valid: ``$ownerProofInputReadinessIsValid``")
$lines.Add("- owner proof input validation failed blocker count: ``$ownerProofInputReadinessValidationFailedBlockerCount``")
$lines.Add("- owner proof input validation can publish publicly: ``$ownerProofInputReadinessValidationCanPublishPublicly``")
$lines.Add("- owner proof input validation can close release issue: ``$ownerProofInputReadinessValidationCanCloseReleaseIssue``")
$lines.Add("- release close preflight: ``$releaseClosePreflightState``")
$lines.Add("- release close preflight failed item count: ``$releaseClosePreflightFailedItemCount``")
$lines.Add("- owner approval status: ``$ownerApprovalStatus``")
$lines.Add("- owner decision state: ``$ownerDecisionState``")
$lines.Add("")
$lines.Add("## One-Screen Release Hold Checklist")
$lines.Add("")
$lines.Add("This section mirrors ``owner-release-execution-package`` for final owner handoff. It is not proof, publication approval, release close approval, or package push.")
$lines.Add("")
$lines.Add("| ID | Owner-visible blocker | Current state | Owner next action | Validator command | Required real inputs | Cannot use |")
$lines.Add("| --- | --- | --- | --- | --- | --- | --- |")
foreach ($item in $oneScreenReleaseHoldChecklist) {
  $requiredInputs = @($item.requiredRealInputs) -join "<br/>"
  $cannotUse = @($item.cannotUse) -join "<br/>"
  $lines.Add("| ``$($item.id)`` | $($item.ownerVisibleBlocker.Replace("|", "\|")) | $($item.currentState.Replace("|", "\|")) | $($item.ownerNextAction.Replace("|", "\|")) | ``$($item.validatorCommand)`` | $($requiredInputs.Replace("|", "\|")) | $($cannotUse.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Post-Publish Required Evidence")
$lines.Add("")
foreach ($field in $postPublishRequiredEvidence) {
  $lines.Add("- ``$field``")
}
$lines.Add("")
$lines.Add("## Blocking Items")
$lines.Add("")
$lines.Add("| ID | State | Current status | Owner action |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($item in $blockingItems) {
  $lines.Add("| ``$($item.id)`` | ``$($item.state)`` | $($item.currentStatus.Replace("|", "\|")) | $($item.ownerAction.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Artifact References")
$lines.Add("")
$lines.Add("| ID | State | Ready | Path |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($artifact in $artifactRefs) {
  $lines.Add("| ``$($artifact.id)`` | ``$($artifact.state)`` | ``$($artifact.ready)`` | ``$($artifact.path)`` |")
}
$lines.Add("")
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $record.safetyNotes) {
  $lines.Add("- $note")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release candidate freeze summary written to $jsonPath"
Write-Host "Release candidate freeze summary written to $markdownPath"
