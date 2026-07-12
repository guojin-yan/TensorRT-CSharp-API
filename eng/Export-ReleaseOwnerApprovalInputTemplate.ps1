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

function New-OwnerInput {
  param(
    [string]$Id,
    [string]$Title,
    [string]$CurrentStatus,
    [string[]]$AllowedDecisionStates,
    [string]$RequiredOwnerInput,
    [string]$RequiredEvidence,
    [string]$Boundary,
    [bool]$RequiredForPublicRelease
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    currentStatus = $CurrentStatus
    requiredForPublicRelease = $RequiredForPublicRelease
    decisionState = "pending"
    allowedDecisionStates = @($AllowedDecisionStates)
    requiredOwnerInput = $RequiredOwnerInput
    requiredEvidence = $RequiredEvidence
    ownerName = ""
    decidedAtUtc = $null
    rationale = ""
    evidenceUri = ""
    boundary = $Boundary
  }
}

$finalRelease = Read-JsonOrNull "artifacts\final-release\final-release-dry-run-summary.json"
$decisionRecord = Read-JsonOrNull "artifacts\final-release\release-owner-decision-record.json"
$promotionIssue = Read-JsonOrNull "artifacts\final-release\release-promotion-issue-record.json"
$staleAudit = Read-JsonOrNull "artifacts\final-release\stale-release-claims-audit.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$finalPackageReview = Read-JsonOrNull "artifacts\final-release\final-package-review-bundle.json"
$externalRuntimeProof = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-record-template.json"
$externalRuntimeProofValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$compatibleHostRunbook = Read-JsonOrNull "artifacts\final-release\compatible-host-runtime-proof-runbook.json"
$compatibleHostCollectionBundle = Read-JsonOrNull "artifacts\final-release\compatible-host-runtime-proof-collection-bundle.json"
$externalRuntimeProofBackfillPlan = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-backfill-plan.json"
$postPublishVerificationBackfillPlan = Read-JsonOrNull "artifacts\final-release\post-publish-verification-backfill-plan.json"
$postPublish = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"

$overallStatus = if ($finalRelease) { [string]$finalRelease.overallStatus } else { "missing-final-release-dry-run" }
$blockingIssueCount = if ($finalRelease) { [int]$finalRelease.blockingIssueCount } else { -1 }
$manualApprovalCount = if ($finalRelease) { [int]$finalRelease.manualApprovalCount } else { -1 }
$smokeStatus = if ($finalRelease) { [string]$finalRelease.packageConsumerSmokeStatus } else { "missing" }
$runtimeProofStatus = if ($finalRelease -and $finalRelease.PSObject.Properties.Name -contains "runtimeProofStatus") { [string]$finalRelease.runtimeProofStatus } else { $smokeStatus }
$runtimeProofRequiredForRelease = if ($finalRelease -and $finalRelease.PSObject.Properties.Name -contains "runtimeProofRequiredForRelease") { [bool]$finalRelease.runtimeProofRequiredForRelease } else { -not [string]::Equals($runtimeProofStatus, "ready", [System.StringComparison]::OrdinalIgnoreCase) }
$allowRuntimeSmokeBlocked = if ($finalRelease -and $finalRelease.PSObject.Properties.Name -contains "allowRuntimeSmokeBlocked") { [bool]$finalRelease.allowRuntimeSmokeBlocked } else { $false }
$runtimeProofBlockerOwnerActionStatus = if ([string]::Equals($runtimeProofStatus, "ready", [System.StringComparison]::OrdinalIgnoreCase)) { "resolved" } else { "owner-action-required" }
$runtimeProofBlockerCategory = switch ($runtimeProofStatus) {
  "ready" { "none"; break }
  "blocked-by-cuda-driver" { "cuda-driver-runtime-compatibility"; break }
  "blocked-by-application-control" { "application-control-policy"; break }
  "not-requested" { "runtime-smoke-not-requested"; break }
  default { "runtime-proof-incomplete"; break }
}
$runtimeProofOwnerCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey $RuntimePackageKey -RunSmoke -AllowSmokeFailure"
$compatibleHostRunbookCommands = Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "commands" -DefaultValue $null
$compatibleHostRunbookState = [string](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "runbookState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookState" -DefaultValue "missing-compatible-host-runtime-proof-runbook")))
$compatibleHostRunbookCompatibleHostRequired = [bool](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "compatibleHostRequired" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookCompatibleHostRequired" -DefaultValue $true)))
$compatibleHostRunbookPerformsPublish = [bool](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "performsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookPerformsPublish" -DefaultValue $false)))
$compatibleHostRunbookApprovesPublicRelease = [bool](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "approvesPublicRelease" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookApprovesPublicRelease" -DefaultValue $false)))
$compatibleHostRunbookCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "canPromoteRuntimeProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookCanPromoteRuntimeProof" -DefaultValue $false)))
$compatibleHostRunbookRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "isRuntimeExecutionEvidence" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookRuntimeExecutionEvidence" -DefaultValue $false)))
$compatibleHostRunbookPromotionBlockedReason = [string](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "promotionBlockedReason" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookPromotionBlockedReason" -DefaultValue "missing-compatible-host-runbook-blocked-reason")))
$compatibleHostRunbookRunPackageConsumerSmokeCommand = [string](Get-PropertyOrDefault -Object $compatibleHostRunbookCommands -Name "runPackageConsumerSmoke" -DefaultValue $runtimeProofOwnerCommand)
$compatibleHostRunbookValidateFilledRecordCommand = [string](Get-PropertyOrDefault -Object $compatibleHostRunbookCommands -Name "validateFilledRecord" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof")
$compatibleHostCollectionBundleCommands = Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "commands" -DefaultValue $null
$compatibleHostCollectionBundleOperatorQuickStart = @(Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "operatorQuickStart" -DefaultValue @())
$compatibleHostCollectionBundlePreflightChecklist = @(Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "preflightChecklist" -DefaultValue @())
$compatibleHostCollectionBundleCopyableExecutionOrder = @(Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "copyableExecutionOrder" -DefaultValue @())
$compatibleHostCollectionBundleOwnerInputArtifacts = @(Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "ownerInputArtifacts" -DefaultValue @())
$compatibleHostCollectionBundleState = [string](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "collectionState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleState" -DefaultValue "missing-compatible-host-runtime-proof-collection-bundle")))
$compatibleHostCollectionBundleCompatibleHostRequired = [bool](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "compatibleHostRequired" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleCompatibleHostRequired" -DefaultValue $true)))
$compatibleHostCollectionBundlePerformsPublish = [bool](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "performsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundlePerformsPublish" -DefaultValue $false)))
$compatibleHostCollectionBundleApprovesPublicRelease = [bool](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "approvesPublicRelease" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease" -DefaultValue $false)))
$compatibleHostCollectionBundleCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "canPromoteRuntimeProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof" -DefaultValue $false)))
$compatibleHostCollectionBundleRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "isRuntimeExecutionEvidence" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence" -DefaultValue $false)))
$compatibleHostCollectionBundlePromotionBlockedReason = [string](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "promotionBlockedReason" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundlePromotionBlockedReason" -DefaultValue "missing-compatible-host-collection-bundle-blocked-reason")))
$compatibleHostCollectionBundleRunPackageConsumerSmokeCommand = [string](Get-PropertyOrDefault -Object $compatibleHostCollectionBundleCommands -Name "runPackageConsumerSmoke" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand" -DefaultValue $compatibleHostRunbookRunPackageConsumerSmokeCommand)))
$compatibleHostCollectionBundleValidateFilledRecordCommand = [string](Get-PropertyOrDefault -Object $compatibleHostCollectionBundleCommands -Name "validateFilledRecord" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand" -DefaultValue $compatibleHostRunbookValidateFilledRecordCommand)))
$compatibleHostCollectionBundleQuickStartCount = $compatibleHostCollectionBundleOperatorQuickStart.Count
$compatibleHostCollectionBundlePreflightCount = $compatibleHostCollectionBundlePreflightChecklist.Count
$compatibleHostCollectionBundleExecutionOrderCount = $compatibleHostCollectionBundleCopyableExecutionOrder.Count
$compatibleHostCollectionBundleOwnerInputArtifactCount = $compatibleHostCollectionBundleOwnerInputArtifacts.Count
$externalRuntimeProofStateFallback = if ($externalRuntimeProofValidation) { [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation") } elseif ($externalRuntimeProof) { [string](Get-PropertyOrDefault -Object $externalRuntimeProof -Name "proofState" -DefaultValue "missing-external-runtime-proof-record-template") } else { "missing-external-runtime-proof-validation" }
$externalRuntimeProofClassificationFallback = if ($externalRuntimeProofValidation) { [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "proofClassification" -DefaultValue "missing-proof-classification") } elseif ($externalRuntimeProof) { [string](Get-PropertyOrDefault -Object $externalRuntimeProof -Name "proofClassification" -DefaultValue "missing-proof-classification") } else { "missing-proof-classification" }
$externalRuntimeProofState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofState" -DefaultValue $externalRuntimeProofStateFallback)
$externalRuntimeProofClassification = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofClassification" -DefaultValue $externalRuntimeProofClassificationFallback)
$externalRuntimeProofRuntimePackageKeyMatches = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofRuntimePackageKeyMatches" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "runtimePackageKeyMatches" -DefaultValue $false)))
$externalRuntimeProofPackageSourceRuntimePackageKeyMatches = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofPackageSourceRuntimePackageKeyMatches" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "packageSourceRuntimePackageKeyMatches" -DefaultValue $false)))
$externalRuntimeProofConsumerProjectIdentityReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofConsumerProjectIdentityReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "consumerProjectIdentityReady" -DefaultValue $false)))
$externalRuntimeProofSmokeCommandRuntimeKeyReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofSmokeCommandRuntimeKeyReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "smokeCommandRuntimeKeyReady" -DefaultValue $false)))
$externalRuntimeProofHostReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofHostReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "hostReady" -DefaultValue $false)))
$externalRuntimeProofCommandsReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofCommandsReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "commandsReady" -DefaultValue $false)))
$externalRuntimeProofManagedNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofManagedNupkgSha256Ready" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "managedNupkgSha256Ready" -DefaultValue $false)))
$externalRuntimeProofRuntimeNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofRuntimeNupkgSha256Ready" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "runtimeNupkgSha256Ready" -DefaultValue $false)))
$externalRuntimeProofLogSha256FormatReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofLogSha256FormatReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "logSha256FormatReady" -DefaultValue $false)))
$externalRuntimeProofLogSha256Matches = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofLogSha256Matches" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "logSha256Matches" -DefaultValue $false)))
$externalRuntimeProofFailedProofItemCount = [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofFailedProofItemCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "failedProofItemCount" -DefaultValue -1)))
$externalRuntimeProofCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canPromoteRuntimeProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)))
$externalRuntimeProofOwnerActionStatus = if ($externalRuntimeProofCanPromoteRuntimeProof) { "resolved" } else { "owner-action-required" }
$externalRuntimeProofDraftState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftState" -DefaultValue "missing-external-runtime-proof-draft")
$externalRuntimeProofDraftCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftCanPromoteRuntimeProof" -DefaultValue $false)
$draftManagedNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftManagedNupkgSha256Ready" -DefaultValue $false)
$draftRuntimeNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftRuntimeNupkgSha256Ready" -DefaultValue $false)
$draftSmokeLogSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftLogSha256Ready" -DefaultValue $false)
$draftNoProjectReference = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftNoProjectReference" -DefaultValue $false)
$draftSmokeStatus = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftSmokeStatus" -DefaultValue "missing-smoke-status")
$externalRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeExecutionEvidence" -DefaultValue $false)
$compatibleHostRequired = $compatibleHostRunbookCompatibleHostRequired -or -not $externalRuntimeExecutionEvidence -or [string]::Equals($runtimeProofStatus, "blocked-by-cuda-driver", [System.StringComparison]::OrdinalIgnoreCase)
$requiredHostAction = "Run package consumer smoke on a CUDA-compatible host and replace the draft with a real external-runtime-proof-record."
$promotionBlockedReason = if ($externalRuntimeProofCanPromoteRuntimeProof -and $externalRuntimeExecutionEvidence) { "none" } elseif ([string]::Equals($runtimeProofStatus, "blocked-by-cuda-driver", [System.StringComparison]::OrdinalIgnoreCase) -or [string]::Equals($draftSmokeStatus, "blocked-by-cuda-driver", [System.StringComparison]::OrdinalIgnoreCase)) { "blocked-by-cuda-driver is not smoke passed" } else { "external runtime proof is not promotable package-consumer-runtime proof" }
$postPublishVerificationState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishVerificationState" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublish -Name "validationState" -DefaultValue "missing-post-publish-validation")))
$postPublishProofClassification = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishProofClassification" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublish -Name "postPublishProofClassification" -DefaultValue "missing-post-publish-proof-classification")))
$postPublishProofClassificationPromotable = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishProofClassificationPromotable" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "postPublishProofClassificationPromotable" -DefaultValue $false)))
$postPublishManagedNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishManagedNupkgSha256Ready" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "managedNupkgSha256Ready" -DefaultValue $false)))
$postPublishRuntimeNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishRuntimeNupkgSha256Ready" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "runtimeNupkgSha256Ready" -DefaultValue $false)))
$postPublishConsumerProjectIdentityReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishConsumerProjectIdentityReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "consumerProjectIdentityReady" -DefaultValue $false)))
$postPublishSmokeCommandRuntimeKeyReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishSmokeCommandRuntimeKeyReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "smokeCommandRuntimeKeyReady" -DefaultValue $false)))
$postPublishHostReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishHostReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "hostReady" -DefaultValue $false)))
$postPublishCommandsReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishCommandsReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "commandsReady" -DefaultValue $false)))
$postPublishStdoutStderrSummaryReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishStdoutStderrSummaryReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "stdoutStderrSummaryReady" -DefaultValue $false)))
$isPostPublishVerificationProof = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "isPostPublishVerificationProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "isPostPublishVerificationProof" -DefaultValue $false)))
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "canCloseReleaseIssue" -DefaultValue $false)))
$signingStatus = if ($finalRelease) { [string]$finalRelease.signingStatus } else { "missing" }
$callbackProof = if ($finalRelease) { [bool]$finalRelease.realCallbackRuntimeProof } else { $false }
$linuxProofState = if ($decisionRecord) { [string]$decisionRecord.linuxEvidenceState } else { "missing-owner-decision-record" }
$isRealLinuxRunnerProof = if ($decisionRecord) { [bool]$decisionRecord.isRealLinuxRunnerProof } else { $false }
$staleFindingCount = if ($staleAudit) { [int]$staleAudit.findingCount } else { -1 }
$finalPackageReviewState = [string](Get-PropertyOrDefault -Object $finalPackageReview -Name "bundleState" -DefaultValue "missing-final-package-review-bundle")
$finalPackageReviewPackageCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "packageCount" -DefaultValue 0)
$finalPackageReviewManagedPackageCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "managedPackageCount" -DefaultValue 0)
$finalPackageReviewRuntimePackageCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "runtimePackageCount" -DefaultValue 0)
$finalPackageReviewSplitRuntimePackageCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "splitRuntimePackageCount" -DefaultValue 0)
$finalPackageReviewNativeAssetCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "nativeAssetCount" -DefaultValue 0)
$finalPackageReviewCanUseAsPublicPackageProof = [bool](Get-PropertyOrDefault -Object $finalPackageReview -Name "canUseAsPublicPackageProof" -DefaultValue $false)
$externalRuntimeProofBackfillPlanState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofBackfillPlan -Name "planState" -DefaultValue "missing-external-runtime-proof-backfill-plan")
$externalRuntimeProofBackfillStepCount = @((Get-PropertyOrDefault -Object $externalRuntimeProofBackfillPlan -Name "backfillSteps" -DefaultValue @())).Count
$externalRuntimeProofBackfillCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofBackfillPlan -Name "canPromoteRuntimeProof" -DefaultValue $false)
$postPublishVerificationBackfillPlanState = [string](Get-PropertyOrDefault -Object $postPublishVerificationBackfillPlan -Name "planState" -DefaultValue "missing-post-publish-verification-backfill-plan")
$postPublishVerificationBackfillStepCount = @((Get-PropertyOrDefault -Object $postPublishVerificationBackfillPlan -Name "backfillSteps" -DefaultValue @())).Count
$postPublishVerificationBackfillCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $postPublishVerificationBackfillPlan -Name "canCloseReleaseIssue" -DefaultValue $false)

$decisionInputs = @(
  New-OwnerInput `
    -Id "release-channel" `
    -Title "Release channel decision" `
    -CurrentStatus "undecided" `
    -AllowedDecisionStates @("approved-public-channel", "approved-private-channel", "blocked") `
    -RequiredOwnerInput "Choose the target channel and confirm rollback policy." `
    -RequiredEvidence "docs/articles/zh-cn/nuget-and-github-packages-release-guide.md; artifacts/final-release/release-promotion-issue-record.md" `
    -Boundary "No public release exists until the chosen channel is pushed and verified." `
    -RequiredForPublicRelease $true
  New-OwnerInput `
    -Id "signing-policy" `
    -Title "Signing and trust decision" `
    -CurrentStatus $signingStatus `
    -AllowedDecisionStates @("approved-signed", "approved-unsigned-rc", "blocked") `
    -RequiredOwnerInput "Approve unsigned RC distribution or attach signing evidence." `
    -RequiredEvidence "docs/articles/zh-cn/signing-and-trust-policy.md; artifacts/final-release/final-release-dry-run-summary.json" `
    -Boundary "unsigned-or-not-requested is not signed output." `
    -RequiredForPublicRelease $true
  New-OwnerInput `
    -Id "nvidia-redistribution" `
    -Title "NVIDIA redistribution decision" `
    -CurrentStatus "pending-legal-or-owner-review" `
    -AllowedDecisionStates @("approved-for-selected-channel", "approved-private-only", "blocked") `
    -RequiredOwnerInput "Confirm whether CUDA, cuDNN, and TensorRT runtime files may be redistributed through the selected channel." `
    -RequiredEvidence "pack/runtime/runtime-packages.manifest.json; docs/articles/zh-cn/signing-and-trust-policy.md" `
    -Boundary "Do not publicly publish NVIDIA runtime components without redistribution approval." `
    -RequiredForPublicRelease $true
  New-OwnerInput `
    -Id "runtime-proof-disposition" `
    -Title "Runtime proof disposition" `
    -CurrentStatus ("runtimeProofStatus=" + $runtimeProofStatus + "; runtimeProofRequiredForRelease=" + $runtimeProofRequiredForRelease + "; packageConsumerSmokeStatus=" + $smokeStatus + "; allowRuntimeSmokeBlocked=" + $allowRuntimeSmokeBlocked + "; ownerAction=" + $runtimeProofBlockerOwnerActionStatus + "; blockerCategory=" + $runtimeProofBlockerCategory + "; externalRuntimeProofState=" + $externalRuntimeProofState + "; externalRuntimeProofClassification=" + $externalRuntimeProofClassification + "; externalRuntimeProofRuntimePackageKeyMatches=" + $externalRuntimeProofRuntimePackageKeyMatches + "; externalRuntimeProofPackageSourceRuntimePackageKeyMatches=" + $externalRuntimeProofPackageSourceRuntimePackageKeyMatches + "; externalRuntimeProofManagedNupkgSha256Ready=" + $externalRuntimeProofManagedNupkgSha256Ready + "; externalRuntimeProofRuntimeNupkgSha256Ready=" + $externalRuntimeProofRuntimeNupkgSha256Ready + "; externalRuntimeProofLogSha256FormatReady=" + $externalRuntimeProofLogSha256FormatReady + "; externalRuntimeProofLogSha256Matches=" + $externalRuntimeProofLogSha256Matches + "; externalRuntimeProofFailedProofItemCount=" + $externalRuntimeProofFailedProofItemCount + "; externalRuntimeProofOwnerAction=" + $externalRuntimeProofOwnerActionStatus + "; draftState=" + $externalRuntimeProofDraftState + "; draftManagedNupkgSha256Ready=" + $draftManagedNupkgSha256Ready + "; draftRuntimeNupkgSha256Ready=" + $draftRuntimeNupkgSha256Ready + "; draftSmokeLogSha256Ready=" + $draftSmokeLogSha256Ready + "; draftNoProjectReference=" + $draftNoProjectReference + "; draftSmokeStatus=" + $draftSmokeStatus + "; compatibleHostRequired=" + $compatibleHostRequired + "; compatibleHostRunbookState=" + $compatibleHostRunbookState + "; compatibleHostRunbookCanPromoteRuntimeProof=" + $compatibleHostRunbookCanPromoteRuntimeProof + "; compatibleHostRunbookRuntimeExecutionEvidence=" + $compatibleHostRunbookRuntimeExecutionEvidence + "; compatibleHostRunbookPerformsPublish=" + $compatibleHostRunbookPerformsPublish + "; compatibleHostRunbookApprovesPublicRelease=" + $compatibleHostRunbookApprovesPublicRelease + "; compatibleHostRunbookPromotionBlockedReason=" + $compatibleHostRunbookPromotionBlockedReason + "; promotionBlockedReason=" + $promotionBlockedReason) `
    -AllowedDecisionStates @("approved-runtime-proof-ready", "approved-known-limitation-for-rc", "rerun-required", "blocked") `
    -RequiredOwnerInput "Decide whether to require a compatible CUDA host rerun before promotion, or approve the current blocker as a documented RC limitation. If external proof is used, attach matching runtimePackageKey, matching packageSource.runtimePackageKey, managed/runtime nupkg SHA256, and a real smoke logSha256. Follow compatible-host-runtime-proof-runbook.json or compatible-host-runtime-proof-collection-bundle.json and run: $compatibleHostCollectionBundleRunPackageConsumerSmokeCommand" `
    -RequiredEvidence "artifacts/final-release/final-release-dry-run-summary.json; artifacts/package-readiness/runtime-package-readiness-summary.json; artifacts/package-consumer/package-consumer-validation-summary.json; artifacts/final-release/external-runtime-proof-validation.json; artifacts/final-release/compatible-host-runtime-proof-runbook.json; artifacts/final-release/compatible-host-runtime-proof-runbook.md; artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json; artifacts/final-release/compatible-host-runtime-proof-collection-bundle.md; docs/articles/zh-cn/cuda-error-35-troubleshooting.md" `
    -Boundary "blocked-by-cuda-driver, dependency-probe-only, runtime-deserialization-dependency-diagnostics, runtime-key-mismatched, package-source-key-mismatched, missing-package-hash, missing-log-hash, compatible-host-runtime-proof-runbook, compatible-host-runtime-proof-collection-bundle, and runtimeProofRequiredForRelease=true are not smoke passed." `
    -RequiredForPublicRelease $true
  New-OwnerInput `
    -Id "linux-runner-proof-disposition" `
    -Title "Linux runner proof disposition" `
    -CurrentStatus ("linuxEvidenceState=" + $linuxProofState + "; isRealLinuxRunnerProof=" + $isRealLinuxRunnerProof) `
    -AllowedDecisionStates @("approved-real-linux-proof", "approved-windows-only-rc", "require-before-public", "blocked") `
    -RequiredOwnerInput "Decide whether Linux proof is required before publication, or whether this RC is Windows-only." `
    -RequiredEvidence "artifacts/linux-dry-run/$LinuxRuntimePackageKey; docs/articles/zh-cn/linux-runner-evidence-record-schema.md" `
    -Boundary "template-only and handoff-only are not Linux runner proof." `
    -RequiredForPublicRelease $true
  New-OwnerInput `
    -Id "final-package-review-acknowledgement" `
    -Title "Final package review acknowledgement" `
    -CurrentStatus ("bundleState=" + $finalPackageReviewState + "; packageCount=" + $finalPackageReviewPackageCount + "; managedPackageCount=" + $finalPackageReviewManagedPackageCount + "; runtimePackageCount=" + $finalPackageReviewRuntimePackageCount + "; splitRuntimePackageCount=" + $finalPackageReviewSplitRuntimePackageCount + "; nativeAssetCount=" + $finalPackageReviewNativeAssetCount + "; canUseAsPublicPackageProof=" + $finalPackageReviewCanUseAsPublicPackageProof) `
    -AllowedDecisionStates @("acknowledged-local-package-inventory", "require-package-rebuild", "blocked") `
    -RequiredOwnerInput "Acknowledge that package ids, versions, SHA256 hashes, and native asset counts in the final package review were reviewed as local package inventory only." `
    -RequiredEvidence "artifacts/final-release/final-package-review-bundle.json; artifacts/final-release/final-package-review-bundle.md" `
    -Boundary "Final package review enumerates local package files and native assets for owner review; it is not public channel proof, runtime execution proof, publication approval, or post-publish verification." `
    -RequiredForPublicRelease $true
  New-OwnerInput `
    -Id "post-publish-verification-disposition" `
    -Title "Post-publish verification disposition" `
    -CurrentStatus ("verificationState=" + $postPublishVerificationState + "; classification=" + $postPublishProofClassification + "; promotable=" + $postPublishProofClassificationPromotable + "; managedNupkgSha256Ready=" + $postPublishManagedNupkgSha256Ready + "; runtimeNupkgSha256Ready=" + $postPublishRuntimeNupkgSha256Ready + "; consumerProjectIdentityReady=" + $postPublishConsumerProjectIdentityReady + "; smokeCommandRuntimeKeyReady=" + $postPublishSmokeCommandRuntimeKeyReady + "; hostReady=" + $postPublishHostReady + "; commandsReady=" + $postPublishCommandsReady + "; stdoutStderrSummaryReady=" + $postPublishStdoutStderrSummaryReady + "; isProof=" + $isPostPublishVerificationProof + "; canCloseReleaseIssue=" + $canCloseReleaseIssue) `
    -AllowedDecisionStates @("acknowledge-post-publish-required", "require-before-closing-issue", "blocked") `
    -RequiredOwnerInput "Acknowledge that release issue closure requires a real post-publish verification record after the selected channel is pushed and package identities/hashes are captured." `
    -RequiredEvidence "artifacts/final-release/post-publish-verification-validation.json; docs/articles/zh-cn/post-publish-verification-record.md" `
    -Boundary "Template, draft, missing package hashes, missing clean consumer identity, missing host metadata, missing command capture, missing --runtime-package-key smoke command, and missing stdout/stderr summaries cannot close the release issue." `
    -RequiredForPublicRelease $false
  New-OwnerInput `
    -Id "backfill-plan-boundary-acknowledgement" `
    -Title "Backfill plan boundary acknowledgement" `
    -CurrentStatus ("externalRuntimeProofBackfillPlanState=" + $externalRuntimeProofBackfillPlanState + "; externalRuntimeProofBackfillStepCount=" + $externalRuntimeProofBackfillStepCount + "; externalRuntimeProofBackfillCanPromoteRuntimeProof=" + $externalRuntimeProofBackfillCanPromoteRuntimeProof + "; postPublishVerificationBackfillPlanState=" + $postPublishVerificationBackfillPlanState + "; postPublishVerificationBackfillStepCount=" + $postPublishVerificationBackfillStepCount + "; postPublishVerificationBackfillCanCloseReleaseIssue=" + $postPublishVerificationBackfillCanCloseReleaseIssue) `
    -AllowedDecisionStates @("acknowledged-guidance-only", "require-proof-backfill-before-publish", "blocked") `
    -RequiredOwnerInput "Acknowledge that external runtime and post-publish backfill plans are execution guidance only, or require real proof backfill before publication." `
    -RequiredEvidence "artifacts/final-release/external-runtime-proof-backfill-plan.json; artifacts/final-release/external-runtime-proof-backfill-plan.md; artifacts/final-release/post-publish-verification-backfill-plan.json; artifacts/final-release/post-publish-verification-backfill-plan.md" `
    -Boundary "Backfill plans are guidance only; they are not runtime proof, post-publish proof, publication approval, release close approval, or package push." `
    -RequiredForPublicRelease $true
  New-OwnerInput `
    -Id "callback-proof-disposition" `
    -Title "Callback proof disposition" `
    -CurrentStatus ("realCallbackRuntimeProof=" + $callbackProof) `
    -AllowedDecisionStates @("approved-real-callback-proof", "approved-known-limitation-for-rc", "require-before-public", "blocked") `
    -RequiredOwnerInput "Decide whether callback proof=false is accepted as a known limitation for this RC." `
    -RequiredEvidence "artifacts/release-candidate/release-candidate-readiness-summary.json; docs/articles/zh-cn/real-callback-runtime-evidence-schema.md" `
    -Boundary "schema-ready, precheck, design gate, and InvocationCount=0 are not real callback runtime proof." `
    -RequiredForPublicRelease $true
)

$template = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "release-owner-approval-input-template"
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  templateOnly = $true
  approvalState = "pending-release-owner-input"
  canPublishPublicly = $false
  requestedCanPublishPublicly = $false
  ownerName = ""
  ownerDecisionId = ""
  overallStatus = $overallStatus
  blockingIssueCount = $blockingIssueCount
  manualApprovalCount = $manualApprovalCount
  staleReleaseClaimsFindingCount = $staleFindingCount
  runtimeProofStatus = $runtimeProofStatus
  runtimeProofRequiredForRelease = $runtimeProofRequiredForRelease
  runtimeProofBlockerOwnerActionStatus = $runtimeProofBlockerOwnerActionStatus
  runtimeProofBlockerCategory = $runtimeProofBlockerCategory
  runtimeProofOwnerCommand = $runtimeProofOwnerCommand
  externalRuntimeProofState = $externalRuntimeProofState
  externalRuntimeProofClassification = $externalRuntimeProofClassification
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
  externalRuntimeProofOwnerActionStatus = $externalRuntimeProofOwnerActionStatus
  externalRuntimeProofCanPromoteRuntimeProof = $externalRuntimeProofCanPromoteRuntimeProof
  externalRuntimeExecutionEvidence = $externalRuntimeExecutionEvidence
  externalRuntimeProofDraftState = $externalRuntimeProofDraftState
  externalRuntimeProofDraftCanPromoteRuntimeProof = $externalRuntimeProofDraftCanPromoteRuntimeProof
  draftManagedNupkgSha256Ready = $draftManagedNupkgSha256Ready
  draftRuntimeNupkgSha256Ready = $draftRuntimeNupkgSha256Ready
  draftSmokeLogSha256Ready = $draftSmokeLogSha256Ready
  draftNoProjectReference = $draftNoProjectReference
  draftSmokeStatus = $draftSmokeStatus
  compatibleHostRequired = $compatibleHostRequired
  compatibleHostRuntimeProofRunbookState = $compatibleHostRunbookState
  compatibleHostRuntimeProofRunbookCompatibleHostRequired = $compatibleHostRunbookCompatibleHostRequired
  compatibleHostRuntimeProofRunbookPerformsPublish = $compatibleHostRunbookPerformsPublish
  compatibleHostRuntimeProofRunbookApprovesPublicRelease = $compatibleHostRunbookApprovesPublicRelease
  compatibleHostRuntimeProofRunbookCanPromoteRuntimeProof = $compatibleHostRunbookCanPromoteRuntimeProof
  compatibleHostRuntimeProofRunbookRuntimeExecutionEvidence = $compatibleHostRunbookRuntimeExecutionEvidence
  compatibleHostRuntimeProofRunbookPromotionBlockedReason = $compatibleHostRunbookPromotionBlockedReason
  compatibleHostRuntimeProofRunbookRunPackageConsumerSmokeCommand = $compatibleHostRunbookRunPackageConsumerSmokeCommand
  compatibleHostRuntimeProofRunbookValidateFilledRecordCommand = $compatibleHostRunbookValidateFilledRecordCommand
  compatibleHostRuntimeProofCollectionBundleState = $compatibleHostCollectionBundleState
  compatibleHostRuntimeProofCollectionBundleCompatibleHostRequired = $compatibleHostCollectionBundleCompatibleHostRequired
  compatibleHostRuntimeProofCollectionBundlePerformsPublish = $compatibleHostCollectionBundlePerformsPublish
  compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease = $compatibleHostCollectionBundleApprovesPublicRelease
  compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof = $compatibleHostCollectionBundleCanPromoteRuntimeProof
  compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence = $compatibleHostCollectionBundleRuntimeExecutionEvidence
  compatibleHostRuntimeProofCollectionBundlePromotionBlockedReason = $compatibleHostCollectionBundlePromotionBlockedReason
  compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand = $compatibleHostCollectionBundleRunPackageConsumerSmokeCommand
  compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand = $compatibleHostCollectionBundleValidateFilledRecordCommand
  compatibleHostRuntimeProofCollectionBundleQuickStartCount = $compatibleHostCollectionBundleQuickStartCount
  compatibleHostRuntimeProofCollectionBundlePreflightChecklistCount = $compatibleHostCollectionBundlePreflightCount
  compatibleHostRuntimeProofCollectionBundleCopyableExecutionOrderCount = $compatibleHostCollectionBundleExecutionOrderCount
  compatibleHostRuntimeProofCollectionBundleOperatorQuickStart = @($compatibleHostCollectionBundleOperatorQuickStart)
  compatibleHostRuntimeProofCollectionBundlePreflightChecklist = @($compatibleHostCollectionBundlePreflightChecklist)
  compatibleHostRuntimeProofCollectionBundleCopyableExecutionOrder = @($compatibleHostCollectionBundleCopyableExecutionOrder)
  compatibleHostRuntimeProofCollectionBundleOwnerInputArtifactCount = $compatibleHostCollectionBundleOwnerInputArtifactCount
  compatibleHostRuntimeProofCollectionBundleOwnerInputArtifacts = @($compatibleHostCollectionBundleOwnerInputArtifacts)
  requiredHostAction = $requiredHostAction
  promotionBlockedReason = $promotionBlockedReason
  finalPackageReviewState = $finalPackageReviewState
  finalPackageReviewPackageCount = $finalPackageReviewPackageCount
  finalPackageReviewManagedPackageCount = $finalPackageReviewManagedPackageCount
  finalPackageReviewRuntimePackageCount = $finalPackageReviewRuntimePackageCount
  finalPackageReviewSplitRuntimePackageCount = $finalPackageReviewSplitRuntimePackageCount
  finalPackageReviewNativeAssetCount = $finalPackageReviewNativeAssetCount
  finalPackageReviewCanUseAsPublicPackageProof = $finalPackageReviewCanUseAsPublicPackageProof
  externalRuntimeProofBackfillPlanState = $externalRuntimeProofBackfillPlanState
  externalRuntimeProofBackfillStepCount = $externalRuntimeProofBackfillStepCount
  externalRuntimeProofBackfillCanPromoteRuntimeProof = $externalRuntimeProofBackfillCanPromoteRuntimeProof
  postPublishVerificationBackfillPlanState = $postPublishVerificationBackfillPlanState
  postPublishVerificationBackfillStepCount = $postPublishVerificationBackfillStepCount
  postPublishVerificationBackfillCanCloseReleaseIssue = $postPublishVerificationBackfillCanCloseReleaseIssue
  postPublishVerificationState = $postPublishVerificationState
  postPublishProofClassification = $postPublishProofClassification
  postPublishProofClassificationPromotable = $postPublishProofClassificationPromotable
  postPublishManagedNupkgSha256Ready = $postPublishManagedNupkgSha256Ready
  postPublishRuntimeNupkgSha256Ready = $postPublishRuntimeNupkgSha256Ready
  postPublishConsumerProjectIdentityReady = $postPublishConsumerProjectIdentityReady
  postPublishSmokeCommandRuntimeKeyReady = $postPublishSmokeCommandRuntimeKeyReady
  postPublishHostReady = $postPublishHostReady
  postPublishCommandsReady = $postPublishCommandsReady
  postPublishStdoutStderrSummaryReady = $postPublishStdoutStderrSummaryReady
  isPostPublishVerificationProof = $isPostPublishVerificationProof
  canCloseReleaseIssue = $canCloseReleaseIssue
  packageConsumerSmokeStatus = $smokeStatus
  realCallbackRuntimeProof = $callbackProof
  isRealLinuxRunnerProof = $isRealLinuxRunnerProof
  sourceDecisionRecordPresent = $null -ne $decisionRecord
  sourcePromotionIssuePresent = $null -ne $promotionIssue
  requiredDecisionIds = @($decisionInputs | ForEach-Object { $_.id })
  decisionInputs = $decisionInputs
  validationCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerApprovalInput.ps1 -InputPath artifacts\final-release\release-owner-approval-input-record.json"
  sourceEvidence = @(
    "artifacts/final-release/compatible-host-runtime-proof-runbook.json",
    "artifacts/final-release/compatible-host-runtime-proof-runbook.md",
    "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json",
    "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.md",
    "artifacts/final-release/release-candidate-package-inventory.json",
    "artifacts/final-release/release-package-proof-bundle.json",
    "artifacts/final-release/final-package-review-bundle.json",
    "artifacts/local-feed-consumer/local-nuget-feed-consumer-summary.json",
    "artifacts/final-release/external-runtime-proof-backfill-plan.json",
    "artifacts/final-release/external-runtime-proof-backfill-plan.md",
    "artifacts/final-release/post-publish-verification-backfill-plan.json",
    "artifacts/final-release/post-publish-verification-backfill-plan.md",
    "artifacts/final-release/final-package-review-bundle.json",
    "artifacts/final-release/final-package-review-bundle.md",
    "artifacts/final-release/final-release-dry-run-summary.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/package-consumer/package-consumer-validation-summary.json"
  )
  safetyNotes = @(
    "This file is a template and cannot approve publication.",
    "Change recordKind to release-owner-approval-input-record only after a real owner fills it.",
    "Keep canPublishPublicly=false unless every required decision is explicitly resolved.",
    "runtimeProofRequiredForRelease=true must be accepted as a limitation or resolved by runtime proof before promotion.",
    "blocked-by-cuda-driver is not smoke passed.",
    "compatible-host-runtime-proof-runbook is an owner-action runbook, not runtime proof.",
    "compatible-host-runtime-proof-collection-bundle is an external execution package, not runtime proof, publication approval, or package push.",
    "Backfill plans are guidance only; they are not runtime proof, post-publish proof, publication approval, release close approval, or package push.",
    "compatibleHostRuntimeProofRunbookCanPromoteRuntimeProof=false keeps runtime proof owner action required.",
    "compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof=false keeps runtime proof owner action required.",
    "compatibleHostRuntimeProofCollectionBundleOwnerInputArtifacts are owner inputs only; package inventory, final package review, release package proof, and local feed consumer summaries cannot promote runtime proof or post-publish proof.",
    "final-package-review-bundle is local package inventory for owner review; it is not public package proof, runtime proof, publication approval, or post-publish verification.",
    "externalRuntimeProofLogSha256FormatReady=false, externalRuntimeProofLogSha256Matches=false, externalRuntimeProofRuntimePackageKeyMatches=false, externalRuntimeProofPackageSourceRuntimePackageKeyMatches=false, externalRuntimeProofConsumerProjectIdentityReady=false, externalRuntimeProofSmokeCommandRuntimeKeyReady=false, externalRuntimeProofHostReady=false, externalRuntimeProofCommandsReady=false, externalRuntimeProofManagedNupkgSha256Ready=false, or externalRuntimeProofRuntimeNupkgSha256Ready=false keeps runtime proof owner action required.",
    "postPublishConsumerProjectIdentityReady=false, postPublishSmokeCommandRuntimeKeyReady=false, postPublishHostReady=false, postPublishCommandsReady=false, postPublishStdoutStderrSummaryReady=false, postPublishManagedNupkgSha256Ready=false, or postPublishRuntimeNupkgSha256Ready=false means the release issue cannot be closed as post-publish verified.",
    "runtime-deserialization-dependency-diagnostics and runtime proof blocker owner action are guidance, not runtime execution proof.",
    "template-only Linux evidence is not Linux runner proof.",
    "realCallbackRuntimeProof=false must remain visible until InvocationCount>0 evidence exists."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "release-owner-approval-input-template.json"
$markdownPath = Join-Path $outputRoot "release-owner-approval-input-template.md"

$template | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Owner Approval Input Template")
$lines.Add("")
$lines.Add("Runtime key: ``$RuntimePackageKey``")
$lines.Add("")
$lines.Add("Linux runtime key: ``$LinuxRuntimePackageKey``")
$lines.Add("")
$lines.Add("Approval state: ``pending-release-owner-input``")
$lines.Add("")
$lines.Add("Can publish publicly: ``false``")
$lines.Add("")
$lines.Add("This file is a template for human owner input. It cannot approve publication by itself.")
$lines.Add("")
$lines.Add("## Evidence Snapshot")
$lines.Add("")
$lines.Add("- overall status: ``$overallStatus``")
$lines.Add("- blocking issues: $blockingIssueCount")
$lines.Add("- manual approvals: $manualApprovalCount")
$lines.Add("- stale release claim findings: $staleFindingCount")
$lines.Add("- runtime proof status: ``$runtimeProofStatus``")
$lines.Add("- runtime proof required for release: ``$runtimeProofRequiredForRelease``")
$lines.Add("- runtime proof blocker owner action: ``$runtimeProofBlockerOwnerActionStatus``")
$lines.Add("- runtime proof blocker category: ``$runtimeProofBlockerCategory``")
$lines.Add("- runtime proof suggested command: ``$runtimeProofOwnerCommand``")
$lines.Add("- external runtime proof state: ``$externalRuntimeProofState``")
$lines.Add("- external runtime proof classification: ``$externalRuntimeProofClassification``")
$lines.Add("- external runtime proof runtime key matches: ``$externalRuntimeProofRuntimePackageKeyMatches``")
$lines.Add("- external runtime proof package source runtime key matches: ``$externalRuntimeProofPackageSourceRuntimePackageKeyMatches``")
$lines.Add("- external runtime proof consumer project identity ready: ``$externalRuntimeProofConsumerProjectIdentityReady``")
$lines.Add("- external runtime proof smoke command runtime key ready: ``$externalRuntimeProofSmokeCommandRuntimeKeyReady``")
$lines.Add("- external runtime proof host ready: ``$externalRuntimeProofHostReady``")
$lines.Add("- external runtime proof commands ready: ``$externalRuntimeProofCommandsReady``")
$lines.Add("- external runtime proof managed nupkg SHA256 ready: ``$externalRuntimeProofManagedNupkgSha256Ready``")
$lines.Add("- external runtime proof runtime nupkg SHA256 ready: ``$externalRuntimeProofRuntimeNupkgSha256Ready``")
$lines.Add("- external runtime proof log SHA256 format ready: ``$externalRuntimeProofLogSha256FormatReady``")
$lines.Add("- external runtime proof log SHA256 matches: ``$externalRuntimeProofLogSha256Matches``")
$lines.Add("- external runtime proof failed proof item count: ``$externalRuntimeProofFailedProofItemCount``")
$lines.Add("- external runtime proof owner action: ``$externalRuntimeProofOwnerActionStatus``")
$lines.Add("- external runtime proof draft state: ``$externalRuntimeProofDraftState``")
$lines.Add("- draft managed nupkg SHA256 ready: ``$draftManagedNupkgSha256Ready``")
$lines.Add("- draft runtime nupkg SHA256 ready: ``$draftRuntimeNupkgSha256Ready``")
$lines.Add("- draft smoke log SHA256 ready: ``$draftSmokeLogSha256Ready``")
$lines.Add("- draft no ProjectReference: ``$draftNoProjectReference``")
$lines.Add("- draft smoke status: ``$draftSmokeStatus``")
$lines.Add("- compatible host required: ``$compatibleHostRequired``")
$lines.Add("- compatible host runtime proof runbook: ``$compatibleHostRunbookState``")
$lines.Add("- compatible host runbook can promote proof: ``$compatibleHostRunbookCanPromoteRuntimeProof``")
$lines.Add("- compatible host runbook runtime execution evidence: ``$compatibleHostRunbookRuntimeExecutionEvidence``")
$lines.Add("- compatible host runbook performs publish: ``$compatibleHostRunbookPerformsPublish``")
$lines.Add("- compatible host runbook approves public release: ``$compatibleHostRunbookApprovesPublicRelease``")
$lines.Add("- compatible host runbook smoke command: ``$compatibleHostRunbookRunPackageConsumerSmokeCommand``")
$lines.Add("- compatible host runbook validation command: ``$compatibleHostRunbookValidateFilledRecordCommand``")
$lines.Add("- compatible host runtime proof collection bundle: ``$compatibleHostCollectionBundleState``")
$lines.Add("- compatible host collection bundle can promote proof: ``$compatibleHostCollectionBundleCanPromoteRuntimeProof``")
$lines.Add("- compatible host collection bundle runtime execution evidence: ``$compatibleHostCollectionBundleRuntimeExecutionEvidence``")
$lines.Add("- compatible host collection bundle performs publish: ``$compatibleHostCollectionBundlePerformsPublish``")
$lines.Add("- compatible host collection bundle approves public release: ``$compatibleHostCollectionBundleApprovesPublicRelease``")
$lines.Add("- compatible host collection bundle quick start items: ``$compatibleHostCollectionBundleQuickStartCount``")
$lines.Add("- compatible host collection bundle preflight items: ``$compatibleHostCollectionBundlePreflightCount``")
$lines.Add("- compatible host collection bundle copyable execution order items: ``$compatibleHostCollectionBundleExecutionOrderCount``")
$lines.Add("- compatible host collection bundle smoke command: ``$compatibleHostCollectionBundleRunPackageConsumerSmokeCommand``")
$lines.Add("- compatible host collection bundle validation command: ``$compatibleHostCollectionBundleValidateFilledRecordCommand``")
$lines.Add("- required host action: $requiredHostAction")
$lines.Add("- promotion blocked reason: $promotionBlockedReason")
$lines.Add("- final package review state: ``$finalPackageReviewState``")
$lines.Add("- final package review package count: ``$finalPackageReviewPackageCount``")
$lines.Add("- final package review managed package count: ``$finalPackageReviewManagedPackageCount``")
$lines.Add("- final package review runtime package count: ``$finalPackageReviewRuntimePackageCount``")
$lines.Add("- final package review split runtime package count: ``$finalPackageReviewSplitRuntimePackageCount``")
$lines.Add("- final package review native asset count: ``$finalPackageReviewNativeAssetCount``")
$lines.Add("- final package review can use as public package proof: ``$finalPackageReviewCanUseAsPublicPackageProof``")
$lines.Add("- external runtime proof backfill plan state: ``$externalRuntimeProofBackfillPlanState``")
$lines.Add("- external runtime proof backfill step count: ``$externalRuntimeProofBackfillStepCount``")
$lines.Add("- external runtime proof backfill can promote runtime proof: ``$externalRuntimeProofBackfillCanPromoteRuntimeProof``")
$lines.Add("- post-publish verification backfill plan state: ``$postPublishVerificationBackfillPlanState``")
$lines.Add("- post-publish verification backfill step count: ``$postPublishVerificationBackfillStepCount``")
$lines.Add("- post-publish verification backfill can close release issue: ``$postPublishVerificationBackfillCanCloseReleaseIssue``")
$lines.Add("- post-publish verification state: ``$postPublishVerificationState``")
$lines.Add("- post-publish proof classification: ``$postPublishProofClassification``")
$lines.Add("- post-publish proof classification promotable: ``$postPublishProofClassificationPromotable``")
$lines.Add("- post-publish managed nupkg SHA256 ready: ``$postPublishManagedNupkgSha256Ready``")
$lines.Add("- post-publish runtime nupkg SHA256 ready: ``$postPublishRuntimeNupkgSha256Ready``")
$lines.Add("- post-publish consumer project identity ready: ``$postPublishConsumerProjectIdentityReady``")
$lines.Add("- post-publish smoke command runtime key ready: ``$postPublishSmokeCommandRuntimeKeyReady``")
$lines.Add("- post-publish host ready: ``$postPublishHostReady``")
$lines.Add("- post-publish commands ready: ``$postPublishCommandsReady``")
$lines.Add("- post-publish stdout/stderr summary ready: ``$postPublishStdoutStderrSummaryReady``")
$lines.Add("- post-publish verification proof: ``$isPostPublishVerificationProof``")
$lines.Add("- can close release issue: ``$canCloseReleaseIssue``")
$lines.Add("- package consumer smoke: ``$smokeStatus``")
$lines.Add("- real callback runtime proof: ``$callbackProof``")
$lines.Add("- real Linux runner proof: ``$isRealLinuxRunnerProof``")
$lines.Add("")
$lines.Add("## Compatible Host Collection Bundle Execution")
$lines.Add("")
$lines.Add("This section mirrors the collection bundle for release owners. It is executable guidance only; it is not runtime proof, publication approval, or package push.")
$lines.Add("")
$lines.Add("### Operator Quick Start")
$lines.Add("")
foreach ($item in $compatibleHostCollectionBundleOperatorQuickStart) {
  $lines.Add("- $item")
}
$lines.Add("")
$lines.Add("### Preflight Checklist")
$lines.Add("")
$lines.Add("| ID | Required input | Why |")
$lines.Add("| --- | --- | --- |")
foreach ($item in $compatibleHostCollectionBundlePreflightChecklist) {
  $lines.Add("| ``$($item.id)`` | $($item.required.Replace("|", "\|")) | $($item.why.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("### Copyable Execution Order")
$lines.Add("")
$lines.Add('```powershell')
foreach ($command in $compatibleHostCollectionBundleCopyableExecutionOrder) {
  $lines.Add($command)
}
$lines.Add('```')
$lines.Add("")
$lines.Add("### Owner Input Artifacts")
$lines.Add("")
$lines.Add("| ID | Path | State | Boundary |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($artifact in $compatibleHostCollectionBundleOwnerInputArtifacts) {
  $lines.Add("| ``$($artifact.id)`` | ``$($artifact.path)`` | $($artifact.state.Replace("|", "\|")) | $($artifact.boundary.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Required Owner Inputs")
$lines.Add("")
$lines.Add("| ID | Current status | Allowed states | Required input | Boundary |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($decision in $decisionInputs) {
  $allowed = ($decision.allowedDecisionStates -join ", ")
  $lines.Add("| ``$($decision.id)`` | ``$($decision.currentStatus)`` | ``$allowed`` | $($decision.requiredOwnerInput.Replace("|", "\|")) | $($decision.boundary.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $template.safetyNotes) {
  $lines.Add("- $note")
}
$lines.Add("")
$lines.Add("## Source Evidence")
$lines.Add("")
foreach ($item in $template.sourceEvidence) {
  $lines.Add("- ``$item``")
}
$lines.Add("")
$lines.Add("## Validation")
$lines.Add("")
$lines.Add("After copying this template to ``release-owner-approval-input-record.json`` and filling it, run:")
$lines.Add("")
$lines.Add('```powershell')
$lines.Add($template.validationCommand)
$lines.Add('```')

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release owner approval input template written to $jsonPath"
Write-Host "Release owner approval input template written to $markdownPath"
